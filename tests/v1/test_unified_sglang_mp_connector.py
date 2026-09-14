# SPDX-License-Identifier: Apache-2.0
"""Small, server-free tests for the unified SGLang MP connector contracts."""

# Standard
from types import ModuleType
from unittest.mock import Mock, patch
import threading
import unittest

# Third Party
import torch

# First Party
from lmcache.integration.sglang.lmcache_mp_metadata import (
    LMCacheLoadOperation,
    LMCacheLookupOperation,
    SGLangKVComponentGroup,
)
from lmcache.integration.sglang.unified_lmcache_mp_connector import (
    UnifiedLMCacheMPConnector,
)
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format import detect_format
import lmcache.lmcache_native as lmcache_native


class _Future:
    def __init__(self, ready: bool):
        self.ready = ready

    def query(self):
        return self.ready

    def retain_reference(self, value):
        self.value = value


class _ResultFuture(_Future):
    def __init__(self, ready: bool, value):
        super().__init__(ready)
        self.value = value

    def result(self, timeout=None):
        del timeout
        if not self.ready:
            raise TimeoutError("future is not ready")
        return self.value

    def wait_on_stream(self, stream, timeout=None):
        del timeout
        self.waited_stream = stream
        return self.result()

    def prepare(self, timeout=None):
        return self.result(timeout)


class _TransferContext:
    def __init__(self):
        self.store_args = None
        self.retrieve_args = None

    def submit_store(self, *args):
        self.store_args = args
        return _Future(True)

    def submit_retrieve(self, *args, **kwargs):
        self.retrieve_args = (args, kwargs)
        return _Future(True)


class _IPCCacheServerKey:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _new_connector() -> UnifiedLMCacheMPConnector:
    """Create a healthy connector shell for server-free unit tests."""
    connector = object.__new__(UnifiedLMCacheMPConnector)
    connector._health_event = threading.Event()
    connector._health_event.set()
    connector.tp_size = 1
    connector.pp_size = 1
    connector.tp_group = None
    connector.pp_group = None
    return connector


class TestUnifiedLMCacheMPConnector(unittest.TestCase):
    def setUp(self):
        heartbeat_patcher = patch.object(
            UnifiedLMCacheMPConnector, "_ensure_heartbeat_started"
        )
        heartbeat_patcher.start()
        self.addCleanup(heartbeat_patcher.stop)
        self.connector = _new_connector()
        self.connector.page_size = 4

    def test_slots_to_blocks_accepts_noncontiguous_pages(self):
        slots = torch.tensor([4, 5, 6, 7, 12, 13, 14, 15])
        self.assertEqual(self.connector._slots_to_blocks(slots), [1, 3])

    def test_pp_parallel_geometry_uses_global_worker_ids(self):
        actual = []
        for pp_rank in range(2):
            for tp_rank in range(2):
                geometry = self.connector._resolve_parallel_geometry(
                    2, tp_rank, 2, pp_rank
                )
                actual.append((geometry[4], geometry[5]))

        self.assertEqual(actual, [(4, 0), (4, 1), (4, 2), (4, 3)])

    def test_mla_kv_geometry_collapses_only_tp_object_identity(self):
        actual = []
        for pp_rank in range(2):
            for tp_rank in range(2):
                geometry = self.connector._resolve_kv_geometry(
                    2, tp_rank, 2, pp_rank, mla_only=True
                )
                actual.append(geometry)

        self.assertEqual(actual, [(2, 0), (2, 0), (2, 1), (2, 1)])

    def test_mla_only_allows_full_and_swa_attention_groups(self):
        groups = [
            SGLangKVComponentGroup("full", ()),
            SGLangKVComponentGroup("swa", (), sliding_window_size=4096),
        ]

        self.assertTrue(self.connector._is_mla_only(True, groups))

    def test_mla_only_rejects_recurrent_hybrid_groups(self):
        groups = [
            SGLangKVComponentGroup("full", ()),
            SGLangKVComponentGroup("mamba", (), recurrent_state=True),
        ]

        self.assertFalse(self.connector._is_mla_only(True, groups))
        self.assertFalse(self.connector._is_mla_only(False, groups[:1]))

    def test_parallel_geometry_rejects_invalid_pp_rank(self):
        with self.assertRaisesRegex(ValueError, "Invalid LMCache PP topology"):
            self.connector._resolve_parallel_geometry(2, 0, 2, 2)

    def test_parallel_all_reduce_covers_tp_then_pp(self):
        self.connector.tp_size = 2
        self.connector.pp_size = 2
        self.connector.tp_group = object()
        self.connector.pp_group = object()
        value = torch.tensor([1], dtype=torch.int32)

        with patch.object(torch.distributed, "all_reduce") as all_reduce:
            self.connector._parallel_all_reduce(value, torch.distributed.ReduceOp.MIN)

        self.assertEqual(all_reduce.call_count, 2)
        self.assertIs(
            all_reduce.call_args_list[0].kwargs["group"], self.connector.tp_group
        )
        self.assertIs(
            all_reduce.call_args_list[1].kwargs["group"], self.connector.pp_group
        )

    def test_health_check_marks_connector_unhealthy_on_ping_failure(self):
        connector = _new_connector()
        connector._heartbeat_stop = threading.Event()
        connector._closed = False
        connector._heartbeat_interval = 1.0
        connector.instance_id = 7
        connector._req_client = Mock()
        connector._req_client.ping.return_value.result.side_effect = TimeoutError
        connector._reregister_kv_cache_after_recovery = Mock(return_value=True)

        connector._run_health_check()

        self.assertFalse(connector.is_healthy)
        connector._reregister_kv_cache_after_recovery.assert_not_called()

    def test_health_check_reregisters_before_marking_recovered(self):
        connector = _new_connector()
        connector._health_event.clear()
        connector._heartbeat_stop = threading.Event()
        connector._closed = False
        connector._heartbeat_interval = 1.0
        connector.instance_id = 7
        connector._req_client = Mock()
        connector._req_client.ping.return_value.result.return_value = True
        connector._reregister_kv_cache_after_recovery = Mock(return_value=True)

        connector._run_health_check()

        connector._reregister_kv_cache_after_recovery.assert_called_once_with()
        self.assertTrue(connector.is_healthy)

    def test_failed_recovery_keeps_connector_unhealthy(self):
        connector = _new_connector()
        connector._health_event.clear()
        connector._heartbeat_stop = threading.Event()
        connector._closed = False
        connector._heartbeat_interval = 1.0
        connector.instance_id = 7
        connector._req_client = Mock()
        connector._req_client.ping.return_value.result.return_value = True
        connector._reregister_kv_cache_after_recovery = Mock(return_value=False)

        connector._run_health_check()

        self.assertFalse(connector.is_healthy)

    def test_recovery_reregisters_and_resets_store_progress(self):
        connector = _new_connector()
        connector._heartbeat_stop = threading.Event()
        connector._closed = False
        connector._registered = True
        connector._send_register_kv_cache_request = Mock()
        lookup = LMCacheLookupOperation(
            request_id="request",
            token_ids=[1, 2, 3, 4],
            local_hit_tokens=0,
            cache_salt="",
            submission_future=object(),
            completion_future=object(),
            locks_held=True,
        )
        connector._lookups = {lookup.request_id: lookup}
        connector._active_sessions = {lookup.request_id}
        connector._store_submitted_tokens = {lookup.request_id: 4}

        connector._mark_unhealthy()
        self.assertTrue(connector._reregister_kv_cache_after_recovery())

        connector._send_register_kv_cache_request.assert_called_once_with()
        self.assertIs(connector._lookups[lookup.request_id], lookup)
        self.assertEqual(connector._active_sessions, {lookup.request_id})
        self.assertEqual(connector._store_submitted_tokens, {})

    def test_unhealthy_lookup_is_a_synchronized_cache_miss(self):
        self.connector._health_event.clear()
        self.connector.chunk_size = 4
        self.connector._lookups = {}
        self.connector._active_sessions = set()
        self.connector._req_client = Mock()

        operation = self.connector.submit_lookup(
            "request",
            [1, 2, 3, 4],
            local_hit_tokens=0,
            cache_salt="",
        )

        self.assertEqual(operation.total_hit_tokens, 0)
        self.connector._req_client.lookup.assert_not_called()
        self.assertNotIn("request", self.connector._active_sessions)

    def test_cross_rank_health_failure_propagates_locally(self):
        connector = _new_connector()

        with patch.object(connector, "_sync_success", return_value=False) as sync:
            self.assertFalse(connector._all_ranks_healthy())

        sync.assert_called_once_with(True)
        self.assertFalse(connector.is_healthy)

    def test_unhealthy_store_is_not_submitted(self):
        connector = _new_connector()
        connector._health_event.clear()

        operation = connector.submit_store(
            "request",
            [1, 2, 3, 4],
            torch.arange(4),
            cache_salt="",
        )

        self.assertIsNone(operation)

    def test_unhealthy_load_returns_failed_placeholder(self):
        connector = _new_connector()
        connector._health_event.clear()
        connector.chunk_size = 4
        connector._kv_groups = (
            SGLangKVComponentGroup("full", (), tokens_per_block=4, slots_per_block=4),
        )
        lookup = LMCacheLookupOperation(
            request_id="request",
            token_ids=[1, 2, 3, 4],
            local_hit_tokens=0,
            cache_salt="",
            total_hit_tokens=4,
            locks_held=True,
        )

        operation = connector.submit_load(
            lookup,
            torch.arange(4),
            local_hit_tokens=0,
        )

        self.assertTrue(operation.query())
        self.assertFalse(operation.future.result())
        self.assertFalse(lookup.locks_held)

    def test_aligned_empty_lookup_does_not_send_end_session(self):
        self.connector.chunk_size = 8
        self.connector._lookups = {}
        self.connector._active_sessions = set()
        self.connector._lookup_leader = True
        self.connector._req_client = Mock()
        self.connector._track_control_future = Mock()

        operation = self.connector.submit_lookup(
            "short-request",
            [1, 2, 3],
            local_hit_tokens=0,
            cache_salt="",
        )
        self.connector.end_session("short-request")

        self.assertEqual(operation.total_hit_tokens, 0)
        self.connector._req_client.lookup.assert_not_called()
        self.connector._req_client.end_session.assert_not_called()
        self.connector._track_control_future.assert_not_called()

    def test_submitted_lookup_tracks_server_session(self):
        self.connector.chunk_size = 4
        self.connector.kv_world_size = 1
        self.connector._lookups = {}
        self.connector._active_sessions = set()
        self.connector._lookup_leader = True
        self.connector._req_client = Mock()
        self.connector._req_client.lookup.return_value = _Future(False)
        self.connector._create_key = Mock(return_value=object())
        self.connector._sync_success = lambda success: success

        self.connector.submit_lookup(
            "lookup-request",
            [1, 2, 3, 4],
            local_hit_tokens=0,
            cache_salt="",
        )

        self.assertIn("lookup-request", self.connector._active_sessions)

    def test_poll_lookup_uses_short_lived_status_queries(self):
        self.connector.chunk_size = 4
        self.connector._lookup_leader = True
        self.connector._req_client = Mock()
        self.connector._sync_leader_int = lambda value: value
        self.connector._store_submitted_tokens = {}
        pending_query = _ResultFuture(False, None)
        completed_query = _ResultFuture(True, 3)
        self.connector._req_client.query_prefetch_status.side_effect = [
            pending_query,
            completed_query,
        ]
        operation = LMCacheLookupOperation(
            request_id="lookup-request",
            token_ids=list(range(16)),
            local_hit_tokens=0,
            cache_salt="",
            submission_future=_ResultFuture(True, None),
        )

        # Submit one non-blocking status query. An outstanding query must
        # not be duplicated by subsequent scheduler polls.
        self.assertIsNone(self.connector.poll_lookup(operation))
        self.assertIsNone(self.connector.poll_lookup(operation))
        self.assertEqual(self.connector._req_client.query_prefetch_status.call_count, 1)

        # A None response means prefetch is still running. The following
        # scheduler pass issues a fresh query, which returns the final hit.
        pending_query.ready = True
        self.assertIsNone(self.connector.poll_lookup(operation))
        self.assertIsNone(self.connector.poll_lookup(operation))
        self.assertEqual(self.connector.poll_lookup(operation), 12)

        self.assertEqual(self.connector._req_client.query_prefetch_status.call_count, 2)
        self.connector._req_client.query_prefetch_status.assert_called_with(
            "lookup-request"
        )
        self.assertTrue(operation.locks_held)
        self.assertEqual(self.connector._store_submitted_tokens["lookup-request"], 12)

    def test_slots_to_blocks_rejects_partial_page(self):
        with self.assertRaisesRegex(ValueError, "complete SGLang pages"):
            self.connector._slots_to_blocks(torch.tensor([4, 5, 6]))

    def test_slots_to_blocks_rejects_unaligned_page(self):
        with self.assertRaisesRegex(ValueError, "page-aligned"):
            self.connector._slots_to_blocks(torch.tensor([5, 6, 7, 8]))

    def test_slots_to_blocks_accepts_explicit_dummy_page_for_load(self):
        slots = torch.tensor([0, 0, 0, 0, 8, 9, 10, 11])
        self.assertEqual(
            self.connector._slots_to_blocks(slots, allow_dummy_page=True), [0, 2]
        )

    def test_component_block_ids_expand_to_kernel_groups(self):
        self.connector._kv_groups = (
            SGLangKVComponentGroup("full", (), tokens_per_block=4, slots_per_block=4),
            SGLangKVComponentGroup("swa", (), tokens_per_block=4, slots_per_block=4),
        )
        self.connector._kernel_group_to_engine_group = (0, 1, 1)
        block_ids = self.connector._block_ids_for_transfer(
            [
                torch.tensor([4, 5, 6, 7, 12, 13, 14, 15]),
                torch.tensor([8, 9, 10, 11, 16, 17, 18, 19]),
            ],
            allow_dummy_page=False,
        )
        self.assertEqual(block_ids, [[1, 3], [2, 4], [2, 4]])

    def test_wire_view_folds_opaque_page_into_single_plane(self):
        tensor = torch.arange(4 * 2 * 3).reshape(4, 2, 3)

        wire = self.connector._to_wire_block_tensor(tensor, slots_per_block=4)

        self.assertEqual(tuple(wire.shape), (1, 1, 1, 24))
        self.assertEqual(wire.data_ptr(), tensor.data_ptr())
        self.assertTrue(torch.equal(wire.reshape(-1), tensor.reshape(-1)))
        fmt, _ = detect_format([wire], EngineType.SGLANG, {"kv_list_layout": "unified"})
        self.assertEqual(fmt, lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_HS)
        self.assertFalse(lmcache_native.is_mla(fmt))

    def test_wire_view_can_preserve_attention_head_geometry(self) -> None:
        tensor = torch.arange(4 * 2 * 3).reshape(4, 2, 3)

        wire = self.connector._to_wire_block_tensor(
            tensor, slots_per_block=4, preserve_head_geometry=True
        )

        self.assertEqual(tuple(wire.shape), (1, 4, 2, 3))
        self.assertEqual(wire.data_ptr(), tensor.data_ptr())

    def test_wire_view_can_preserve_mla_block_geometry(self) -> None:
        tensor = torch.arange(4 * 1 * 6).reshape(4, 1, 6)

        wire = self.connector._to_wire_block_tensor(
            tensor, slots_per_block=4, preserve_mla_geometry=True
        )

        self.assertEqual(tuple(wire.shape), (1, 4, 6))
        self.assertEqual(wire.data_ptr(), tensor.data_ptr())

    def test_wire_view_keeps_one_mamba_state_slot_per_block(self):
        tensor = torch.arange(5 * 2 * 3).reshape(5, 2, 3)

        wire = self.connector._to_wire_block_tensor(tensor, slots_per_block=1)

        self.assertEqual(tuple(wire.shape), (5, 1, 1, 6))
        self.assertEqual(wire.data_ptr(), tensor.data_ptr())

    def test_group_info_specs_preserve_component_address_spaces(self):
        connector = _new_connector()
        connector.page_size = 4
        connector._kv_groups = (
            SGLangKVComponentGroup(
                "full",
                (
                    torch.empty(5, 4, 1, 8),
                    torch.empty(5, 4, 1, 8),
                ),
                tokens_per_block=4,
                slots_per_block=4,
            ),
            SGLangKVComponentGroup(
                "swa",
                (
                    torch.empty(12, 1, 8),
                    torch.empty(12, 1, 16),
                ),
                sliding_window_size=8,
                tokens_per_block=4,
                slots_per_block=4,
            ),
        )

        specs, kernel_to_engine = connector._build_engine_group_info_specs()

        self.assertEqual(kernel_to_engine, (0, 1, 1))
        self.assertEqual(
            [spec["layer_indices"] for spec in specs], [(0, 1), (2,), (3,)]
        )
        self.assertEqual([spec["sw_size_tokens"] for spec in specs], [-1, 8, 8])

    def test_group_info_specs_mark_mamba_as_recurrent_one_block_window(self):
        connector = _new_connector()
        connector._kv_groups = (
            SGLangKVComponentGroup(
                "mamba",
                (torch.empty(12, 1, 32),),
                sliding_window_size=256,
                tokens_per_block=256,
                slots_per_block=1,
                recurrent_state=True,
            ),
        )

        specs, kernel_to_engine = connector._build_engine_group_info_specs()

        self.assertEqual(kernel_to_engine, (0,))
        self.assertEqual(specs[0]["tokens_per_block"], 256)
        self.assertEqual(specs[0]["sw_size_tokens"], 256)
        self.assertTrue(specs[0]["recurrent_state"])

    def test_group_info_specs_keep_dsa_sidecar_in_full_address_space(self):
        connector = _new_connector()
        connector._kv_groups = (
            SGLangKVComponentGroup(
                "full",
                (
                    torch.empty(3, 1, 64, dtype=torch.bfloat16),
                    torch.empty(3, 1, 528, dtype=torch.uint8),
                ),
                tokens_per_block=4,
                slots_per_block=4,
                tensor_rows_per_block=(1, 1),
            ),
        )

        specs, kernel_to_engine = connector._build_engine_group_info_specs()

        self.assertEqual(kernel_to_engine, (0, 0))
        self.assertEqual([spec["layer_indices"] for spec in specs], [(0,), (1,)])
        self.assertEqual([spec["engine_group_id"] for spec in specs], [0, 0])

    def test_submit_store_passes_list_of_block_ids_per_group(self):
        connector = _new_connector()
        connector.page_size = 4
        connector.chunk_size = 8
        connector.blocks_in_chunk = 2
        connector.sglang_worker_id = 0
        connector.kv_worker_id = 0
        connector.instance_id = 1
        connector._kv_groups = (
            SGLangKVComponentGroup("full", (), tokens_per_block=4, slots_per_block=4),
            SGLangKVComponentGroup("swa", (), tokens_per_block=4, slots_per_block=4),
        )
        connector._kernel_group_to_engine_group = (0, 1)
        connector._store_submitted_tokens = {}
        connector._active_sessions = set()
        connector._kv_caches = {}
        connector._transfer_ctx = _TransferContext()
        connector._is_kv_writer = True
        connector._new_event = lambda: object()
        connector._create_key = Mock(return_value=object())
        connector._sync_success = lambda success: success
        connector._sync_leader_int = lambda value: value

        operation = connector.submit_store(
            "request",
            list(range(8)),
            [
                torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]),
                torch.tensor([12, 13, 14, 15, 20, 21, 22, 23]),
            ],
            cache_salt="",
        )

        self.assertIsNotNone(operation)
        self.assertIn("request", connector._active_sessions)
        self.assertEqual(
            connector._transfer_ctx.store_args[4],
            [[1, 2], [3, 5]],
        )
        self.assertEqual(connector._create_key.call_args.kwargs["worker_id"], 0)

    def test_mla_non_writer_uses_collective_placeholder(self):
        connector = _new_connector()
        connector.page_size = 4
        connector.chunk_size = 8
        connector.blocks_in_chunk = 2
        connector.sglang_worker_id = 0
        connector.kv_worker_id = 0
        connector.instance_id = 1
        connector._kv_groups = (
            SGLangKVComponentGroup("full", (), tokens_per_block=4, slots_per_block=4),
        )
        connector._kernel_group_to_engine_group = (0,)
        connector._store_submitted_tokens = {}
        connector._active_sessions = set()
        connector._kv_caches = {}
        connector._transfer_ctx = _TransferContext()
        connector._is_kv_writer = False
        connector._sync_success = lambda success: success
        # A peer TP0 submitted the shared MLA object.
        connector._sync_leader_int = lambda value: 1

        operation = connector.submit_store(
            "request",
            list(range(8)),
            [torch.tensor([4, 5, 6, 7, 8, 9, 10, 11])],
            cache_salt="",
        )

        self.assertIsNotNone(operation)
        self.assertIsNone(connector._transfer_ctx.store_args)
        self.assertTrue(operation.query())
        self.assertTrue(connector.complete_store(operation))
        self.assertIn("request", connector._active_sessions)

    def test_submit_store_accepts_all_null_historical_swa_chunk(self):
        connector = _new_connector()
        connector.page_size = 4
        connector.chunk_size = 8
        connector.blocks_in_chunk = 2
        connector.sglang_worker_id = 0
        connector.kv_worker_id = 0
        connector.instance_id = 1
        connector._kv_groups = (
            SGLangKVComponentGroup("full", (), tokens_per_block=4, slots_per_block=4),
            SGLangKVComponentGroup(
                "swa",
                (),
                sliding_window_size=8,
                tokens_per_block=4,
                slots_per_block=4,
            ),
        )
        connector._kernel_group_to_engine_group = (0, 1)
        connector._store_submitted_tokens = {}
        connector._active_sessions = set()
        connector._kv_caches = {}
        connector._transfer_ctx = _TransferContext()
        connector._is_kv_writer = True
        connector._new_event = lambda: object()
        connector._create_key = Mock(return_value=object())
        connector._sync_success = lambda success: success
        connector._sync_leader_int = lambda value: value

        operation = connector.submit_store(
            "request",
            list(range(16)),
            [
                torch.arange(4, 20),
                torch.tensor([0] * 8 + list(range(12, 20))),
            ],
            cache_salt="",
        )

        self.assertIsNotNone(operation)
        self.assertEqual(
            connector._transfer_ctx.store_args[4],
            [[1, 2, 3, 4], [0, 0, 3, 4]],
        )

    def test_submit_store_defers_partly_unmapped_retained_swa_chunk(self):
        connector = _new_connector()
        connector.page_size = 4
        connector.chunk_size = 8
        connector._kv_groups = (
            SGLangKVComponentGroup("full", (), tokens_per_block=4, slots_per_block=4),
            SGLangKVComponentGroup(
                "swa",
                (),
                sliding_window_size=8,
                tokens_per_block=4,
                slots_per_block=4,
            ),
        )
        connector._kernel_group_to_engine_group = (0, 1)
        connector._store_submitted_tokens = {}
        connector._active_sessions = set()
        connector._is_kv_writer = True
        connector._sync_success = lambda success: success

        operation = connector.submit_store(
            "request",
            list(range(8)),
            [
                torch.arange(4, 12),
                torch.tensor([0, 0, 0, 0, 12, 13, 14, 15]),
            ],
            cache_salt="",
        )

        self.assertIsNone(operation)
        self.assertNotIn("request", connector._active_sessions)
        self.assertNotIn("request", connector._store_submitted_tokens)

    def test_submit_store_accepts_dummy_mamba_blocks(self):
        connector = _new_connector()
        connector.page_size = 1
        connector.chunk_size = 8
        connector.blocks_in_chunk = 8
        connector.sglang_worker_id = 0
        connector.kv_worker_id = 0
        connector.instance_id = 1
        connector._kv_groups = (
            SGLangKVComponentGroup("full", (), tokens_per_block=1, slots_per_block=1),
            SGLangKVComponentGroup(
                "mamba",
                (),
                sliding_window_size=4,
                tokens_per_block=4,
                slots_per_block=1,
                recurrent_state=True,
            ),
        )
        connector._kernel_group_to_engine_group = (0, 1)
        connector._store_submitted_tokens = {}
        connector._active_sessions = set()
        connector._kv_caches = {}
        connector._transfer_ctx = _TransferContext()
        connector._is_kv_writer = True
        connector._new_event = lambda: object()
        connector._create_key = lambda *args, **kwargs: object()
        connector._sync_success = lambda success: success
        connector._sync_leader_int = lambda value: value

        operation = connector.submit_store(
            "request",
            list(range(8)),
            [torch.arange(1, 9), torch.tensor([0, 7])],
            cache_salt="",
        )

        self.assertIsNotNone(operation)
        self.assertEqual(
            connector._transfer_ctx.store_args[4],
            [list(range(1, 9)), [0, 7]],
        )

    def test_submit_load_uses_compressed_mamba_block_ids(self):
        connector = _new_connector()
        connector.page_size = 1
        connector.chunk_size = 8
        connector.blocks_in_chunk = 8
        connector.sglang_worker_id = 0
        connector.kv_worker_id = 0
        connector.instance_id = 1
        connector._kv_groups = (
            SGLangKVComponentGroup("full", (), tokens_per_block=1, slots_per_block=1),
            SGLangKVComponentGroup(
                "mamba",
                (),
                sliding_window_size=4,
                tokens_per_block=4,
                slots_per_block=1,
                recurrent_state=True,
            ),
        )
        connector._kernel_group_to_engine_group = (0, 1)
        connector._kv_caches = {}
        connector._transfer_ctx = _TransferContext()
        connector._new_event = lambda: object()
        connector._create_key = lambda *args, **kwargs: object()
        lookup = LMCacheLookupOperation(
            request_id="request",
            token_ids=list(range(8)),
            local_hit_tokens=3,
            cache_salt="",
            total_hit_tokens=8,
            locks_held=True,
        )

        operation = connector.submit_load(
            lookup,
            [torch.arange(4, 9), torch.tensor([0, 7])],
            local_hit_tokens=3,
        )

        args, kwargs = connector._transfer_ctx.retrieve_args
        self.assertEqual(args[4], [[0, 0, 0, 4, 5, 6, 7, 8], [0, 7]])
        self.assertEqual(kwargs["skip_first_n_tokens"], 3)
        self.assertEqual(operation.start, 0)
        self.assertEqual(operation.end, 8)

    def test_prepare_load_orders_forward_stream_without_completing_future(self):
        connector = _new_connector()
        connector._mq_timeout = 10
        connector._sync_success = lambda success: success
        lookup = LMCacheLookupOperation(
            request_id="request",
            token_ids=list(range(8)),
            local_hit_tokens=0,
            cache_salt="",
            total_hit_tokens=8,
            locks_held=True,
        )
        future = _ResultFuture(True, True)
        operation = LMCacheLoadOperation(
            request_id="request",
            token_ids=list(range(8)),
            start=0,
            end=8,
            local_hit_tokens=0,
            device_indices=torch.arange(8),
            future=future,
            lookup=lookup,
        )

        self.assertTrue(connector.prepare_load_on_stream(operation, "forward"))
        self.assertEqual(future.waited_stream, "forward")
        self.assertIsNone(operation.result)
        self.assertTrue(lookup.locks_held)

    def test_free_lookup_locks_sends_one_leader_prefix_range(self):
        connector = _new_connector()
        connector._lookup_leader = True
        connector.kv_world_size = 2
        connector._req_client = Mock()
        connector._req_client.free_lookup_locks.return_value = _Future(True)
        connector._track_control_future = Mock()
        connector._create_key = Mock(return_value="prefix-key")
        lookup = LMCacheLookupOperation(
            request_id="request",
            token_ids=list(range(8)),
            local_hit_tokens=4,
            cache_salt="",
            total_hit_tokens=8,
            locks_held=True,
        )
        connector._lookups = {lookup.request_id: lookup}
        connector.free_lookup_locks("request", start=0, end=4)

        connector._create_key.assert_called_once_with(
            lookup, start=0, end=4, worker_id=None
        )
        connector._req_client.free_lookup_locks.assert_called_once_with("prefix-key", 2)
        self.assertEqual(lookup.lock_start, 4)
        self.assertTrue(lookup.locks_held)
        self.assertIs(connector._lookups["request"], lookup)

    def test_complete_load_relies_on_retrieve_to_release_read_locks(self):
        connector = _new_connector()
        connector._sync_success = lambda success: success
        connector._free_lookup_locks = Mock()
        lookup = LMCacheLookupOperation(
            request_id="request",
            token_ids=list(range(8)),
            local_hit_tokens=0,
            cache_salt="",
            total_hit_tokens=8,
            locks_held=True,
        )
        connector._lookups = {lookup.request_id: lookup}
        operation = LMCacheLoadOperation(
            request_id="request",
            token_ids=list(range(8)),
            start=0,
            end=8,
            local_hit_tokens=0,
            device_indices=torch.arange(8),
            future=_ResultFuture(True, True),
            lookup=lookup,
        )

        self.assertTrue(connector.complete_load(operation))

        connector._free_lookup_locks.assert_not_called()
        self.assertFalse(lookup.locks_held)
        self.assertNotIn("request", connector._lookups)

    def test_completed_operation_does_not_requery_future(self):
        operation = object.__new__(LMCacheLoadOperation)
        operation.result = False
        operation.future = _Future(ready=True)
        self.assertTrue(operation.query())

    def test_create_key_declares_single_kv_reader(self):
        self.connector.model_name = "model"
        self.connector.kv_world_size = 2
        self.connector.num_kv_readers = 1
        operation = LMCacheLookupOperation(
            request_id="request",
            token_ids=[1, 2, 3, 4],
            local_hit_tokens=0,
            cache_salt="salt",
        )
        custom_types = ModuleType("lmcache.v1.multiprocess.custom_types")
        custom_types.IPCCacheServerKey = _IPCCacheServerKey

        with patch.dict(
            "sys.modules",
            {"lmcache.v1.multiprocess.custom_types": custom_types},
        ):
            key = self.connector._create_key(operation, start=0, end=4, worker_id=None)

        self.assertEqual(key.num_kv_readers, 1)
        self.assertEqual(key.world_size, 2)

    def test_create_key_reserves_one_read_lock_per_tp_replica(self):
        self.connector.model_name = "model"
        self.connector.kv_world_size = 1
        self.connector.num_kv_readers = 4
        operation = LMCacheLookupOperation(
            request_id="request",
            token_ids=[1, 2, 3, 4],
            local_hit_tokens=0,
            cache_salt="salt",
        )
        custom_types = ModuleType("lmcache.v1.multiprocess.custom_types")
        custom_types.IPCCacheServerKey = _IPCCacheServerKey

        with patch.dict(
            "sys.modules",
            {"lmcache.v1.multiprocess.custom_types": custom_types},
        ):
            key = self.connector._create_key(operation, start=0, end=4, worker_id=None)

        self.assertEqual(key.num_kv_readers, 4)
        self.assertEqual(key.world_size, 1)
