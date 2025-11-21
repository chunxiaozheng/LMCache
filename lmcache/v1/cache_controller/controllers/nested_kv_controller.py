# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from typing import Optional

# First Party
from lmcache.v1.cache_controller.controllers import KVController
from lmcache.v1.cache_controller.message import (
    BatchedP2PLookupMsg,
    BatchedP2PLookupRetMsg,
    KVAdmitMsg,
    KVEvictMsg,
    LookupMsg,
    LookupRetMsg,
)


class NestedKVController(KVController):
    def __init__(self):
        super().__init__()
        # Mapping from `(instance_id, worker_id)` -> [location -> set[chunk_hash]]
        self.nested_kv_pool: dict[tuple[str, int], dict[str, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )

    async def admit(self, msg: KVAdmitMsg) -> None:
        report_id = (msg.instance_id, msg.worker_id)
        self.nested_kv_pool[report_id][msg.location].add(msg.key)

    async def evict(self, msg: KVEvictMsg) -> None:
        report_id = (msg.instance_id, msg.worker_id)
        location = msg.location
        key = msg.key

        if (
            report_id not in self.nested_kv_pool
            or location not in self.nested_kv_pool[report_id]
            or key not in self.nested_kv_pool[report_id][location]
        ):
            return

        self.nested_kv_pool[report_id][location].remove(key)
        if not self.nested_kv_pool[report_id][location]:
            del self.nested_kv_pool[report_id][location]
        if not self.nested_kv_pool[report_id]:
            del self.nested_kv_pool[report_id]

    async def deregister(self, instance_id: str, worker_id: int) -> None:
        report_id = (instance_id, worker_id)
        if report_id in self.nested_kv_pool:
            del self.nested_kv_pool[report_id]

    async def lookup(self, msg: LookupMsg) -> LookupRetMsg:
        tokens = msg.tokens
        layout_info = {}
        for start, end, key in self.token_database.process_tokens(
            tokens, make_key=False
        ):
            result = self.exists(key)
            if result is None:
                break
            matched_instance = result[0]
            matched_location = result[2]
            layout_info[matched_instance] = (matched_location, end)
        return LookupRetMsg(layout_info=layout_info, event_id=msg.event_id)

    async def batched_p2p_lookup(
        self, msg: BatchedP2PLookupMsg
    ) -> BatchedP2PLookupRetMsg:
        #
        result = self.exists(msg.hashes[0], msg.instance_id)
        if result is None:
            return BatchedP2PLookupRetMsg(layout_info=[("", "", 0, "")])

        instance_id = result[0]
        worker_id = result[1]
        location = result[2]
        peer_init_url = self.reg_controller.get_distributed_url(instance_id, worker_id)
        assert peer_init_url is not None
        current_instance_keys = self.nested_kv_pool[(instance_id, worker_id)][location]
        num_hit_chunks = 0
        for key in msg.hashes:
            if key not in current_instance_keys:
                break
            num_hit_chunks += 1

        return BatchedP2PLookupRetMsg(
            layout_info=[
                (instance_id, location, num_hit_chunks, peer_init_url),
            ]
        )

    def exists(
        self,
        key: int,
        exclude_instance_id: Optional[str] = None,
        exclude_location: Optional[str] = None,
    ) -> Optional[tuple[str, int, str]]:
        for (instance_id, worker_id), location_kvs in self.nested_kv_pool.items():
            for location, kvs in location_kvs.items():
                if (
                    key in kvs
                    and instance_id != exclude_instance_id
                    and location != exclude_location
                ):
                    return instance_id, worker_id, location
        return None
