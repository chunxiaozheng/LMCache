# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict, deque
from dataclasses import dataclass

# First Party
from lmcache.v1.cache_controller.message import (
    BatchedP2PLookupMsg,
    BatchedP2PLookupRetMsg,
    CheckFinishMsg,
    CheckFinishRetMsg,
    ClearMsg,
    ClearRetMsg,
    CompressMsg,
    CompressRetMsg,
    DecompressMsg,
    DecompressRetMsg,
    KVAdmitMsg,
    KVEvictMsg,
    LookupMsg,
    LookupRetMsg,
    MoveMsg,
    MoveRetMsg,
    PinMsg,
    PinRetMsg,
)
from lmcache.v1.cache_controller.observability import PrometheusLogger
from lmcache.v1.token_database import ChunkedTokenDatabase


@dataclass
class KVChunkMetadata:
    """
    A class representing a KV chunk metadata.
    """

    instance_id: str
    worker_id: int
    location: str

    def __hash__(self) -> int:
        """
        Hash method.
        """
        return hash((self.instance_id, self.worker_id, self.location))

    def __eq__(self, other) -> bool:
        """
        Equality comparison method.
        """
        if not isinstance(other, KVChunkMetadata):
            return False
        return (
            self.instance_id == other.instance_id
            and self.worker_id == other.worker_id
            and self.location == other.location
        )


class KVController:
    def __init__(self) -> None:
        self.kv_pool: dict[int, deque[KVChunkMetadata]] = defaultdict(deque)
        self.reverse_index: dict[KVChunkMetadata, set[int]] = defaultdict(set)
        # TODO(Jiayi): remove this hardcode
        self.token_database = ChunkedTokenDatabase()
        self._setup_metrics()

    def _setup_metrics(self):
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.kv_pool_keys_count.set_function(lambda: len(self.kv_pool))

    def post_init(self, reg_controller, cluster_executor):
        """
        Post initialization of the KV controller.
        """
        self.reg_controller = reg_controller
        self.cluster_executor = cluster_executor

    async def admit(self, msg: KVAdmitMsg) -> None:
        """
        Admit a new kv chunk.
        """
        chunk_meta = KVChunkMetadata(msg.instance_id, msg.worker_id, msg.location)
        self.kv_pool[msg.key].append(chunk_meta)
        self.reverse_index[chunk_meta].add(msg.key)

    async def evict(self, msg: KVEvictMsg) -> None:
        """
        Evict a kv chunk.
        """
        chunk_meta = KVChunkMetadata(msg.instance_id, msg.worker_id, msg.location)
        key = msg.key

        if key not in self.kv_pool:
            return

        try:
            self.kv_pool[key].remove(chunk_meta)
        except ValueError:
            pass
        try:
            self.reverse_index[chunk_meta].remove(key)
        except KeyError:
            pass

        if len(self.kv_pool[key]) == 0:
            del self.kv_pool[key]
        if (
            chunk_meta in self.reverse_index
            and len(self.reverse_index[chunk_meta]) == 0
        ):
            del self.reverse_index[chunk_meta]

    async def clear(self, msg: ClearMsg) -> ClearRetMsg:
        """
        Clear kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("clear", msg)

    async def pin(self, msg: PinMsg) -> PinRetMsg:
        """
        Pin kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("pin", msg)

    async def compress(self, msg: CompressMsg) -> CompressRetMsg:
        """
        Compress kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("compress", msg)

    async def decompress(self, msg: DecompressMsg) -> DecompressRetMsg:
        """
        Decompress kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("decompress", msg)

    async def move(self, msg: MoveMsg) -> MoveRetMsg:
        """
        Move kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("move", msg)

    async def check_finish(self, msg: CheckFinishMsg) -> CheckFinishRetMsg:
        """
        Check if an event is finished.
        """
        return await self.cluster_executor.execute("check_finish", msg)

    async def deregister(self, instance_id: str, worker_id: int) -> None:
        """
        Deregister all kv chunks of an instance-worker.
        """
        for chunk_meta, keys in list(self.reverse_index.items()):
            if (
                chunk_meta.instance_id == instance_id
                and chunk_meta.worker_id == worker_id
            ):
                # delete the key from the kv pool
                for key in keys:
                    if key not in self.kv_pool:
                        continue

                    self.kv_pool[key].remove(chunk_meta)

                    if len(self.kv_pool[key]) == 0:
                        del self.kv_pool[key]

                # delete the reverse index
                del self.reverse_index[chunk_meta]

    # TODO(Jiayi): The current implementation does not handle
    # the case where the prefix chunks are evicted while the
    # suffix chunk is still in the system. LMCache should guarantee
    # this does not happen.
    # TODO(Jiayi): The current implementation does not consider
    # the location of the kv chunks. It simply returns the
    # `instance_id` with longest prefix.
    # TODO(Jiayi): Need to get rid of the hash somehow
    async def lookup(self, msg: LookupMsg) -> LookupRetMsg:
        tokens = msg.tokens
        layout_info = {}
        for start, end, key in self.token_database.process_tokens(
            tokens, make_key=False
        ):
            if key not in self.kv_pool:
                break
            matched_instance = self.kv_pool[key][0].instance_id
            matched_location = self.kv_pool[key][0].location
            layout_info[matched_instance] = (matched_location, end)
        return LookupRetMsg(layout_info=layout_info, event_id=msg.event_id)

    async def batched_p2p_lookup(
        self, msg: BatchedP2PLookupMsg
    ) -> BatchedP2PLookupRetMsg:
        """
        Perform batched P2P lookup for multiple keys.

        :param BatchedP2PLookupMsg msg: The batched P2P lookup message containing keys.

        :return: A BatchedP2PLookupRetMsg containing the lookup results.
        """

        worker_id = msg.worker_id
        query_instance_id = msg.instance_id
        num_hit_chunks = 0
        instance_id = ""
        location = ""
        peer_init_url = ""
        for key in msg.hashes:
            # TODO(Jiayi): remove this string conversion
            if key not in self.kv_pool:
                break

            # TODO(Jiayi): Currently, we use the first matched
            # kv chunk metadata to do matching. The matching
            # logic can be improved.
            # TODO(Jiayi): The KV Cache could be from different
            # instances. We need to handle this case as well.
            matched_kv_chunk_meta = None
            for kv_chunk_meta in self.kv_pool[key]:
                if kv_chunk_meta.instance_id != query_instance_id:
                    # Found a matching instance_id that's not the
                    # same as the query_instance_id.
                    matched_kv_chunk_meta = kv_chunk_meta
                    break

            if matched_kv_chunk_meta is None:
                break
            if instance_id != "" and (
                instance_id != matched_kv_chunk_meta.instance_id
                or location != matched_kv_chunk_meta.location
            ):
                # We have already found a different instance_id
                # before. Stop here.
                break
            elif instance_id == "":
                instance_id = matched_kv_chunk_meta.instance_id
                location = matched_kv_chunk_meta.location
                peer_init_url = self.reg_controller.get_peer_init_url(
                    instance_id, worker_id
                )
                assert peer_init_url is not None
            num_hit_chunks += 1

        return BatchedP2PLookupRetMsg(
            layout_info=[
                (instance_id, location, num_hit_chunks, peer_init_url),
            ]
        )
