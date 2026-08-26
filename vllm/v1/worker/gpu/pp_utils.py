# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm import envs
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    event: torch.cuda.Event | None

    sampled_tokens: torch.Tensor  # [num_reqs, max_sample_len]
    combined: torch.Tensor  # [2, num_reqs]: num_sampled, num_rejected
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    with outputs that might be needed in a subsequent (decode) step.
    Returns None if no sampled outputs are needed in the requests' next step."""

    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    max_seq_len = input_batch.max_seq_len_np
    assert max_seq_len is not None  # always populated under PP
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    # Exclude requests that we know are finished.
    not_finishing = np.maximum(old_computed, prefill_len) + 1 < max_seq_len
    need_sampled_mask = produces_sample & not_finishing
    return need_sampled_mask if need_sampled_mask.any() else None


class PPHandler:
    """Runs the PP sampled-token broadcast/recv on a side stream so the
    default stream isn't gated by the matching peer call. Step T's recv is
    consumed at step T+pp_size via `get_prev_sampled_outputs`.

    Uses a dedicated NCCL communicator (sibling of the PP `device_group`)
    for the broadcast so it does not serialize on the wire with the
    inter-stage hidden-state p2p send/recv ops.
    """

    def __init__(
        self,
        max_num_reqs: int,
        num_speculative_steps: int,
        device: torch.device,
        use_async_scheduling: bool,
    ):
        pp_group = get_pp_group()
        self.is_last_rank = pp_group.is_last_rank
        self.last_rank = pp_group.last_rank
        self.max_sample_len = num_speculative_steps + 1
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        self.requested_recv_launch_delay = envs.VLLM_PP_DEFER_SAMPLED_TOKEN_RECV
        if not 0 <= self.requested_recv_launch_delay < pp_group.world_size:
            raise ValueError(
                "VLLM_PP_DEFER_SAMPLED_TOKEN_RECV must be in "
                f"[0, {pp_group.world_size - 1}] for PP size "
                f"{pp_group.world_size}; got {self.requested_recv_launch_delay}"
            )
        if self.requested_recv_launch_delay and not current_platform.is_cuda():
            raise ValueError(
                "VLLM_PP_DEFER_SAMPLED_TOKEN_RECV is currently supported only on CUDA"
            )
        if self.requested_recv_launch_delay and not use_async_scheduling:
            raise ValueError(
                "VLLM_PP_DEFER_SAMPLED_TOKEN_RECV requires async scheduling"
            )

        # Warmup must retain upstream collective timing. The worker enables
        # the requested delay only after all compilation and warmup completes.
        # The last PP rank always broadcasts immediately.
        self.recv_launch_delay = 0
        self.deferred_collectives_active = False

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
        # pushed by step T's `receive` is consumed pp_size steps later. Pre-seeded
        # with pp_size None placeholders so the first pp_size consumes are no-ops.
        # None means no postprocess is pending for that step (broadcast skipped).
        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * pp_group.world_size)
        )

        # Per req-index generation counter, incremented every time a request
        # index is freed in RequestStats. Used for invalidating freed req data
        # between PP decodes.
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = pp_group.make_sibling_device_group(
            group_desc="pp_broadcast"
        )

    def enable_deferred_collectives(self) -> bool:
        """Enable the requested receive delay after worker warmup."""
        if not self.requested_recv_launch_delay or self.deferred_collectives_active:
            return False

        self.recv_launch_delay = (
            0 if self.is_last_rank else self.requested_recv_launch_delay
        )
        self.deferred_collectives_active = True
        pp_group = get_pp_group()
        logger.info_once(
            "Enabled deferred PP sampled-token receive: "
            "pp_rank=%d/%d recv_delay_steps=%d",
            pp_group.rank_in_group,
            pp_group.world_size,
            self.recv_launch_delay,
        )
        return True

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def _launch_receive(self, slot: PendingRecv) -> None:
        """Post one receiver's broadcasts exactly once."""
        if slot.event is not None:
            return
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                slot.sampled_tokens,
                src=self.last_rank,
                group=self.broadcast_group,
            )
            torch.distributed.broadcast(
                slot.combined,
                src=self.last_rank,
                group=self.broadcast_group,
            )
            slot.event = self.broadcast_stream.record_event()
            # The tensors are populated on the broadcast stream and consumed
            # later on the main stream.
            slot.sampled_tokens.record_stream(self.main_stream)
            slot.combined.record_stream(self.main_stream)

    def _advance_receive_queue(self) -> PendingRecv | None:
        """Launch the configured slot and return the slot due this step."""
        if self.recv_launch_delay:
            launch_slot = self.queue[-self.recv_launch_delay]
            if launch_slot is not None:
                self._launch_receive(launch_slot)

        due_slot = self.queue.popleft()
        # Reserve this step's slot; `receive` overwrites it if applicable.
        self.queue.append(None)
        return due_slot

    def flush_pending_collectives(self) -> int:
        """Post all deferred receives before an idle boundary or shutdown."""
        if self.is_last_rank:
            return 0

        launched = 0
        for slot in self.queue:
            if slot is not None and slot.event is None:
                self._launch_receive(slot)
                launched += 1
        return launched

    def get_prev_sampled_outputs(self) -> dict[str, torch.Tensor] | None:
        """Consume the entry from pp_size steps ago and wait for its recv event,
        then filter out entries whose request was freed since `receive`.
        """
        if not self.queue:
            return None
        slot = self._advance_receive_queue()
        if slot is None:
            return None

        if slot.event is None:
            # Defensive fallback for a changed cadence or an early finite
            # drain: never consume a slot without posting its collective.
            logger.warning_once(
                "Deferred PP sampled-token receive reached consumption before "
                "its configured launch step; posting it now"
            )
            self._launch_receive(slot)
        assert slot.event is not None

        # Skip requests which did not need sampled output and/or those already
        # finished. The post_update kernel skips the -1 entries.
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                # No states require update anymore.
                return None
            # Filter excluded request indices.
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        self.main_stream.wait_event(slot.event)
        num_sampled, num_rejected = slot.combined.unbind(dim=0)
        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            idx_mapping=idx_mapping,
        )

    def receive(self, input_batch: InputBatch) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch."""
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
            return False

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            sampled_tokens = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
        slot = PendingRecv(
            None,
            sampled_tokens,
            combined,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
        )
        self.queue[-1] = slot
        if self.recv_launch_delay == 0:
            self._launch_receive(slot)
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                sampled_token_ids.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)
