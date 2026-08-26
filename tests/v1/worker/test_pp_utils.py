# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import pytest

from vllm.v1.worker.gpu.pp_utils import PendingRecv, PPHandler


@dataclass
class _FakeSlot:
    sequence: int
    event: object | None = None


def _make_queue_handler(
    pp_size: int, delay: int
) -> tuple[PPHandler, list[tuple[int, int]], Callable[[int], None]]:
    handler = PPHandler.__new__(PPHandler)
    handler.queue = deque([None] * pp_size)
    handler.recv_launch_delay = delay

    launches: list[tuple[int, int]] = []
    current_step = -1

    def launch(slot: PendingRecv) -> None:
        fake_slot = cast(_FakeSlot, slot)
        launches.append((fake_slot.sequence, current_step))
        fake_slot.event = object()

    handler._launch_receive = launch  # type: ignore[method-assign]

    def set_current_step(step: int) -> None:
        nonlocal current_step
        current_step = step

    return handler, launches, set_current_step


@pytest.mark.parametrize(("pp_size", "delay"), [(2, 1), (4, 1), (4, 2), (4, 3)])
def test_deferred_receive_launch_and_consume_cadence(pp_size: int, delay: int) -> None:
    handler, launches, set_current_step = _make_queue_handler(pp_size, delay)
    consumed: list[tuple[int, int]] = []
    num_origin_steps = 8

    for step in range(num_origin_steps + pp_size):
        set_current_step(step)
        due_slot = handler._advance_receive_queue()
        if due_slot is not None:
            fake_slot = cast(_FakeSlot, due_slot)
            consumed.append((fake_slot.sequence, step))
        if step < num_origin_steps:
            handler.queue[-1] = cast(PendingRecv, _FakeSlot(step))

    assert launches == [(origin, origin + delay) for origin in range(num_origin_steps)]
    assert consumed == [
        (origin, origin + pp_size) for origin in range(num_origin_steps)
    ]


def test_deferred_receive_empty_steps_preserve_collective_order() -> None:
    handler, launches, set_current_step = _make_queue_handler(pp_size=4, delay=3)
    sampled_steps = {0, 2, 5}
    consumed: list[int] = []

    for step in range(10):
        set_current_step(step)
        due_slot = handler._advance_receive_queue()
        if due_slot is not None:
            consumed.append(cast(_FakeSlot, due_slot).sequence)
        if step in sampled_steps:
            handler.queue[-1] = cast(PendingRecv, _FakeSlot(step))

    assert [sequence for sequence, _ in launches] == sorted(sampled_steps)
    assert consumed == sorted(sampled_steps)


def test_flush_pending_collectives_is_idempotent() -> None:
    handler, launches, set_current_step = _make_queue_handler(pp_size=4, delay=3)
    handler.is_last_rank = False
    slots = [_FakeSlot(0), _FakeSlot(1)]
    handler.queue = deque(
        [cast(PendingRecv, slots[0]), None, cast(PendingRecv, slots[1]), None]
    )
    set_current_step(7)

    assert handler.flush_pending_collectives() == 2
    assert handler.flush_pending_collectives() == 0
    assert launches == [(0, 7), (1, 7)]


def test_last_rank_has_no_receives_to_flush() -> None:
    handler, launches, _ = _make_queue_handler(pp_size=4, delay=3)
    handler.is_last_rank = True

    assert handler.flush_pending_collectives() == 0
    assert launches == []
