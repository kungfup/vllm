# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_tp_group
# Insert microbatch yield helpers. These are no-ops if microbatching is disabled
# or no UBatchContext is active for the current thread.
from vllm.v1.worker.ubatching import (
    yield_and_switch_from_compute_to_comm,
    yield_and_switch_from_comm_to_compute,
)


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group.

    When microbatching (DBO) is enabled, insert yield points around the
    all-reduce to allow overlapping compute and communication across
    microbatches. If no microbatch context is active, these calls are no-ops.
    """
    # Switch from compute stream to comm stream and yield to the sibling
    # microbatch (no-op if microbatching is not active).
    yield_and_switch_from_compute_to_comm(schedule="default")
    out = get_tp_group().all_reduce(input_)
    # Switch back from comm stream to compute stream and wait on comm (no-op if
    # microbatching is not active).
    yield_and_switch_from_comm_to_compute(schedule="default")
    return out


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_reduce_scatter(input_: torch.Tensor,
                                         dim: int = -1) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(tensor_dict: Optional[dict[Any, Union[torch.Tensor,
                                                                Any]]] = None,
                          src: int = 0):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
