# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_tp_group


def _get_yield_funcs():
    """Lazily import microbatch yield helpers to avoid circular imports.

    Returns a pair of callables (to_comm, to_compute). If microbatching is not
    available or import fails during early init, returns no-op functions.
    """
    try:
        from vllm.v1.worker.ubatching import (  # type: ignore
            yield_and_switch_from_compute_to_comm as _to_comm,
            yield_and_switch_from_comm_to_compute as _to_compute,
        )
        return _to_comm, _to_compute
    except Exception:
        # Fallback no-ops (e.g., during early import or when microbatching is
        # not enabled). Keep signature compatible.
        def _noop_to_comm(schedule: str = "default"):
            return None

        def _noop_to_compute(schedule: str = "default"):
            return None

        return _noop_to_comm, _noop_to_compute


def tensor_model_parallel_all_reduce(input_: torch.Tensor,
                                     *,
                                     schedule: str = "default") -> torch.Tensor:
    """All-reduce the input tensor across model parallel group.

    When microbatching (DBO) is enabled, insert yield points around the
    all-reduce to allow overlapping compute and communication across
    microbatches. If no microbatch context is active, these calls are no-ops.
    """
    # Fast path: if ubatching is not globally enabled, skip any yield logic to
    # avoid introducing Dynamo-unfriendly symbols (e.g., threading.get_ident).
    try:
        from vllm.v1.worker.ubatching import (  # type: ignore
            is_ubatching_globally_enabled as _is_enabled,
        )
        if not _is_enabled():
            return get_tp_group().all_reduce(input_)
    except Exception:
        # Any failure to import or query the flag: fall back to vanilla all_reduce.
        return get_tp_group().all_reduce(input_)

    to_comm, to_compute = _get_yield_funcs()
    to_comm(schedule=schedule)
    out = get_tp_group().all_reduce(input_)
    to_compute(schedule=schedule)
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
