"""Single source of truth for how a (dp, cp, pp, tp) coordinate tuple maps to
one global NPU rank, and vice versa -- shared by trace_generator.py,
comm_generator.py, gen_dgx_topology.py and gen_opt_topology.py so the four of
them can never independently drift out of sync on this (as they did once
already, by hand, when the nesting order changed from d,p,c,t to d,c,p,t).

A "global rank" is a mixed-radix number: given an axis_order from outermost
to innermost (e.g. the default ("dp","cp","pp","tp")), the innermost axis
changes fastest (stride 1) and each axis further out has a stride equal to
the product of every axis inside it. This is exactly the generalization of
trace_generator.py's old hardcoded `d*CP*PP*TP + c*PP*TP + p*TP + t` --  that
formula is just strides() + coords_to_rank() with axis_order fixed to
("dp","cp","pp","tp").

Making axis_order configurable (via --axis-order on every generator) is what
lets a topology choose e.g. ("dp","pp","cp","tp") instead, so cp ends up
adjacent to tp and can validly form a contiguous inner domain alongside it --
see inner_axes_for_domain_size.
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

DEFAULT_AXIS_ORDER: Tuple[str, ...] = ("dp", "cp", "pp", "tp")
AXES = frozenset(DEFAULT_AXIS_ORDER)


def axis_order_arg(s: str) -> Tuple[str, ...]:
    """argparse type= for --axis-order: comma-separated axis names, outermost
    to innermost (e.g. "dp,cp,pp,tp"). Must be a permutation of dp/cp/pp/tp."""
    order = tuple(s.split(","))
    if set(order) != AXES or len(order) != len(AXES):
        raise argparse.ArgumentTypeError(
            f"--axis-order must be a permutation of {','.join(DEFAULT_AXIS_ORDER)}, got {s!r}"
        )
    return order


def strides(sizes: Dict[str, int], axis_order: Tuple[str, ...] = DEFAULT_AXIS_ORDER) -> Dict[str, int]:
    """Per-axis stride (how much the global rank changes per +1 on that axis)
    given axis_order outermost-to-innermost -- the innermost axis always has
    stride 1, same as the ones place of a mixed-radix number."""
    result: Dict[str, int] = {}
    stride = 1
    for axis in reversed(axis_order):
        result[axis] = stride
        stride *= sizes[axis]
    return result


def coords_to_rank(coords: Dict[str, int], sizes: Dict[str, int], axis_order: Tuple[str, ...] = DEFAULT_AXIS_ORDER) -> int:
    s = strides(sizes, axis_order)
    return sum(coords[axis] * s[axis] for axis in axis_order)


def rank_to_coords(rank: int, sizes: Dict[str, int], axis_order: Tuple[str, ...] = DEFAULT_AXIS_ORDER) -> Dict[str, int]:
    s = strides(sizes, axis_order)
    remaining = rank
    result: Dict[str, int] = {}
    for axis in axis_order:
        result[axis] = remaining // s[axis]
        remaining %= s[axis]
    return result


def achievable_domain_sizes(sizes: Dict[str, int], axis_order: Tuple[str, ...] = DEFAULT_AXIS_ORDER) -> list[int]:
    """Every trailing-suffix product of axis_order, innermost-axis-alone
    first and the whole order's full product last -- the complete list of
    domain sizes a single contiguous inner domain could validly be formed at
    under this axis_order (see inner_axes_for_domain_size). A sweep over
    candidate npus_per_domain values should draw from this list, not from an
    arbitrary range, since only these sizes correspond to a real trailing
    suffix of axis_order."""
    sizes_list = []
    product = 1
    for axis in reversed(axis_order):
        product *= sizes[axis]
        sizes_list.append(product)
    return sizes_list


def inner_axes_for_domain_size(
    npus_per_domain: int, sizes: Dict[str, int], axis_order: Tuple[str, ...] = DEFAULT_AXIS_ORDER
) -> Tuple[str, ...]:
    """Finds the longest trailing suffix of axis_order whose size-product
    equals npus_per_domain exactly -- the set of axes that can physically sit
    together inside a domain of that size, since only a trailing suffix is
    numbered contiguously under this axis_order (see this module's
    docstring). Returned in axis_order's own relative order.

    Raises ValueError, listing the actual achievable domain sizes, if no
    trailing suffix matches -- silently grouping by a non-suffix set would
    produce a domain whose "NPUs" aren't actually contiguous ranks, i.e. a
    topology that doesn't correspond to the trace's own numbering at all.
    """
    achievable = achievable_domain_sizes(sizes, axis_order)
    if npus_per_domain in achievable:
        count = achievable.index(npus_per_domain) + 1
        return tuple(axis_order[len(axis_order) - count:])
    raise ValueError(
        f"npus_per_domain={npus_per_domain} doesn't match any trailing-suffix product of "
        f"axis_order={','.join(axis_order)} with sizes={sizes}; achievable domain sizes are {achievable}. "
        f"Reorder --axis-order so your desired inner axes are last, or change --npus-per-domain."
    )
