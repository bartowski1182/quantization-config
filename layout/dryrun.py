"""llama-quantize `--dry-run` — the argv to run and the parser for its output.

The dry run is what makes a map measurement-free: it reports, per tensor, the
type the built-in heuristic would have chosen and how big the file would be, so
the solver can hand the body exactly the heuristic's bytes and pin a
shape-incompatible tensor to the heuristic's own fallback type.

This module never runs anything — `dry_run_args` builds the argv a caller
executes inside the pinned image, `parse_inventory` turns the captured output
into the inventory the generator consumes.
"""
from __future__ import annotations

import re

from . import LayoutError
from .solver import BLOCK

# `[  n/  N] NAME  - [ne0, ne1, ne2, ne3], type = SRC, <tail>`; the tail carries the new type
# either as the old `(TYPE, N bytes)` or as the current `size = A MiB -> B MiB (TYPE)`. A tail
# without a new type means the tensor is copied through: its type stays the source type.
DRY_RE = re.compile(r"^\[ *\d+/ *\d+\] (\S+) +- \[ *(\d+), *(\d+), *(\d+), *(\d+)\], type = *([^,]+), *(.*)$")
DRY_OLD_TAIL = re.compile(r"\(([^,)]+), *(\d+) bytes\)")
DRY_NEW_TAIL = re.compile(r"->\s*[\d.]+\s*\w+\s*\((\S+?)\)")


def type_bytes(ty: str, ne0: int, rows: int, line: str) -> int:
    """ggml on-disk size of a tensor, from the block table (the current dry run prints MiB only)."""
    if ty not in BLOCK:
        raise LayoutError(
            f"unknown tensor type {ty!r} in the dry-run output (add it to layout.solver.BLOCK):\n  {line}"
        )
    blk, bpb = BLOCK[ty]
    return rows * (ne0 // blk) * bpb


def dry_run_args(
    ftype: str,
    model_in: str,
    out_in: str,
    imatrix_in: str | None,
    pins: list[tuple[str, str]],
    threads: int,
) -> list[str]:
    """llama-quantize argv for a dry run, with container-side paths.

    Only class-A pins (the tensors without imatrix data) belong on the command
    line, as (pattern, type) pairs carrying the same type the map will give them,
    so the inventory's bytes are the real file's: a shape-incompatible tensor must
    show up in the inventory carrying the type the built-in heuristic falls back
    to, which then becomes its pin line.
    """
    cmd = ["./llama-quantize", "--dry-run"]
    if imatrix_in:
        cmd += ["--imatrix", imatrix_in]
    for p, ty in pins:
        cmd += ["--tensor-type", f"{p}={ty}"]
    cmd += [model_in, out_in, ftype, str(threads)]
    return cmd


def parse_inventory(text: str) -> list[dict]:
    """dry-run output -> per-tensor inventory [{name, ne0, ne1, ne2, rows, src, type, bytes}]."""
    inv = []
    for line in text.splitlines():
        m = DRY_RE.match(line)
        if not m:
            continue
        name, ne0, ne1, ne2, ne3, src, tail = m.groups()
        ne0, rows = int(ne0), int(ne1) * int(ne2) * int(ne3)
        src = src.strip().lower()
        old = DRY_OLD_TAIL.search(tail)
        if old:
            ty, nbytes = old.group(1).strip().lower(), int(old.group(2))
        else:
            new = DRY_NEW_TAIL.search(tail)
            ty = new.group(1).strip().lower() if new else src
            nbytes = type_bytes(ty, ne0, rows, line)
        inv.append({"name": name, "ne0": ne0, "ne1": int(ne1), "ne2": int(ne2), "rows": rows,
                    "src": src, "type": ty, "bytes": nbytes})
    if not inv:
        raise LayoutError("could not parse the dry-run inventory")
    return inv
