# src/auto_quant_v2/layout/shape.py
"""The body-shape key: what makes two models the same shape for a canary.

A hash of the architecture string plus the sorted `name=dims` lines of the
model's *body* tensors — the embedding class left out, so a tune that extended
the vocabulary still matches its base, and every tensor of an MTP block left
out, so a tune that dropped the NextN layers its base carried still matches.
A different size, tied-ness, layer count or expert count lands on a different
key.

Deliberately not part of the generator key's hashed set (`generate.
generator_key()` covers dryrun.py, generate.py and solver.py only): this file
describes the model, not the rules that lay a model out, so adding to it must
not move the generator key and must not invalidate cached verdicts.

Stdlib only.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, Sequence

from .generate import EMBD

_BLOCK_RE = re.compile(r"^blk\.(\d+)\.")


def shape_key(
    arch: str | None,
    tensors: Sequence[tuple[str, Sequence[int]]],
    *,
    mtp_blocks: Iterable[int] = (),
) -> str:
    """First 16 hex characters of a SHA-256 over the architecture and the
    body tensors' names and dims, like the generator key."""
    skip_blocks = {int(b) for b in mtp_blocks}
    lines = []
    for name, dims in tensors:
        if name in EMBD:
            continue
        m = _BLOCK_RE.match(name)
        if m is not None and int(m.group(1)) in skip_blocks:
            continue
        lines.append(f"{name}={'x'.join(str(d) for d in dims)}")
    payload = (arch or "") + "\n" + "\n".join(sorted(lines))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
