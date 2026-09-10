"""Per-tensor layout maps: pins, the embedding rule, and the body solve.

Given a model's q8_0 dry-run inventory, the ftype's own dry-run inventory and
the imatrix's tensor list, this module produces the text of one llama-quantize
`--tensor-type-file` plus the metadata describing how it was built:

  * pins, in three classes and in this order — class A, every quantizable tensor
    the imatrix does not cover (an IQ type on one aborts llama-quantize): a whole
    block the request's MTP detection names is pinned to q4_0, any other one (a
    single tensor, or a whole block not detected as MTP) to the ftype's
    imatrix-free floor; class B, every weight whose ne0 is not a multiple of 256
    (embeddings included), pinned to the type the ftype's own dry run fell back
    to, which is exactly what the built-in heuristic would have chosen, so the
    map is byte-neutral there; class C, every tiny body tensor (under
    TINY_PARAMS parameters), pinned to f32. All three classes stay out of the
    body solve and out of the embedding rule, and pins are the first lines of
    the file: in llama-quantize the first matching pattern wins.
  * the embedding rule: each embedding-class tensor (token_embd / output /
    per_layer_token_embd) is placed by its share of the predicted file — at or
    below EMBD_LO to q8_0, at or above EMBD_HI to the target-based rule's type
    for the ftype and tied/untied, between them one notch above that rule, except
    that at the Q4 rungs and above the notch never exceeds the heuristic's own
    type for the tensor.
  * the body rule (every model class at every share rung; plain dense
    all-attention models without per-layer embeddings are solved from the prior's
    own dense table, the `dense|...` cells, while per-layer-embedding models
    (Gemma-E) classify as dense but stay on the pooled cells): the ftype-identity search — the base type is a floor and the rung is the tier's
    minimum share of body bytes held at it (S/M/L in TIER_SHARE), with no byte
    budget at all; everything above that share is spent by damage per byte from
    the frozen cross-model prior and the bitrate falls where it falls.
  * K-quant ftypes never receive IQ types; on MoE and per-layer-embedding models
    an IQ ftype (except IQ4_NL, which stays repackable) may also crush below the
    base type, in a two-pass solve: bump-only at the tier's share, then re-solved
    at those bytes with the crush candidates allowed, the share still held.

Stdlib only and free of I/O beyond reading the packaged prior: the caller runs
the dry runs and writes the files.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, NamedTuple

from . import LayoutError
from .solver import (
    ALL_TYPES,
    IQ_TYPES,
    K_TYPES,
    NO_IMATRIX_OK,
    bits_per_weight,
    prior_value,
    solve_majority,
    tensor_bytes,
)

EMBD = {"token_embd.weight", "output.weight", "per_layer_token_embd.weight"}
# the recipe for a draft-only MTP/NextN block: a block is MTP because the model's MTP
# detection said so, never because of the pin's shape. The quantize command line pins those
# blocks the same way, and the map's line agrees with it so the map alone reproduces the file
BLOCK_PIN_TYPE = "q4_0"
SRC_TYPES = {"bf16", "f16", "f32"}
# a body tensor below this many parameters costs at most 1 MiB at f32, so it is kept at
# full precision instead of being solved like a real weight
TINY_PARAMS = 1 << 18

# ftype -> (base tensor type, family). Beware llama-quantize's naming: ftype iq2_s produces
# IQ2_XS tensors, iq2_m produces IQ2_S, iq3_xs/iq3_m produce IQ3_S.
FTYPES = {
    "iq1_s": ("iq1_s", "iq"), "iq1_m": ("iq1_m", "iq"),
    "iq2_xxs": ("iq2_xxs", "iq"), "iq2_xs": ("iq2_xs", "iq"), "iq2_s": ("iq2_xs", "iq"), "iq2_m": ("iq2_s", "iq"),
    "iq3_xxs": ("iq3_xxs", "iq"), "iq3_xs": ("iq3_s", "iq"), "iq3_s": ("iq3_s", "iq"), "iq3_m": ("iq3_s", "iq"),
    "q2_k": ("q2_k", "k"), "q2_k_s": ("q2_k", "k"),
    "q3_k_s": ("q3_k", "k"), "q3_k_m": ("q3_k", "k"), "q3_k_l": ("q3_k", "k"),
    "q4_k_s": ("q4_k", "k"), "q4_k_m": ("q4_k", "k"),
    "q5_k_s": ("q5_k", "k"), "q5_k_m": ("q5_k", "k"), "q6_k": ("q6_k", "k"),
    "iq4_xs": ("iq4_xs", "iq"), "iq4_nl": ("iq4_nl", "iq"),
    # the virtual tier-L rungs: llama-quantize has no such ftype — the dry run and the
    # quantize command use the base ftype the caller resolves (Q4_K_M / Q6_K), and the map
    # covers every body tensor, so that ftype only sets the pins' fallback types
    "q4_k_l": ("q4_k", "k"), "q6_k_l": ("q6_k", "k"),
}
# every ftype is the same recipe — a share rung (`ident`): the embedding class placed by the
# share rule, the body solved at the tier's minimum base share of body bytes with no byte
# budget at all. There is no per-ftype exception left.
BODY_TYPES = {"iq4_nl": ["iq4_nl", "q5_k", "q6_k", "q8_0"]}  # keep IQ4_NL files repackable: no q4_k/iq4_xs in the body

# tier -> the minimum share of body bytes that stays at the base type. S is "the base type
# with a few choice bumps", L several high-value bumps, M in between; the rung is that share
# and nothing else — there is no byte budget, and the bitrate is reported, not targeted.
TIER_SHARE = {"S": 0.90, "M": 0.70, "L": 0.50}
TIER_S = {"q3_k_s", "q4_k_s", "q2_k_s", "q5_k_s", "q6_k", "iq3_xs", "iq2_xs", "iq4_xs"}
TIER_L = {"q3_k_l", "iq3_m", "q4_k_l", "q6_k_l"}
# every other share ftype (q3_k_m, q4_k_m, q2_k, q5_k_m, iq3_s, iq3_xxs, iq2_s, iq2_m,
# iq2_xxs, iq4_nl, iq1_s, iq1_m) is tier M

# the embedding-class thresholds, as a share of the predicted file: at or below EMBD_LO a
# table is cheap enough to keep at q8_0, at or above EMBD_HI it dominates the file and stays
# on the target-based rule, between them it moves one notch up this ladder — at the Q4 rungs
# and above capped at the heuristic's own type for the tensor, which the notch never exceeds
EMBD_LO, EMBD_HI = 0.03, 0.15
EMBD_LADDER = ["q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_0"]

PRIOR_PATH = Path(__file__).with_name("prior.json")
# the packaged prior is identical to the research campaign's copy
# (research/layout-maps/scripts/layout-prior.json) and is never hand-edited: it is
# rebuilt only by that campaign's --dump-prior over its measurement CSV.
PRIOR_NAME = "layout/prior.json"
GENERATOR = "auto_quant_v2.layout"
# the files whose content defines what a map is: the prior and the three rule modules.
_KEY_SOURCES = ("dryrun.py", "generate.py", "solver.py")


def _rule_digest(src: str) -> str:
    """one rule module's contribution to the key: the module's AST dump with every
    docstring removed, so comments, docstrings and formatting are invisible to it."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


@lru_cache(maxsize=None)
def generator_key() -> str:
    """the generator's content key: the first 16 hex characters of a SHA-256 over
    prior.json's bytes and the docstring-stripped AST dump of each rule module, in
    that order. Comments, docstrings and formatting do not move it; any change to a
    constant, a table, a rule or the solver does."""
    h = hashlib.sha256()
    h.update(PRIOR_PATH.read_bytes())
    for name in _KEY_SOURCES:
        h.update(_rule_digest(Path(__file__).with_name(name).read_text()).encode())
    return h.hexdigest()[:16]


class Prior(NamedTuple):
    """the frozen cross-model prior: kind x band cells, kind cells and r(t)."""

    prior: dict
    kind_prior: dict
    r: dict
    raw: dict


@dataclass
class Pins:
    """the three pin classes, in file order; `shape` maps a class-B pattern to its tensor,
    `fixed` a class-C pattern to the type its line carries, and `mtp_blocks` holds the block
    ids the model's MTP detection named."""

    patterns: list[str] = field(default_factory=list)
    why: dict[str, str] = field(default_factory=dict)
    shape: dict[str, str] = field(default_factory=dict)
    fixed: dict[str, str] = field(default_factory=dict)
    mtp_blocks: frozenset[int] = frozenset()

    @property
    def class_a(self) -> list[str]:
        """the no-imatrix pins — the only ones that go on a dry-run command line."""
        return [p for p in self.patterns if p not in self.shape and p not in self.fixed]

    def class_a_types(self, ftype: str) -> list[tuple[str, str]]:
        """the class-A pins as (pattern, type) for one ftype, in file order.

        The patterns themselves are ftype-independent; the type is not — a whole
        block the MTP detection named goes to BLOCK_PIN_TYPE, every other class-A
        pin (a single tensor, or a whole block not detected as MTP) to the ftype's
        imatrix-free floor.
        """
        floor = no_imatrix_type(ftype)
        return [(p, BLOCK_PIN_TYPE if block_id(p) in self.mtp_blocks else floor)
                for p in self.class_a]


@dataclass
class MapResult:
    lines: list[str]
    text: str
    info: dict
    variant: str


def load_prior(path=None) -> Prior:
    """the packaged prior.json (or another copy of it) as a Prior."""
    if path is None:
        return _load_prior_cached(PRIOR_PATH)
    return _load_prior(Path(path))


def _load_prior(path: Path) -> Prior:
    d = json.loads(path.read_text())

    def key(s):
        p = s.split("|")
        return tuple(None if x == "-" else x for x in p)

    return Prior({key(k): v for k, v in d["prior"].items()},
                 {key(k): v for k, v in d["kind_prior"].items()},
                 d["r"], d)


@lru_cache(maxsize=None)
def _load_prior_cached(path: Path) -> Prior:
    return _load_prior(path)


def supported_ftypes() -> list[str]:
    """the ftypes the generator can build a map for, upper-case."""
    return [ft.upper() for ft in FTYPES]


def maps_apply(quant_base: str) -> bool:
    """True when a layout map changes anything for this ftype.

    Every ftype the generator knows is a share rung, so this is exactly
    membership in FTYPES. Q8_0, the legacy and full-precision types, and the
    `_L` names with no rung under maps (Q5_K_L, Q2_K_L, Q3_K_XL) are False.
    """
    return quant_base.lower() in FTYPES


def covered_names(tensor_names: Iterable[str]) -> set[str]:
    """tensor names covered by an imatrix, from the imatrix GGUF's tensor list."""
    return {re.sub(r"\.(in_sum2|in_sum|counts)$", "", n) for n in tensor_names}


def tier_of(ft: str) -> str:
    """the ftype's rung tier: S, M or L."""
    ft = ft.lower()
    return "S" if ft in TIER_S else ("L" if ft in TIER_L else "M")


def notch_up(ty: str) -> str:
    """the next embedding type above `ty` on the embedding ladder (q8_0 is the top)."""
    for cand in EMBD_LADDER:
        if bits_per_weight(cand) > bits_per_weight(ty) + 1e-9:
            return cand
    return "q8_0"


def embd_rule(ftype, tied):
    """embedding class -> type, by target ftype and tied/untied;
    per_layer_token_embd follows token_embd. Monotone in the ftype's bitrate."""
    if ftype.startswith(("iq1", "iq2")):
        return {"token_embd": "q3_k"} if tied else {"token_embd": "q2_k", "output": "q4_k"}
    if ftype.startswith("q2_k"):
        return {"token_embd": "q4_k"} if tied else {"token_embd": "q2_k", "output": "q4_k"}
    if ftype.startswith(("q3_k", "iq3")):
        # untied IQ3: q3_k, not q2_k — token_embd q2_k costs ~12 % of a 9B IQ3_S file's KLD
        return {"token_embd": "q4_k"} if tied else {"token_embd": "q3_k", "output": "q4_k"}
    if ftype.startswith(("q4_k", "iq4")):
        # IQ4 shares Q4_K's line: without it an IQ4 rung falls back to the heuristic's
        # richer choice and gets a bigger embedding than Q4_K_M
        return {"token_embd": "q5_k"} if tied else {"token_embd": "q4_k", "output": "q6_k"}
    return None


def block_id(pattern: str) -> int | None:
    """the block id of a class-A whole-block pattern (`^blk\\.N\\.`), None for anything
    else — a single-tensor pattern ends in `$`."""
    m = re.fullmatch(r"\^blk\\\.(\d+)\\\.", pattern)
    return int(m.group(1)) if m else None


def no_imatrix_type(ftype: str) -> str:
    """the ftype's imatrix-free floor: the smallest type llama-quantize accepts without
    imatrix data, inside the ftype's own family, at or above the base type's bitrate.

    q2_k is excluded for Q2_K_S — llama-quantize requires an imatrix for q2_k tensors
    inside a Q2_K_S file.
    """
    ft = ftype.lower()
    if ft not in FTYPES:
        raise LayoutError(f"unsupported ftype {ftype}: choose from {', '.join(FTYPES)}")
    base, fam = FTYPES[ft]
    cands = list(IQ_TYPES if fam == "iq" else K_TYPES)
    if base not in cands:
        cands.append(base)
    cands = [ty for ty in cands if ty in NO_IMATRIX_OK
             and bits_per_weight(ty) >= bits_per_weight(base) - 1e-9]
    if ft == "q2_k_s":
        cands = [ty for ty in cands if ty != "q2_k"]
    if not cands:
        raise LayoutError(f"no imatrix-free type at or above {base} for ftype {ftype}")
    return min(cands, key=bits_per_weight)


def find_pins(inv_q8, covered: set[str], mtp_blocks: Iterable[int] = ()) -> Pins:
    """quantizable tensors the imatrix does not cover (+ ne0 % 256 != 0 weights) -> patterns.

    Three classes, in this order: class A — no imatrix data (a whole block, or a single tensor);
    an IQ type on them aborts llama-quantize, so a whole block the model's MTP detection names
    (`mtp_blocks`) is pinned to BLOCK_PIN_TYPE and every other class-A pin — a single tensor, or a
    whole block not detected as MTP — to the ftype's imatrix-free floor (resolved per ftype by
    build_map). Class B — a row width
    that is not a multiple of 256, embeddings included; no K/IQ type applies to them at all, so
    they carry the type the ftype's own dry run fell back to (filled in per ftype by build_map).
    Class C — a body tensor under TINY_PARAMS parameters; structurally sensitive and trivially
    cheap, so it is kept at f32.
    """
    mtp = frozenset(int(b) for b in mtp_blocks)
    quant = [t for t in inv_q8 if t["src"] in SRC_TYPES and t["type"] != t["src"]]
    missing = [t["name"] for t in quant if t["name"] not in covered and t["name"] not in EMBD]
    by_block = defaultdict(list)
    for t in quant:
        m = re.match(r"blk\.(\d+)\.", t["name"])
        if m:
            by_block[int(m.group(1))].append(t["name"])
    pins, done, why = [], set(), {}
    for n in missing:
        m = re.match(r"blk\.(\d+)\.", n)
        if m:
            b = int(m.group(1))
            if all(x in missing for x in by_block[b]):
                if b not in done:
                    pins.append(f"^blk\\.{b}\\."); done.add(b)
                    why[pins[-1]] = ("MTP block without imatrix data" if b in mtp
                                     else "block without imatrix data")
                continue
        pins.append("^" + n.replace(".", "\\.") + "$"); why[pins[-1]] = "no imatrix data"
    shape_pins = {}
    for t in quant:
        if t["ne0"] % 256 and t["name"] not in missing:
            p = "^" + t["name"].replace(".", "\\.") + "$"
            pins.append(p)
            why[p] = (f"ne0={t['ne0']} not a multiple of 256"
                      + (" (embedding: excluded from the embedding rule)" if t["name"] in EMBD else ""))
            shape_pins[p] = t["name"]
    fixed_pins = {}
    for t in quant:
        name = t["name"]
        params = t["ne0"] * t["rows"]
        m = re.match(r"blk\.(\d+)\.", name)
        if not m or params >= TINY_PARAMS or name in missing or int(m.group(1)) in done:
            continue
        p = "^" + name.replace(".", "\\.") + "$"
        if p in shape_pins:
            continue
        pins.append(p)
        why[p] = f"tiny: {params:,} params, kept at f32"
        fixed_pins[p] = "f32"
    return Pins(pins, why, shape_pins, fixed_pins, mtp)


def annotate(inv, pin_res):
    """kind / layer / layer-kind / pinned flags."""
    # a block is linear when it holds an ssm_* tensor; attention spellings never decide it
    # (Qwen's GDN blocks carry attn_qkv, Ling's KDA blocks carry attn_q)
    blocks_ssm = {int(m.group(1)) for t in inv for m in [re.match(r"blk\.(\d+)\.ssm_", t["name"])] if m}
    blocks_body = {int(m.group(1)) for t in inv for m in [re.match(r"blk\.(\d+)\.", t["name"])]
                   if m and t["src"] in SRC_TYPES and t["type"] != t["src"] and not any(p.search(t["name"]) for p in pin_res)}
    hybrid = any(b in blocks_ssm for b in blocks_body) and any(b not in blocks_ssm for b in blocks_body)
    for t in inv:
        t["pinned"] = any(p.search(t["name"]) for p in pin_res)
        t["quant"] = t["src"] in SRC_TYPES and t["type"] != t["src"]
        t["kind"] = t["layer"] = t["lk"] = None
        m = re.match(r"blk\.(\d+)\.(.+)\.weight$", t["name"])
        if m:
            t["layer"] = int(m.group(1)); t["kind"] = m.group(2)
            if hybrid:
                t["lk"] = "gdn" if t["layer"] in blocks_ssm else "attn"
        elif t["name"] in EMBD:
            t["kind"] = t["name"].replace(".weight", "")
    return inv


def classify(inv):
    kinds = {t["kind"] for t in inv if t["kind"]}
    if any(re.match(r"ffn_(down|gate|up)_exps$", k) for k in kinds):
        return "moe"
    if any(k.startswith("ssm_") for k in kinds):
        return "hybrid"
    return "dense"


def natural(name):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", name)]


def build_map(ftype, inv, pins: Pins, *, prior: Prior | None = None,
              model: str | None = None, imatrix: str | None = None) -> MapResult:
    """the map text and metadata for one ftype.

    `inv` is the parsed inventory of that ftype's own dry run — for a virtual
    tier-L ftype, its base ftype's dry run — run with the class-A pins on the
    command line, so a shape-pinned tensor carries the heuristic's fallback type.
    An `_L` ftype is just another rung: tier L of its base type, its embeddings
    placed by the same share rule as every other rung. `model` and `imatrix` are
    recorded in the metadata only.
    """
    ft = ftype.lower()
    if ft not in FTYPES:
        raise LayoutError(f"unsupported ftype {ftype} (legacy/near-lossless types stay on the built-in "
                          f"heuristic): choose from {', '.join(FTYPES)}")
    if prior is None:
        prior = load_prior()
    pr, kind_prior, r = prior.prior, prior.kind_prior, prior.r
    base, fam = FTYPES[ft]
    pin_res = [re.compile(p) for p in pins.patterns]
    inv = annotate([dict(t) for t in inv], pin_res)
    inv_type = {t["name"]: t["type"] for t in inv}
    shape_type = {}  # class-B pattern -> the fallback type this ftype's dry run assigned
    for p, name in pins.shape.items():
        ty = inv_type.get(name)
        if ty is None:
            raise LayoutError(f"{ft}: shape-pinned tensor {name} is missing from the dry-run inventory")
        if ty in SRC_TYPES:
            raise LayoutError(f"{ft}: shape-pinned tensor {name} was copied through as {ty} in the dry run — "
                              f"no quantized fallback to pin it to")
        shape_type[p] = ty
    pin_why = dict(pins.why)
    for p in pins.shape:
        pin_why[p] = f"{pin_why[p]} -> heuristic {shape_type[p]}"
    arch = classify(inv)
    tied = not any(t["name"] == "output.weight" for t in inv)
    per_layer = any(t["name"] == "per_layer_token_embd.weight" for t in inv)
    if arch == "dense" and not per_layer:
        # a plain dense all-attention model is solved from the prior's own dense table: its
        # attention tensors are shaped unlike the pooled table's, whose per-tensor shares
        # would read as several times the per-byte sensitivity here. A per-layer-embedding
        # model (Gemma-E) classifies as dense too but is measured better on the pooled cells,
        # so it keeps layer-kind None
        for t in inv:
            if t["layer"] is not None:
                t["lk"] = "dense"
    bpw = bits_per_weight
    # every ftype is a share rung: the tier's base share of body bytes replaces the byte
    # budget entirely, and the body is solved for every model class
    recipe = "ident"
    rung_share = TIER_SHARE[tier_of(ft)]
    n_layer = 1 + max(t["layer"] for t in inv if t["layer"] is not None and t["quant"] and not t["pinned"])

    # class-C tensor name -> its pin line's type, for the byte prediction
    fixed_names = {}
    for p, ty in pins.fixed.items():
        rx = re.compile(p)
        for t in inv:
            if rx.search(t["name"]):
                fixed_names[t["name"]] = ty
    # the embedding rule's inputs: each embedding-class tensor's share of the whole predicted
    # file (the heuristic's own inventory) and the type the target-based rule gives it
    file_bytes_heur = sum(t["bytes"] for t in inv) or 1
    target_rule = embd_rule(ft, tied)

    def rule_type(t):
        """the target-based rule's type; at >= Q5_K, where it has none, the heuristic's own."""
        if not target_rule:
            return t["type"]
        return target_rule["token_embd"] if t["kind"] == "per_layer_token_embd" else target_rule[t["kind"]]

    # the middle band's notch cap applies only at the Q4 rungs and above
    cap_ftype = ft.startswith(("q4_k", "iq4", "q5_k", "q6_k"))

    embd_choice, embd_shares = {}, {}
    for t in inv:
        if t["name"] in EMBD and t["quant"] and not t["pinned"]:
            share = t["bytes"] / file_bytes_heur
            # value per overall bit: a table that is a small share of the file is cheap
            # to keep at q8_0, one that dominates it stays on the target-based rule
            base_ty = rule_type(t)
            ty = "q8_0" if share <= EMBD_LO else (base_ty if share >= EMBD_HI else notch_up(base_ty))
            # at the Q4 rungs and above the middle band's notch is a ceiling, not a raise:
            # the tensor never gets more bits per weight than the heuristic's own type for it
            if (EMBD_LO < share < EMBD_HI and cap_ftype
                    and bits_per_weight(ty) > bits_per_weight(t["type"])):
                ty = t["type"]
            if ty not in NO_IMATRIX_OK:
                ty = "q4_k"
            embd_choice[t["name"]] = ty
            embd_shares[t["name"]] = {"share": round(share, 4), "type": ty}
    # a shape-incompatible embedding is pinned, not chosen by the rule; it is still an embedding
    # for the record's purposes, at the type its pin line carries
    embd_meta = dict(embd_choice)
    for p, name in pins.shape.items():
        if name in EMBD:
            embd_meta[name] = shape_type[p]

    fam_types = BODY_TYPES.get(ft, K_TYPES if fam == "k" else ALL_TYPES)
    if base not in fam_types:
        fam_types = fam_types + [base]
    fam_types = [ty for ty in fam_types if ty in r]
    # crushing below the base type (majority still >= 50 % at it): measured as a win on MoE and
    # per-layer-embedding models at IQ ftypes, a wash on hybrids, a loss on plain dense. K types
    # never crush; IQ4_NL keeps its floor so the file stays repackable.
    auto_crush = fam == "iq" and ft != "iq4_nl" and (arch == "moe" or per_layer)
    # bump-only (`ident`) unless the rule above applies
    weight_types = (fam_types if auto_crush
                    else [ty for ty in fam_types if bpw(ty) >= bpw(base) - 1e-9])

    items, info, sens, body_ref, embd_ref, body_params = [], {}, {}, 0, 0, 0
    for t in inv:
        if not t["quant"] or t["pinned"]:
            continue
        name = t["name"]
        if name in EMBD:
            ty = embd_choice[name]
            items.append((name, [(ty, 0.0, tensor_bytes(ty, t["ne0"], t["rows"]))]))
            embd_ref += t["bytes"]
        elif t["layer"] is not None:
            body_ref += t["bytes"]
            body_params += t["ne0"] * t["rows"]
            cands = weight_types
            s = prior_value(pr, kind_prior, t, n_layer)
            sens[name] = s
            opts = [(ty, s * r[ty], tensor_bytes(ty, t["ne0"], t["rows"])) for ty in cands]
            opts.sort(key=lambda o: o[2])
            items.append((name, opts))
        info[name] = t
    crush_pass_bytes = None
    frac = rung_share
    # a share rung has no byte target at all: the share cap alone defines it
    budget = float("inf")
    if auto_crush:
        # the IQ crush pass has no heuristic budget to be neutral against: pass 1 solves
        # bump-only at the tier's share, pass 2 re-solves at those bytes with the crush
        # candidates allowed, so a crush only ever funds a bump
        bump_items = [(name, opts if name in EMBD
                       else [o for o in opts if bpw(o[0]) >= bpw(base) - 1e-9] or opts)
                      for name, opts in items]
        pass1 = solve_majority(bump_items, budget, base, frac)
        crush_pass_bytes = sum(next(o[2] for o in opts if o[0] == pass1[name])
                               for name, opts in bump_items)
        budget = crush_pass_bytes
    choice = solve_majority(items, budget, base, frac)
    pred = sum(next(o[2] for o in opts if o[0] == choice[name]) for name, opts in items)
    heur = body_ref + embd_ref

    # A -> q4_0 for a whole block the MTP detection named, the ftype's imatrix-free floor for
    # any other one; B -> the ftype's own fallback; C -> f32
    class_a_type = dict(pins.class_a_types(ft))
    lines = [f"{p}=" + (shape_type.get(p) or pins.fixed.get(p) or class_a_type[p]) for p in pins.patterns]
    for name in sorted(embd_choice, key=natural):
        lines.append("^" + name.replace(".", "\\.") + "$=" + embd_choice[name])
    body_lines = [("^" + name.replace(".", "\\.") + "$=" + choice[name]) for name, _ in items if name not in EMBD]
    lines += sorted(body_lines, key=natural)
    hist = defaultdict(lambda: defaultdict(int))
    for name, _ in items:
        if name not in EMBD:
            hist[info[name]["kind"]][choice[name]] += 1
    base_share = 0.0
    bb = [(name, next(o[2] for o in opts if o[0] == choice[name]), choice[name]) for name, opts in items if name not in EMBD]
    if bb:
        base_share = sum(b for _, b, ty in bb if ty == base) / sum(b for _, b, _ in bb)
    body_bytes = sum(b for _, b, _ in bb)
    # the whole file at the types this map produces: solved tensors at their choice, pinned
    # ones at their pin line's type, everything else (f32 norms, copied-through) unchanged
    chosen_bytes = {name: next(o[2] for o in opts if o[0] == choice[name]) for name, opts in items}
    pin_type = {pins.shape[p]: shape_type[p] for p in pins.patterns if p in pins.shape}
    pin_type.update(fixed_names)
    for p, ty in class_a_type.items():
        rx = re.compile(p)
        for t in inv:
            if t["quant"] and rx.search(t["name"]):
                pin_type[t["name"]] = ty
    file_bytes_pred, total_params = 0, 0
    for t in inv:
        total_params += t["ne0"] * t["rows"]
        if t["name"] in chosen_bytes:
            file_bytes_pred += chosen_bytes[t["name"]]
        elif t["pinned"] and t["quant"]:
            file_bytes_pred += tensor_bytes(pin_type.get(t["name"], "q8_0"), t["ne0"], t["rows"])
        else:
            file_bytes_pred += t["bytes"]
    variant = f"shr{rung_share:.2f}" + ("+crush" if auto_crush else "")
    meta = {"model": model, "imatrix": imatrix, "ftype": ft.upper(), "base_type": base, "family": fam,
            "recipe": recipe + ("+crush" if auto_crush else ""),
            "variant": variant,
            "auto_crush": bool(auto_crush),
            "rung_mode": "share", "tier": tier_of(ft),
            "rung_share": rung_share, "crush_pass_bytes": crush_pass_bytes,
            "embd_shares": embd_shares,
            "arch": arch, "tied": tied, "n_layer": n_layer, "body_rule": True,
            "pins": {p: pin_why[p] for p in pins.patterns},
            "shape_pins": {pins.shape[p]: shape_type[p] for p in pins.patterns if p in pins.shape},
            "fixed_pins": {name: fixed_names[name] for name in sorted(fixed_names, key=natural)},
            "embeddings": embd_meta,
            "bytes": {"heuristic_body+embd": heur, "map_body+embd": pred, "ratio": round(pred / heur, 4) if heur else None,
                      "note": "quantizable, unpinned tensors only; pinned/unquantized tensors are identical in both"},
            "body_base_share": round(base_share, 3) if bb else None,
            "body_bpw": round(8.0 * body_bytes / body_params, 3) if (bb and body_params) else None,
            "body_bpw_heuristic": round(8.0 * body_ref / body_params, 3) if body_params else None,
            "total_params": total_params, "file_bytes_pred": file_bytes_pred,
            "file_bpw_pred": round(8.0 * file_bytes_pred / total_params, 3) if total_params else None,
            "histogram": {k: dict(sorted(v.items(), key=lambda x: -x[1])) for k, v in sorted(hist.items())},
            "prior": PRIOR_NAME, "pin_rule": "floor", "embd_cap": "heur", "generator": GENERATOR,
            "generator_key": generator_key()}
    return MapResult(lines=lines, text="\n".join(lines) + "\n", info=meta, variant=variant)
