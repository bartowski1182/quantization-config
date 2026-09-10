"""Per-tensor type solver — the search half of the layout maps.

A port of the research campaign's `layout-search.py` (the parts the map
generator uses): the ggml block table, the candidate type lists, the
structural sensitivity prior lookup and the two Lagrangian solvers. The
damage model is additive:

    damage(layout) = sum_i  s_i * r(t_i)

with `s_i` the tensor's sensitivity from the frozen cross-model prior and
`r(t)` the type's damage relative to q2_k. `solve_majority` keeps at least the
given fraction of body bytes at the base type while meeting a byte budget; the
budget may be infinite, in which case the fraction alone bounds the solution.

Pure stdlib and side-effect free: no I/O, no Docker, no subprocesses.
"""
from __future__ import annotations

import re

# ggml block size (elements) and bytes per block
BLOCK = {
    "q2_k": (256, 84), "q3_k": (256, 110), "q4_k": (256, 144), "q5_k": (256, 176),
    "q6_k": (256, 210), "q8_0": (32, 34), "iq1_s": (256, 50), "iq1_m": (256, 56),
    "iq2_xxs": (256, 66), "iq2_xs": (256, 74), "iq2_s": (256, 82), "iq3_xxs": (256, 98),
    "iq3_s": (256, 110), "iq4_xs": (256, 136), "iq4_nl": (32, 18),
    "f16": (1, 2), "bf16": (1, 2), "f32": (1, 4),
    # legacy/fallback types the built-in heuristic can still emit in a dry run
    "q4_0": (32, 18), "q4_1": (32, 20), "q5_0": (32, 22), "q5_1": (32, 24), "mxfp4": (32, 17),
}
K_TYPES = ["q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_0"]
IQ_TYPES = ["iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq4_xs"]
ALL_TYPES = IQ_TYPES + K_TYPES
NO_IMATRIX_OK = {"q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_0", "iq3_s", "iq4_xs", "iq4_nl", "f16", "f32"}

# embedding-class tensors: chosen by the embedding rule, never part of the body
EMBD_KINDS = ("token_embd", "output", "per_layer_token_embd")


def is_embd_name(name):
    return name.replace(".weight", "") in EMBD_KINDS


def tensor_bytes(t, ne0, rows):
    blk, bpb = BLOCK[t]
    if ne0 % blk:
        raise ValueError(f"ne0 {ne0} not a multiple of block {blk} for {t}")
    return rows * (ne0 // blk) * bpb


def bits_per_weight(ty):
    return 8.0 * BLOCK[ty][1] / BLOCK[ty][0]


def band(layer, n_layer):
    if layer == 0:
        return "first"
    if layer == n_layer - 1:
        return "last"
    rel = layer / (n_layer - 1)
    return "early" if rel < 0.25 else ("mid" if rel < 0.75 else "late")


def dense_kind(kind):
    """map MoE kinds onto the dense ffn kinds for the structural prior."""
    return re.sub(r"^(ffn_(?:down|gate|up))_(?:exps|shexp)$", r"\1", kind)


def prior_value(prior, kind_prior, t, n_layer):
    key = (t["lk"], dense_kind(t["kind"]), band(t["layer"], n_layer))
    if key in prior:
        return prior[key]
    k2 = (t["lk"], dense_kind(t["kind"]))
    if k2 in kind_prior:
        return kind_prior[k2]
    # unknown kind: the median of the Qwen-fit attn / gdn cells, for every class — the dense
    # table is a second fit and must not move any other class's maps
    vals = sorted(v for k, v in prior.items() if k[0] is not None and k[0] != "dense")
    return vals[len(vals) // 2]


def solve(items, budget):
    """items: list of (name, [(type, damage, bytes), ...]) — options sorted by bytes asc.
    Returns {name: type}."""
    def pick(lmb):
        choice = {}
        tot_b = tot_d = 0.0
        for name, opts in items:
            best = min(opts, key=lambda o: o[1] + lmb * o[2])
            choice[name] = best
            tot_b += best[2]; tot_d += best[1]
        return choice, tot_b, tot_d

    lo, hi = 0.0, 1.0
    c, b, _ = pick(hi)
    while b > budget:
        hi *= 4
        c, b, _ = pick(hi)
        if hi > 1e12:
            break
    c0, b0, _ = pick(lo)
    if b0 <= budget:
        return {n: o[0] for n, o in c0.items()}
    for _ in range(80):
        mid = (lo + hi) / 2
        c, b, _ = pick(mid)
        if b <= budget:
            hi = mid
        else:
            lo = mid
    choice, tot_b, _ = pick(hi)
    # greedy fill of the slack: best damage reduction per extra byte that fits
    opts_by = dict(items)
    while True:
        slack = budget - tot_b
        best = None
        for name, cur in choice.items():
            for o in opts_by[name]:
                db = o[2] - cur[2]
                dd = cur[1] - o[1]
                if db <= 0 or db > slack or dd <= 0:
                    continue
                ratio = dd / db
                if best is None or ratio > best[0]:
                    best = (ratio, name, o)
        if best is None:
            break
        _, name, o = best
        tot_b += o[2] - choice[name][2]
        choice[name] = o
    return {n: o[0] for n, o in choice.items()}


def solve_majority(items, budget, base, frac):
    """two-multiplier Lagrangian: every tensor picks argmin damage + lam*bytes + mu*bytes*[type != base]
    (embedding items have one fixed option). lam is bisected to meet the byte budget; mu is
    raised until at least `frac` of body bytes sit at `base`. Crushing below base and bumping
    above it are both available, so a crush can fund a bump. Greedy fill of the slack at the end.

    `budget` may be float("inf"): nothing exceeds it, lam settles at 0 and the greedy fill
    spends against an infinite slack, so `frac` is the only bound on the solution."""
    body = [n for n, _ in items if not is_embd_name(n)]
    opts_by = dict(items)

    def pick(lam, mu):
        ch = {}
        for name, opts in items:
            if name in body:
                ch[name] = min(opts, key=lambda o: o[1] + lam * o[2] + (mu * o[2] if o[0] != base else 0.0))
            else:
                ch[name] = opts[0]
        return ch

    def stats(ch):
        tot = sum(o[2] for o in ch.values())
        bb = sum(ch[n][2] for n in body)
        at = sum(ch[n][2] for n in body if ch[n][0] == base)
        return tot, (at / bb if bb else 1.0)

    def solve_lam(mu):
        lo, hi = 0.0, 1.0
        while stats(pick(hi, mu))[0] > budget and hi < 1e12:
            hi *= 4
        if stats(pick(lo, mu))[0] <= budget:
            return pick(lo, mu)
        for _ in range(80):
            mid = (lo + hi) / 2
            if stats(pick(mid, mu))[0] <= budget:
                hi = mid
            else:
                lo = mid
        return pick(hi, mu)

    mu_lo, mu_hi = 0.0, 1e-12
    ch = solve_lam(0.0)
    if stats(ch)[1] < frac:
        while stats(solve_lam(mu_hi))[1] < frac and mu_hi < 1e3:
            mu_hi *= 4
        for _ in range(60):
            mid = (mu_lo + mu_hi) / 2
            if stats(solve_lam(mid))[1] >= frac:
                mu_hi = mid
            else:
                mu_lo = mid
        ch = solve_lam(mu_hi)
    # greedy fill of the slack, keeping the majority
    cur = dict(ch)
    tot_b = stats(cur)[0]
    while True:
        slack = budget - tot_b
        best = None
        for name in body:
            c = cur[name]
            for o in opts_by[name]:
                db = o[2] - c[2]; dd = c[1] - o[1]
                if db <= 0 or db > slack or dd <= 0:
                    continue
                r = dd / db
                if best is None or r > best[0]:
                    best = (r, name, o)
        if best is None:
            break
        _, name, o = best
        prev = cur[name]; cur[name] = o
        if stats(cur)[1] < frac:
            cur[name] = prev
            opts_by[name] = [x for x in opts_by[name] if x is not o]
            continue
        tot_b += o[2] - prev[2]
    return {n: o[0] for n, o in cur.items()}
