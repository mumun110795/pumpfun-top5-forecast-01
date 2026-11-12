#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry: Monte Carlo forecaster (Python 3.13, stdlib only).

Reads /input/task.json and /input/history.ndjson (10-minute snapshots of every
pump.fun coin launched since the opening, ranked by market cap) and writes one
forecast to /output/forecast.json: place probabilities for places 1..5, a top-5
guess, and a market-cap size distribution per place.

Model (dynamics measured on the rehearsal recording; horizons read from
task.json, so the same code serves the 2-hour practice and the 24-hour final):
  * Per-coin log10 market-cap (in SOL) changes are close to a random walk with
    slight 10-minute mean reversion (measured corr ~ -0.14), so momentum is
    ignored and each coin's future size is its last size plus drift plus noise.
  * Volatility is strongly age- and venue-dependent in the rehearsal data:
    coins younger than ~15 minutes move ~3-4x as much per step as older ones,
    graduated (pool) coins in the top 20 persist while curve coins churn, and
    coins seen in many marks are steadier than one-mark wonders. Per-coin
    noise sd therefore scales with sqrt(horizon) and is multiplied by age,
    venue and persistence factors; a horizon-growing "fade" hazard can knock
    a coin down by 0.6-2.2 dex (pump.fun coins die), with a higher hazard for
    curve coins and for coins already far below their peak.
  * A saturating ecosystem drift is estimated from the recent trajectory of
    the top order statistics and shrunk hard toward zero.
  * Coins born after the cut (and risers never ranked in recent marks) are
    modelled as a Poisson stream of newcomers whose sizes are drawn around
    the current leaders' sizes; in the rehearsal 3 of the 9 distinct top-5
    coins across the answer reads were not in the history at all. Round-2
    recalibration on held-out rehearsal slices (every cut/target pair of
    marks, scored with the official formulas) showed the first version
    understated this churn: the newcomer arrival rate is raised (base
    coefficient 3.24/hour before the history-age decay), newcomer sizes are
    drawn wider (sd 0.60), and every place keeps a larger explicit unnamed
    reserve (base 0.24, cap 0.40) on top of the simulated newcomer mass.
    Blending the size distributions toward a static anchor at the last
    mark's sizes was tested and rejected: it scored worse on every
    held-out slice, because place sizes themselves move with the flow of
    new leaders, which the simulation's order statistics already track.
  * Each Monte Carlo replicate first draws one of the seven answer reads
    uniformly (the score is the mean over reads), simulates every candidate's
    size at that horizon plus the newcomers, and ranks them. Place
    frequencies, shrunk by the reserve, become the probabilities; the
    simulated order statistics give the size distributions.

Deterministic: fixed seed, sorted iteration everywhere, floored 6-decimal
probabilities, so two runs on the same history are byte-identical.

Robustness: history rows are validated at load time (see row_usable); a row
whose fields cannot be parsed downstream is skipped instead of crashing the
whole run on the final's much longer history.
"""

import json
import math
import os
import random
from datetime import datetime, timezone

SEED = 20261112
SIMS = 8000
MICRO = 1_000_000
MAX_CANDIDATES = 400

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def size_of(lamports):
    """log10 of market cap in SOL."""
    return math.log10(int(lamports)) - 9.0


def floor6(x):
    return math.floor(x * MICRO) / MICRO


def row_usable(r):
    """True when a history row carries every field the model parses, in a
    parseable form: a positive integer market cap (log10 is taken of it), a
    rank of at least 1, a mint, and a created_at timestamp that parses.
    Anything else is dropped at load time so one malformed row in a long
    history cannot abort the run and forfeit the forecast entirely."""
    try:
        if int(r["market_cap_lamports"]) <= 0:
            return False
        if int(r["rank"]) < 1:
            return False
        if not r["mint"]:
            return False
        parse_ts(r["created_at"])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    n_bins = len(edges) + 1
    reads = [parse_ts(r) for r in task["reads"]]
    cadence_h = float(task.get("cadence_minutes", 10)) / 60.0

    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list) and m["rows"]:
                m["rows"] = [r for r in m["rows"] if row_usable(r)]
                if m["rows"]:
                    marks.append(m)
    if not marks:
        raise SystemExit("history holds no usable mark")
    marks.sort(key=lambda m: m["at"])
    cut = parse_ts(marks[-1]["at"])
    t0 = parse_ts(task["t0"])

    # ---- per-coin trajectories -------------------------------------------
    # obs[mint] = list of (mark_index, size, rank, venue, complete, created_at)
    obs = {}
    for idx, m in enumerate(marks):
        for r in m["rows"]:
            obs.setdefault(r["mint"], []).append(
                (idx, size_of(r["market_cap_lamports"]), int(r["rank"]),
                 r.get("venue", "curve"), bool(r.get("complete")),
                 parse_ts(r["created_at"]))
            )
    last_idx = len(marks) - 1

    # ---- pooled step volatility (per cadence step), survivor sample -------
    diffs = []
    by_mark = []
    for m in marks:
        by_mark.append({r["mint"]: size_of(r["market_cap_lamports"]) for r in m["rows"]})
    for i in range(len(marks) - 1):
        a, b = by_mark[i], by_mark[i + 1]
        for mint in a.keys() & b.keys():
            diffs.append(b[mint] - a[mint])
    if len(diffs) >= 30:
        mean_d = sum(diffs) / len(diffs)
        step_sd = math.sqrt(sum((d - mean_d) ** 2 for d in diffs) / len(diffs))
    else:
        step_sd = 0.12
    # Survivors that stayed in the top 200 understate true vol; inflate, then
    # convert to a per-hour sd.
    hourly_sd = min(0.65, max(0.25, 1.45 * step_sd / math.sqrt(cadence_h)))

    # ---- ecosystem drift: recent slope of top order statistics ------------
    recent = marks[-min(12, len(marks)):]
    drift_rate = 0.0
    if len(recent) >= 3:
        xs, ys = [], []
        base_t = parse_ts(recent[0]["at"])
        for m in recent:
            rows = sorted(m["rows"], key=lambda r: r["rank"])[:10]
            if len(rows) >= 5:
                xs.append((parse_ts(m["at"]) - base_t).total_seconds() / 3600.0)
                ys.append(sum(size_of(r["market_cap_lamports"]) for r in rows) / len(rows))
        if len(xs) >= 3 and xs[-1] > xs[0]:
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            denom = sum((x - mx) ** 2 for x in xs)
            if denom > 0:
                slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
                drift_rate = max(-0.12, min(0.12, 0.4 * slope))

    # ---- candidate set -----------------------------------------------------
    cand = {}
    for mint, hist in obs.items():
        last = hist[-1]
        seen_recently = last[0] >= last_idx - 2
        best_rank = min(h[2] for h in hist)
        if seen_recently or best_rank <= 40:
            peak = max(h[1] for h in hist)
            cand[mint] = {
                "last_size": last[1],
                "venue": last[3],
                "complete": last[4],
                "created": last[5],
                "peak": peak,
                "drawdown": peak - last[1],
                "appearances": len(hist),
            }
    if len(cand) > MAX_CANDIDATES:
        keep = sorted(cand.keys(), key=lambda m: (-cand[m]["last_size"], m))[:MAX_CANDIDATES]
        cand = {m: cand[m] for m in keep}
    mints = sorted(cand.keys())

    last_rows = sorted(marks[-1]["rows"], key=lambda r: r["rank"])
    second_size = size_of(last_rows[min(1, len(last_rows) - 1)]["market_cap_lamports"])
    history_span_h = max(0.1, (cut - t0).total_seconds() / 3600.0)

    horizons = []
    for r in reads:
        h = (r - cut).total_seconds() / 3600.0
        horizons.append(max(0.05, h))
    mean_h = sum(horizons) / len(horizons)

    # Newcomer intensity: serious new contenders per hour. The rehearsal's
    # flow of new coins into the top 200 stayed high (~60-100 per mark), and
    # held-out slices showed the threat to the top 5 from post-cut births is
    # larger than a naive rate suggests; the base coefficient (3.24) was
    # recalibrated on those slices in round 2. Longer histories face more
    # post-cut births but also a more entrenched leader set, so the rate
    # decays with the age of the history.
    newcomer_rate = 3.24 / (1.0 + 0.35 * max(0.0, history_span_h - 1.0))
    newcomer_mu = second_size - 0.02 + drift_rate * mean_h * 0.5
    newcomer_sd = 0.60

    # Unnamed reserve per place: share held back for coins outside the
    # candidate set (deep risers) plus a model-error cushion, on top of the
    # simulated post-cut newcomers. Grows with horizon; slightly larger for
    # lower places (more churn there). Round-2 held-out calibration raised
    # the base from 0.08 to 0.24 (named mass per place on the rehearsal now
    # totals ~0.47-0.51 instead of ~0.65-0.68).
    reserve = []
    for k in range(places):
        r_k = 0.24 + 0.05 * (1.0 - math.exp(-mean_h / 3.0)) + 0.012 * k
        reserve.append(min(0.40, r_k))

    # ---- per-coin simulation parameters ------------------------------------
    params = []
    for mint in mints:
        c = cand[mint]
        age_h = max(0.0, (cut - c["created"]).total_seconds() / 3600.0)
        if age_h < 0.25:
            age_f = 1.75
        elif age_h < 1.0:
            age_f = 1.25
        elif age_h < 4.0:
            age_f = 1.0
        else:
            age_f = 0.85
        venue_f = 0.9 if c["venue"] == "pool" else (1.05 if c["venue"] == "migrating" else 1.3)
        persist_f = 0.92 if c["appearances"] >= 4 else 1.0
        lam = 0.035 if c["venue"] == "pool" else 0.09
        if c["drawdown"] > 0.45:
            lam *= 1.8
        params.append((mint, c["last_size"], hourly_sd * age_f * venue_f * persist_f, lam))

    # ---- Monte Carlo ---------------------------------------------------------
    rng = random.Random(SEED)
    place_count = {mint: [0] * places for mint in mints}
    size_hist = [[0] * n_bins for _ in range(places)]

    def bin_of(size):
        if size < edges[0]:
            return 0
        if size >= edges[-1]:
            return n_bins - 1
        lo, hi = 0, len(edges) - 1
        # binary search: bin = number of edges <= size
        while lo < hi:
            mid = (lo + hi) // 2
            if edges[mid] <= size:
                lo = mid + 1
            else:
                hi = mid
        return lo

    n_reads = len(horizons)
    for _ in range(SIMS):
        h = horizons[rng.randrange(n_reads)]
        drift = drift_rate * h / (1.0 + h / 6.0)
        root_h = math.sqrt(h)
        scored = []
        for mint, base, vol, lam in params:
            sd = min(1.15, vol * root_h)
            v = base + drift + rng.gauss(0.0, sd)
            if rng.random() < 1.0 - math.exp(-lam * h):
                v -= rng.uniform(0.6, 2.2)
            scored.append((v, mint))
        # Poisson draw of post-cut newcomers (Knuth; the mean is small).
        mean_new = newcomer_rate * h
        k_new = 0
        if mean_new > 0:
            limit = math.exp(-mean_new)
            p_acc = 1.0
            while True:
                p_acc *= rng.random()
                if p_acc <= limit:
                    break
                k_new += 1
                if k_new >= 14:
                    break
        for j in range(k_new):
            scored.append((rng.gauss(newcomer_mu + drift, newcomer_sd), "\x00new%d" % j))
        scored.sort(key=lambda t: (-t[0], t[1]))
        for k in range(places):
            if k >= len(scored):
                break
            v, mint = scored[k]
            size_hist[k][bin_of(v)] += 1
            if not mint.startswith("\x00new"):
                place_count[mint][k] += 1

    # ---- assemble forecast ---------------------------------------------------
    probabilities = []
    for k in range(places):
        named = []
        for mint in mints:
            freq = place_count[mint][k] / SIMS
            p = floor6(freq * (1.0 - reserve[k]))
            if p >= 0.004:
                named.append((mint, p))
        named.sort(key=lambda t: (-t[1], t[0]))
        named = named[:60]
        total = sum(p for _, p in named)
        if total > 1.0 - reserve[k] + 1e-9:  # safety: rescale down if flooring drifted
            scale = (1.0 - reserve[k]) / total
            named = [(m, floor6(p * scale)) for m, p in named if floor6(p * scale) >= 0.004]
        for mint, p in named:
            if p > 0:
                probabilities.append({"mint": mint, "place": k + 1, "p": p})
    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["mint"]))

    # top5 by Borda score over simulated places, tie-broken by place-1
    # frequency then mint for determinism.
    borda = []
    for mint in mints:
        s = sum(place_count[mint][k] * (places - k) for k in range(places))
        borda.append((s, place_count[mint][0], mint))
    borda.sort(key=lambda t: (-t[0], -t[1], t[2]))
    top5 = [mint for _, _, mint in borda[:places]]
    if len(top5) < places:  # degenerate history fallback: last mark's order
        for r in last_rows:
            if r["mint"] not in top5:
                top5.append(r["mint"])
            if len(top5) == places:
                break

    sizes = []
    for k in range(places):
        total_n = sum(size_hist[k])
        if total_n == 0:
            continue
        raw = [c / total_n for c in size_hist[k]]
        # smooth: 0.70 self + 0.15 each neighbour + uniform epsilon
        sm = []
        for b in range(n_bins):
            v = 0.70 * raw[b]
            if b > 0:
                v += 0.15 * raw[b - 1]
            if b < n_bins - 1:
                v += 0.15 * raw[b + 1]
            v = 0.995 * v + 0.005 / n_bins
            sm.append(v)
        tot = sum(sm)
        sm = [v / tot for v in sm]
        for b, v in enumerate(sm):
            p = floor6(v)
            if p > 0:
                sizes.append({"place": k + 1, "bin": b, "p": p})
    sizes.sort(key=lambda r: (r["place"], r["bin"]))

    out = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task["output"]), "w", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
