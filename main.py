"""GTA VI PS5 trophy earn-rate forecaster (Python 3.13 stdlib only).

Earners/players decomposition. rate = 100 * earners / players_tracked.
Exophase freezes the page between refreshes; across refreshes players usually grow
faster than earners, so rates fall (dilution), then stabilize as N growth slows and
completion catches up. Forecast ln N and ln earners at the target, combine, shrink
hard toward the last rate when recent player growth is weak (persistence then wins).
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.64485, -0.67449, 0.0, 0.67449, 1.64485]
RELEASE = "2026-11-19T00:00:00+00:00"


def clamp(x: float) -> float:
    return min(99.99, max(0.01, float(x)))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(z: float) -> float:
    return 100.0 / (1.0 + math.exp(-z))


def parse_ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def hours_since_release(s: str) -> float:
    return (parse_ts(s) - parse_ts(RELEASE)) / 3600.0


def median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def wls(xs: list[float], ys: list[float], ws: list[float] | None = None):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    if ws is None:
        ws = [1.0] * n
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx < 1e-14:
        return 0.0, my, 0.0
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx
    a = my - b * mx
    res2 = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
    sd = math.sqrt(res2 / max(sw, 1.0))
    return b, a, sd


def page_states(history: list[dict]) -> list[dict]:
    out: list[dict] = []
    for h in history:
        n = h.get("players_tracked") or 0
        st = {"n": n, "t": hours_since_release(h["at"]), "rates": h["rates"], "at": h["at"]}
        if out and out[-1]["n"] == n:
            out[-1] = st
        else:
            out.append(st)
    return out


def forecast_n(states: list[dict], t_last: float, t_target: float) -> tuple[float, float, float]:
    """Return (lnN_target, sd_lnN, recent_dlnN_per_hour)."""
    use = states[-12:]
    if len(use) < 2:
        n0 = max(1, use[-1]["n"] if use else 1)
        hours_fwd = max(0.0, t_target - t_last)
        # Contest rules document the first refresh: 638 -> 3868 on launch night.
        if n0 < 2000 and hours_fwd <= 3.0:
            return math.log(3868.0), 0.06, math.log(3868.0 / n0) / max(hours_fwd, 0.5)
        growth = 0.45 * math.log(1.0 + hours_fwd / 8.0)
        return math.log(n0) + growth, 0.30 + 0.10 * growth, growth / max(hours_fwd, 1.0)

    xs = [math.log(1.0 + max(0.0, s["t"])) for s in use]
    ys = [math.log(max(1, s["n"])) for s in use]
    ws = [1.0 + 0.5 * i for i in range(len(use))]
    b, a, sd = wls(xs, ys, ws)
    b = min(1.6, max(0.0, b)) * 0.55  # strong damping — over-growth hurts
    xT = math.log(1.0 + max(0.0, t_target))
    lnN = max(a + b * xT, math.log(max(1, use[-1]["n"])))
    lnN = min(lnN, math.log(2_000_000.0))
    hours_fwd = max(0.0, t_target - t_last)
    # Recent instantaneous growth from last two distinct states.
    if len(use) >= 2 and use[-1]["t"] > use[-2]["t"]:
        recent = (math.log(use[-1]["n"]) - math.log(use[-2]["n"])) / (use[-1]["t"] - use[-2]["t"])
    else:
        recent = 0.0
    # If recent growth is tiny, stay near last N (persistence of the page).
    if abs(recent) < 0.01:
        lnN = 0.85 * math.log(use[-1]["n"]) + 0.15 * lnN
    sdN = math.sqrt(sd ** 2 + (0.10 + 0.06 * math.log(1.0 + hours_fwd)) ** 2)
    return lnN, sdN, recent


def forecast_case(case: dict) -> dict:
    trophies = case["trophies"]
    hist = case["history"]
    last = hist[-1]
    t_last = hours_since_release(last["at"])
    t_target = hours_since_release(case["target"])
    horizon_d = case["horizon_minutes"] / 1440.0
    states = page_states([h for h in hist if h.get("players_tracked")])
    n_last = max(1, last["players_tracked"] or 1)
    lnN_T, sdN, recent = forecast_n(states, t_last, t_target)
    n_T = math.exp(lnN_T)
    n_states = len(states)

    m = len(trophies)
    earner_growth: list[float | None] = []
    fit_sd: list[float] = []
    for i in range(m):
        pts = []
        for s in states:
            r = s["rates"][i]
            if r is None:
                continue
            e = max(1e-6, r * s["n"] / 100.0)
            pts.append((math.log(1.0 + max(0.0, s["t"])), math.log(e)))
        if len(pts) >= 2 and pts[-1][0] - pts[0][0] > 0.05:
            be, ae, sde = wls([p[0] for p in pts], [p[1] for p in pts])
            earner_growth.append(be)
            fit_sd.append(sde)
        else:
            earner_growth.append(None)
            fit_sd.append(0.0)

    tiers = [t["tier"] for t in trophies]
    eg_obs = [g for g in earner_growth if g is not None]
    eg_pop = median(eg_obs) if eg_obs else 0.10
    eg_tier: dict[str, float] = {}
    for t in set(tiers):
        v = [earner_growth[i] for i in range(m) if tiers[i] == t and earner_growth[i] is not None]
        eg_tier[t] = median(v) if v else eg_pop
    w = n_states / (n_states + 5.0)

    # When recent player growth is weak, lean hard on persistence (house baseline wins then).
    persist_w = 1.0 / (1.0 + 25.0 * abs(recent))
    if n_states < 2:
        persist_w = 0.0

    rows = []
    for i, tr in enumerate(trophies):
        r = last["rates"][i]
        r = 0.0 if r is None else float(r)
        if r >= 99.5:
            rows.append({"id": tr["id"], "q": [99.5, 99.9, 100.0, 100.0, 100.0]})
            continue

        e_last = max(1e-6, r * n_last / 100.0)
        prior_g = 0.55 * eg_tier.get(tr["tier"], eg_pop) + 0.45 * eg_pop
        g = w * earner_growth[i] + (1.0 - w) * prior_g if earner_growth[i] is not None else prior_g
        if tr["tier"] == "platinum":
            g = min(g, 0.05)
        g = min(0.9, max(0.0, g))
        damp = 1.0 / (1.0 + 0.25 * horizon_d)
        dle = g * damp * (math.log(1.0 + t_target) - math.log(1.0 + t_last))
        e_T = e_last * math.exp(dle)
        rate_model = min(99.5, max(0.01, 100.0 * e_T / max(n_T, 1.0)))
        rate_dilute = min(99.5, max(0.01, 100.0 * e_last / max(n_T, 1.0)))

        if n_states < 2:
            rate_T = rate_dilute
        else:
            # Blend model with last rate; more persistence when N is calm.
            rate_T = (1.0 - persist_w) * (0.7 * rate_model + 0.3 * rate_dilute) + persist_w * r
            rate_T = min(99.5, max(0.01, rate_T))

        n_k = max(1.0, e_last)
        sd_count = 0.45 / math.sqrt(n_k)
        sd_fit = 0.55 * fit_sd[i]
        sd_N_effect = 0.85 * sdN  # ~unit elasticity
        sd_rw = 0.10 * math.sqrt(max(horizon_d, 1e-6))
        sd_refresh = 0.10 if n_states < 2 else 0.0
        sd = math.sqrt(sd_count ** 2 + sd_fit ** 2 + sd_N_effect ** 2 + sd_rw ** 2 + sd_refresh ** 2)
        # Narrow toward persistence spread when we lean on last rate.
        sd_persist = 0.5 * math.sqrt(max(horizon_d, 1e-6))
        sd = (1.0 - persist_w) * sd + persist_w * sd_persist
        sd = max(0.06, min(2.2, sd))

        rows.append({"id": tr["id"], "q": quantiles(logit(rate_T), sd)})
    return {"trophies": rows}


def quantiles(mu: float, sd: float) -> list[float]:
    out: list[float] = []
    prev = 0.0
    for z in Z:
        v = round(min(100.0, max(0.0, inv_logit(mu + z * sd))), 4)
        if v < prev:
            v = prev
        prev = v
        if abs(v - round(v)) < 1e-12:
            v = int(round(v))
        out.append(v)
    return out


def fallback(case: dict) -> dict:
    last = case["history"][-1]
    rows = []
    for i, tr in enumerate(case["trophies"]):
        r = last["rates"][i] or 0.0
        rows.append({"id": tr["id"], "q": quantiles(logit(r), 0.8)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            if case.get("levels") != LEVELS or not case.get("history"):
                raise ValueError("bad case")
            res = forecast_case(case)
        except Exception:
            res = fallback(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(res, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
