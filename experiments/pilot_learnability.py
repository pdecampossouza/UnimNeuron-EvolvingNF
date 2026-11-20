#!/usr/bin/env python
"""Kick-off learnability pilot (PROTOCOL.md, "Learnability rule").

Question: on which synthetic streams does the model learn anything, so that accuracy-based claims
(accuracy recovery, SQ4 ablations) are meaningful?

Rule, fixed before running:
  * pilot seed 99 only (never used in the experiments, seeds 0-9);
  * accuracy only: relevance is NOT logged or looked at;
  * each stream family is tested on its stationary version (first concept), n = 20,000;
  * a family is "predictive" if the `open` variant (max_rules=None) beats the majority-class rate
    by >= 5 percentage points; otherwise it is "relevance-only";
  * Gaussian mean shift: delta = the smallest of 0.3 / 0.4 / 0.5 that passes; if none passes, 0.5
    is used and the Gaussian streams are relevance-only as well.

    python experiments/pilot_learnability.py           -> results_pilot/pilot.csv, pilot_decision.json
"""
import csv, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_relevance import run_one, _git_info, _versions
from relevance.streams import gauss_shift_stream, relevance_drift_stream, sea_stream

PILOT_SEED = 99
N = 20000
MARGIN = 0.05
DELTAS = (0.3, 0.4, 0.5)
BASE = {"tau_merge": 0.62, "safe_threshold": 0.07, "min_firing_for_update": 0.05, "drift_sensitivity": 0.35}
VARIANTS = {"ref": dict(max_rules=5, **BASE), "open": dict(max_rules=None, **BASE)}
CFG = {"checkpoint_every": 50, "acc_window": 500}      # run_one needs these; nothing is logged (log=False)

CANDIDATES = [(f"gauss_delta{d}", "gauss_shift", d,
               lambda d=d: gauss_shift_stream(n=N, d=10, drift_at=None, delta=d, seed=PILOT_SEED)) for d in DELTAS]
CANDIDATES += [("uniform_sum", "relevance_drift", None,
                lambda: relevance_drift_stream(n=N, d=10, drift_at=None, seed=PILOT_SEED)),
               ("sea_variant0", "sea", None, lambda: sea_stream(n=N, drift_at=None, seed=PILOT_SEED))]


def main():
    out = ROOT / "results_pilot"
    out.mkdir(exist_ok=True)
    rows = []
    for name, family, delta, make in CANDIDATES:
        st = make()
        y = st.y
        majority = float(max(np.mean(y), 1 - np.mean(y)))
        no_change = float(np.mean(y[1:] == y[:-1]))
        for vname, params in VARIANTS.items():
            t0 = time.perf_counter()
            res = run_one(CFG, PILOT_SEED, log=False, stream=st, model_cfg=params)
            row = dict(stream=name, family=family, delta=delta, variant=vname, acc=res["acc_final"],
                       majority=majority, no_change=no_change, margin_vs_majority=res["acc_final"] - majority,
                       rules_final=int(res["n_rules"][-1]), seconds=round(time.perf_counter() - t0, 1))
            rows.append(row)
            print(f"{name:16s} {vname:4s} acc={row['acc']:.4f} majority={majority:.4f} "
                  f"margin={row['margin_vs_majority']:+.4f} rules={row['rules_final']} ({row['seconds']}s)", flush=True)

    with open(out / "pilot.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    passes = {r["stream"]: r["margin_vs_majority"] >= MARGIN for r in rows if r["variant"] == "open"}
    ok_deltas = [d for d in DELTAS if passes[f"gauss_delta{d}"]]
    delta = ok_deltas[0] if ok_deltas else DELTAS[-1]
    roles = {"gauss_shift": "predictive" if ok_deltas else "relevance-only",
             "relevance_drift": "predictive" if passes["uniform_sum"] else "relevance-only",
             "sea": "predictive" if passes["sea_variant0"] else "relevance-only",
             "river:Elec2": "relevance-only (not piloted: needs river; no predictive claims until WF4's "
                            "majority/no-change check says otherwise)"}
    decision = dict(rule=__doc__.split("Rule, fixed before running:")[1].split("python experiments")[0].strip(),
                    pilot_seed=PILOT_SEED, n=N, margin=MARGIN, gauss_delta=delta, roles=roles,
                    open_variant_passes=passes, git=_git_info(), versions=_versions())
    json.dump(decision, open(out / "pilot_decision.json", "w"), indent=2)
    print(json.dumps(dict(gauss_delta=delta, roles=roles), indent=2))


if __name__ == "__main__":
    main()
