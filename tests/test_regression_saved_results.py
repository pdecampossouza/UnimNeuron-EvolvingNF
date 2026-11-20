"""Guards the results already in the repository against changes in the core code.

1. analyze() on the saved checkpoints reproduces every saved metric (all seeds, both exploratory configs).
2. Re-running configs/mve.json with the current code reproduces the saved checkpoints bit for bit (seed 0).

Run with `python tests/test_regression_saved_results.py` or `pytest tests/`.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relevance.analysis import analyze
from relevance.streams import build_stream
from run_relevance import run_one

SAVED = ("mve", "stationary_control")                      # Minês's exploratory runs (19 Nov 2026)
RUNNER_KEYS = {"acc_final", "f1_final", "rules_final", "proc_time_s", "log_time_s"}   # timing / runner fields
CHECKPOINT_KEYS = ("t", "G", "A", "n_rules", "acc_cum", "acc_win", "created", "removed")


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return (np.isnan(a) and np.isnan(b)) or a == b


def test_saved_metrics_are_reproduced_by_analysis():
    for name in SAVED:
        summ = json.load(open(ROOT / "results_relevance" / name / "summary.json"))
        cfg = summ["config"]
        for s in summ["seeds"]:
            d = ROOT / "results_relevance" / name / f"seed_{s}"
            arr = dict(np.load(d / "checkpoints.npz"))
            st = build_stream(cfg["stream"], s)
            got = analyze(arr, st.truth, st.drift_points, cfg["analysis"], cfg["checkpoint_every"], st.drift_widths)
            saved = json.load(open(d / "metrics.json"))
            for k, v in saved.items():
                if k not in RUNNER_KEYS:
                    assert _same(got.get(k), v), (name, s, k, got.get(k), v)


def test_rerun_reproduces_saved_checkpoints():
    summ = json.load(open(ROOT / "results_relevance" / "mve" / "summary.json"))
    saved = dict(np.load(ROOT / "results_relevance" / "mve" / "seed_0" / "checkpoints.npz"))
    arr = run_one(summ["config"], 0, log=True)["logger"].arrays()
    for k in CHECKPOINT_KEYS:
        assert np.array_equal(saved[k], arr[k]), k


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
