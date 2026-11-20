"""Tests for the kick-off additions to the core: streams, gradual-drift analysis, macro-F1,
rule-event counts, run metadata and config variants.

Run with `python tests/test_kickoff_core.py` or `pytest tests/`.
"""
import json, sys, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relevance.streams import (gauss_shift_stream, sea_stream, relevance_drift_stream, stream_from_river,
                               build_stream)
from relevance.analysis import analyze
from relevance.metrics import macro_f1, macro_f1_from_counts
from run_relevance import run_one, run_config

ANALYSIS = {"lags_samples": [50, 200], "topk": 2, "recovery_precision": 1.0, "recovery_consecutive": 3,
            "warmup_flush": 200, "acc_tol": 0.02}


def test_gauss_stream_truth_bounds_and_signal():
    st = gauss_shift_stream(n=4000, d=6, drift_at=2000, rel_before=(0, 1), rel_after=(3, 4), delta=0.4, seed=1)
    assert st.X.min() >= 0 and st.X.max() <= 1
    assert st.truth[:2000, [0, 1]].all() and not st.truth[:2000, [2, 3, 4, 5]].any()
    assert st.truth[2000:, [3, 4]].all() and not st.truth[2000:, [0, 1, 2, 5]].any()
    assert st.drift_points == [2000] and st.drift_widths == [0]
    pre, post = slice(0, 2000), slice(2000, 4000)
    gap = lambda sl, j: st.X[sl][st.y[sl] == 1, j].mean() - st.X[sl][st.y[sl] == 0, j].mean()
    assert gap(pre, 0) > 0.3 and abs(gap(pre, 3)) < 0.05      # relevant before, irrelevant before
    assert gap(post, 3) > 0.3 and abs(gap(post, 0)) < 0.05    # and swapped after


def test_gradual_drift_mixes_concepts():
    st = gauss_shift_stream(n=6000, d=6, drift_at=3000, rel_before=(0, 1), rel_after=(3, 4), delta=0.4,
                            drift_width=1000, seed=2)
    assert st.drift_widths == [1000]
    gap = lambda sl, j: st.X[sl][st.y[sl] == 1, j].mean() - st.X[sl][st.y[sl] == 0, j].mean()
    assert gap(slice(0, 2400), 3) < 0.05 and gap(slice(3600, 6000), 3) > 0.3   # pure before/after
    mid = gap(slice(2900, 3100), 3)
    assert 0.1 < mid < 0.3                                                      # mixture in the middle


def test_abrupt_streams_unchanged_by_gradual_support():
    """drift_width=0 must not draw extra random numbers: same stream as before gradual drift existed."""
    a = relevance_drift_stream(n=3000, d=10, drift_at=1500, seed=7)
    b = relevance_drift_stream(n=3000, d=10, drift_at=1500, seed=7, drift_width=0)
    rng = np.random.default_rng(7)
    X = rng.random((3000, 10))
    assert np.array_equal(a.X, X) and np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y)


def test_sea_negative_control():
    st = sea_stream(n=4000, drift_at=2000, variant_before=0, variant_after=2, seed=0)
    assert st.X.min() >= 0 and st.X.max() <= 1 and st.X.shape[1] == 3
    assert st.truth[:, [0, 1]].all() and not st.truth[:, 2].any()            # relevance never changes
    s = st.X[:, 0] + st.X[:, 1]
    assert np.array_equal(st.y[:2000], (s[:2000] * 10 > 8).astype(int))
    assert np.array_equal(st.y[2000:], (s[2000:] * 10 > 7).astype(int))


def test_river_loader_drops_features_without_leakage():
    fake = [({"date": float(i), "a": float(i % 7), "b": float((3 * i) % 5)}, "UP" if i % 2 else "DOWN")
            for i in range(50)]
    st = stream_from_river(fake, n_max=40, drop=["date"], name="fake")
    assert st.feature_names == ["a", "b"] and st.X.shape == (40, 2)
    assert set(np.unique(st.y)) == {0, 1}
    raw = np.array([[r[0]["a"], r[0]["b"]] for r in fake[:40]])
    t = 10                                              # row t is scaled with rows 0..t only
    lo, hi = raw[:t + 1].min(0), raw[:t + 1].max(0)
    assert np.allclose(st.X[t], (raw[t] - lo) / np.where(hi > lo, hi - lo, 1.0))


def test_macro_f1():
    y = np.array([0, 0, 1, 1, 1, 2])
    p = np.array([0, 1, 1, 1, 0, 0])
    # class 0: tp1 fp2 fn1 -> 2/5 ; class 1: tp2 fp1 fn1 -> 4/6 ; class 2: tp0 fp0 fn1 -> 0
    assert abs(macro_f1(y, p) - np.mean([2 / 5, 4 / 6, 0.0])) < 1e-12
    assert np.isnan(macro_f1_from_counts([0, 0], [0, 0], [0, 0]))


def _fake_run(T_ckpt=120, every=50, d=4, switch_at=3000, lag_ckpts=4, width=0):
    """Checkpoints where G ranks {0,1} first until `switch_at + lag`, then {2,3}."""
    t = np.arange(1, T_ckpt + 1) * every
    G = np.where((t < switch_at + width / 2 + lag_ckpts * every)[:, None],
                 np.array([1.0, 0.9, 0.2, 0.3]), np.array([0.2, 0.3, 1.0, 0.9]))
    arr = dict(t=t, G=G, A=np.full((T_ckpt, d), 0.25), acc_win=np.full(T_ckpt, 0.8),
               n_rules=np.full(T_ckpt, 5), created=np.arange(T_ckpt), removed=np.arange(T_ckpt))
    truth = np.zeros((t[-1], d), dtype=int)
    truth[:switch_at, [0, 1]] = 1
    truth[switch_at:, [2, 3]] = 1
    return arr, truth


def test_recovery_abrupt_and_gradual():
    arr, truth = _fake_run(width=0)
    m = analyze(arr, truth, [3000], ANALYSIS, 50)
    assert m["recovery/G/drift1"] == 200.0 and m["recovered/G/drift1"] == 1.0
    assert m["recovered/A/drift1"] == 0.0 and np.isnan(m["recovery/A/drift1"])   # uniform A: chance, never 1
    assert m["recovery_censored/A/drift1"] == arr["t"][-1] - 3000
    arr, truth = _fake_run(width=1000)
    m = analyze(arr, truth, [3000], ANALYSIS, 50, drift_widths=[1000])
    assert m["recovery/G/drift1"] == 200.0          # counted from the END of the transition (3500)


def test_gradual_transition_is_not_clean():
    arr, truth = _fake_run(width=1000)
    m0 = analyze(arr, truth, [3000], ANALYSIS, 50, drift_widths=[0])
    m1 = analyze(arr, truth, [3000], ANALYSIS, 50, drift_widths=[1000])
    # G re-orients at t=3700. Treated as abrupt, the clean post-drift segment starts at t=3200 and
    # includes stale rankings; with the 1000-sample transition excluded it starts at t=3700.
    assert m0["valid/G/prec_at_k/seg0"] == 1.0 and m1["valid/G/prec_at_k/seg0"] == 1.0
    assert m0["valid/G/prec_at_k/seg1"] < 1.0
    assert m1["valid/G/prec_at_k/seg1"] == 1.0


def test_rule_events_reference_config_freezes_rule_base():
    cfg = json.load(open(ROOT / "configs" / "mve.json"))
    cfg["stream"].update(n=1000, drift_at=500)
    ev = run_one(cfg, 0, log=True)["events"]
    assert ev["ev_created"] - ev["ev_pruned"] - ev["ev_merged"] == 5
    assert ev["ev_merged"] == 0                                   # merging cannot fire with max_rules=5
    assert ev["ev_newborn_pruned"] > 0.9 * ev["ev_created"]       # nearly every new rule is deleted at once


def test_run_config_writes_meta_and_variants():
    cfg = json.load(open(ROOT / "configs" / "protocol" / "s1_gauss_abrupt.json"))
    cfg["stream"].update(n=600, drift_at=300)
    with tempfile.TemporaryDirectory() as tmp:
        run_config(cfg, seeds=[0], plot=False, outdir=tmp, quiet=True)
        for v in cfg["variants"]:
            d = Path(tmp) / cfg["name"] / v / "seed_0"
            meta = json.load(open(d / "meta.json"))
            assert meta["variant"] == v and meta["seed"] == 0 and "commit" in meta["git"]
            assert meta["stream"]["family"] == "gauss_shift"
            assert (d / "checkpoints.csv").exists() and (d / "metrics.json").exists()
            assert (Path(tmp) / cfg["name"] / v / "summary.json").exists()


def test_protocol_configs_are_valid():
    for p in sorted((ROOT / "configs" / "protocol").glob("*.json")):
        cfg = json.load(open(p))
        assert cfg["seeds"] and cfg["checkpoint_every"] % 50 == 0 and "variants" in cfg, p.name
        assert set(cfg["variants"]) == {"ref", "open"}, p.name
        if cfg["stream"]["type"] != "river":
            st = build_stream(dict(cfg["stream"], n=min(cfg["stream"]["n"], 400),
                                   drift_at=None if cfg["stream"].get("drift_at") is None else 200), 0)
            assert st.X.shape[0] == 400 and st.truth is not None, p.name


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
