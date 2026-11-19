"""Run with `python tests/test_nonintrusive.py` or `pytest tests/`."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_relevance import run_one
from relevance.streams import relevance_drift_stream, expanding_minmax
from relevance.metrics import tau_b, auc, topk_jaccard, first_sustained, topk_precision

CFG = json.load(open(ROOT / "configs" / "mve.json"))
CFG["stream"].update(n=3000, drift_at=1500)
CFG["model"]["random_state"] = 0   # rng is created but never used by the model; fix it so states are comparable


def test_logging_is_nonintrusive():
    a = run_one(CFG, seed=0, log=False)
    b = run_one(CFG, seed=0, log=True)
    assert np.array_equal(a["y_pred"], b["y_pred"])
    assert np.array_equal(a["n_rules"], b["n_rules"])
    assert np.array_equal(a["model"].current_feature_weights, b["model"].current_feature_weights)
    for ra, rb in zip(a["model"].rules, b["model"].rules):
        assert np.array_equal(ra.center, rb.center) and np.array_equal(ra.w, rb.w) and ra.support == rb.support
    assert a["model"].rng.bit_generator.state == b["model"].rng.bit_generator.state


def test_no_label_leakage():
    """Flipping the label at t0 must not change any prediction up to and including t0."""
    st = relevance_drift_stream(n=1500, d=10, drift_at=None, seed=1)
    t0 = 1000
    y2 = st.y.copy(); y2[t0] = 1 - y2[t0]
    st2 = type(st)(X=st.X, y=y2, truth=st.truth, drift_points=[], feature_names=st.feature_names)
    a = run_one(CFG, 0, log=False, stream=st)
    b = run_one(CFG, 0, log=False, stream=st2)
    assert np.array_equal(a["y_pred"][:t0 + 1], b["y_pred"][:t0 + 1])


def _import_baseline_loop():
    try:
        import enfs_uni0_evolving  # noqa: F401
    except ModuleNotFoundError:
        import types
        stub = types.ModuleType("enfs_uni0_evolving")
        stub.ENFSUni0Evolving = object   # imported by exp_sota_comparison but never used in main()
        sys.modules["enfs_uni0_evolving"] = stub
    from exp_sota_comparison import prequential_nf_generic
    return prequential_nf_generic


def test_matches_baseline_loop_if_available():
    try:
        prequential_nf_generic = _import_baseline_loop()
    except Exception as e:
        print("SKIP baseline comparison:", repr(e))
        return
    from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe
    st = relevance_drift_stream(n=2000, d=10, drift_at=1000, seed=0)
    ref_model = EvolvingNeuroFuzzyUnimSafe(n_features=10, n_classes=2, **CFG["model"])
    ref = prequential_nf_generic(ref_model, st.X, st.y, desc="ref")
    ours = run_one(CFG, 0, log=True, stream=st)
    assert np.array_equal(ref["y_pred"], ours["y_pred"])
    assert np.array_equal(ref["n_rules_time"], ours["n_rules"])
    print("RAN baseline comparison: identical predictions and rule counts")


def test_metrics_and_scaling():
    assert abs(tau_b([1, 2, 3, 4], [1, 2, 3, 4]) - 1) < 1e-12
    assert abs(tau_b([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12
    assert auc(np.array([.9, .8, .1, .2]), np.array([1, 1, 0, 0])) == 1.0
    assert topk_jaccard(np.array([3, 2, 1, 0]), np.array([3, 2, 0, 1]), 2) == 1.0
    assert first_sustained([0, 1, 1, 1, 0], 3) == 1 and first_sustained([1, 0, 1], 2) is None
    tv = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    assert abs(topk_precision(np.full(10, .1), tv, 3) - 0.3) < 1e-12      # uniform weights: chance, not 1.0
    assert topk_precision(np.array([9, 8, 7, 1, 1, 1, 1, 1, 1, 1.]), tv, 3) == 1.0
    X = np.array([[0.], [10.], [5.]])
    assert np.allclose(expanding_minmax(X).ravel(), [0.0, 1.0, 0.5])   # row 2 uses only rows 0-2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)