"""
Experiment – Stream experiments with evolving fuzzy models
=============================================================================

This script runs *prequential* (predict-then-update) evaluations on a collection
of real and synthetic data streams, comparing:

Baselines (evolvingfuzzysystems.eFS)
-----------------------------------
- ePL, ePL_plus, exTS, Simpl_eTS, eMG

Neuro-fuzzy evolving models (local code)
----------------------------------------
- ENF_UnimNeuron_Safe (optional)
- ENF_UnimNeuron_PA (optional)

Datasets (river)
----------------
Real:
- Elec2_real      -> river.datasets.Elec2
- CreditCard_real -> river.datasets.CreditCard
- Phishing_real   -> river.datasets.Phishing
- Bananas_real    -> river.datasets.Bananas

Synthetic:
- ConceptDrift_Agrawal -> river.datasets.synth.ConceptDriftStream(Agrawal->Agrawal)
- Agrawal_synth        -> river.datasets.synth.Agrawal
- RandomRBFDrift_synth -> river.datasets.synth.RandomRBFDrift
- Sine_synth           -> river.datasets.synth.Sine

Protocol (reproducible, conference-friendly)
--------------------------------------------
1) Stream -> NumPy (X, y_int)
2) Min-max scaling X to [0, 1]
3) Prequential evaluation
4) Save:
   - Console summary lines ([RESULT])
   - Per-dataset plots (accuracy and rule evolution)
   - LaTeX tables (summary + statistical tests)
   - Optional: exported fuzzy rules (best-effort) for interpretability

Anonymous submission note
-------------------------
This file contains NO author-identifying information by design (anonymous review).

How to run
----------
python exp_sota_comparison.py

Output folder
-------------
results_exp/
  <dataset_name>/
     <dataset>_acc_all_models.png
     <dataset>_rules_all_models.png
     rules_<model>.txt          (optional, when export is available)
  exp_stream_summary.tex
  exp_stats_friedman.tex
  exp_stats_posthoc.tex        (if posthoc available)
"""

from __future__ import annotations

import os
import time
from typing import Dict, Any, Tuple, List

import numpy as np

# Non-interactive backend (no Tkinter issues)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from river import datasets
from river.datasets import synth

# -------------------------------------------------------------------------
# Optional stats (recommended). If unavailable, script still runs.
# -------------------------------------------------------------------------
HAS_SCIPY = False
HAS_SCPH = False
try:
    from scipy import stats  # type: ignore

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import scikit_posthocs as sp  # type: ignore

    HAS_SCPH = True
except Exception:
    HAS_SCPH = False

# -------------------------------------------------------------------------
# Your models (local files)
# -------------------------------------------------------------------------
# Keep imports optional to avoid breaking the experiment suite.

try:
    from evolving_nf_advanced import EvolvingNeuroFuzzyAdvanced  # optional

    HAS_ENF_ADVANCED = True
except Exception:
    HAS_ENF_ADVANCED = False

from enfs_uni0_evolving import ENFSUni0Evolving  # optional, if you re-enable

# Optional: UnimNeuron-based variants (if present in repository)
try:
    from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe

    HAS_UNIM_SAFE = True
except Exception:
    HAS_UNIM_SAFE = False

try:
    from ENF_UnimNeuron_PA import EvolvingNeuroFuzzyUnimPA

    HAS_UNIM_PA = True
except Exception:
    HAS_UNIM_PA = False

# Kaike's library
from evolvingfuzzysystems.eFS import (
    ePL,
    ePL_plus,
    exTS,
    Simpl_eTS,
    eMG,
    ePL_KRLS_DISCO,
)

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import os


def _latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _linguistic_term(u: float) -> str:
    # deterministic, simple, paper-friendly
    if u < 1.0 / 3.0:
        return "Low"
    elif u < 2.0 / 3.0:
        return "Medium"
    else:
        return "High"


def _topk_indices_by_weight(w: np.ndarray, k: int = 5) -> List[int]:
    w = np.asarray(w, dtype=float).ravel()
    k = int(max(1, min(k, w.size)))
    return list(np.argsort(-w)[:k])


# -----------------------------
# Linguistic labeling utilities
# -----------------------------
def _linguistic_term(u: float) -> str:
    """Map scalar in [0,1] to a simple linguistic label (IEEE-friendly, deterministic)."""
    if u < 1.0 / 3.0:
        return "Low"
    elif u < 2.0 / 3.0:
        return "Medium"
    else:
        return "High"


def _topk_indices_by_weight(w: np.ndarray, k: int = 5) -> List[int]:
    w = np.asarray(w, dtype=float).ravel()
    k = int(max(1, min(k, w.size)))
    return list(np.argsort(-w)[:k])


# -----------------------------
# UnimNeuron regime + rho from h
# -----------------------------
def _unim_regime_and_rho(h: np.ndarray, e: float) -> Tuple[str, float]:
    """
    Decide regime AND/OR/COMP based on h_j relative to e
    and compute rho in [-1,1] using the same indicator definition you used.

    rho = (1/n) sum( 1{h>=e} - 1{h<=e} ).
    Note: if h==e, both indicators are 1 -> cancels to 0 (neutral).
    """
    h = np.asarray(h, dtype=float).ravel()
    if h.size == 0:
        return "COMP", 0.0

    all_le = np.all(h <= e)
    all_ge = np.all(h >= e)

    if all_le and not all_ge:
        reg = "AND"
    elif all_ge and not all_le:
        reg = "OR"
    else:
        reg = "COMP"

    ge = (h >= e).astype(float)
    le = (h <= e).astype(float)
    rho = float(np.mean(ge - le))
    return reg, rho


# -----------------------------
# Rule stats accumulator
# -----------------------------
@dataclass
class RuleStats:
    act_sum: float = 0.0
    rho_sum: float = 0.0
    n_and: float = 0.0
    n_or: float = 0.0
    n_comp: float = 0.0

    def add(self, act: float, reg: str, rho: float) -> None:
        # Use activation-weighted accounting (more faithful for evolving rules)
        a = float(max(act, 0.0))
        self.act_sum += a
        self.rho_sum += a * float(rho)
        if reg == "AND":
            self.n_and += a
        elif reg == "OR":
            self.n_or += a
        else:
            self.n_comp += a

    def summary(self) -> Dict[str, float]:
        denom = self.act_sum if self.act_sum > 0 else 1.0
        return {
            "rho_mean": self.rho_sum / denom,
            "AND_pct": 100.0 * self.n_and / denom,
            "OR_pct": 100.0 * self.n_or / denom,
            "COMP_pct": 100.0 * self.n_comp / denom,
            "act_sum": self.act_sum,
        }


# -------------------------------------------------------------------------
# Global configuration
# -------------------------------------------------------------------------

# Default is "complete" for final paper results.
EXPERIMENT_MODE = "complete"

BASE_OUTDIR = "results_exp"
os.makedirs(BASE_OUTDIR, exist_ok=True)

# Warmup and batch sizes for Kaike eFS models
WARMUP = 200
BATCH_EVOLVE = 20

# Plot aesthetics (IEEE-friendly)
PLOT_LINEWIDTH = 1.10  # thinner than before
PLOT_GRID_ALPHA = 0.30
PLOT_DPI = 300

# -------------------------------------------------------------------------
# Dataset definitions
# -------------------------------------------------------------------------

REAL_DATASETS = {
    "Elec2_real": datasets.Elec2(),
    "CreditCard_real": datasets.CreditCard(),
    "Phishing_real": datasets.Phishing(),
    "Bananas_real": datasets.Bananas(),  # classic 2D benchmark
}

SYNTH_DATASETS = {
    "ConceptDrift_Agrawal": synth.ConceptDriftStream(
        stream=synth.Agrawal(seed=1),
        drift_stream=synth.Agrawal(seed=2),
        seed=42,
        position=2500,  # known drift point
        width=1000,  # gradual drift window
    ),
    "Agrawal_synth": synth.Agrawal(seed=42),
    "RandomRBFDrift_synth": synth.RandomRBFDrift(),
    "Sine_synth": synth.Sine(seed=42),
}

DATASETS_ALL = {**REAL_DATASETS, **SYNTH_DATASETS}

# Stream sizes per dataset and per mode
STREAM_LENGTHS_PRELIM = {
    "Elec2_real": 3000,
    "CreditCard_real": 3000,
    "Phishing_real": 3000,
    "Bananas_real": 3000,
    "ConceptDrift_Agrawal": 3000,
    "Agrawal_synth": 5000,
    "RandomRBFDrift_synth": 5000,
    "Sine_synth": 5000,
}

STREAM_LENGTHS_COMPLETE = {
    "Elec2_real": 10000,
    "CreditCard_real": 5000,
    "Phishing_real": 5000,
    "Bananas_real": 5000,
    "ConceptDrift_Agrawal": 20000,
    "Agrawal_synth": 20000,
    "RandomRBFDrift_synth": 20000,
    "Sine_synth": 40000,
}


def get_n_max_for_dataset(name: str, mode: str) -> int:
    if mode == "prelim":
        return STREAM_LENGTHS_PRELIM.get(name, 3000)
    return STREAM_LENGTHS_COMPLETE.get(name, 5000)


# -------------------------------------------------------------------------
# Drift markers (best-effort)
# -------------------------------------------------------------------------
# For ConceptDriftStream we know position and width (from the constructor above).
# Real datasets: drift point unknown (do NOT mark).
# RandomRBFDrift: drift exists but no single "official" timestamp is exposed by river;
# only mark if YOU define one explicitly.
DRIFT_MARKERS = {
    "ConceptDrift_Agrawal": {"points": [2500], "width": 1000},
    # Example of manual marker (only if you decide a point):
    # "RandomRBFDrift_synth": {"points": [10000], "width": 0},
}


# -------------------------------------------------------------------------
# Utility: load stream to NumPy
# -------------------------------------------------------------------------
def load_stream_to_numpy(ds, n_max: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convert a river dataset/stream into NumPy arrays (X, y_int).

    Returns
    -------
    X : (N, d) float
    y_int : (N,) int labels encoded as {0,...,C-1}
    feature_names : list[str] feature key order used for vectorization

    Notes
    -----
    Assumes numeric features. Text datasets require vectorization.
    """
    X_list = []
    y_raw_list = []

    it = ds.__iter__() if hasattr(ds, "__iter__") else iter(ds)

    try:
        x0, y0 = next(it)
    except StopIteration:
        raise RuntimeError("Empty stream/dataset")

    feature_names = list(x0.keys())
    x0_vec = np.array([x0[f] for f in feature_names], dtype=float)
    X_list.append(x0_vec)
    y_raw_list.append(y0)

    for i, (x, y) in enumerate(it, start=1):
        if i >= n_max:
            break
        x_vec = np.array([x[f] for f in feature_names], dtype=float)
        X_list.append(x_vec)
        y_raw_list.append(y)

    X = np.vstack(X_list).astype(float)

    # encode labels to integers
    unique_labels = list(dict.fromkeys(y_raw_list))  # preserves order
    label_to_int = {lab: idx for idx, lab in enumerate(unique_labels)}
    y_int = np.array([label_to_int[lab] for lab in y_raw_list], dtype=int)

    return X, y_int, feature_names


def min_max_scale(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max scaling to [0, 1]."""
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    denom = x_max - x_min
    denom[denom == 0.0] = 1.0
    X_scaled = (X - x_min) / denom
    return X_scaled, x_min, x_max


# -------------------------------------------------------------------------
# Safe wrapper for evolvingfuzzysystems (eFS) models
# -------------------------------------------------------------------------
class SafeEFSWrapper:
    """
    Wraps an evolvingfuzzysystems model so that:
      - internal index/value errors do NOT crash the experiment
      - non-finite predictions fall back to class 0
      - keeps counters:
            stabilizations: internal errors caught
            fallback_preds: NaN/inf predictions
    """

    def __init__(self, model, name="eFS"):
        self.model = model
        self.name = name
        self.stabilizations = 0
        self.fallback_preds = 0

    def _safe_call(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (IndexError, ValueError, FloatingPointError):
            self.stabilizations += 1
            return None

    def fit(self, X, y):
        return self._safe_call(self.model.fit, X, y)

    def evolve(self, X, y):
        if hasattr(self.model, "evolve"):
            return self._safe_call(self.model.evolve, X, y)
        return self.fit(X, y)

    def predict(self, X):
        try:
            y = self.model.predict(X)
            y = np.asarray(y).ravel()
            if not np.isfinite(y).all():
                raise FloatingPointError
            return y
        except Exception:
            self.fallback_preds += 1
            return np.zeros(len(X), dtype=int)

    def n_rules(self):
        try:
            return self.model.n_rules()
        except Exception:
            return -1


# -------------------------------------------------------------------------
# Prequential evaluation for neuro-fuzzy (sample-by-sample)
# -------------------------------------------------------------------------
def prequential_nf_generic(model, X, y, desc="NF") -> Dict[str, Any]:
    """
    Prequential evaluation (predict-then-update) for NF-type models.

    Additionally logs per-rule UnimNeuron regime (AND/OR/COMP) and rho when possible.
    It stores stats into: model._rule_stats (list[RuleStats]) for later export.

    Expected API:
      - predict(X) -> label(s)
      - partial_fit(X, y)
      - optional: n_rls_resets (count of repaired numerical instabilities)
      - optional: rules list with (center, sigma, w), and scalar neutral element 'e'
    """
    n = len(y)
    y_pred = np.zeros(n, int)
    correct = np.zeros(n, float)
    acc = np.zeros(n, float)
    rules_time = []

    pred_fallbacks = 0

    # Initialize rule stats holder (best-effort)
    # We refresh size dynamically because rules can be added/removed.
    if not hasattr(model, "_rule_stats"):
        model._rule_stats = []  # type: ignore

    for t in tqdm(range(n), desc=desc, leave=False):
        x_t = X[t : t + 1]
        yt = int(y[t])

        # ---- predict
        if t == 0:
            y_hat = 0
        else:
            try:
                yhat_raw = np.asarray(model.predict(x_t)).ravel()
                if yhat_raw.size == 0 or not np.isfinite(yhat_raw[0]):
                    raise FloatingPointError
                y_hat = int(yhat_raw[0])
            except Exception:
                y_hat = 0
                pred_fallbacks += 1

        y_pred[t] = y_hat
        correct[t] = 1.0 if y_hat == yt else 0.0
        acc[t] = correct[: t + 1].mean()

        # ---- update model
        model.partial_fit(x_t, np.array([yt]))

        # ---- track rule count over time
        if hasattr(model, "history_n_rules") and getattr(model, "history_n_rules"):
            rules_time.append(model.history_n_rules[-1])
        elif hasattr(model, "n_rules_"):
            rules_time.append(model.n_rules_)
        elif hasattr(model, "rules"):
            rules_time.append(len(model.rules))
        else:
            rules_time.append(np.nan)

        # ---- collect UnimNeuron semantics per rule (best-effort)
        # Works for your ENF_Unim_* models because they have:
        #   model.rules : list of rules with center, sigma, w
        #   model.e : neutral element
        if hasattr(model, "rules") and hasattr(model, "e"):
            try:
                e = float(getattr(model, "e"))
                rules_list = list(getattr(model, "rules"))
                R = len(rules_list)

                # ensure stats list matches current rule count
                stats_list: List[RuleStats] = list(getattr(model, "_rule_stats"))
                if len(stats_list) != R:
                    # Resize conservatively: keep prefix; new rules get fresh stats
                    new_stats = [RuleStats() for _ in range(R)]
                    m = min(len(stats_list), R)
                    for i in range(m):
                        new_stats[i] = stats_list[i]
                    setattr(model, "_rule_stats", new_stats)
                    stats_list = new_stats

                # compute per-rule activation + regime on current sample
                x_vec = np.asarray(x_t, dtype=float).ravel()
                for i, r in enumerate(rules_list):
                    # Rule must have center/sigma/w
                    c = np.asarray(getattr(r, "center"), dtype=float).ravel()
                    sig = float(getattr(r, "sigma"))
                    w = np.asarray(getattr(r, "w"), dtype=float).ravel()

                    # Gaussian membership per feature
                    sig_eff = max(sig, 1e-6)
                    a = np.exp(-((x_vec - c) ** 2) / (2.0 * sig_eff**2))

                    # feature contributions h_j (same as in your models)
                    h = w * a + (1.0 - w) * e

                    # UnimNeuron regime + rho
                    reg, rho = _unim_regime_and_rho(h, e=e)

                    # Activation value of the rule: use the same unim aggregation
                    # (matches your implementation)
                    # AND: min, OR: max, COMP: mean
                    if reg == "AND":
                        act = float(np.min(h))
                    elif reg == "OR":
                        act = float(np.max(h))
                    else:
                        act = float(np.mean(h))

                    stats_list[i].add(act=act, reg=reg, rho=rho)

                setattr(model, "_rule_stats", stats_list)

            except Exception:
                # do not break experiments
                pass

    rules_arr = np.array(rules_time, float)
    n_rules_final = int(rules_arr[-1]) if not np.isnan(rules_arr[-1]) else -1
    P_instabilities = int(getattr(model, "n_rls_resets", 0))

    return dict(
        accuracy=acc,
        y_pred=y_pred,
        n_rules_time=rules_arr,
        n_rules_final=n_rules_final,
        pred_fallbacks=int(pred_fallbacks),
        P_instabilities=P_instabilities,
    )


# -------------------------------------------------------------------------
# Prequential evaluation for eFS models (warmup + batched evolve)
# -------------------------------------------------------------------------
def prequential_efs(model, X, y, warmup=200, batch_evolve=20, desc="eFS"):
    """
    Prequential evaluation for evolvingfuzzysystems (Kaike eFS models).
    """
    n = len(y)
    y_pred = np.zeros(n, int)
    correct = np.zeros(n, float)
    acc = np.zeros(n, float)

    if warmup >= n:
        warmup = max(1, n // 10)

    model.fit(X[:warmup], y[:warmup])

    bufX = []
    bufY = []

    for t in tqdm(range(warmup, n), desc=desc, leave=False):
        x_t = X[t : t + 1]
        yt = int(y[t])

        yhat_raw = model.predict(x_t)
        yhat_raw = np.asarray(yhat_raw).ravel()

        if yhat_raw.size == 0:
            yh = 0
        else:
            # Binary-friendly fallback:
            if 0.0 <= yhat_raw[0] <= 1.0:
                yh = int(yhat_raw[0] >= 0.5)
            else:
                yh = int(yhat_raw[0])

        y_pred[t] = yh
        correct[t] = 1.0 if yh == yt else 0.0
        acc[t] = correct[: t + 1].mean()

        bufX.append(x_t.ravel())
        bufY.append(yt)

        if len(bufY) >= batch_evolve:
            X_block = np.vstack(bufX)
            y_block = np.array(bufY, int)
            model.fit(X_block, y_block)
            bufX.clear()
            bufY.clear()

    if n > warmup:
        acc[:warmup] = acc[warmup]

    try:
        n_rules_final = int(model.n_rules())
    except Exception:
        n_rules_final = -1

    rules_time = np.full_like(acc, n_rules_final, float)

    pred_fallbacks = int(getattr(model, "fallback_preds", 0))
    P_instabilities = int(getattr(model, "stabilizations", 0))

    return dict(
        accuracy=acc,
        y_pred=y_pred,
        n_rules_time=rules_time,
        n_rules_final=n_rules_final,
        pred_fallbacks=pred_fallbacks,
        P_instabilities=int(P_instabilities),
    )


# -------------------------------------------------------------------------
# Plotting helpers (with drift markers when available)
# -------------------------------------------------------------------------
def _draw_drift_markers(ax, dataset_name: str):
    info = DRIFT_MARKERS.get(dataset_name, None)
    if not info:
        return

    points = info.get("points", [])
    width = int(info.get("width", 0))

    for p in points:
        # drift instant marker
        ax.axvline(p, color="red", linestyle="--", linewidth=1.0, alpha=0.75)

        # optional drift window shading
        if width and width > 0:
            left = max(0, p - width // 2)
            right = p + width // 2
            ax.axvspan(left, right, color="red", alpha=0.08, linewidth=0)


def plot_all_models_accuracy(dataset_name, model_results, outdir):
    """One figure per dataset: prequential accuracy curves for all models."""
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for model_name, res in model_results.items():
        acc = res["accuracy"]
        t = np.arange(len(acc))
        ax.plot(t, acc, label=model_name, linewidth=PLOT_LINEWIDTH)

    _draw_drift_markers(ax, dataset_name)

    ax.set_xlabel("Time step", fontsize=9)
    ax.set_ylabel("Prequential accuracy", fontsize=9)
    ax.set_title(dataset_name, fontsize=9)
    ax.grid(alpha=PLOT_GRID_ALPHA)

    ax.legend(
        fontsize=7,
        loc="lower right",
        ncol=2,
        frameon=True,
        framealpha=0.90,
        borderpad=0.2,
        handlelength=1.6,
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, f"{dataset_name}_acc_all_models.png"), dpi=PLOT_DPI
    )
    plt.close(fig)


def plot_all_models_rules(dataset_name, model_results, outdir):
    """One figure per dataset: rule-count evolution curves (when available)."""
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    has_any = False
    for model_name, res in model_results.items():
        rules = res.get("n_rules_time", None)
        if rules is None or np.all(np.isnan(rules)):
            continue
        t = np.arange(len(rules))
        ax.plot(t, rules, label=model_name, linewidth=PLOT_LINEWIDTH)
        has_any = True

    if not has_any:
        plt.close(fig)
        return

    _draw_drift_markers(ax, dataset_name)

    ax.set_xlabel("Time step", fontsize=9)
    ax.set_ylabel("Number of rules", fontsize=9)
    ax.set_title(dataset_name + " – rule evolution", fontsize=9)
    ax.grid(alpha=PLOT_GRID_ALPHA)

    ax.legend(
        fontsize=7,
        loc="upper left",
        ncol=2,
        frameon=True,
        framealpha=0.90,
        borderpad=0.2,
        handlelength=1.6,
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, f"{dataset_name}_rules_all_models.png"), dpi=PLOT_DPI
    )
    plt.close(fig)


# -------------------------------------------------------------------------
# LaTeX: summary table
# -------------------------------------------------------------------------
def build_latex_table(summary, outpath: str):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Prequential accuracy, rule growth and stability indicators on real and synthetic data streams.}"
    )
    lines.append(r"\begin{tabular}{llllcccccc} \toprule")
    lines.append(
        r"Mode & Dataset & Model & $N$ & Acc$_{final}$ & Acc$_{mean}$ & "
        r"$N_{\text{rules}}^{final}$ & Time [s] & Fallbacks & Instabilities \\ \midrule"
    )

    for mode, dataset_name, model_name in sorted(summary.keys()):
        s = summary[(mode, dataset_name, model_name)]
        lines.append(
            f"{mode} & {dataset_name} & {model_name} & "
            f"{int(s['N'])} & {s['acc_final']:.3f} & {s['acc_mean']:.3f} & "
            f"{int(s['n_rules_final'])} & {s['time_s']:.1f} & "
            f"{int(s['pred_fallbacks'])} & {int(s['P_instabilities'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -------------------------------------------------------------------------
# LaTeX: statistical tests (Friedman + posthoc, best-effort)
# -------------------------------------------------------------------------
def _latex_escape(s: str) -> str:
    return (
        s.replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def build_stats_tables(
    acc_mean_matrix: np.ndarray,
    model_names: List[str],
    outdir: str,
):
    os.makedirs(outdir, exist_ok=True)

    friedman_path = os.path.join(outdir, "exp_stats_friedman.tex")
    posthoc_path = os.path.join(outdir, "exp_stats_posthoc.tex")

    if not HAS_SCIPY:
        with open(friedman_path, "w", encoding="utf-8") as f:
            f.write("% SciPy not available: Friedman test was not computed.\n")
        with open(posthoc_path, "w", encoding="utf-8") as f:
            f.write("% scikit-posthocs/SciPy not available: Post-hoc not computed.\n")
        return

    cols = [acc_mean_matrix[:, j] for j in range(acc_mean_matrix.shape[1])]
    stat, p = stats.friedmanchisquare(*cols)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Global statistical comparison via Friedman test (Acc$_{mean}$).}"
    )
    lines.append(r"\begin{tabular}{lc} \toprule")
    lines.append(r"Statistic & $p$-value \\ \midrule")
    lines.append(f"$\\chi^2_F = {stat:.4f}$ & ${p:.4g}$ \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(friedman_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    if not HAS_SCPH:
        with open(posthoc_path, "w", encoding="utf-8") as f:
            f.write("% scikit-posthocs not available: Nemenyi post-hoc not computed.\n")
        return

    nemenyi = sp.posthoc_nemenyi_friedman(acc_mean_matrix)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Post-hoc Nemenyi test p-values (pairwise).}")
    colspec = "l" + "c" * len(model_names)
    lines.append(rf"\begin{{tabular}}{{{colspec}}} \toprule")
    header = (
        "Model & "
        + " & ".join([_latex_escape(m) for m in model_names])
        + r" \\ \midrule"
    )
    lines.append(header)

    for i, mi in enumerate(model_names):
        row = [_latex_escape(mi)]
        for j in range(len(model_names)):
            row.append(f"{nemenyi.iloc[i, j]:.4f}")
        lines.append(" & ".join(row) + r" \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(posthoc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -------------------------------------------------------------------------
# Interpretability: export fuzzy rules (best-effort, non-breaking)
# -------------------------------------------------------------------------
def export_rules_best_effort(
    model: Any,
    model_name: str,
    dataset_outdir: str,
    feature_names: List[str],
    max_rules_to_write: int = 5,
    top_features_per_rule: int = 6,
) -> None:
    """
    Export human-readable fuzzy rules for UnimNeuron-based models.

    Produces:
      - Linguistic antecedents: feature is {Low,Medium,High} based on rule center in [0,1]
      - Feature relevance weights w_j
      - UnimNeuron semantics per rule: AND/OR/COMP percentages + rho_mean
      - Consequent:
          * Safe: argmax class_probs
          * PA: argmax row W[1+idx,:] (per-rule consequent weights)

    Notes
    -----
    - Requires that during evaluation `prequential_nf_generic` stored `model._rule_stats`.
    - This function is best-effort: never crashes experiments.
    """
    os.makedirs(dataset_outdir, exist_ok=True)
    outpath = os.path.join(dataset_outdir, f"rules_{model_name}.txt")

    try:
        if not hasattr(model, "rules") or len(getattr(model, "rules")) == 0:
            return

        rules_list = list(getattr(model, "rules"))
        k = min(int(max_rules_to_write), len(rules_list))

        # Stats collected online
        stats_list: List[RuleStats] = []
        if hasattr(model, "_rule_stats"):
            stats_list = list(getattr(model, "_rule_stats"))

        # Prepare consequent accessors
        has_safe_probs = False
        has_pa_W = False
        W = None
        if hasattr(model, "W") and getattr(model, "W") is not None:
            W = np.asarray(getattr(model, "W"), dtype=float)
            has_pa_W = True

        # Safe model rules have class_probs method
        if hasattr(rules_list[0], "class_probs"):
            has_safe_probs = True

        lines: List[str] = []
        lines.append(f"Model: {model_name}")
        lines.append(
            "Interpretable rule export (UnimNeuron semantics + linguistic antecedents)\n"
        )
        lines.append("Legend:")
        lines.append("  - center[j] in [0,1] -> {Low, Medium, High}")
        lines.append("  - w_j is feature relevance in [0,1]")
        lines.append("  - rho_mean in [-1,1]: +1 OR-like, -1 AND-like, 0 COMP-like\n")

        # Sort rules by support (descending) to show the most relevant first
        supports = np.array(
            [int(getattr(r, "support", 0)) for r in rules_list], dtype=int
        )
        order = list(np.argsort(-supports))

        shown = 0
        for ridx in order:
            if shown >= k:
                break
            r = rules_list[ridx]

            center = np.asarray(getattr(r, "center"), dtype=float).ravel()
            sigma = float(getattr(r, "sigma"))
            support = int(getattr(r, "support", 0))
            w = np.asarray(getattr(r, "w"), dtype=float).ravel()

            # Rule semantics summary if available
            sem = None
            if ridx < len(stats_list):
                sem = stats_list[ridx].summary()

            # Consequent
            consequent_str = "THEN class = ?"
            if has_safe_probs:
                probs = np.asarray(r.class_probs(), dtype=float).ravel()
                c = int(np.argmax(probs))
                consequent_str = f"THEN class = {c} (p={probs[c]:.3f})"
            elif has_pa_W and W is not None:
                # Row per rule is W[1+ridx,:] (bias at row 0)
                if W.shape[0] > 1 + ridx:
                    row = W[1 + ridx, :].ravel()
                    c = int(np.argmax(row))
                    # show top-2 classes as extra info
                    top2 = list(np.argsort(-row)[:2])
                    consequent_str = (
                        f"THEN class = {c}  "
                        f"[top: {top2[0]}({row[top2[0]]:.3f}), {top2[1]}({row[top2[1]]:.3f})]"
                    )

            # Antecedent: show only top features by relevance weight to keep it readable
            top_idx = _topk_indices_by_weight(w, k=min(top_features_per_rule, w.size))
            antecedents = []
            for j in top_idx:
                fname = feature_names[j] if j < len(feature_names) else f"x{j+1}"
                term = _linguistic_term(float(center[j]))
                antecedents.append(f"{fname} is {term} (w={w[j]:.2f})")

            # Build rule text
            lines.append(f"Rule {shown+1}  [internal_id={ridx}]")
            lines.append(f"  support = {support}, sigma = {sigma:.4f}")
            if sem is not None:
                lines.append(
                    f"  semantics: rho_mean={sem['rho_mean']:.3f} | "
                    f"AND={sem['AND_pct']:.1f}% OR={sem['OR_pct']:.1f}% COMP={sem['COMP_pct']:.1f}%"
                )
            else:
                lines.append(
                    "  semantics: (not available)  -- run via prequential_nf_generic to log regimes"
                )

            lines.append("  IF " + " AND ".join(antecedents))
            lines.append("  " + consequent_str)
            lines.append("")

            shown += 1

        with open(outpath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    except Exception:
        return

def export_rules_latex(
    model: Any,
    model_name: str,
    dataset_outdir: str,
    feature_names: List[str],
    max_rules_to_write: int = 5,
    top_features_per_rule: int = 6,
    latex_env: str = "enumerate",  # "enumerate" or "align"
) -> None:
    """
    Export interpretable fuzzy rules to a LaTeX .tex file.

    Output:
      <dataset_outdir>/rules_<model_name>.tex

    Contents (per rule):
      - support, sigma
      - semantics: rho_mean and AND/OR/COMP percentages (if model._rule_stats exists)
      - IF antecedent using linguistic terms derived from center[j]
      - THEN consequent (Safe: class_probs; PA: W row per rule)

    Notes:
      - Best-effort: never breaks experiment if some info is missing.
      - For anonymous submission, this contains no author info.
    """
    os.makedirs(dataset_outdir, exist_ok=True)
    outpath = os.path.join(dataset_outdir, f"rules_{model_name}.tex")

    try:
        if not hasattr(model, "rules") or len(getattr(model, "rules")) == 0:
            return

        rules_list = list(getattr(model, "rules"))
        n_rules = len(rules_list)

        # stats collected online (optional)
        stats_list = []
        if hasattr(model, "_rule_stats"):
            try:
                stats_list = list(getattr(model, "_rule_stats"))
            except Exception:
                stats_list = []

        # consequent: Safe vs PA
        has_safe_probs = hasattr(rules_list[0], "class_probs")
        W = None
        has_pa_W = False
        if hasattr(model, "W") and getattr(model, "W") is not None:
            try:
                W = np.asarray(getattr(model, "W"), dtype=float)
                has_pa_W = True
            except Exception:
                W = None
                has_pa_W = False

        # sort by support descending (more publishable)
        supports = np.array([int(getattr(r, "support", 0)) for r in rules_list], dtype=int)
        order = list(np.argsort(-supports))

        k = min(int(max_rules_to_write), n_rules)

        lines: List[str] = []
        lines.append(r"% Auto-generated interpretable rules (UnimNeuron models)")
        lines.append(r"% Copy/paste this block into the paper where rules are discussed.")
        lines.append("")

        if latex_env == "align":
            lines.append(r"\begin{align}")
        else:
            lines.append(r"\begin{enumerate}")

        shown = 0
        for ridx in order:
            if shown >= k:
                break

            r = rules_list[ridx]
            center = np.asarray(getattr(r, "center"), dtype=float).ravel()
            sigma = float(getattr(r, "sigma", np.nan))
            support = int(getattr(r, "support", 0))
            w = np.asarray(getattr(r, "w", np.ones_like(center)), dtype=float).ravel()

            # semantics summary (optional)
            sem_str = r"\textit{semantics: not available}"
            if ridx < len(stats_list):
                try:
                    sem = stats_list[ridx].summary()
                    sem_str = (
                        rf"\textit{{semantics: }} $\bar{{\rho}}={sem['rho_mean']:.3f}$, "
                        rf"AND={sem['AND_pct']:.1f}\%, OR={sem['OR_pct']:.1f}\%, COMP={sem['COMP_pct']:.1f}\%"
                    )
                except Exception:
                    pass

            # consequent string
            then_str = r"\text{THEN class } = ?"
            if has_safe_probs:
                try:
                    probs = np.asarray(r.class_probs(), dtype=float).ravel()
                    c = int(np.argmax(probs))
                    then_str = rf"\text{{THEN class}} = {c}\, (p={probs[c]:.3f})"
                except Exception:
                    pass
            elif has_pa_W and W is not None:
                try:
                    if W.shape[0] > 1 + ridx:
                        row = W[1 + ridx, :].ravel()
                        c = int(np.argmax(row))
                        top2 = list(np.argsort(-row)[:2])
                        then_str = (
                            rf"\text{{THEN class}} = {c}"
                            rf"\, [\text{{top: }} {top2[0]}({row[top2[0]]:.3f}), {top2[1]}({row[top2[1]]:.3f})]"
                        )
                except Exception:
                    pass

            # antecedent (top-K by weight)
            top_idx = _topk_indices_by_weight(w, k=min(top_features_per_rule, w.size))
            ants = []
            for j in top_idx:
                fname = feature_names[j] if j < len(feature_names) else f"x{j+1}"
                fname = _latex_escape(fname)
                term = _linguistic_term(float(center[j]))
                ants.append(rf"{fname}\ \text{{is}}\ {term}\ (w={w[j]:.2f})")

            # build rule latex
            if_part = r"\ \wedge\ ".join(ants) if len(ants) > 0 else r"\text{(no antecedent available)}"
            meta = rf"\textit{{support}}={support},\ \sigma={sigma:.3f}"

            if latex_env == "align":
                lines.append(
                    rf"\text{{Rule {shown+1}:}}\ & \text{{IF }} {if_part}\ \text{{ THEN }} {then_str} \\"
                )
                lines.append(rf"& {meta},\ {sem_str} \\")
                lines.append(r"")
            else:
                lines.append(r"\item")
                lines.append(rf"\textbf{{Rule {shown+1}.}} {meta}. {sem_str}.")
                lines.append(r"\\")
                lines.append(rf"$\textbf{{IF}}\ {if_part}\ \ \textbf{{THEN}}\ {then_str}.$")
                lines.append("")

            shown += 1

        if latex_env == "align":
            lines.append(r"\end{align}")
        else:
            lines.append(r"\end{enumerate}")

        with open(outpath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    except Exception:
        return


# -------------------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------------------
def main():
    summary: Dict[tuple, Dict[str, Any]] = {}

    MODELS: List[str] = [
        "ePL",
        "ePL_plus",
        "exTS",
        "Simpl_eTS",
        "eMG",
    ]
    if HAS_UNIM_SAFE:
        MODELS.append("ENF_Unim_Safe")
    if HAS_UNIM_PA:
        MODELS.append("ENF_Unim_PA")

    # stats matrix: rows=datasets, cols=models (only when complete row exists)
    acc_means_for_stats: List[List[float]] = []
    model_names_for_stats = MODELS[:]  # fixed order

    for ds_name, ds in DATASETS_ALL.items():
        print(f"\n=== DATASET: {ds_name} ===")
        n_max = get_n_max_for_dataset(ds_name, EXPERIMENT_MODE)

        X_raw, y, feature_names = load_stream_to_numpy(ds, n_max=n_max)
        n_samples, n_features = X_raw.shape
        n_classes = len(np.unique(y))

        X, _, _ = min_max_scale(X_raw)

        outdir_ds = os.path.join(BASE_OUTDIR, ds_name)
        os.makedirs(outdir_ds, exist_ok=True)

        results_for_ds: Dict[str, Dict[str, Any]] = {}

        for model_name in MODELS:
            print(f"  -> Model: {model_name}")

            if model_name == "ENF_Unim_Safe" and HAS_UNIM_SAFE:
                model = EvolvingNeuroFuzzyUnimSafe(
                    n_features=n_features,
                    n_classes=n_classes,
                    max_rules=5,
                    safe_threshold=0.07,
                    min_firing_for_update=0.05,
                    tau_merge=0.62,
                    drift_sensitivity=0.35,
                )
                eval_fn = prequential_nf_generic

            elif model_name == "ENF_Unim_PA" and HAS_UNIM_PA:
                model = EvolvingNeuroFuzzyUnimPA(
                    n_features=n_features,
                    n_classes=n_classes,
                    max_rules=5,
                    C=0.25,
                    use_PA_I=True,
                    margin_threshold=0.8,
                    decay_rate=1e-2,
                    tau_merge=0.68,
                )
                eval_fn = prequential_nf_generic

            elif model_name == "ePL":
                model = ePL()
                eval_fn = lambda m, X_, y_, desc: prequential_efs(
                    m, X_, y_, warmup=WARMUP, batch_evolve=BATCH_EVOLVE, desc=desc
                )

            elif model_name == "ePL_plus":
                model = SafeEFSWrapper(ePL_plus(), name="ePL_plus")
                eval_fn = lambda m, X_, y_, desc: prequential_efs(
                    m, X_, y_, warmup=WARMUP, batch_evolve=BATCH_EVOLVE, desc=desc
                )

            elif model_name == "exTS":
                model = SafeEFSWrapper(exTS(), name="exTS")
                eval_fn = lambda m, X_, y_, desc: prequential_efs(
                    m, X_, y_, warmup=WARMUP, batch_evolve=BATCH_EVOLVE, desc=desc
                )

            elif model_name == "Simpl_eTS":
                model = SafeEFSWrapper(Simpl_eTS(), name="Simpl_eTS")
                eval_fn = lambda m, X_, y_, desc: prequential_efs(
                    m, X_, y_, warmup=WARMUP, batch_evolve=BATCH_EVOLVE, desc=desc
                )

            elif model_name == "eMG":
                model = SafeEFSWrapper(eMG(), name="eMG")
                eval_fn = lambda m, X_, y_, desc: prequential_efs(
                    m, X_, y_, warmup=WARMUP, batch_evolve=BATCH_EVOLVE, desc=desc
                )

            else:
                print(f"    [WARN] Model {model_name} not available, skipping.")
                continue

            t0 = time.time()
            res = eval_fn(
                model, X, y, desc=f"{model_name} ({ds_name}, mode={EXPERIMENT_MODE})"
            )
            exec_time = time.time() - t0

            acc = res["accuracy"]
            n_rules_final = res.get("n_rules_final", -1)
            pred_fallbacks = res.get("pred_fallbacks", 0)
            P_instabilities = res.get("P_instabilities", 0)

            final_acc = float(acc[-1])
            mean_acc = float(acc.mean())

            res["time_s"] = exec_time
            res["N"] = n_samples
            results_for_ds[model_name] = res

            summary[(EXPERIMENT_MODE, ds_name, model_name)] = dict(
                acc_final=final_acc,
                acc_mean=mean_acc,
                n_rules_final=float(n_rules_final),
                time_s=float(exec_time),
                pred_fallbacks=int(pred_fallbacks),
                P_instabilities=int(P_instabilities),
                N=int(n_samples),
            )

            print(
                f"[RESULT] mode={EXPERIMENT_MODE:8s} | dataset={ds_name:18s} | "
                f"model={model_name:15s} | N={n_samples:5d} | "
                f"Acc_final={final_acc:.4f} | Acc_mean={mean_acc:.4f} | "
                f"Rules_final={n_rules_final} | Time={exec_time:.1f}s | "
                f"Pred_fallbacks={pred_fallbacks} | P_instabilities={P_instabilities}"
            )

            export_rules_latex(
                model=model,
                model_name=model_name,
                dataset_outdir=outdir_ds,
                feature_names=feature_names,
                max_rules_to_write=5,
                top_features_per_rule=6,
                latex_env="enumerate",  # ou "align"
            )


        # plots (now with drift markers when available)
        plot_all_models_accuracy(ds_name, results_for_ds, outdir_ds)
        plot_all_models_rules(ds_name, results_for_ds, outdir_ds)

        # stats row only if all models exist for this dataset
        row = []
        ok = True
        for m in model_names_for_stats:
            if m not in results_for_ds:
                ok = False
                break
            row.append(float(results_for_ds[m]["accuracy"].mean()))
        if ok and len(row) >= 2:
            acc_means_for_stats.append(row)

    tex_path = os.path.join(BASE_OUTDIR, "exp_stream_summary.tex")
    build_latex_table(summary, tex_path)
    print(f"\nLaTeX summary table saved to: {tex_path}")

    # Stats
    if len(acc_means_for_stats) >= 2 and len(model_names_for_stats) >= 2:
        acc_mat = np.array(acc_means_for_stats, dtype=float)
        build_stats_tables(
            acc_mean_matrix=acc_mat,
            model_names=model_names_for_stats,
            outdir=BASE_OUTDIR,
        )
        print("LaTeX stats saved to:")
        print(f"  - {os.path.join(BASE_OUTDIR, 'exp_stats_friedman.tex')}")
        print(f"  - {os.path.join(BASE_OUTDIR, 'exp_stats_posthoc.tex')}")
    else:
        print("\n[WARN] Not enough complete rows to run Friedman/posthoc.")


if __name__ == "__main__":
    main()
