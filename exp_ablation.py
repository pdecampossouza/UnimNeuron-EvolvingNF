"""
Experiment 3 – Ablation study for UnimNeuron-based evolving neuro-fuzzy models
============================================================================

This script runs an *ablation* study focused on the proposed UnimNeuron-based
models (ENF_UnimNeuron_Safe and ENF_UnimNeuron_PA).

Ablations (antecedent side)
---------------------------
We evaluate how much each component contributes by selectively disabling it:

1) FULL (UnimNeuron): adaptive AND/OR/COMP regime (default unim_aggregate).
2) FIXED_AND: force antecedent aggregation to behave as AND (min).
3) FIXED_OR:  force antecedent aggregation to behave as OR  (max).
4) FIXED_COMP: force antecedent aggregation to behave as COMP (mean).
5) NO_W: disable feature relevance weights by forcing w_j = 1 for all rules
         (keeps FULL UnimNeuron regime unless combined manually).

Notes:
- This file is designed for anonymous submission (no author identifiers).
- The script is "best-effort" with River dataset availability: if a dataset class
  does not exist in your installed River version, it is skipped gracefully.
- Only numeric-feature datasets are compatible with the simple loader below.

Outputs
-------
results_exp3/
  <dataset_name>/
     <dataset>_acc_ablation.png
     rules_<model>__<ablation>.txt           (optional, best-effort)
  exp3_ablation_summary.tex

How to run
----------
python exp3_ablation.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Callable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from river import datasets
from river.datasets import synth

# ---------------------------------------------------------------------
# Optional: your UnimNeuron-based models (local files in same folder)
# ---------------------------------------------------------------------
try:
    import ENF_UnimNeuron_Safe as unim_safe_mod
    from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe

    HAS_UNIM_SAFE = True
except Exception:
    HAS_UNIM_SAFE = False

try:
    import ENF_UnimNeuron_PA as unim_pa_mod
    from ENF_UnimNeuron_PA import EvolvingNeuroFuzzyUnimPA

    HAS_UNIM_PA = True
except Exception:
    HAS_UNIM_PA = False


# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------
EXPERIMENT_MODE = "complete"  # keep consistent with your paper setup
BASE_OUTDIR = "results_exp3"
os.makedirs(BASE_OUTDIR, exist_ok=True)

# Plot aesthetics (IEEE-friendly)
PLOT_LINEWIDTH = 1.10
PLOT_GRID_ALPHA = 0.30
PLOT_DPI = 300

# ---------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------
# We use a robust "safe instantiation" approach: list candidate dataset class names;
# if a class does not exist in your River version, it is skipped.
#
# Tip: Prefer classification datasets with numeric features.
REAL_DATASET_CANDIDATES = [
    # Already used in exp2 (kept because they are stable and interpretable)
    "Elec2",
    "Phishing",
    "CreditCard",
    "Bananas",
    # Extra candidates (skip gracefully if missing)
    "ImageSegments",
]

SYNTH_DATASETS = {
    # Known drift point for marker
    "ConceptDrift_Agrawal": synth.ConceptDriftStream(
        stream=synth.Agrawal(seed=1),
        drift_stream=synth.Agrawal(seed=2),
        seed=42,
        position=2500,
        width=1000,
    ),
    # Additional synthetic streams (no single known drift marker)
    "RandomRBFDrift_synth": synth.RandomRBFDrift(),
    "Sine_synth": synth.Sine(seed=42),
}

# Stream sizes per mode
STREAM_LENGTHS_COMPLETE = {
    "Elec2_real": 10000,
    "CreditCard_real": 5000,
    "Phishing_real": 5000,
    "Bananas_real": 5000,
    "ImageSegments_real": 5000,
    "ConceptDrift_Agrawal": 20000,
    "RandomRBFDrift_synth": 20000,
    "Sine_synth": 40000,
}

STREAM_LENGTHS_PRELIM = {
    "Elec2_real": 3000,
    "CreditCard_real": 3000,
    "Phishing_real": 3000,
    "Bananas_real": 3000,
    "ImageSegments_real": 3000,
    "ConceptDrift_Agrawal": 3000,
    "RandomRBFDrift_synth": 5000,
    "Sine_synth": 5000,
}

DRIFT_MARKERS = {
    "ConceptDrift_Agrawal": {"points": [2500], "width": 1000},
}


def get_n_max_for_dataset(name: str, mode: str) -> int:
    if mode == "prelim":
        return STREAM_LENGTHS_PRELIM.get(name, 3000)
    return STREAM_LENGTHS_COMPLETE.get(name, 5000)


def safe_instantiate_real_datasets() -> Dict[str, Any]:
    """Instantiate candidate River real datasets if available in this River version."""
    out: Dict[str, Any] = {}
    for cls_name in REAL_DATASET_CANDIDATES:
        cls = getattr(datasets, cls_name, None)
        if cls is None:
            continue
        try:
            ds = cls()
        except Exception:
            continue
        key = f"{cls_name}_real"
        out[key] = ds
    return out


DATASETS_ALL: Dict[str, Any] = {}
DATASETS_ALL.update(safe_instantiate_real_datasets())
DATASETS_ALL.update(SYNTH_DATASETS)


# ---------------------------------------------------------------------
# Stream loader (numeric only)
# ---------------------------------------------------------------------
def load_stream_to_numpy(ds, n_max: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert a river dataset/stream into NumPy arrays (X, y_int)."""
    X_list: List[np.ndarray] = []
    y_raw_list: List[Any] = []

    it = ds.__iter__() if hasattr(ds, "__iter__") else iter(ds)

    try:
        x0, y0 = next(it)
    except StopIteration:
        raise RuntimeError("Empty stream/dataset")

    feature_names = list(x0.keys())

    def vec(x: dict) -> np.ndarray:
        return np.array([x[f] for f in feature_names], dtype=float)

    X_list.append(vec(x0))
    y_raw_list.append(y0)

    for i, (x, y) in enumerate(it, start=1):
        if i >= n_max:
            break
        X_list.append(vec(x))
        y_raw_list.append(y)

    X = np.vstack(X_list).astype(float)

    unique_labels = list(dict.fromkeys(y_raw_list))
    label_to_int = {lab: idx for idx, lab in enumerate(unique_labels)}
    y_int = np.array([label_to_int[lab] for lab in y_raw_list], dtype=int)

    return X, y_int, feature_names


def min_max_scale(X: np.ndarray) -> np.ndarray:
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    denom = x_max - x_min
    denom[denom == 0.0] = 1.0
    return (X - x_min) / denom


# ---------------------------------------------------------------------
# Prequential evaluation (sample-by-sample)
# ---------------------------------------------------------------------
def prequential_nf_generic(model, X, y, desc="NF") -> Dict[str, Any]:
    n = len(y)
    y_pred = np.zeros(n, int)
    correct = np.zeros(n, float)
    acc = np.zeros(n, float)
    rules = []

    pred_fallbacks = 0

    for t in tqdm(range(n), desc=desc, leave=False):
        x_t = X[t : t + 1]
        yt = int(y[t])

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

        model.partial_fit(x_t, np.array([yt]))

        if hasattr(model, "history_n_rules") and getattr(model, "history_n_rules"):
            rules.append(model.history_n_rules[-1])
        elif hasattr(model, "n_rules_"):
            rules.append(model.n_rules_)
        elif hasattr(model, "rules"):
            rules.append(len(model.rules))
        else:
            rules.append(np.nan)

    rules_arr = np.array(rules, float)
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


# ---------------------------------------------------------------------
# Ablation controls
# ---------------------------------------------------------------------
def make_fixed_aggregate(mode: str) -> Callable[[np.ndarray, float], float]:
    """Return a unim_aggregate-like function (h, e) -> float."""
    mode = mode.upper()

    def full(h: np.ndarray, e: float = 0.5) -> float:
        z = np.asarray(h, dtype=float)
        if z.size == 0:
            return float(e)
        all_le = np.all(z <= e)
        all_ge = np.all(z >= e)
        if all_le and not all_ge:
            return float(np.min(z))
        if all_ge and not all_le:
            return float(np.max(z))
        return float(np.mean(z))

    if mode == "FULL":
        return full
    if mode == "FIXED_AND":
        return lambda h, e=0.5: (
            float(np.min(np.asarray(h, dtype=float)))
            if np.asarray(h).size
            else float(e)
        )
    if mode == "FIXED_OR":
        return lambda h, e=0.5: (
            float(np.max(np.asarray(h, dtype=float)))
            if np.asarray(h).size
            else float(e)
        )
    if mode == "FIXED_COMP":
        return lambda h, e=0.5: (
            float(np.mean(np.asarray(h, dtype=float)))
            if np.asarray(h).size
            else float(e)
        )
    raise ValueError(f"Unknown aggregation mode: {mode}")


@dataclass(frozen=True)
class AblationConfig:
    name: str
    agg_mode: str
    force_no_w: bool


ABLATIONS: List[AblationConfig] = [
    AblationConfig(name="FULL", agg_mode="FULL", force_no_w=False),
    AblationConfig(name="FIXED_AND", agg_mode="FIXED_AND", force_no_w=False),
    AblationConfig(name="FIXED_OR", agg_mode="FIXED_OR", force_no_w=False),
    AblationConfig(name="FIXED_COMP", agg_mode="FIXED_COMP", force_no_w=False),
    AblationConfig(name="NO_W", agg_mode="FULL", force_no_w=True),
]


def apply_ablation_patches(ab: AblationConfig):
    fn = make_fixed_aggregate(ab.agg_mode)
    if HAS_UNIM_PA:
        unim_pa_mod.unim_aggregate = fn  # type: ignore
    if HAS_UNIM_SAFE:
        unim_safe_mod.unim_aggregate = fn  # type: ignore


def enforce_no_w(model: Any):
    if not hasattr(model, "rules"):
        return
    try:
        for r in model.rules:
            if hasattr(r, "w"):
                r.w = np.ones_like(r.w, dtype=float)
    except Exception:
        return


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def _draw_drift_markers(ax, dataset_name: str):
    info = DRIFT_MARKERS.get(dataset_name, None)
    if not info:
        return
    points = info.get("points", [])
    width = int(info.get("width", 0))
    for p in points:
        ax.axvline(p, color="red", linestyle="--", linewidth=1.0, alpha=0.70)
        if width and width > 0:
            ax.axvspan(
                max(0, p - width // 2),
                p + width // 2,
                color="red",
                alpha=0.08,
                linewidth=0,
            )


def plot_ablation_accuracy(
    dataset_name: str, curves: Dict[str, np.ndarray], outdir: str
):
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    for label, acc in curves.items():
        t = np.arange(len(acc))
        ax.plot(t, acc, label=label, linewidth=PLOT_LINEWIDTH)

    _draw_drift_markers(ax, dataset_name)

    ax.set_xlabel("Time step", fontsize=9)
    ax.set_ylabel("Prequential accuracy", fontsize=9)
    ax.set_title(dataset_name, fontsize=9)
    ax.grid(alpha=PLOT_GRID_ALPHA)

    leg = ax.legend(
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),  # abaixo do eixo
        ncol=2,
        frameon=True,
        framealpha=0.90,
        borderpad=0.25,
        handlelength=1.6,
        columnspacing=1.2,
    )

    fig.subplots_adjust(bottom=0.32)

    fig.savefig(os.path.join(outdir, f"{dataset_name}_acc_ablation.png"), dpi=PLOT_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------
def build_latex_table(rows: List[Dict[str, Any]], outpath: str):
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Ablation study on UnimNeuron components (prequential evaluation).}"
    )
    lines.append(r"\begin{tabular}{llllccccc} \toprule")
    lines.append(
        r"Mode & Dataset & Model & Ablation & $N$ & Acc$_{final}$ & Acc$_{mean}$ & $N_{\mathrm{rules}}$ & Time [s] \\ \midrule"
    )

    for r in rows:
        lines.append(
            f"{r['mode']} & {r['dataset']} & {r['model']} & {r['ablation']} & "
            f"{int(r['N'])} & {r['acc_final']:.3f} & {r['acc_mean']:.3f} & "
            f"{int(r['n_rules_final'])} & {r['time_s']:.1f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    if not (HAS_UNIM_SAFE or HAS_UNIM_PA):
        raise RuntimeError(
            "No UnimNeuron models found. Place ENF_UnimNeuron_Safe.py and/or ENF_UnimNeuron_PA.py in this folder."
        )

    model_specs: List[Tuple[str, Any]] = []
    if HAS_UNIM_SAFE:
        model_specs.append(("ENF_Unim_Safe", EvolvingNeuroFuzzyUnimSafe))
    if HAS_UNIM_PA:
        model_specs.append(("ENF_Unim_PA", EvolvingNeuroFuzzyUnimPA))

    summary_rows: List[Dict[str, Any]] = []

    for ds_name, ds in DATASETS_ALL.items():
        print(f"\n=== DATASET: {ds_name} ===")
        n_max = get_n_max_for_dataset(ds_name, EXPERIMENT_MODE)

        try:
            X_raw, y, _ = load_stream_to_numpy(ds, n_max=n_max)
        except ValueError as e:
            print(f"  [SKIP] {ds_name}: non-numeric feature encountered ({e}).")
            continue
        except Exception as e:
            print(f"  [SKIP] {ds_name}: failed to load ({e}).")
            continue

        X = min_max_scale(X_raw)
        n_samples, n_features = X.shape
        n_classes = int(len(np.unique(y)))

        outdir_ds = os.path.join(BASE_OUTDIR, ds_name)
        os.makedirs(outdir_ds, exist_ok=True)

        curves_for_plot: Dict[str, np.ndarray] = {}

        for model_name, model_cls in model_specs:
            for ab in ABLATIONS:
                label = f"{model_name}-{ab.name}"
                print(f"  -> {label}")

                apply_ablation_patches(ab)

                # Instantiate with tuned defaults
                if model_name == "ENF_Unim_Safe":
                    model = model_cls(
                        n_features=n_features,
                        n_classes=n_classes,
                        max_rules=5,
                        safe_threshold=0.07,
                        min_firing_for_update=0.05,
                        tau_merge=0.62,
                        drift_sensitivity=0.35,
                    )
                else:
                    model = model_cls(
                        n_features=n_features,
                        n_classes=n_classes,
                        max_rules=5,
                        C=0.25,
                        use_PA_I=True,
                        margin_threshold=0.8,
                        decay_rate=1e-2,
                        tau_merge=0.68,
                    )

                # Custom prequential loop to allow NO_W enforcement without altering your main evaluator
                n = len(y)
                correct = np.zeros(n, float)
                acc = np.zeros(n, float)

                t0 = time.time()
                for t in tqdm(range(n), desc=f"{label} ({ds_name})", leave=False):
                    x_t = X[t : t + 1]
                    yt = int(y[t])

                    if t == 0:
                        y_hat = 0
                    else:
                        try:
                            yhat_raw = np.asarray(model.predict(x_t)).ravel()
                            y_hat = int(yhat_raw[0]) if yhat_raw.size else 0
                        except Exception:
                            y_hat = 0

                    correct[t] = 1.0 if y_hat == yt else 0.0
                    acc[t] = correct[: t + 1].mean()

                    model.partial_fit(x_t, np.array([yt]))
                    if ab.force_no_w:
                        enforce_no_w(model)

                exec_time = time.time() - t0

                n_rules_final = len(getattr(model, "rules", []))
                curves_for_plot[label] = acc

                summary_rows.append(
                    dict(
                        mode=EXPERIMENT_MODE,
                        dataset=ds_name,
                        model=model_name,
                        ablation=ab.name,
                        N=n_samples,
                        acc_final=float(acc[-1]),
                        acc_mean=float(acc.mean()),
                        n_rules_final=int(n_rules_final),
                        time_s=float(exec_time),
                    )
                )

                print(
                    f"     [RESULT] Acc_final={acc[-1]:.4f} | Acc_mean={acc.mean():.4f} | Rules_final={n_rules_final} | Time={exec_time:.1f}s"
                )

        if curves_for_plot:
            plot_ablation_accuracy(ds_name, curves_for_plot, outdir_ds)

    tex_path = os.path.join(BASE_OUTDIR, "exp3_ablation_summary.tex")
    build_latex_table(summary_rows, tex_path)
    print(f"\nLaTeX ablation table saved to: {tex_path}")
    print("Figures saved under:", BASE_OUTDIR)


if __name__ == "__main__":
    main()
