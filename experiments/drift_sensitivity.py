import copy
import json
import sys
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_relevance import run_one
from relevance.analysis import analyze

WINDOW_SIZES = [50, 100, 200, 500, 1000]
SEEDS = list(range(10))

with open(ROOT / "configs" / "mve.json") as f:
    base_config = json.load(f)

results = []

for window in WINDOW_SIZES:

    for seed in SEEDS:

        config = copy.deepcopy(base_config)

        config["model"]["buffer_size_separability"] = window

        # Use a common washout period for fair comparisons
        config["analysis"]["warmup_flush"] = max(WINDOW_SIZES)

        result = run_one(config, seed, log=True)

        stream = result["stream"]
        checkpoints = result["logger"].arrays()

        metrics = analyze(
            checkpoints,
            stream.truth,
            stream.drift_points,
            config["analysis"],
            config["checkpoint_every"]
        )

        recovery_g = metrics["recovery/G/drift1"]
        recovery_a = metrics["recovery/A/drift1"]

        row = {
            "window": window,
            "seed": seed,
            "recovery_G": recovery_g,
            "recovery_A": recovery_a,
            "precision_G": metrics["valid/G/prec_at_k/seg1"],
            "precision_A": metrics["valid/A/prec_at_k/seg1"],
            "stability_G": metrics["stab/G/tau_b/lag200"],
            "accuracy": result["acc_final"],
            "runtime": result["proc_time_s"]
        }

        results.append(row)

        print(
            f"Window={window:4d} | "
            f"Seed={seed} | "
            f"Recovery G={recovery_g} | "
            f"Recovery A={recovery_a} | "
            f"Accuracy={result['acc_final']:.4f}"
        )

print("\n--- Mean Global Recovery Times ---")

for window in WINDOW_SIZES:
    values = [
        row["recovery_G"]
        for row in results
        if row["window"] == window
        and np.isfinite(row["recovery_G"])
    ]

    mean = np.mean(values) if values else float("nan")

    print(f"Window {window}: {mean:.2f} samples ({len(values)}/3 recovered)")


    # Export individual results

output_dir = ROOT / "results_q2" / "window_sensitivity"
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# Aggregate results by window size

summary = []

for window in WINDOW_SIZES:
    rows = [r for r in results if r["window"] == window]

    recovery = [
        r["recovery_G"] for r in rows
        if np.isfinite(r["recovery_G"])
    ]

    summary.append({
        "window": window,
        "recovery_mean": np.mean(recovery) if recovery else np.nan,
        "recovery_std": np.std(recovery) if recovery else np.nan,
        "recovered_G": len(recovery),
        "recovered_A": sum(
            np.isfinite(r["recovery_A"]) for r in rows
        ),
        "precision_G": np.mean([r["precision_G"] for r in rows]),
        "precision_A": np.mean([r["precision_A"] for r in rows]),
        "stability_G": np.mean([r["stability_G"] for r in rows]),
        "accuracy": np.mean([r["accuracy"] for r in rows])
    })

with open(output_dir / "summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

windows = [r["window"] for r in summary]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Recovery time
axes[0].errorbar(
    windows,
    [r["recovery_mean"] for r in summary],
    yerr=[r["recovery_std"] for r in summary],
    marker="o",
    capsize=5
)

axes[0].set_title("Global relevance recovery")
axes[0].set_ylabel("Samples after drift")

# Stability
axes[1].plot(
    windows,
    [r["stability_G"] for r in summary],
    marker="o",
    color="orange"
)

axes[1].set_title("Global ranking stability")
axes[1].set_ylabel("Kendall tau-b (lag 200)")
axes[1].set_ylim(-1, 1)

# Feature relevance validity
axes[2].plot(
    windows,
    [r["precision_G"] for r in summary],
    marker="o",
    label="Global weights"
)

axes[2].plot(
    windows,
    [r["precision_A"] for r in summary],
    marker="s",
    label="Rule-base weights"
)

axes[2].set_title("Post-drift relevance validity")
axes[2].set_ylabel("Precision@3")
axes[2].set_ylim(0, 1.05)
axes[2].legend()

for ax in axes:
    ax.set_xlabel("Window size")
    ax.set_xscale("log")
    ax.set_xticks(windows)
    ax.set_xticklabels(windows)
    ax.axvline(200, linestyle="--", color="gray")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    output_dir / "window_sensitivity.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print(f"\nResults saved to {output_dir}")