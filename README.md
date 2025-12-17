# UnimNeuron: Automatic Connective Selection for Interpretable Evolving Neuro-Fuzzy Rules

This repository provides a reproducible experimental pipeline for evaluating **UnimNeuron-based evolving neuro-fuzzy models** under a **prequential (test-then-train)** protocol on **real and synthetic data streams**.  
It is intentionally **anonymous** for double-blind review. After publication, this repository can be updated with author/project metadata.

---

## 1) Repository structure

```
.
├── ENF_UnimNeuron_PA.py          # Proposed model (PA-based local adaptation)
├── ENF_UnimNeuron_Safe.py        # Proposed model (stability-oriented local adaptation)
├── exp_sota_comparison.py        # Experiment 1: SOTA comparison (streams)
├── exp_ablation.py               # Experiment 2: Ablation study (UnimNeuron components)
├── results_exp/                  # Outputs of Experiment 1 (SOTA comparison)
│   ├── <dataset_name>/
│   │   ├── <dataset>_acc_all_models.png
│   │   ├── <dataset>_rules_all_models.png
│   │   ├── rules_<model>.txt / rules_<model>.tex (if enabled)
│   │   └── ...
│   ├── exp_stream_summary.tex
│   ├── exp_stats_friedman.tex
│   └── exp_stats_posthoc.tex
└── results_exp2/                 # Outputs of Experiment 2 (ablation)
    ├── <dataset_name>/
    │   ├── <dataset>_acc_ablation.png
    │   └── ...
    └── exp3_ablation_summary.tex
```

---

## 2) Environment and dependencies

### Python
- Recommended: Python **3.10+**

### Main dependencies
- `numpy`
- `matplotlib`
- `tqdm`
- `river`
- `evolvingfuzzysystems`

### Optional (recommended for statistics)
- `scipy`
- `scikit-posthocs`

Example installation:
```bash
pip install numpy matplotlib tqdm river scipy scikit-posthocs
pip install evolvingfuzzysystems
```

---

## 3) How to reproduce the experiments

### 3.1 Experiment 2 — SOTA comparison
```bash
python exp_sota_comparison.py
```

Outputs are stored in `results_exp/` and include figures and LaTeX tables.

### 3.2 Experiment 3 — Ablation study
```bash
python exp_ablation.py
```

Outputs are stored in `results_exp2/`.

---

## 4) Experimental protocol

All streams follow a **prequential test-then-train** protocol.  
Features are min–max normalized to `[0,1]`.  
Reported metrics include accuracy, rule growth, runtime, and stability indicators.

---

## 5) Interpretability

The repository supports exporting **human-readable fuzzy rules** for UnimNeuron-based models, including:
- feature relevance weights,
- adaptive connective behavior (AND / OR / COMP),
- rule support and dispersion,
- class preferences.

These exports are intended to support interpretability analysis in the paper.

---

## 6) Reproducibility notes

- Synthetic datasets use fixed random seeds.
- Minor numerical differences may occur across platforms.
- Repository is anonymized for double-blind review.

---

## 7) License and citation

License: to be defined.  
Citation details will be added after publication.
