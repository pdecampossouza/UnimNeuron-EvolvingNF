# UnimNeuron: Automatic Connective Selection for Interpretable Evolving Neuro-Fuzzy Systems
# Minês was here - test


This repository provides a **reproducible and anonymous research codebase** accompanying the paper:

> *UnimNeuron: Automatic Connective Selection for Interpretable Evolving Neuro-Fuzzy Rules*

The project introduces the **UnimNeuron**, a novel neuro-fuzzy neuron whose logical connective
(AND / OR / COMP) is **automatically selected from data**, enabling transparent and adaptive
rule-based reasoning in **non-stationary data streams**.

---

## 🔹 Quick Start

```bash
git clone https://github.com/pdecampossouza/UnimNeuron-EvolvingNF.git
cd UnimNeuron-EvolvingNF
pip install -r requirements.txt
python exp_sota_comparison.py
python exp_ablation.py
```

All figures, LaTeX tables, and exported rules will be generated automatically.

---

## 📁 Repository Structure

```
UnimNeuron-EvolvingNF/
├── ENF_UnimNeuron_PA.py
├── ENF_UnimNeuron_Safe.py
├── exp_sota_comparison.py
├── exp_ablation.py
├── results_exp/
│   └── <dataset_name>/
│       ├── *_acc_all_models.png
│       ├── *_rules_all_models.png
│       ├── rules_*.tex
│       └── summary tables (.tex)
├── results_exp2/
│   └── <dataset_name>/
│       ├── *_acc_ablation.png
│       └── ablation summary (.tex)
├── README.md
```

---

## 🔬 Experiments

### Experiment 1 – State-of-the-Art Comparison
- Benchmarks UnimNeuron models against evolving fuzzy systems
- Evaluated under **prequential (test-then-train)** protocol
- Includes accuracy, rule growth, drift markers, and statistical tests

Run:
```bash
python exp_sota_comparison.py
```

---

### Experiment 2 – Ablation Study
Evaluates the contribution of each UnimNeuron component:
- FULL, FIXED_AND, FIXED_OR, FIXED_COMP, NO_W

Run:
```bash
python exp_ablation.py
```

---

## 🔍 Interpretability & Rules

Each UnimNeuron corresponds to the **antecedent of a fuzzy rule**.
Rules are automatically exported in LaTeX, including:
- Feature names
- Linguistic labels
- Feature relevance weights
- Logical regime statistics (AND / OR / COMP)

---

## 📦 Dependencies

Main dependencies:
```
numpy
matplotlib
tqdm
river
evolvingfuzzysystems
scipy
scikit-posthocs
```

---

## 📝 Anonymity & Reproducibility

This repository is structured for **anonymous peer review**.
All results reported in the paper can be reproduced by running the scripts.


---

## 🔍 Experimental protocol

All streams follow a **prequential test-then-train** protocol.  
Features are min–max normalized to `[0,1]`.  
Reported metrics include accuracy, rule growth, runtime, and stability indicators.

---

## 🔍 Interpretability

The repository supports exporting **human-readable fuzzy rules** for UnimNeuron-based models, including:
- feature relevance weights,
- adaptive connective behavior (AND / OR / COMP),
- rule support and dispersion,
- class preferences.

These exports are intended to support interpretability analysis in the paper.

---

## 📝 Reproducibility notes

- Synthetic datasets use fixed random seeds.
- Minor numerical differences may occur across platforms.
- Repository is anonymized for double-blind review.

---

## 📖 License and citation

License: to be defined.  
Citation details will be added after publication.
