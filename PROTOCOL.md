# T17 Protocol: feature-relevance stability in UnimNeuron

**Group 17 · Theme 17 (Track D: interpretability and structural explanation)**
**Status: DRAFT for team approval (kick-off, Fri 20 Nov 2026).** The protocol is frozen when the team tags the
approved commit `protocol-freeze`. After that, every change goes into [§14 Deviations](#14-deviations) with a
date and a reason. Nothing here may be changed because of a result.

---

## 1. Research question, sub-questions and hypotheses

**Near-final RQ.** Does UnimNeuron's class-separability feature relevance (global window, recursive and
per-rule local) remain stable in stationary segments and re-orient to the true relevant features after drift,
compared with stationary/shuffled controls and a shuffled-weights ablation, under abrupt and gradual
feature-relevance drift, as measured by top-k overlap and weighted rank correlation across lags, top-k
precision against ground truth, and ranking recovery time?

| SQ | Question | Primary metric (expected direction) | Owner |
|---|---|---|---|
| SQ1 stability | How much do relevance rankings fluctuate when nothing changes? | Top-k Jaccard of R1 at lag 200: **Elec2 vs shuffled Elec2** (Elec2 lower = real temporal change on top of noise) | WF1 Minês |
| SQ2 drift sensitivity | After drift, how fast does the ranking re-orient? | Ranking recovery time of R1 on S1 (expected 100–250 samples for the 200-sample window, growing with window size) | WF2 Bruno + Luis |
| SQ3 validity | Do the weights rank the truly relevant features first? | Top-k precision of R1 after warm-up in the clean segments of S1 (expected 1 for mean-shift relevance) | WF2 Luis |
| SQ4 information content | Do the weights carry information the model uses? | Mean prequential accuracy, `open` FULL vs `open` SHUFFLED_W, predictive streams only (FULL > SHUFFLED_W if they do) | WF4 Diogo |

Everything else in §5 is secondary and reported as such.

**Hypotheses**
- **H1 (stability):** in stationary segments the top-k set of R1 is stable (Jaccard ≈ 1 at lags ≥ 200). The
  ordering of irrelevant features is noise (τ-b over all features stays well below 1, equally on the control).
- **H2 (drift sensitivity):** after an abrupt relevance swap, R1 reaches top-k precision 1 against the new truth
  within 100–250 samples (window 200), and the delay grows with the window size.
- **H3 (global vs local):** per-rule local relevance (R3) is less stable than global relevance (R1).
- **H4 (validity):** after warm-up, relevant features rank above irrelevant ones (top-k precision 1, AUC 1)
  when relevance shows up as a class-mean difference; it fails when it does not (Agrawal F0, §6).
- **H5 (information content):** with `open`, shuffling the weights (SHUFFLED_W) lowers accuracy.
- **H6 (outdated explanations):** after drift, the weights the rules actually use (R4) lag behind R1/R3.

Negative or inconclusive results are valid findings if the protocol is followed.

## 2. What the baseline does (read from the code)

Facts that the protocol depends on. Line numbers refer to the unmodified baseline.

- **Global weights** `current_feature_weights` ([ENF_UnimNeuron_Safe.py:85](ENF_UnimNeuron_Safe.py)): a
  between/within-class variance ratio per feature over the **last 200 samples**, recomputed when
  `K % 50 == 0` (line 277), floored at 0.2 and scaled so the maximum is 1 (line 113). They depend only on the
  data, not on the model state.
- **Rule weights** `rule.w` are a copy of the global weights when the rule is created (line 283) and change only
  when rules merge.
- **The rule cap freezes the rule base.** With `max_rules=5`, a new rule has support 1 and is the one pruned
  (`argmin(support)`, line 292). Measured with `RuleEventCounter` on configs/mve.json (20,000 samples, seeds
  0–2): about 19,800 rules created per run, of which **99.9 % were pruned in the same step**. Merging
  never fires (`min_rules_for_merge=5`). The surviving rules are created in the first ~20 samples with uniform
  weights `1/d`.
- `safe_threshold`, `min_firing_for_update` and `drift_sensitivity`, passed by the baseline scripts, are
  **ignored** (`**kwargs`, line 165). The configs keep them to mirror the baseline exactly.
- The model is **deterministic** (`self.rng` is never used): seeds only change the stream.
- The baseline's `min_max_scale` ([exp_sota_comparison.py:395](exp_sota_comparison.py)) uses the whole stream:
  **future-information leakage**. Not used here (§8).
- `exp_sota_comparison.py:111` imports `enfs_uni0_evolving`, which is not in the repository (WF4 fixes this
  without editing the baseline).

## 3. Relevance definitions

| Id | Name | Definition | Where | Priority |
|---|---|---|---|---|
| R1 | Global window | `model.current_feature_weights` (copy, read-only) | `relevance/extract.py` (`G`) | Must, done |
| R2 | Recursive feature distances | Per-class running mean/variance with forgetting λ ∈ {0.99, 0.995, 1.0}; per feature, between-class distance ÷ within-class spread, mapped to [0, 1]; updated only after the prediction | WF3 | Should |
| R3 | Local per rule | R1's formula on a 200-point window of the samples each rule wins (highest activation); NaN if < 20 points or one class; coverage logged | WF3 | Must |
| R4 | Effective (rule base) | Support-weighted mean of `rule.w` = what prediction uses | `relevance/extract.py` (`A`) | Must, done |
| R5 | Permutation importance | Accuracy drop when one feature is permuted on the last 200 seen samples, every 500 samples, prediction only, separate RNG, 5 repeats | WF2 | Could |

## 4. Checkpoints and windows

- A checkpoint is taken right after `partial_fit` whenever `t + 1 = K` is a multiple of **50**, which is
  when the model refreshes R1. Checkpoints in between would only repeat values.
- **Clean checkpoints:** the R1 buffer (200 samples) must contain data from one concept only. This means
  at least `warmup_flush = 200` samples after the start of the segment, and no sample from a gradual
  transition. Stability and validity use clean checkpoints only.
- **Lags** for stability: 50, 200, 500 and 2,000 samples. Lag 50 is reported but flagged, because consecutive
  snapshots share 150 of 200 samples. Lag 200 is the first lag with disjoint buffers.
- **Window analysis (WF5):** non-overlapping windows of 1,000 samples, mean relevance per window; compare
  consecutive windows, the first vs the last window, and, on drift streams, the last window before the drift
  vs the last window.

## 5. Metrics

**Primary:** one per SQ (§1).
**Secondary:**
- *Stability:* weighted τ (`scipy.stats.weightedtau`, top-weighted), τ-b over all features, Spearman,
  per-feature variance in clean segments, top-k switch rate per 1,000 samples, between-seed vs within-run
  agreement. τ-b over all features measures mostly the random order of irrelevant features (§13), so it is
  never used alone.
- *Validity:* ROC-AUC of the weights vs the ground-truth mask.
- *Drift:* restabilisation time (WF5): weighted τ at lag 200 back above the 10th percentile of its pre-drift
  values for 3 checkpoints in a row.
- *Predictive:* cumulative and 500-sample windowed accuracy and macro-F1; accuracy drop and recovery (§10).
- *Structural:* rule count, rule creations / prunings / newborn prunings / merges (`RuleEventCounter`).
- *Resources:* processing time per sample, logging overhead, peak memory (WF4, tracemalloc).

`k` = number of truly relevant features (3 on S1/S2/S3/S1u/S3u, 2 on S4) and 3 on Elec2.
Top-k precision is tie-aware: ties at the k-th place count at their expected value, so uniform weights score
chance, never 1.

## 6. Streams

All synthetic streams: n = 20,000, seeds 0–9 (seed = generator seed, stream order fixed by it), fixed known
bounds (no data-dependent scaling), ground-truth relevance mask per time step. Configs in
[configs/protocol/](configs/protocol).

| Id | Config | Stream | Relevant set | Role (§7) |
|---|---|---|---|---|
| S1 | `s1_gauss_abrupt` | Gaussian mean shift (sd 0.12, class means 0.5 ± δ/2 on relevant features, clipped to [0,1]), d = 10 | {0,1,2} → {5,6,7} at 10,000, abrupt | predictive (δ = 0.3) |
| S1u | `s1u_uniform_abrupt` | Uniform features, y = 1[sum of relevant > 1.5] (Minês's stream, = configs/mve.json) | {0,1,2} → {5,6,7} at 10,000 | relevance-only |
| S2 | `s2_gauss_gradual` | S1 with sigmoid transition of width 1,000 (as river's ConceptDriftStream) | same; transition 9,500–10,500 | predictive (δ = 0.3) |
| S3 | `s3_gauss_stationary` | S1's first concept only | {0,1,2} | predictive (δ = 0.3) |
| S3u | `s3u_uniform_stationary` | S1u's first concept only (= configs/stationary_control.json) | {0,1,2} | relevance-only |
| S4 | `s4_sea_drift` | SEA, x ∈ U[0,10]³ (divided by 10), y = 1[x0 + x1 > θ], θ 8 → 7 at 10,000 | {0,1} throughout (negative control) | relevance-only |
| S5 | WF2 | Agrawal pair, chosen by the rule below | from the generator | WF2 |
| S6 | WF2 | S1 plus x10 = x0 + small noise (non-causal copy) | {0,1,2} → {5,6,7}; x10 never | WF2 |
| E | `elec2` | river Elec2, first 10,000 samples in real order, `date` dropped, expanding min-max | unknown | descriptive (SQ1) |
| E-sh | `elec2_shuffled` | E shuffled (seed = permutation) | unknown | noise floor for E |

**Agrawal selection rule (S5, fixed now, run by WF2):** run the oracle check (R1's formula on 100,000
stationary samples per function). Choose the two functions whose relevant sets (from the generator code) are
detected with top-k precision 1 and are disjoint; ties go to the lowest function index. Function 0 is
reported as a documented failure case: age matters in a non-monotone way that a class-mean ratio cannot see.

## 7. Learnability rule and pilot outcome

Accuracy-based claims (accuracy recovery, SQ4) are made only on **predictive** streams. The rule was
fixed before the pilot ran ([experiments/pilot_learnability.py](experiments/pilot_learnability.py)):
- Seed 99 only (never used in experiments) and accuracy only; relevance is not looked at.
- Each family is tested on its stationary version, n = 20,000.
- A family is *predictive* if `open` (`max_rules=None`) beats the majority-class rate by ≥ 5 percentage
  points.
- Gaussian δ = the smallest of 0.3 / 0.4 / 0.5 that passes.

**Pilot outcome** (20 Nov 2026; [results_pilot/pilot.csv](results_pilot/pilot.csv),
[pilot_decision.json](results_pilot/pilot_decision.json)):

| Stream (stationary, seed 99) | `ref` accuracy | `open` accuracy | Majority | `open` margin | Role |
|---|---|---|---|---|---|
| Gaussian δ = 0.3 | 0.494 | **0.627** | 0.501 | +12.6 pp | **predictive → δ = 0.3** |
| Gaussian δ = 0.4 | 0.494 | 0.726 | 0.501 | +22.5 pp | (not needed) |
| Gaussian δ = 0.5 | 0.495 | 0.650 | 0.501 | +14.9 pp | (not needed) |
| Uniform sum (S1u/S3u) | 0.505 | 0.512 | 0.501 | +1.1 pp | relevance-only |
| SEA variant 0 (S4) | 0.678 | 0.678 | 0.678 | 0.0 pp | relevance-only |

- `ref` is at majority level on every stream, which is consistent with the frozen rule base (§2).
- Only the Gaussian family supports accuracy-based claims, and only with `open`.
- Accuracy is not monotone in δ (0.4 > 0.5). This is recorded, not acted on: the rule picks the smallest
  passing δ.
- Elec2 was not piloted (it needs `river`). It stays relevance-only until WF4's majority/no-change check
  says otherwise.

## 8. Preprocessing and leakage

- Synthetic streams are generated inside fixed bounds; no scaling is fitted to data.
- Real streams use **expanding min-max**: sample t is scaled with the minimum and maximum of samples 0..t
  only. Its label is never used before its prediction.
- Prequential order: predict, then `partial_fit` (t = 0 predicts class 0), exactly as
  `prequential_nf_generic`. `tests/test_nonintrusive.py` checks that flipping the label at t changes no
  prediction up to t.

## 9. Model variants

| Variant | Settings | Purpose |
|---|---|---|
| `ref` | baseline settings (`max_rules=5`, `tau_merge=0.62`, ignored kwargs kept) | the published configuration; structural finding (frozen rule base) |
| `open` | as `ref` with `max_rules=None` | rules created after t = 50 survive, so rule weights can change |
| NO_W (WF4) | weights set to 1 after every `partial_fit` (as `exp_ablation.py`) | tests weight *scale* + information together |
| SHUFFLED_W (WF4) | each rule's weights in a fixed random order (permutation drawn once per rule, separate RNG) | same scale, no information → SQ4 primary |

Ablations are wrappers around the unmodified model. The baseline files are never edited.

## 10. Drift ground truth, recovery and censoring

- **Synthetic:** drift positions and widths come from the generator. A gradual transition is
  [dp − w/2, dp + w/2]. Checkpoints whose buffer touches it are excluded from stability and validity, and
  recovery is counted from the **end** of the transition.
- **Elec2:** no ground truth. ADWIN (δ = 0.002) on the `ref` error stream gives estimated change points,
  used only as an overlay in descriptive plots. No recovery claims on Elec2.
- **Ranking recovery:** first checkpoint after the drift (or transition end) where top-k precision vs the
  new truth = 1 for **3 checkpoints in a row**.
- **Accuracy recovery** (predictive streams only):
  - Pre-drift level = mean 500-window accuracy over the last 1,000 samples before the drift (or
    transition start).
  - Trough = minimum within 1,000 samples after the drift.
  - Recovered = the first point from the trough on where windowed accuracy ≥ pre-drift − 0.02 for
    **3 checkpoints in a row**.
- **Censoring:** if never reached, the run is censored at the end of the segment. Report the % censored, use
  the censoring value (`recovery_censored/*`) in a Wilcoxon test, and add a sign test as a check.

## 11. Statistical plan

- 10 seeds (0–9). Report mean ± SD and a 95 % bootstrap CI (10,000 resamples) of the per-seed values.
- Elec2 has one real order. Its CI uses a block bootstrap over checkpoints, with blocks of 500 samples.
- **Paired comparisons** (same seed = same stream): Wilcoxon signed-rank + matched-pairs rank-biserial
  effect size, Holm correction within each SQ. With 10 seeds the smallest two-sided p is 0.002 (5 seeds
  could never reach 0.05).
- **SQ1 noise floor:** compare Elec2 with the 10 shuffled-Elec2 values (empirical percentile).
  - S3 within-run vs between-seed agreement is a sanity check only. For a window statistic on an iid
    stream the two match by construction.

## 12. Are the relevance scores meaningful?

- Ground-truth recovery on S1, S2 and S5 (SQ3).
- Negative controls:
  - S3 / S3u: nothing changes.
  - S4: accuracy can drop but the relevant set doesn't change. On S4, `recovery/*` is trivially the first
    checkpoint after the drift and is not reported. The check is that the top-2 set before and after the
    drift is identical (window analysis, WF5). Report macro-F1 next to accuracy on S4, because its classes
    are unbalanced (about 68/32 before the drift, 76/24 after).
  - Elec2 shuffled: no temporal structure.
- The oracle check separates two failure modes: the statistic can't see the feature, or the 200-sample
  estimate is noisy.
- R5 permutation importance is a model-based cross-check (Could).
- SHUFFLED_W tests whether the information in the weights matters for prediction.

## 13. Pre-registration status (what was seen before the freeze)

To be stated in the report. These results are **exploratory**; confirmatory results come only from
`configs/protocol/` with seeds 0–9 after the tag.
- `results_relevance/mve` and `results_relevance/stationary_control` (Minês, 19 Nov, `ref`, seeds 0–2,
  uniform-sum stream):
  - R1 top-3 precision 1.0 in both segments; ranking recovery 150 samples in every seed.
  - τ-b ≈ 0.47 at lags ≥ 200 on both streams, against top-3 Jaccard 1.0.
  - R4 frozen; accuracy ≈ 0.50.
- `results_q2/window_sensitivity` (Bruno, 19 Nov, `ref`, seeds 0–9, same stream): recovery 100 / 100 / 150 /
  310 / 580 samples for windows 50 / 100 / 200 / 500 / 1,000.
- Kick-off checks (20 Nov):
  - The `ref` model stays at chance on uniform-sum streams.
  - On a Gaussian mean-shift stream, `open` reached 0.612 against 0.506 majority (one run, not a protocol
    seed).
- So H2's "100–250 samples" was derived from the mechanism and is *consistent with* the exploratory runs.
  The confirmatory tests of H2 are S1 (Gaussian), S2 (gradual) and the window-size scaling on S1.

## 14. Deviations

| Date | Section | Change | Reason | Approved by |
|---|---|---|---|---|
| | | | | |

## 15. Workflows, priorities and report

| Workflow | Owner(s) | Delivers |
|---|---|---|
| WF1 Stability (SQ1) | Minês | SQ1 row of Table 1, Figure 2a |
| WF2 Drift + validity (SQ2, SQ3) | Bruno + Luis | SQ2/SQ3 rows, Figure 1, S5/S6, window-size study (Bruno, already explored) |
| WF3 Feature relevance (R2, R3) | João | R2, R3, H3/H6 results |
| WF4 Baseline + information content (SQ4) | Diogo | runnable baseline, `REPRODUCE.md`, ablations, SQ4 row, demo |
| WF5 Windows + restabilisation (SQ1×SQ2) | Minês + Bruno | Figure 2b (first vs last window), restabilisation |

- **Priorities:**
  - Must = kick-off, baseline + tests, S1/S3 results, first vs last window, SQ4 with `open`, R3.
  - Should = R2, H6, S5 Agrawal, S6, window-size trade-off.
  - Could = R5, proposal Alternative 3.
  - On Tue 24 Nov, if a Must item is late, Could items are dropped.
- **Progress report (2–3 pages):**
  - Figure 1: S1 heatmap + accuracy + rule count.
  - Figure 2: (a) stability over time, (b) first vs last window.
  - Table 1: one row per SQ with its primary metric.

## 16. How to run

```powershell
python -m pip install -r requirements.txt
python tests/test_nonintrusive.py
python tests/test_regression_saved_results.py
python tests/test_kickoff_core.py
python experiments/pilot_learnability.py
python run_relevance.py --config configs/protocol/s1_gauss_abrupt.json --plot
python run_relevance.py --config configs/protocol/s3_gauss_stationary.json --variants open --outdir results_wf1
```

Each run writes `<outdir>/<config>/<variant>/seed_<s>/`:
- `checkpoints.npz` and `checkpoints.csv` (the same numbers).
- `per_rule.csv`.
- `metrics.json`.
- `meta.json`: git commit, versions, stream parameters, rule-event counts.
