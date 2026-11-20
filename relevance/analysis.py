"""Turns one run's checkpoints into the metrics of SQ1-SQ3 (+ structural). Pure function, no model access.

Backward compatibility: with the defaults (abrupt drift, no `acc_*` keys in the analysis config) every
metric is computed exactly as in the first version of this file; tests/test_regression_saved_results.py
recomputes the saved results of configs/mve.json and configs/stationary_control.json and checks this.

Gradual drift (drift_widths > 0): the transition is [dp - w/2, dp + w/2]. Checkpoints whose 200-sample
buffer touches the transition are not "clean" (excluded from stability/validity), and recovery times are
counted from the END of the transition (PROTOCOL.md, "Drift ground truth").
"""
from __future__ import annotations
import numpy as np
from .metrics import tau_b, topk_jaccard, topk_precision, auc, first_sustained, is_constant

NAN = float("nan")


def _nm(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if x.size and not np.all(np.isnan(x)) else NAN


def analyze(arr: dict, truth, drift_points, a: dict, every: int, drift_widths=None) -> dict:
    t, G, A, acc = arr["t"], arr["G"], arr["A"], arr["acc_win"]
    T, k, flush = len(t), a["topk"], a["warmup_flush"]
    dps = list(drift_points)
    widths = list(drift_widths) if drift_widths else [0] * len(dps)
    tr_start = [dp - w / 2.0 for dp, w in zip(dps, widths)]   # transition start (= dp when abrupt)
    tr_end = [dp + w / 2.0 for dp, w in zip(dps, widths)]     # transition end   (= dp when abrupt)

    starts = np.array([0] + tr_end, dtype=float)
    seg = np.searchsorted(starts, t - 1, side="right") - 1      # segment of the newest sample seen
    next_tr = np.array(tr_start + [np.inf], dtype=float)        # where the following transition begins
    clean = ((t - starts[seg]) >= flush) & ((t - 1) < next_tr[seg])   # buffer holds only this segment's data
    out = {"n_checkpoints": T, "struct/rules_mean": _nm(arr["n_rules"]),
           "struct/created_total": float(arr["created"][-1]), "struct/removed_total": float(arr["removed"][-1])}

    for name, W in (("G", G), ("A", A)):
        # SQ1: stability within stationary segments, at several lags (consecutive snapshots overlap 75%)
        for L in a["lags_samples"]:
            lc = max(1, L // every)
            taus, jacs = [], []
            for i in range(lc, T):
                j = i - lc
                if clean[i] and clean[j] and seg[i] == seg[j] and not is_constant(W[i]) and not is_constant(W[j]):
                    taus.append(tau_b(W[j], W[i])); jacs.append(topk_jaccard(W[j], W[i], k))
            out[f"stab/{name}/tau_b/lag{L}"] = _nm(taus)
            out[f"stab/{name}/jaccard/lag{L}"] = _nm(jacs)
        out[f"frac_constant/{name}"] = float(np.mean([is_constant(W[i]) for i in range(T)]))
        for s in range(len(starts)):
            m = clean & (seg == s)
            out[f"var/{name}/seg{s}"] = float(W[m].var(axis=0).mean()) if m.sum() > 1 else NAN

    if truth is not None:
        tr = truth[t - 1]
        for name, W in (("G", G), ("A", A)):
            prec = np.array([topk_precision(W[i], tr[i], k) for i in range(T)])
            auc_ = np.array([auc(W[i], tr[i]) for i in range(T)])
            for s in range(len(starts)):                         # SQ3: validity
                m = clean & (seg == s)
                out[f"valid/{name}/prec_at_k/seg{s}"] = _nm(prec[m])
                out[f"valid/{name}/auc/seg{s}"] = _nm(auc_[m])
            for n in range(1, len(dps) + 1):                     # SQ2: ranking recovery time
                base, end = tr_end[n - 1], next_tr[n]
                idx = np.where((t > base) & (t < end))[0]         # t > dp: first sample after drift is index dp
                j = first_sustained(prec[idx] >= a["recovery_precision"], a["recovery_consecutive"]) if idx.size else None
                out[f"recovery/{name}/drift{n}"] = NAN if j is None else float(t[idx[j]] - base)
                # censoring: when never recovered, the time until the end of the segment is a lower bound
                out[f"recovered/{name}/drift{n}"] = 0.0 if j is None else 1.0
                out[f"recovery_censored/{name}/drift{n}"] = (
                    float(t[idx[j]] - base) if j is not None else (float(t[idx[-1]] - base) if idx.size else NAN))

        pre_samples = a.get("acc_pre_samples", 5 * every)       # default = first version (5 checkpoints)
        acc_consec = a.get("acc_consecutive", 1)                 # default = first version (no persistence)
        trough_samples = a.get("acc_trough_samples")             # default = first version (whole segment)
        for n in range(1, len(dps) + 1):                         # accuracy recovery (separate from ranking recovery)
            base, end, ts = tr_end[n - 1], next_tr[n], tr_start[n - 1]
            pre = _nm(acc[(t <= ts) & (t > ts - pre_samples)])
            idx = np.where((t > base) & (t < end))[0]
            if idx.size == 0:
                out[f"acc/drop/drift{n}"] = out[f"acc/recovery/drift{n}"] = NAN
                out[f"acc/recovered/drift{n}"] = 0.0
                continue
            post = acc[idx]
            m = idx.size if trough_samples is None else max(1, int(np.sum(t[idx] <= base + trough_samples)))
            jm = int(np.argmin(post[:m]))
            out[f"acc/drop/drift{n}"] = float(pre - post[jm])
            j = first_sustained(post[jm:] >= pre - a["acc_tol"], acc_consec)
            out[f"acc/recovery/drift{n}"] = NAN if j is None else float(t[idx[jm + j]] - base)
            out[f"acc/recovered/drift{n}"] = 0.0 if j is None else 1.0
    return out
