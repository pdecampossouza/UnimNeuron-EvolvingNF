"""Turns one run's checkpoints into the metrics of SQ1-SQ3 (+ structural). Pure function, no model access."""
from __future__ import annotations
import numpy as np
from .metrics import tau_b, topk_jaccard, topk_precision, auc, first_sustained, is_constant

NAN = float("nan")


def _nm(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if x.size and not np.all(np.isnan(x)) else NAN


def analyze(arr: dict, truth, drift_points, a: dict, every: int) -> dict:
    t, G, A, acc = arr["t"], arr["G"], arr["A"], arr["acc_win"]
    T, k, flush = len(t), a["topk"], a["warmup_flush"]
    starts = np.array([0] + list(drift_points))
    seg = np.searchsorted(starts, t - 1, side="right") - 1      # segment of the newest sample seen
    clean = (t - starts[seg]) >= flush                          # buffer holds only this segment's data
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
            for n, dp in enumerate(drift_points, 1):             # SQ2: ranking recovery time
                end = drift_points[n] if n < len(drift_points) else np.inf
                idx = np.where((t > dp) & (t < end))[0]           # t > dp: first sample after drift is index dp
                j = first_sustained(prec[idx] >= a["recovery_precision"], a["recovery_consecutive"])
                out[f"recovery/{name}/drift{n}"] = NAN if j is None else float(t[idx[j]] - dp)

        for n, dp in enumerate(drift_points, 1):                 # accuracy recovery (separate from ranking recovery)
            end = drift_points[n] if n < len(drift_points) else np.inf
            pre = _nm(acc[(t <= dp) & (t > dp - 5 * every)])
            idx = np.where((t > dp) & (t < end))[0]
            post = acc[idx]
            jm = int(np.argmin(post))
            out[f"acc/drop/drift{n}"] = float(pre - post[jm])
            back = np.where(post[jm:] >= pre - a["acc_tol"])[0]
            out[f"acc/recovery/drift{n}"] = NAN if back.size == 0 else float(t[idx[jm + back[0]]] - dp)
    return out
