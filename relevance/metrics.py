"""Stability and validity metrics for feature-relevance vectors."""
from __future__ import annotations
import numpy as np
from scipy.stats import kendalltau, rankdata


def tau_b(a, b) -> float:
    """Kendall tau-b (tie-corrected). NaN if either vector is constant (e.g. uniform warm-up weights)."""
    return float(kendalltau(a, b, variant="b")[0])


def topk_idx(w, k):
    return np.argsort(-np.asarray(w), kind="stable")[:k]


def topk_jaccard(a, b, k) -> float:
    A, B = set(topk_idx(a, k).tolist()), set(topk_idx(b, k).tolist())
    return len(A & B) / len(A | B)


def is_constant(w) -> bool:
    return float(np.ptp(np.asarray(w, dtype=float))) == 0.0


def topk_precision(w, truth_vec, k) -> float:
    """Tie-aware: if the k-th boundary falls inside a group of tied weights, the expected precision
    under random tie-breaking is used. Plain argsort would silently pick the lowest indices, which
    can coincide with the true relevant set and fake a perfect score for uninformative (uniform) weights."""
    w = np.asarray(w, dtype=float); tv = np.asarray(truth_vec, dtype=float)
    thr = np.sort(w)[::-1][k - 1]
    above, tied = w > thr, w == thr
    need = k - int(above.sum())
    return float((tv[above].sum() + need * tv[tied].mean()) / k)


def auc(w, truth_vec) -> float:
    pos = np.asarray(truth_vec).astype(bool)
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(w)
    return float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def macro_f1_from_counts(tp, fp, fn) -> float:
    """Macro-F1 over the classes that occurred in y_true or y_pred (same convention as
    sklearn's f1_score(average='macro') on those labels; F1 = 0 for a class never predicted correctly)."""
    tp, fp, fn = (np.asarray(v, dtype=float) for v in (tp, fp, fn))
    den = 2 * tp + fp + fn
    seen = den > 0
    if not seen.any():
        return float("nan")
    return float(np.mean(2 * tp[seen] / den[seen]))


def macro_f1(y_true, y_pred, n_classes=None) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    C = n_classes or int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    tp = np.bincount(y_true[y_true == y_pred], minlength=C)
    fp = np.bincount(y_pred[y_true != y_pred], minlength=C)
    fn = np.bincount(y_true[y_true != y_pred], minlength=C)
    return macro_f1_from_counts(tp, fp, fn)


def first_sustained(cond, consecutive):
    """Index of the first run of `consecutive` True values, or None (right-censored)."""
    cond = np.asarray(cond, dtype=bool)
    for i in range(len(cond) - consecutive + 1):
        if cond[i:i + consecutive].all():
            return i
    return None
