"""Streams with known feature relevance (synthetic) and a leakage-free loader (real).

River's Agrawal/SEA drift changes the decision rule, not necessarily WHICH features
matter, so we use our own generator where the relevant set is explicit.

Stream families (see PROTOCOL.md, section "Streams"):
  relevance_drift : uniform features, y = 1[sum of relevant > |R|/2]  (Minês's original stream)
  gauss_shift     : Gaussian features whose class means differ only on the relevant set
                    (a concept the model can actually learn; see the learnability pilot)
  sea             : SEA concepts; the threshold drifts but the relevant set {x0, x1} never changes
                    (negative control: accuracy can drop, the ranking should not move)
All synthetic streams use fixed, known bounds, so no data-dependent scaling is needed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Stream:
    X: np.ndarray                      # (n, d), already in [0, 1] without future information
    y: np.ndarray                      # (n,) int labels
    truth: Optional[np.ndarray]        # (n, d) binary relevance at each time step, or None (real data)
    drift_points: List[int] = field(default_factory=list)   # sample indices where relevance changes
    feature_names: Optional[List[str]] = None
    drift_widths: List[int] = field(default_factory=list)   # 0 = abrupt; >0 = gradual transition width
    meta: dict = field(default_factory=dict)                  # generator parameters, for meta.json


def _new_concept_mask(n: int, drift_at, width: int, rng) -> np.ndarray:
    """True where the post-drift concept generates the sample.

    Abrupt (width 0): t >= drift_at, and NO random numbers are drawn, so abrupt streams are
    bit-identical to the generators used before gradual drift existed.
    Gradual: P(new concept at t) = 1 / (1 + exp(-4 (t - drift_at) / width)), the same sigmoid as
    river's ConceptDriftStream; the transition is taken as [drift_at - width/2, drift_at + width/2]."""
    t = np.arange(n)
    if drift_at is None or drift_at >= n:
        return np.zeros(n, dtype=bool)
    if not width:
        return t >= drift_at
    p = 1.0 / (1.0 + np.exp(-4.0 * (t - drift_at) / float(width)))
    return rng.random(n) < p


def _truth(n, d, drift_at, rel_before, rel_after) -> np.ndarray:
    truth = np.zeros((n, d), dtype=int)
    truth[:, list(rel_before)] = 1
    if drift_at is not None and drift_at < n:
        truth[drift_at:, :] = 0
        truth[drift_at:, list(rel_after)] = 1
    return truth


def relevance_drift_stream(n=20000, d=10, drift_at=10000, rel_before=(0, 1, 2),
                           rel_after=(5, 6, 7), label_noise=0.0, seed=0, drift_width=0) -> Stream:
    """y = 1[ sum of the relevant features > |R|/2 ]. At `drift_at` the relevant set R switches
    from rel_before to rel_after (abruptly, or gradually if drift_width > 0).
    drift_at=None gives a stationary control stream."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))

    def label(rel):
        rel = list(rel)
        return (X[:, rel].sum(axis=1) > len(rel) / 2.0).astype(int)

    y = label(rel_before)
    truth = _truth(n, d, drift_at, rel_before, rel_after)
    drift_points: List[int] = []
    widths: List[int] = []
    if drift_at is not None and drift_at < n:
        new = _new_concept_mask(n, drift_at, drift_width, rng)
        y = np.where(new, label(rel_after), y)
        drift_points, widths = [int(drift_at)], [int(drift_width)]
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y = np.where(flip, 1 - y, y)
    return Stream(X=X, y=y.astype(int), truth=truth, drift_points=drift_points,
                  feature_names=[f"x{j}" for j in range(d)], drift_widths=widths,
                  meta=dict(family="relevance_drift", n=n, d=d, drift_at=drift_at, drift_width=drift_width,
                            rel_before=list(rel_before), rel_after=list(rel_after),
                            label_noise=label_noise, seed=seed))


def gauss_shift_stream(n=20000, d=10, drift_at=10000, rel_before=(0, 1, 2), rel_after=(5, 6, 7),
                       delta=0.3, sd=0.12, drift_width=0, label_noise=0.0, seed=0) -> Stream:
    """Balanced binary classes. Every feature is N(0.5, sd); on the features that are relevant for the
    active concept the class means are 0.5 - delta/2 (class 0) and 0.5 + delta/2 (class 1).
    Values are clipped to [0, 1] (fixed bounds, no data-dependent scaling).
    The relevant set switches from rel_before to rel_after at drift_at."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(0.5, sd, (n, d))
    new = _new_concept_mask(n, drift_at, drift_width, rng)
    shift = np.where(y == 1, delta / 2.0, -delta / 2.0)
    for mask, rel in ((~new, rel_before), (new, rel_after)):
        idx = np.where(mask)[0]
        if idx.size:
            X[np.ix_(idx, list(rel))] += shift[idx, None]
    X = np.clip(X, 0.0, 1.0)
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y = np.where(flip, 1 - y, y)
    has_drift = drift_at is not None and drift_at < n
    return Stream(X=X, y=y.astype(int), truth=_truth(n, d, drift_at, rel_before, rel_after),
                  drift_points=[int(drift_at)] if has_drift else [],
                  feature_names=[f"x{j}" for j in range(d)],
                  drift_widths=[int(drift_width)] if has_drift else [],
                  meta=dict(family="gauss_shift", n=n, d=d, drift_at=drift_at, drift_width=drift_width,
                            rel_before=list(rel_before), rel_after=list(rel_after), delta=delta, sd=sd,
                            label_noise=label_noise, seed=seed))


SEA_THRESHOLDS = {0: 8.0, 1: 9.0, 2: 7.0, 3: 9.5}   # same variants as river.datasets.synth.SEA


def sea_stream(n=20000, drift_at=10000, variant_before=0, variant_after=2, drift_width=0,
               label_noise=0.0, seed=0) -> Stream:
    """SEA concepts: 3 features U[0, 10], y = 1[x0 + x1 > threshold]; x2 is never relevant.
    The drift changes the threshold, NOT the relevant set, so the ground truth is {x0, x1} throughout.
    Features are divided by 10 (fixed bounds) so they lie in [0, 1]."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 10.0, (n, 3))
    new = _new_concept_mask(n, drift_at, drift_width, rng)
    thr = np.where(new, SEA_THRESHOLDS[variant_after], SEA_THRESHOLDS[variant_before])
    y = (X[:, 0] + X[:, 1] > thr).astype(int)
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y = np.where(flip, 1 - y, y)
    truth = np.zeros((n, 3), dtype=int)
    truth[:, [0, 1]] = 1
    has_drift = drift_at is not None and drift_at < n
    return Stream(X=X / 10.0, y=y, truth=truth,
                  drift_points=[int(drift_at)] if has_drift else [],
                  feature_names=["x0", "x1", "x2"],
                  drift_widths=[int(drift_width)] if has_drift else [],
                  meta=dict(family="sea", n=n, drift_at=drift_at, drift_width=drift_width,
                            variant_before=variant_before, variant_after=variant_after,
                            label_noise=label_noise, seed=seed))


def expanding_minmax(X: np.ndarray) -> np.ndarray:
    """Min-max scaling using only samples seen up to and including t (no future leakage).
    The baseline's min_max_scale uses the whole stream, which leaks future information."""
    lo = np.minimum.accumulate(X, axis=0)
    hi = np.maximum.accumulate(X, axis=0)
    den = hi - lo
    den[den == 0.0] = 1.0
    return (X - lo) / den


def stream_from_river(ds, n_max: int, drop=None, name: str = "river") -> Stream:
    """Vectorise a river dataset, drop the named features, then scale with expanding min-max."""
    drop = set(drop or [])
    it = iter(ds)
    x0, y0 = next(it)
    names = [f for f in x0.keys() if f not in drop]
    rows, labels = [[x0[f] for f in names]], [y0]
    for i, (x, y) in enumerate(it, start=1):
        if i >= n_max:
            break
        rows.append([x[f] for f in names]); labels.append(y)
    uniq = list(dict.fromkeys(labels))
    y_int = np.array([uniq.index(v) for v in labels], dtype=int)
    X = expanding_minmax(np.asarray(rows, dtype=float))
    return Stream(X=X, y=y_int, truth=None, drift_points=[], feature_names=names,
                  meta=dict(family="river", name=name, n=n_max, dropped=sorted(drop), scaling="expanding_minmax"))


def load_river_stream(name: str, n_max: int, drop=None) -> Stream:
    from river import datasets  # lazy import
    return stream_from_river(getattr(datasets, name)(), n_max, drop=drop, name=name)


def shuffle_control(st: Stream, seed: int) -> Stream:
    """Shuffled copy of a stream: destroys temporal structure, gives a noise floor."""
    p = np.random.default_rng(seed).permutation(len(st.y))
    return Stream(X=st.X[p], y=st.y[p], truth=None, drift_points=[], feature_names=st.feature_names,
                  meta=dict(st.meta, shuffled=True, shuffle_seed=seed))


def build_stream(scfg: dict, seed: int) -> Stream:
    if scfg["type"] == "relevance_drift":
        return relevance_drift_stream(
            n=scfg["n"], d=scfg["d"], drift_at=scfg.get("drift_at"),
            rel_before=tuple(scfg["rel_before"]), rel_after=tuple(scfg["rel_after"]),
            label_noise=scfg.get("label_noise", 0.0), seed=seed, drift_width=scfg.get("drift_width", 0))
    if scfg["type"] == "gauss_shift":
        return gauss_shift_stream(
            n=scfg["n"], d=scfg["d"], drift_at=scfg.get("drift_at"),
            rel_before=tuple(scfg["rel_before"]), rel_after=tuple(scfg["rel_after"]),
            delta=scfg["delta"], sd=scfg.get("sd", 0.12), drift_width=scfg.get("drift_width", 0),
            label_noise=scfg.get("label_noise", 0.0), seed=seed)
    if scfg["type"] == "sea":
        return sea_stream(
            n=scfg["n"], drift_at=scfg.get("drift_at"), variant_before=scfg.get("variant_before", 0),
            variant_after=scfg.get("variant_after", 2), drift_width=scfg.get("drift_width", 0),
            label_noise=scfg.get("label_noise", 0.0), seed=seed)
    if scfg["type"] == "river":
        st = load_river_stream(scfg["name"], scfg["n"], drop=scfg.get("drop"))
        return shuffle_control(st, seed) if scfg.get("shuffle") else st
    raise ValueError(f"unknown stream type {scfg['type']}")
