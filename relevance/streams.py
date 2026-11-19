"""Streams with known feature relevance (synthetic) and a leakage-free loader (real).

River's Agrawal/SEA drift changes the decision rule, not necessarily WHICH features
matter, so we use our own generator where the relevant set is explicit.
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


def relevance_drift_stream(n=20000, d=10, drift_at=10000, rel_before=(0, 1, 2),
                           rel_after=(5, 6, 7), label_noise=0.0, seed=0) -> Stream:
    """y = 1[ sum of the relevant features > |R|/2 ]. At `drift_at` the relevant set R switches
    abruptly from rel_before to rel_after. drift_at=None gives a stationary control stream."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))

    def label(rel):
        rel = list(rel)
        return (X[:, rel].sum(axis=1) > len(rel) / 2.0).astype(int)

    y = label(rel_before)
    truth = np.zeros((n, d), dtype=int)
    truth[:, list(rel_before)] = 1
    drift_points: List[int] = []
    if drift_at is not None and drift_at < n:
        y = np.where(np.arange(n) < drift_at, y, label(rel_after))
        truth[drift_at:, :] = 0
        truth[drift_at:, list(rel_after)] = 1
        drift_points = [int(drift_at)]
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y = np.where(flip, 1 - y, y)
    return Stream(X=X, y=y.astype(int), truth=truth, drift_points=drift_points,
                  feature_names=[f"x{j}" for j in range(d)])


def expanding_minmax(X: np.ndarray) -> np.ndarray:
    """Min-max scaling using only samples seen up to and including t (no future leakage).
    The baseline's min_max_scale uses the whole stream, which leaks future information."""
    lo = np.minimum.accumulate(X, axis=0)
    hi = np.maximum.accumulate(X, axis=0)
    den = hi - lo
    den[den == 0.0] = 1.0
    return (X - lo) / den


def load_river_stream(name: str, n_max: int) -> Stream:
    from river import datasets  # lazy import
    ds = getattr(datasets, name)()
    it = iter(ds)
    x0, y0 = next(it)
    names = list(x0.keys())
    rows, labels = [[x0[f] for f in names]], [y0]
    for i, (x, y) in enumerate(it, start=1):
        if i >= n_max:
            break
        rows.append([x[f] for f in names]); labels.append(y)
    uniq = list(dict.fromkeys(labels))
    y_int = np.array([uniq.index(v) for v in labels], dtype=int)
    X = expanding_minmax(np.asarray(rows, dtype=float))
    return Stream(X=X, y=y_int, truth=None, drift_points=[], feature_names=names)


def shuffle_control(st: Stream, seed: int) -> Stream:
    """Shuffled copy of a stream: destroys temporal structure, gives a noise floor."""
    p = np.random.default_rng(seed).permutation(len(st.y))
    return Stream(X=st.X[p], y=st.y[p], truth=None, drift_points=[], feature_names=st.feature_names)


def build_stream(scfg: dict, seed: int) -> Stream:
    if scfg["type"] == "relevance_drift":
        return relevance_drift_stream(
            n=scfg["n"], d=scfg["d"], drift_at=scfg.get("drift_at"),
            rel_before=tuple(scfg["rel_before"]), rel_after=tuple(scfg["rel_after"]),
            label_noise=scfg.get("label_noise", 0.0), seed=seed)
    if scfg["type"] == "river":
        st = load_river_stream(scfg["name"], scfg["n"])
        return shuffle_control(st, seed) if scfg.get("shuffle") else st
    raise ValueError(f"unknown stream type {scfg['type']}")
