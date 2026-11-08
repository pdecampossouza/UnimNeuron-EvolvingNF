"""
EvolvingNeuroFuzzyAdvanced + UnimNeuron

A simple evolving neuro-fuzzy classifier with:
- Gaussian antecedents (one Gaussian per feature, per rule),
- a Unim-style fuzzy neuron aggregating feature-level memberships
  (automatic AND / OR / COMP behavior),
- online feature-weight estimation using a Lughofer-style separability criterion,
- online rule similarity based on functional overlap over a buffer of samples,
- rule merging driven by a similarity threshold,
- optional hard cap on the number of rules (max_rules).

This implementation is designed to be:
- readable,
- suitable for experimentation in data streams,
- consistent with the Experiment 1 / 2 scripts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable

import numpy as np


# -------------------------------------------------------------------------
# Basic t-/s-/c-operators for UnimNary
# -------------------------------------------------------------------------


def T_min(z: np.ndarray) -> float:
    """T-norm: minimum."""
    if z.size == 0:
        return 0.0
    return float(np.min(z))


def S_max(z: np.ndarray) -> float:
    """S-norm: maximum."""
    if z.size == 0:
        return 0.0
    return float(np.max(z))


def C_mean(z: np.ndarray) -> float:
    """Compensative operator: arithmetic mean."""
    if z.size == 0:
        return 0.0
    return float(np.mean(z))


@dataclass
class UnimNary:
    """
    Unim-style n-ary connective with automatic regime selection.

    Given inputs z in [0,1]^d and a neutral element e in (0,1):

        - If all z_i <= e: AND-like regime (T-norm, here min),
        - If all z_i >= e: OR-like regime (S-norm, here max),
        - Otherwise: COMP regime (compensative, here mean).

    Returns (value, regime, rho), where:
        regime in {"AND", "OR", "COMP"}
        rho in [-1, 1]  (heuristic regime index)
            ~ -1  -> AND-like
            ~  0  -> compensative
            ~ +1  -> OR-like
    """

    e: float = 0.5
    T: Callable[[np.ndarray], float] = T_min
    S: Callable[[np.ndarray], float] = S_max
    C: Callable[[np.ndarray], float] = C_mean

    def aggregate(self, z: np.ndarray) -> Tuple[float, str, float]:
        z = np.asarray(z, dtype=float).ravel()
        if z.size == 0:
            return float(self.e), "COMP", 0.0

        all_le = np.all(z <= self.e)
        all_ge = np.all(z >= self.e)

        if all_le and not all_ge:
            val = self.T(z)
            regime = "AND"
            rho = -1.0
        elif all_ge and not all_le:
            val = self.S(z)
            regime = "OR"
            rho = +1.0
        else:
            val = self.C(z)
            regime = "COMP"
            rho = 0.0

        # clamp for safety
        val = float(max(0.0, min(1.0, val)))
        return val, regime, rho


# -------------------------------------------------------------------------
# Helper: Gaussian memberships per feature
# -------------------------------------------------------------------------


def gaussian_memberships_per_feature(
    x: np.ndarray, center: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """
    Compute per-feature Gaussian memberships:

        a_j = exp( -0.5 * ((x_j - c_j) / sigma_j)^2 )

    using a scalar per-feature sigma_j > 0.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    center = np.asarray(center, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    sigma = np.maximum(sigma, 1e-6)

    diff = (x - center) / sigma
    diff2 = diff * diff
    a = np.exp(-0.5 * diff2)
    return a


# -------------------------------------------------------------------------
# Helper: Neuro-fuzzy rule
# -------------------------------------------------------------------------


@dataclass
class NeuroFuzzyRule:
    """
    Single fuzzy rule with Gaussian antecedents and class statistics.

    Antecedent (UnimNeuron version):
        - Per-feature membership a_j(x) via Gaussian
        - Feature relevance weights w_j in [0,1]
        - Local mixing:
              h_j = w_j * a_j + (1 - w_j) * e
        - Aggregation by UnimNary over h_j, with automatic
          AND / OR / COMP regime selection.

    Consequent:
        - Categorical distribution over classes, estimated from counts.
    """

    center: np.ndarray
    sigma: np.ndarray
    n_features: int
    n_classes: int
    initial_class: int
    e: float = 0.5  # neutral element for UnimNary
    support: int = 1

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float).reshape(-1)
        self.sigma = np.maximum(np.asarray(self.sigma, dtype=float).reshape(-1), 1e-3)
        self.n_features = int(self.n_features)
        self.n_classes = int(self.n_classes)

        self.class_counts = np.zeros(self.n_classes, dtype=float)
        self.class_counts[int(self.initial_class)] += 1.0

        # Unim n-ary connective (same e for all features of this rule)
        self.unim = UnimNary(e=float(self.e))

        # For interpretability
        self.last_regime: Optional[str] = None
        self.last_rho: Optional[float] = None

    # ------------------------------------------------------------------

    def activation(
        self,
        x: np.ndarray,
        feature_weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute rule activation for a single input x (1D vector) using
        the UnimNeuron at feature level.

        If feature_weights is not None, they are used as relevance
        weights w_j in [0,1]. Otherwise, all features are equally
        weighted.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        d = self.n_features

        # Per-feature memberships
        a = gaussian_memberships_per_feature(x, self.center, self.sigma)

        if feature_weights is None:
            w = np.ones(d, dtype=float)
        else:
            w = np.asarray(feature_weights, dtype=float).reshape(-1)
            if w.size != d:
                raise ValueError("feature_weights has wrong dimension")
        # Normalize w into [0,1] if needed
        w = np.clip(w, 0.0, 1.0)

        # Local mixing with neutral e (UnimNeuron-style)
        e = float(self.unim.e)
        h = w * a + (1.0 - w) * e
        h = np.clip(h, 0.0, 1.0)

        # Aggregate with UnimNary
        val, regime, rho = self.unim.aggregate(h)
        self.last_regime = regime
        self.last_rho = rho

        return float(val)

    # ------------------------------------------------------------------

    def update_premise(self, x: np.ndarray, lr_center: float = 0.1) -> None:
        """
        Simple incremental update of the premise parameters:
        - center updated by exponential moving average,
        - sigma updated towards absolute deviation.

        This is deliberately simple; more advanced schemes can be plugged in.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        self.support += 1

        # Update center
        lr = float(max(1.0 / self.support, lr_center))
        self.center = (1.0 - lr) * self.center + lr * x

        # Update sigma as smoothed absolute deviation
        dev = np.abs(x - self.center) + 1e-3
        self.sigma = 0.9 * self.sigma + 0.1 * dev

    # ------------------------------------------------------------------

    def update_consequent(self, y: int) -> None:
        """Update class counts (consequent parameters)."""
        self.class_counts[int(y)] += 1.0

    # ------------------------------------------------------------------

    @property
    def class_probs(self) -> np.ndarray:
        """Return normalized class probabilities for this rule."""
        total = float(self.class_counts.sum())
        if total <= 0.0:
            return np.ones(self.n_classes, dtype=float) / float(self.n_classes)
        return self.class_counts / total


# -------------------------------------------------------------------------
# Helper: Lughofer-style separability buffer
# -------------------------------------------------------------------------


class FeatureSeparabilityBuffer:
    """
    Maintains a sliding window of samples and computes global feature
    weights based on a single-feature separability criterion
    (between/within-class scatter).

    The calculation is inspired by Edwin Lughofer's separability approach,
    but simplified for clarity.
    """

    def __init__(self, n_features: int, n_classes: int, buffer_size: int = 200):
        self.n_features = n_features
        self.n_classes = n_classes
        self.buffer_size = buffer_size

        self.X_buffer: List[np.ndarray] = []
        self.y_buffer: List[int] = []

    # ------------------------------------------------------------------

    def add_sample(self, x: np.ndarray, y: int) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        self.X_buffer.append(x)
        self.y_buffer.append(int(y))

        if len(self.X_buffer) > self.buffer_size:
            self.X_buffer.pop(0)
            self.y_buffer.pop(0)

    # ------------------------------------------------------------------

    def compute_feature_weights(self) -> np.ndarray:
        """
        Compute global feature weights in [0,1]^d based on a between/within
        separability measure per feature. If the buffer is too small or
        has degenerate statistics, returns uniform weights.
        """
        if len(self.X_buffer) < 2:
            return np.ones(self.n_features, dtype=float) / float(self.n_features)

        X = np.stack(self.X_buffer, axis=0)  # (N, d)
        y = np.array(self.y_buffer, dtype=int)
        N, d = X.shape

        sep = np.zeros(d, dtype=float)

        for j in range(d):
            xj = X[:, j]
            mean_all = float(xj.mean())

            within = 0.0
            between = 0.0

            for c in range(self.n_classes):
                idx_c = np.where(y == c)[0]
                if idx_c.size == 0:
                    continue
                xj_c = xj[idx_c]
                n_c = xj_c.size
                if n_c <= 1:
                    var_c = 0.0
                else:
                    var_c = float(xj_c.var(ddof=1))
                within += n_c * var_c
                mean_c = float(xj_c.mean())
                between += n_c * (mean_all - mean_c) ** 2

            if within > 0.0:
                sep[j] = between / within
            else:
                sep[j] = 0.0

        max_sep = float(sep.max())
        if max_sep <= 0.0:
            # fall back to uniform if no discrimination
            w = np.ones(d, dtype=float) / float(d)
        else:
            w = sep / max_sep

        return w


# -------------------------------------------------------------------------
# Main evolving neuro-fuzzy model
# -------------------------------------------------------------------------


class EvolvingNeuroFuzzyAdvanced:
    """
    Evolving neuro-fuzzy classifier with UnimNeuron antecedents.

    Key components:
    - Gaussian rules with UnimNeuron aggregation over features,
    - feature weights from a separability buffer,
    - simple rule similarity based on functional overlap,
    - optional hard cap on the number of rules (max_rules).
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        alpha_add: float = 0.4,
        tau_merge: float = 0.9,
        lambda_sim: float = 0.5,
        buffer_size_similarity: int = 200,
        buffer_size_separability: int = 200,
        max_rules: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        n_features : int
            Number of input features.
        n_classes : int
            Number of classes.
        alpha_add : float, default=0.4
            Threshold on maximum activation to decide whether to create
            a new rule. If max_activation < alpha_add, a new rule is
            created at x.
        tau_merge : float, default=0.9
            Similarity threshold for merging rules.
        lambda_sim : float, default=0.5
            Decay factor for exponential moving average of similarities.
            (Disabled in this simplified version, kept for compatibility.)
        buffer_size_similarity : int, default=200
            Number of samples kept in the buffer to estimate functional
            similarity between rules.
        buffer_size_separability : int, default=200
            Number of samples used in the separability buffer.
        max_rules : int or None, default=None
            Maximum number of rules; if exceeded, the least supported
            rule is pruned.
        random_state : int or None
            Seed for reproducible random initialization.
        """
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)

        self.alpha_add = float(alpha_add)
        self.tau_merge = float(tau_merge)
        self.lambda_sim = float(lambda_sim)
        self.buffer_size_similarity = int(buffer_size_similarity)
        self.max_rules = max_rules

        self.rng = np.random.default_rng(random_state)

        # Rules
        self.rules: List[NeuroFuzzyRule] = []

        # Feature separability buffer
        self.sep_buffer = FeatureSeparabilityBuffer(
            n_features=self.n_features,
            n_classes=self.n_classes,
            buffer_size=buffer_size_separability,
        )

        # Similarity buffer (inputs only; labels not needed here)
        self.sim_X_buffer: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "EvolvingNeuroFuzzyAdvanced":
        """
        Incremental training on a batch of samples.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()

        for xi, yi in zip(X, y):
            self._update_one(xi, int(yi))

        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EvolvingNeuroFuzzyAdvanced":
        """
        Full pass over data (same as partial_fit for this model).
        """
        return self.partial_fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for a batch of samples.
        """
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        y_hat = np.zeros(n_samples, dtype=int)

        if len(self.rules) == 0:
            # no rules -> predict majority class 0
            return y_hat

        # Use current feature weights
        w = self.sep_buffer.compute_feature_weights()

        for i, xi in enumerate(X):
            activations = np.array(
                [rule.activation(xi, feature_weights=w) for rule in self.rules],
                dtype=float,
            )

            if np.all(activations <= 0.0):
                # fallback: majority class across rules
                total_counts = np.sum(
                    [rule.class_counts for rule in self.rules], axis=0
                )
                if total_counts.sum() <= 0.0:
                    y_hat[i] = 0
                else:
                    y_hat[i] = int(np.argmax(total_counts))
                continue

            # Rule-wise class distributions
            class_mat = np.stack(
                [rule.class_probs for rule in self.rules], axis=0
            )  # (n_rules, n_classes)

            # Aggregate by weighted sum
            agg = activations[:, None] * class_mat
            agg_sum = agg.sum(axis=0)
            y_hat[i] = int(np.argmax(agg_sum))

        return y_hat

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_one(self, x: np.ndarray, y: int) -> None:
        """
        Process a single sample (x, y):
        - update separability buffer,
        - update / create rules,
        - update similarity buffer and possibly merge rules.
        """
        x = np.asarray(x, dtype=float).reshape(-1)

        # Update buffers
        self.sep_buffer.add_sample(x, y)
        self._add_similarity_sample(x)

        # Current feature weights
        w = self.sep_buffer.compute_feature_weights()

        # If no rules yet -> create first rule
        if len(self.rules) == 0:
            self._create_rule(x, y)
            return

        # Assign sample to best rule
        best_idx, best_act = self._assign_sample_to_rule(x, w)

        if best_idx is None or best_act < self.alpha_add:
            # Create new rule
            self._create_rule(x, y)
        else:
            # Adapt existing rule
            rule = self.rules[best_idx]
            rule.update_premise(x)
            rule.update_consequent(y)

        # Similarity-based merging and max_rules pruning
        self._maybe_merge_rules(w)
        self._enforce_max_rules()

    # ------------------------------------------------------------------

    def _create_rule(self, x: np.ndarray, y: int) -> None:
        """
        Create a new rule centered at x with initial sigma based on a
        small random spread, and class y.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        # Initialize sigma as small fraction of range or 1.0
        sigma_init = np.full(self.n_features, 0.5, dtype=float)

        # Neutral element e for this rule (could be random or fixed)
        e_rule = 0.5

        rule = NeuroFuzzyRule(
            center=x,
            sigma=sigma_init,
            n_features=self.n_features,
            n_classes=self.n_classes,
            initial_class=y,
            e=e_rule,
            support=1,
        )
        self.rules.append(rule)

    # ------------------------------------------------------------------

    def _add_similarity_sample(self, x: np.ndarray) -> None:
        """
        Add one sample to the similarity buffer.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        self.sim_X_buffer.append(x)
        if len(self.sim_X_buffer) > self.buffer_size_similarity:
            self.sim_X_buffer.pop(0)

    # ------------------------------------------------------------------

    def _assign_sample_to_rule(
        self, x: np.ndarray, feature_weights: Optional[np.ndarray]
    ) -> Tuple[Optional[int], float]:
        """
        Return (idx, activation) of the best matching rule for x,
        or (None, 0.0) if there are no rules.
        """
        if len(self.rules) == 0:
            return None, 0.0

        acts = np.array(
            [
                rule.activation(x, feature_weights=feature_weights)
                for rule in self.rules
            ],
            dtype=float,
        )
        best_idx = int(np.argmax(acts))
        best_act = float(acts[best_idx])
        return best_idx, best_act

    # ------------------------------------------------------------------

    def _compute_similarity_matrix(
        self, feature_weights: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Compute a simple functional similarity matrix between rules
        using Jaccard-like overlap of activations over the similarity
        buffer samples.
        """
        r = len(self.rules)
        if r <= 1 or len(self.sim_X_buffer) == 0:
            return None

        Xb = np.stack(self.sim_X_buffer, axis=0)  # (N, d)
        N = Xb.shape[0]

        # Activations of each rule on buffer samples
        A = np.zeros((r, N), dtype=float)
        for i, rule in enumerate(self.rules):
            for n in range(N):
                A[i, n] = rule.activation(Xb[n], feature_weights=feature_weights)

        S = np.eye(r, dtype=float)
        for i in range(r):
            for j in range(i + 1, r):
                ai = A[i]
                aj = A[j]
                num = np.minimum(ai, aj).sum()
                den = np.maximum(ai, aj).sum()
                if den <= 0.0:
                    sim = 0.0
                else:
                    sim = float(num / den)
                S[i, j] = S[j, i] = sim

        return S

    # ------------------------------------------------------------------

    def _maybe_merge_rules(self, feature_weights: Optional[np.ndarray]) -> None:
        """
        Merge rules whose similarity exceeds tau_merge.
        """
        r = len(self.rules)
        if r <= 1:
            return

        S = self._compute_similarity_matrix(feature_weights)
        if S is None:
            return

        # Find the most similar pair
        np.fill_diagonal(S, -np.inf)
        i, j = divmod(int(np.argmax(S)), S.shape[1])
        max_sim = float(S[i, j])

        if max_sim < self.tau_merge:
            return

        # Merge rule j into rule i (keep i)
        ri = self.rules[i]
        rj = self.rules[j]

        total_support = ri.support + rj.support
        if total_support <= 0:
            return

        wi = ri.support / total_support
        wj = rj.support / total_support

        # Merge centers and sigmas
        ri.center = wi * ri.center + wj * rj.center
        ri.sigma = wi * ri.sigma + wj * rj.sigma

        # Merge class counts and support
        ri.class_counts = ri.class_counts + rj.class_counts
        ri.support = total_support

        # Remove rule j
        del self.rules[j]

    # ------------------------------------------------------------------

    def _enforce_max_rules(self) -> None:
        """
        If max_rules is set and exceeded, remove the rule with the
        smallest support.
        """
        if self.max_rules is None:
            return
        if len(self.rules) <= self.max_rules:
            return

        supports = np.array([r.support for r in self.rules], dtype=float)
        idx = int(np.argmin(supports))
        del self.rules[idx]
