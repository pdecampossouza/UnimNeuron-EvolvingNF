
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# ---------------------------------------------------------------------
# Helpers: Gaussian membership + UnimNeuron aggregation
# ---------------------------------------------------------------------

def gaussian_membership(x: np.ndarray, center: np.ndarray, sigma: float) -> np.ndarray:
    """Per-feature Gaussian membership with scalar sigma."""
    x = np.asarray(x, dtype=float)
    center = np.asarray(center, dtype=float)
    sigma_eff = float(max(sigma, 1e-6))
    diff2 = (x - center) ** 2
    return np.exp(-diff2 / (2.0 * sigma_eff**2))


def unim_aggregate(h: np.ndarray, e: float = 0.5) -> float:
    """
    UnimNeuron-style aggregation of feature contributions h_j in [0,1].

    - If all h_j <= e -> AND-like (min)
    - If all h_j >= e -> OR-like  (max)
    - Otherwise       -> COMP-like (mean)
    """
    z = np.asarray(h, dtype=float)
    if z.size == 0:
        return float(e)

    all_le = np.all(z <= e)
    all_ge = np.all(z >= e)

    if all_le and not all_ge:
        return float(np.min(z))
    elif all_ge and not all_le:
        return float(np.max(z))
    else:
        return float(np.mean(z))


# ---------------------------------------------------------------------
# Rule structure
# ---------------------------------------------------------------------

@dataclass
class ENFRuleSafe:
    center: np.ndarray          # (d,)
    sigma: float
    support: int
    w: np.ndarray               # (d,) feature relevance weights in [0,1]
    class_counts: np.ndarray    # (C,) exponentially weighted counts

    def class_probs(self) -> np.ndarray:
        s = float(self.class_counts.sum())
        if s <= 0.0:
            return np.ones_like(self.class_counts) / len(self.class_counts)
        return self.class_counts / s


# ---------------------------------------------------------------------
# Feature separability buffer (Edwin-style inspiration)
# ---------------------------------------------------------------------

class FeatureSeparabilityBuffer:
    """
    Maintains a sliding window of (x,y) and computes simple separability
    weights per feature using a between/within-class variance ratio.
    """

    def __init__(self, n_features: int, buffer_size: int = 200):
        self.n_features = int(n_features)
        self.buffer_size = int(buffer_size)
        self.X: List[np.ndarray] = []
        self.y: List[int] = []

    def add(self, x: np.ndarray, y: int) -> None:
        x = np.asarray(x, dtype=float).ravel()
        self.X.append(x)
        self.y.append(int(y))
        if len(self.X) > self.buffer_size:
            self.X.pop(0)
            self.y.pop(0)

    def compute_feature_weights(self) -> np.ndarray:
        if len(self.X) < 10:
            return np.ones(self.n_features, dtype=float) / float(self.n_features)

        X = np.stack(self.X, axis=0)
        y = np.asarray(self.y, dtype=int)
        classes = np.unique(y)
        if classes.size < 2:
            return np.ones(self.n_features, dtype=float) / float(self.n_features)

        overall_mean = X.mean(axis=0)
        sep = np.zeros(self.n_features, dtype=float)

        for c in classes:
            Xc = X[y == c]
            if Xc.shape[0] < 2:
                continue
            mean_c = Xc.mean(axis=0)
            between = (mean_c - overall_mean) ** 2 * Xc.shape[0]
            within = ((Xc - mean_c) ** 2).sum(axis=0)
            sep += between / (within + 1e-6)

        if np.all(sep <= 0.0):
            return np.ones(self.n_features, dtype=float) / float(self.n_features)

        sep = np.maximum(sep, 0.0)
        sep_norm = sep / sep.max()
        # add a floor so no feature becomes irrelevant
        w = 0.2 + 0.8 * sep_norm   # in [0.2, 1.0]
        w = w / w.max()
        return w


# ---------------------------------------------------------------------
# Similarity between rules (for optional merging)
# ---------------------------------------------------------------------

def rule_similarity(r1: ENFRuleSafe, r2: ENFRuleSafe) -> float:
    """
    Simple Gaussian-like similarity between two rules based on centers/sigmas.
    """
    diff2 = float(np.sum((r1.center - r2.center) ** 2))
    denom = r1.sigma**2 + r2.sigma**2 + 1e-6
    return float(np.exp(-diff2 / denom))


# ---------------------------------------------------------------------
# Main model: ENF with UnimNeuron + safe local update
# ---------------------------------------------------------------------

class EvolvingNeuroFuzzyUnimSafe:
    """
    Evolving neuro-fuzzy classifier with:
    - UnimNeuron-style fuzzy neuron in the antecedent,
    - rule-based class histograms with exponential forgetting,
    - safe local update inspired by Lughofer's evolving TS models,
    - optional rule merging and max-rule constraint.

    API is compatible with prequential_nf_generic:
        - predict(X) -> labels
        - partial_fit(X, y)
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        alpha_add: float = 0.3,
        tau_merge: float = 0.7,
        lambda_sim: float = 0.5,              # kept for API compatibility (not heavily used)
        buffer_size_similarity: int = 200,    # not used explicitly here
        buffer_size_separability: int = 200,
        max_rules: Optional[int] = None,
        e: float = 0.5,
        act_min_update: float = 0.3,
        dist_safe_factor: float = 3.0,
        forget_counts: float = 0.99,
        min_rules_for_merge: int = 5,
        min_support_for_merge: int = 10,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.alpha_add = float(alpha_add)
        self.tau_merge = float(tau_merge)
        self.lambda_sim = float(lambda_sim)
        self.max_rules = max_rules
        self.e = float(e)
        self.act_min_update = float(act_min_update)
        self.dist_safe_factor = float(dist_safe_factor)
        self.forget_counts = float(forget_counts)
        self.min_rules_for_merge = int(min_rules_for_merge)
        self.min_support_for_merge = int(min_support_for_merge)
        self.rng = np.random.default_rng(random_state)

        # ADPA-style global statistics
        self.global_mean: Optional[np.ndarray] = None
        self.global_X2: float = 0.0
        self.K: int = 0  # number of samples

        # Rule base
        self.rules: List[ENFRuleSafe] = []

        # Feature separability buffer (Edwin-inspired)
        self.sep_buffer = FeatureSeparabilityBuffer(
            n_features=self.n_features,
            buffer_size=buffer_size_separability,
        )
        self.current_feature_weights = np.ones(self.n_features, dtype=float) / float(self.n_features)

    # --------------------------------------------------------------
    # Properties and public API
    # --------------------------------------------------------------

    @property
    def n_rules_(self) -> int:
        return len(self.rules)

    # batch API used in experiment 2
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        for xi, yi in zip(X, y):
            self._update_one(xi, int(yi))

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        y_hat = np.zeros(X.shape[0], dtype=int)
        for i, xi in enumerate(X):
            y_hat[i] = self._predict_one(xi)
        return y_hat

    # --------------------------------------------------------------
    # Single-sample update
    # --------------------------------------------------------------

    def _update_one(self, x: np.ndarray, y: int) -> None:
        x = np.asarray(x, dtype=float).ravel()

        # update global stats (ADPA-style)
        if self.K == 0:
            self.global_mean = x.copy()
            self.global_X2 = float(np.sum(x**2))
            self.K = 1
            self._create_rule(x, y, sigma=1.0)
            self.sep_buffer.add(x, y)
            return

        t = self.K + 1
        self.global_mean = (self.global_mean * (t - 1) + x) / t
        self.global_X2 = (self.global_X2 * (t - 1) + float(np.sum(x**2))) / t
        self.K = t

        mean_norm2 = float(np.sum(self.global_mean**2))
        VC = float(np.sqrt(max(2.0 * (self.global_X2 - mean_norm2), 1e-12)))
        SIGMAg = VC / 2.0

        if self.n_rules_ == 0:
            self._create_rule(x, y, sigma=SIGMAg)
            self.sep_buffer.add(x, y)
            return

        # decide whether to create new rule or adapt nearest
        centers = np.stack([r.center for r in self.rules], axis=0)
        dist_centers = np.max(np.abs(centers - self.global_mean), axis=1)
        dist_x = float(np.max(np.abs(x - self.global_mean)))

        if float(np.min(dist_centers)) > dist_x or float(np.max(dist_centers)) < dist_x:
            # new region -> create rule
            self._create_rule(x, y, sigma=SIGMAg)
            self._maybe_merge_last_rule()
        else:
            # adapt nearest rule
            cheb = np.max(np.abs(centers - x), axis=1)
            idx = int(np.argmin(cheb))
            V = float(cheb[idx])
            if V < SIGMAg:
                r = self.rules[idx]
                new_support = r.support + 1
                r.center = (r.center * r.support + x) / float(new_support)
                r.support = new_support
                r.sigma = 0.8 * r.sigma + 0.2 * SIGMAg if r.sigma > 0 else SIGMAg
            else:
                self._create_rule(x, y, sigma=SIGMAg)
                self._maybe_merge_last_rule()

        # safe local update of consequents
        self._safe_update_consequents(x, y)

        # update separability buffer and refresh feature weights from time to time
        self.sep_buffer.add(x, y)
        if self.K % 50 == 0:
            self.current_feature_weights = self.sep_buffer.compute_feature_weights()

    # --------------------------------------------------------------

    def _create_rule(self, x: np.ndarray, y: int, sigma: float) -> None:
        w = self.current_feature_weights.copy()
        class_counts = np.zeros(self.n_classes, dtype=float)
        class_counts[y] = 1.0
        rule = ENFRuleSafe(center=x.copy(), sigma=float(max(sigma, 1e-3)), support=1, w=w, class_counts=class_counts)
        self.rules.append(rule)

        # enforce max_rules by removing least supported rule
        if self.max_rules is not None and self.n_rules_ > self.max_rules:
            supports = np.array([r.support for r in self.rules], dtype=int)
            idx = int(np.argmin(supports))
            del self.rules[idx]

    # --------------------------------------------------------------

    def _maybe_merge_last_rule(self) -> None:
        if self.n_rules_ <= self.min_rules_for_merge:
            return

        idx = self.n_rules_ - 1
        r_new = self.rules[idx]

        # find most similar older rule
        sims = []
        for j in range(self.n_rules_ - 1):
            r_old = self.rules[j]
            sims.append(rule_similarity(r_new, r_old))
        sims = np.asarray(sims, dtype=float)
        j_best = int(np.argmax(sims))
        sim_best = float(sims[j_best])

        if sim_best < self.tau_merge:
            return

        # check support constraints
        r_old = self.rules[j_best]
        if min(r_old.support, r_new.support) < self.min_support_for_merge:
            return

        # merge new into old
        total_support = r_old.support + r_new.support
        if total_support <= 0:
            return

        w_old = r_old.support / float(total_support)
        w_new = r_new.support / float(total_support)

        r_old.center = w_old * r_old.center + w_new * r_new.center
        r_old.sigma = float(w_old * r_old.sigma + w_new * r_new.sigma)
        r_old.w = w_old * r_old.w + w_new * r_new.w
        r_old.class_counts = r_old.class_counts * w_old + r_new.class_counts * w_new
        r_old.support = total_support

        # remove last rule
        self.rules.pop(idx)

    # --------------------------------------------------------------
    # Safe local update of class_counts
    # --------------------------------------------------------------

    def _safe_update_consequents(self, x: np.ndarray, y: int) -> None:
        if self.n_rules_ == 0:
            return

        x = np.asarray(x, dtype=float).ravel()
        activations, dists = self._compute_rule_activations_and_dists(x)
        max_act = float(activations.max())
        if max_act < self.act_min_update:
            return

        for r, act, dist in zip(self.rules, activations, dists):
            if act < 0.5 * self.act_min_update * max_act:
                continue
            if dist > self.dist_safe_factor * r.sigma:
                # point is too far from the rule -> treat as unsafe
                continue

            # exponential forgetting
            r.class_counts *= self.forget_counts
            # weighted increment for observed class
            r.class_counts[y] += act

    # --------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------

    def _compute_rule_activations_and_dists(self, x: np.ndarray):
        x = np.asarray(x, dtype=float).ravel()
        if self.n_rules_ == 0:
            return np.zeros(0), np.zeros(0)

        acts = np.zeros(self.n_rules_, dtype=float)
        dists = np.zeros(self.n_rules_, dtype=float)

        for idx, r in enumerate(self.rules):
            a = gaussian_membership(x, r.center, r.sigma)
            h = r.w * a + (1.0 - r.w) * self.e
            y_r = unim_aggregate(h, e=self.e)
            acts[idx] = y_r
            dists[idx] = float(np.max(np.abs(x - r.center)))

        return acts, dists

    def _predict_one(self, x: np.ndarray) -> int:
        if self.n_rules_ == 0:
            return 0

        acts, _ = self._compute_rule_activations_and_dists(x)
        if np.all(acts <= 0.0):
            return 0

        # normalize activations
        s = float(acts.sum())
        if s > 0.0:
            w_acts = acts / s
        else:
            w_acts = acts

        # aggregate rule-wise class probabilities
        agg = np.zeros(self.n_classes, dtype=float)
        for w, r in zip(w_acts, self.rules):
            agg += w * r.class_probs()

        return int(np.argmax(agg))
