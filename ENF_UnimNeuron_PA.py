
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# ---------------------------------------------------------------------
# Helpers: Gaussian membership + UnimNeuron aggregation
# ---------------------------------------------------------------------

def gaussian_membership(x: np.ndarray, center: np.ndarray, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    center = np.asarray(center, dtype=float)
    sigma_eff = float(max(sigma, 1e-6))
    diff2 = (x - center) ** 2
    return np.exp(-diff2 / (2.0 * sigma_eff**2))


def unim_aggregate(h: np.ndarray, e: float = 0.5) -> float:
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
class ENFRulePA:
    center: np.ndarray
    sigma: float
    support: int
    w: np.ndarray


# ---------------------------------------------------------------------
# Main model: ENF with UnimNeuron + Passive-Aggressive consequent
# ---------------------------------------------------------------------

class EvolvingNeuroFuzzyUnimPA:
    """
    Evolving neuro-fuzzy classifier with:
    - UnimNeuron antecedents,
    - Passive–Aggressive (PA) linear classifier on rule activations,
    - simple safe update based on activation and distance.

    This is intentionally simpler than the Safe variant; it focuses on
    a margin-based consequent learner instead of class histograms.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        alpha_add: float = 0.3,
        tau_merge: float = 0.7,
        lambda_sim: float = 0.5,
        buffer_size_similarity: int = 200,
        buffer_size_separability: int = 200,
        max_rules: Optional[int] = None,
        e: float = 0.5,
        act_min_update: float = 0.3,
        dist_safe_factor: float = 3.0,
        C: float = 1.0,
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
        self.C = float(C)
        self.rng = np.random.default_rng(random_state)

        # global stats for ADPA-like radius
        self.global_mean: Optional[np.ndarray] = None
        self.global_X2: float = 0.0
        self.K: int = 0

        # rules
        self.rules: List[ENFRulePA] = []

        # PA weight matrix W (M x C), M = 1 + n_rules
        self.W: Optional[np.ndarray] = None

    # --------------------------------------------------------------

    @property
    def n_rules_(self) -> int:
        return len(self.rules)

    # batch API
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

    def _ensure_W_dim(self):
        M_needed = self.n_rules_ + 1  # bias + rules
        if self.W is None:
            self.W = np.zeros((M_needed, self.n_classes), dtype=float)
            return
        M_current = self.W.shape[0]
        if M_current == M_needed:
            return
        if M_needed > M_current:
            W_new = np.zeros((M_needed, self.n_classes), dtype=float)
            W_new[:M_current, :] = self.W
            self.W = W_new
        else:
            self.W = self.W[:M_needed, :]

    # --------------------------------------------------------------

    def _update_one(self, x: np.ndarray, y: int) -> None:
        x = np.asarray(x, dtype=float).ravel()

        # ADPA-like radius
        if self.K == 0:
            self.global_mean = x.copy()
            self.global_X2 = float(np.sum(x**2))
            self.K = 1
            self._create_rule(x, sigma=1.0)
            self._ensure_W_dim()
            return

        t = self.K + 1
        self.global_mean = (self.global_mean * (t - 1) + x) / t
        self.global_X2 = (self.global_X2 * (t - 1) + float(np.sum(x**2))) / t
        self.K = t

        mean_norm2 = float(np.sum(self.global_mean**2))
        VC = float(np.sqrt(max(2.0 * (self.global_X2 - mean_norm2), 1e-12)))
        SIGMAg = VC / 2.0

        if self.n_rules_ == 0:
            self._create_rule(x, sigma=SIGMAg)
            self._ensure_W_dim()
            return

        centers = np.stack([r.center for r in self.rules], axis=0)
        dist_centers = np.max(np.abs(centers - self.global_mean), axis=1)
        dist_x = float(np.max(np.abs(x - self.global_mean)))

        if float(np.min(dist_centers)) > dist_x or float(np.max(dist_centers)) < dist_x:
            self._create_rule(x, sigma=SIGMAg)
            self._ensure_W_dim()
        else:
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
                self._create_rule(x, sigma=SIGMAg)
                self._ensure_W_dim()

        # PA update on consequent with safe gating
        self._pa_update(x, y)

    # --------------------------------------------------------------

    def _create_rule(self, x: np.ndarray, sigma: float) -> None:
        w = np.ones(self.n_features, dtype=float)
        rule = ENFRulePA(center=x.copy(), sigma=float(max(sigma, 1e-3)), support=1, w=w)
        self.rules.append(rule)

        # enforce max_rules
        if self.max_rules is not None and self.n_rules_ > self.max_rules:
            # remove least supported rule
            supports = np.array([r.support for r in self.rules], dtype=int)
            idx = int(np.argmin(supports))
            del self.rules[idx]

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

    # --------------------------------------------------------------

    def _phi(self, x: np.ndarray) -> np.ndarray:
        acts, _ = self._compute_rule_activations_and_dists(x)
        M = self.n_rules_ + 1
        phi = np.zeros(M, dtype=float)
        phi[0] = 1.0
        if acts.size > 0:
            s = float(acts.sum())
            if s > 0.0:
                phi[1:] = acts / s
            else:
                phi[1:] = acts
        return phi

    # --------------------------------------------------------------

    def _pa_update(self, x: np.ndarray, y: int) -> None:
        if self.W is None or self.n_rules_ == 0:
            self._ensure_W_dim()
            return

        acts, dists = self._compute_rule_activations_and_dists(x)
        if acts.size == 0:
            return

        max_act = float(acts.max())
        if max_act < self.act_min_update:
            return

        # simple distance gating: require at least one rule reasonably close
        min_dist = float(dists.min())
        if min_dist > self.dist_safe_factor * max(r.sigma for r in self.rules):
            return

        phi = self._phi(x)  # (M,)
        scores = phi @ self.W  # (C,)
        y_hat = int(np.argmax(scores))

        if y_hat == y:
            # Check margin; if already good, no update
            sorted_scores = np.sort(scores)
            if self.n_classes >= 2:
                margin = sorted_scores[-1] - sorted_scores[-2]
                if margin >= 1.0:
                    return

        # PA update (multiclass, one-vs-wrong-class)
        # Find most competitive incorrect class
        scores_y = scores[y]
        scores_others = scores.copy()
        scores_others[y] = -np.inf
        j = int(np.argmax(scores_others))
        loss = max(0.0, 1.0 - (scores_y - scores_others[j]))
        if loss <= 0.0:
            return

        tau = min(self.C, loss / (2.0 * np.dot(phi, phi) + 1e-12))

        # Update: push y up, j down
        self.W[:, y] += tau * phi
        self.W[:, j] -= tau * phi

    # --------------------------------------------------------------

    def _predict_one(self, x: np.ndarray) -> int:
        if self.W is None or self.n_rules_ == 0:
            return 0
        phi = self._phi(x)
        scores = phi @ self.W
        return int(np.argmax(scores))
