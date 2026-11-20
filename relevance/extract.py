"""READ-ONLY extraction of feature relevance from EvolvingNeuroFuzzyUnimSafe.
Nothing here writes to the model.

Three relevance views are available in the model:
  G : model.current_feature_weights  (global; recomputed from the last 200 samples every 50 samples)
  A : support-weighted mean of rule.w (what the rule base actually uses at prediction time)
  r.w per rule : COPY of G at the moment the rule was created; never refreshed afterwards
                 (only changed by merging, which never fires when max_rules == min_rules_for_merge == 5).
"""
from __future__ import annotations
import numpy as np

MODEL_REFRESH = 50   # `if self.K % 50 == 0` in ENF_UnimNeuron_Safe._update_one


class RuleTracker:
    """Rules have no IDs in the model (plain list, indices shift on deletion). We assign IDs by object
    identity. References are kept so Python can never reuse an id() for a different rule."""

    def __init__(self):
        self._uid, self._keep, self._alive = {}, [], set()
        self._next = 0
        self.birth = {}
        self.n_created = 0
        self.n_removed = 0

    def observe(self, model, t: int) -> None:
        cur = {id(r): r for r in model.rules}
        for k, r in cur.items():
            if k not in self._uid:
                self._uid[k] = self._next
                self.birth[self._next] = int(t)
                self._keep.append(r)
                self._next += 1
                self.n_created += 1
        self.n_removed += len(self._alive - cur.keys())
        self._alive = set(cur.keys())

    def uid(self, rule) -> int:
        return self._uid[id(rule)]


class RuleEventCounter:
    """Exact counts of rule-base events that happen INSIDE partial_fit.

    RuleTracker only sees the rule list between samples, so a rule that is created and pruned within
    the same partial_fit call (the usual case once max_rules is reached) is invisible to it.
    This counter wraps `_create_rule` and `_maybe_merge_last_rule` on this model instance only:
    each wrapper calls the original method with the same arguments and returns its result, so the
    model's behaviour is unchanged. tests/test_nonintrusive.py checks this on both configs.
    """

    def __init__(self, model):
        self.created = 0          # calls to _create_rule (every rule birth)
        self.pruned = 0           # rules removed by the max_rules constraint
        self.newborn_pruned = 0   # ... of which the rule that had just been created
        self.merged = 0           # rules removed by merging
        create, merge = model._create_rule, model._maybe_merge_last_rule

        def _create_rule(*args, **kwargs):
            before = {id(r) for r in model.rules}
            n0 = len(model.rules)
            out = create(*args, **kwargs)
            removed = n0 + 1 - len(model.rules)
            self.created += 1
            self.pruned += removed
            if removed and all(id(r) in before for r in model.rules):
                self.newborn_pruned += 1
            return out

        def _maybe_merge_last_rule(*args, **kwargs):
            n0 = len(model.rules)
            out = merge(*args, **kwargs)
            self.merged += n0 - len(model.rules)
            return out

        model._create_rule = _create_rule
        model._maybe_merge_last_rule = _maybe_merge_last_rule

    def counts(self) -> dict:
        return dict(ev_created=self.created, ev_pruned=self.pruned,
                    ev_newborn_pruned=self.newborn_pruned, ev_merged=self.merged)


def take_snapshot(model, tracker: RuleTracker) -> dict:
    g = np.array(model.current_feature_weights, dtype=float, copy=True)
    rules = list(model.rules)
    if rules:
        sup = np.array([r.support for r in rules], dtype=float)
        W = np.stack([np.asarray(r.w, dtype=float) for r in rules])
        agg = (sup[:, None] * W).sum(axis=0) / sup.sum()
        uids = [tracker.uid(r) for r in rules]
    else:
        sup, W, agg, uids = np.zeros(0), np.zeros((0, g.size)), g.copy(), []
    return dict(G=g, A=agg, support=sup, W=W, uids=uids, n_rules=len(rules))
