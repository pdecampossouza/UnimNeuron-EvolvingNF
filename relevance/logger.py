"""Checkpoint logger. Only reads the model; call it from the runner, never from the model."""
from __future__ import annotations
import csv, os
import numpy as np
from .extract import RuleTracker, take_snapshot


class RelevanceLogger:
    def __init__(self, d: int):
        self.d = d
        self.tracker = RuleTracker()
        self.rec = {k: [] for k in ("t", "G", "A", "n_rules", "acc_cum", "acc_win",
                                    "proc_time_s", "created", "removed")}
        self.per_rule = []

    def observe(self, model, t: int) -> None:          # call every sample (cheap)
        self.tracker.observe(model, t)

    def checkpoint(self, model, t_count: int, acc_cum: float, acc_win: float, proc_time_s: float) -> None:
        s = take_snapshot(model, self.tracker)
        r = self.rec
        r["t"].append(t_count); r["G"].append(s["G"]); r["A"].append(s["A"])
        r["n_rules"].append(s["n_rules"]); r["acc_cum"].append(acc_cum); r["acc_win"].append(acc_win)
        r["proc_time_s"].append(proc_time_s)
        r["created"].append(self.tracker.n_created); r["removed"].append(self.tracker.n_removed)
        for i, u in enumerate(s["uids"]):
            self.per_rule.append([t_count, u, self.tracker.birth[u], float(s["support"][i]), *s["W"][i].tolist()])

    def arrays(self) -> dict:
        return {k: np.asarray(v) for k, v in self.rec.items()}

    def save(self, outdir: str) -> None:
        os.makedirs(outdir, exist_ok=True)
        np.savez_compressed(os.path.join(outdir, "checkpoints.npz"), **self.arrays())
        with open(os.path.join(outdir, "per_rule.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "uid", "birth_t", "support"] + [f"w{j}" for j in range(self.d)])
            w.writerows(self.per_rule)
