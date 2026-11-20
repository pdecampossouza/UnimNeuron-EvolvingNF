"""Checkpoint logger. Only reads the model; call it from the runner, never from the model."""
from __future__ import annotations
import csv, os
import numpy as np
from .extract import RuleTracker, take_snapshot

EVENT_KEYS = ("ev_created", "ev_pruned", "ev_newborn_pruned", "ev_merged")
SCALAR_KEYS = ("t", "acc_cum", "acc_win", "f1_cum", "f1_win", "n_rules", "created", "removed",
               *EVENT_KEYS, "proc_time_s")


class RelevanceLogger:
    def __init__(self, d: int, feature_names=None):
        self.d = d
        self.feature_names = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(d)]
        self.tracker = RuleTracker()
        self.rec = {k: [] for k in ("t", "G", "A", "n_rules", "acc_cum", "acc_win",
                                    "proc_time_s", "created", "removed",
                                    "f1_cum", "f1_win", *EVENT_KEYS)}
        self.per_rule = []

    def observe(self, model, t: int) -> None:          # call every sample (cheap)
        self.tracker.observe(model, t)

    def checkpoint(self, model, t_count: int, acc_cum: float, acc_win: float, proc_time_s: float,
                   f1_cum: float = float("nan"), f1_win: float = float("nan"), events: dict = None) -> None:
        s = take_snapshot(model, self.tracker)
        r = self.rec
        r["t"].append(t_count); r["G"].append(s["G"]); r["A"].append(s["A"])
        r["n_rules"].append(s["n_rules"]); r["acc_cum"].append(acc_cum); r["acc_win"].append(acc_win)
        r["proc_time_s"].append(proc_time_s)
        r["created"].append(self.tracker.n_created); r["removed"].append(self.tracker.n_removed)
        r["f1_cum"].append(f1_cum); r["f1_win"].append(f1_win)
        events = events or {}
        for k in EVENT_KEYS:
            r[k].append(events.get(k, -1))       # -1 = not counted in this run
        for i, u in enumerate(s["uids"]):
            self.per_rule.append([t_count, u, self.tracker.birth[u], float(s["support"][i]), *s["W"][i].tolist()])

    def arrays(self) -> dict:
        return {k: np.asarray(v) for k, v in self.rec.items()}

    def save(self, outdir: str) -> None:
        os.makedirs(outdir, exist_ok=True)
        arr = self.arrays()
        np.savez_compressed(os.path.join(outdir, "checkpoints.npz"), **arr)
        with open(os.path.join(outdir, "per_rule.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "uid", "birth_t", "support"] + [f"w{j}" for j in range(self.d)])
            w.writerows(self.per_rule)
        # flat, spreadsheet-friendly copy of every checkpoint (same numbers as checkpoints.npz)
        with open(os.path.join(outdir, "checkpoints.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(SCALAR_KEYS) + [f"G_{n}" for n in self.feature_names] + [f"A_{n}" for n in self.feature_names])
            for i in range(len(arr["t"])):
                w.writerow([arr[k][i].item() for k in SCALAR_KEYS] + arr["G"][i].tolist() + arr["A"][i].tolist())
