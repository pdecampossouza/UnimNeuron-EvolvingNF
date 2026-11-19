#!/usr/bin/env python
"""Config- and seed-driven prequential run with relevance logging.

    python run_relevance.py --config configs/mve.json --plot
    python run_relevance.py --config configs/stationary_control.json --seeds 0 1

The model is imported untouched. The loop mirrors prequential_nf_generic in exp_sota_comparison.py:
predict (t==0 -> class 0) THEN partial_fit, so no label is used before its prediction.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe
from relevance.streams import build_stream
from relevance.logger import RelevanceLogger
from relevance.extract import MODEL_REFRESH
from relevance.analysis import analyze


def run_one(cfg, seed, log=True, stream=None):
    st = stream if stream is not None else build_stream(cfg["stream"], seed)
    X, y = st.X, st.y
    n, d = X.shape
    every, win = cfg["checkpoint_every"], cfg["acc_window"]
    assert every % MODEL_REFRESH == 0, "weights only refresh every 50 samples inside the model"

    model = EvolvingNeuroFuzzyUnimSafe(n_features=d, n_classes=int(y.max()) + 1, **cfg["model"])
    logger = RelevanceLogger(d) if log else None
    y_pred = np.zeros(n, int); correct = np.zeros(n); n_rules = np.zeros(n, int)
    t_log = 0.0
    t_start = time.perf_counter()
    for t in range(n):
        x_t, yt = X[t:t + 1], int(y[t])
        y_hat = 0 if t == 0 else int(np.asarray(model.predict(x_t)).ravel()[0])   # predict first
        y_pred[t], correct[t] = y_hat, float(y_hat == yt)
        model.partial_fit(x_t, np.array([yt]))                                     # then learn
        n_rules[t] = model.n_rules_
        if log:
            t0 = time.perf_counter()
            logger.observe(model, t)
            if (t + 1) % every == 0:
                logger.checkpoint(model, t + 1, correct[:t + 1].mean(),
                                  correct[max(0, t + 1 - win):t + 1].mean(),
                                  time.perf_counter() - t_start - t_log)
            t_log += time.perf_counter() - t0
    total = time.perf_counter() - t_start
    return dict(stream=st, model=model, logger=logger, y_pred=y_pred, n_rules=n_rules,
                acc_final=float(correct.mean()), proc_time_s=total - t_log, log_time_s=t_log)


def plot_run(arr, st, outpath, every):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, d = arr["t"], arr["G"].shape[1]
    colors = ["gray"] * d
    if st.truth is not None:
        for j in range(d):
            b, a = st.truth[0, j], st.truth[-1, j]
            colors[j] = "purple" if (a and b) else "tab:blue" if b else "tab:red" if a else "lightgray"
    fig, ax = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    for axx, key, title in ((ax[0], "G", "Global weights (current_feature_weights)"),
                            (ax[1], "A", "Rule-base weights (support-weighted mean of rule.w)")):
        for j in range(d):
            axx.plot(t, arr[key][:, j], color=colors[j], lw=1.0, label=f"x{j}" if key == "G" else None)
        axx.set_title(title, fontsize=9); axx.set_ylabel("weight")
    ax[2].plot(t, arr["acc_win"], label="windowed acc"); ax[2].set_ylabel("accuracy")
    r = ax[2].twinx(); r.plot(t, arr["n_rules"], color="k", lw=0.8); r.set_ylabel("rules")
    for a_ in ax:
        for dp in st.drift_points:
            a_.axvline(dp, color="red", ls="--", lw=0.8)
    ax[2].set_xlabel("samples")
    ax[0].legend(ncol=5, fontsize=6)
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)


def aggregate(per_seed):
    keys = sorted({k for m in per_seed for k in m})
    out = {}
    for k in keys:
        v = np.array([m.get(k, np.nan) for m in per_seed], dtype=float)
        out[k] = dict(mean=float(np.nanmean(v)) if not np.all(np.isnan(v)) else None,
                      std=float(np.nanstd(v)) if not np.all(np.isnan(v)) else None,
                      n_valid=int((~np.isnan(v)).sum()), n_seeds=len(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    seeds = args.seeds if args.seeds else cfg["seeds"]
    root = Path(cfg["outdir"]) / cfg["name"]
    per_seed = []
    for s in seeds:
        res = run_one(cfg, s, log=True)
        st, lg = res["stream"], res["logger"]
        arr = lg.arrays()
        m = analyze(arr, st.truth, st.drift_points, cfg["analysis"], cfg["checkpoint_every"])
        m.update(acc_final=res["acc_final"], rules_final=int(res["n_rules"][-1]),
                 proc_time_s=res["proc_time_s"], log_time_s=res["log_time_s"])
        d = root / f"seed_{s}"
        lg.save(str(d))
        json.dump(m, open(d / "metrics.json", "w"), indent=2)
        if args.plot:
            plot_run(arr, st, str(d / "trajectories.png"), cfg["checkpoint_every"])
        per_seed.append(m)
        print(f"seed {s}: acc={m['acc_final']:.4f} rules={m['rules_final']} time={m['proc_time_s']:.1f}s")
    root.mkdir(parents=True, exist_ok=True)
    summ = aggregate(per_seed)
    json.dump(dict(config=cfg, seeds=seeds, summary=summ), open(root / "summary.json", "w"), indent=2)
    for k, v in summ.items():
        if v["mean"] is not None:
            print(f"{k:40s} {v['mean']:9.4f} ± {v['std']:.4f}  (valid {v['n_valid']}/{v['n_seeds']})")


if __name__ == "__main__":
    main()
