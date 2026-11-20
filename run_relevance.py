#!/usr/bin/env python
"""Config- and seed-driven prequential run with relevance logging.

    python run_relevance.py --config configs/mve.json --plot
    python run_relevance.py --config configs/stationary_control.json --seeds 0 1
    python run_relevance.py --config configs/protocol/s1_gauss_abrupt.json --variants open --plot
    python run_relevance.py --config configs/protocol/s3_gauss_stationary.json --outdir results_wf1

The model is imported untouched. The loop mirrors prequential_nf_generic in exp_sota_comparison.py:
predict (t==0 -> class 0) THEN partial_fit, so no label is used before its prediction.

A config either has one "model" block (output: <outdir>/<name>/seed_<s>/) or a "variants" block with
several named model settings (output: <outdir>/<name>/<variant>/seed_<s>/). Each seed folder holds
checkpoints.npz, checkpoints.csv, per_rule.csv, metrics.json and meta.json.
"""
import argparse, datetime, json, platform, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe
from relevance.streams import build_stream
from relevance.logger import RelevanceLogger
from relevance.extract import MODEL_REFRESH, RuleEventCounter
from relevance.analysis import analyze
from relevance.metrics import macro_f1, macro_f1_from_counts


def run_one(cfg, seed, log=True, stream=None, model_cfg=None):
    """One prequential run. `model_cfg` overrides cfg["model"] (used for config variants)."""
    st = stream if stream is not None else build_stream(cfg["stream"], seed)
    X, y = st.X, st.y
    n, d = X.shape
    every, win = cfg["checkpoint_every"], cfg["acc_window"]
    assert every % MODEL_REFRESH == 0, "weights only refresh every 50 samples inside the model"

    params = dict(cfg["model"] if model_cfg is None else model_cfg)
    C = int(y.max()) + 1
    model = EvolvingNeuroFuzzyUnimSafe(n_features=d, n_classes=C, **params)
    logger = RelevanceLogger(d, st.feature_names) if log else None
    events = RuleEventCounter(model) if (log and cfg.get("count_rule_events", True)) else None
    y_pred = np.zeros(n, int); correct = np.zeros(n); n_rules = np.zeros(n, int)
    tp, fp, fn = np.zeros(C), np.zeros(C), np.zeros(C)
    t_log = 0.0
    t_start = time.perf_counter()
    for t in range(n):
        x_t, yt = X[t:t + 1], int(y[t])
        y_hat = 0 if t == 0 else int(np.asarray(model.predict(x_t)).ravel()[0])   # predict first
        y_pred[t], correct[t] = y_hat, float(y_hat == yt)
        if y_hat == yt:
            tp[yt] += 1
        else:
            fp[y_hat] += 1; fn[yt] += 1
        model.partial_fit(x_t, np.array([yt]))                                     # then learn
        n_rules[t] = model.n_rules_
        if log:
            t0 = time.perf_counter()
            logger.observe(model, t)
            if (t + 1) % every == 0:
                lo = max(0, t + 1 - win)
                logger.checkpoint(model, t + 1, correct[:t + 1].mean(), correct[lo:t + 1].mean(),
                                  time.perf_counter() - t_start - t_log,
                                  f1_cum=macro_f1_from_counts(tp, fp, fn),
                                  f1_win=macro_f1(y[lo:t + 1], y_pred[lo:t + 1], C),
                                  events=events.counts() if events else None)
            t_log += time.perf_counter() - t0
    total = time.perf_counter() - t_start
    return dict(stream=st, model=model, logger=logger, y_pred=y_pred, n_rules=n_rules, y=y,
                acc_final=float(correct.mean()), f1_final=macro_f1_from_counts(tp, fp, fn),
                events=events.counts() if events else None,
                proc_time_s=total - t_log, log_time_s=t_log)


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
    widths = st.drift_widths or [0] * len(st.drift_points)
    for a_ in ax:
        for dp, w in zip(st.drift_points, widths):
            a_.axvline(dp, color="red", ls="--", lw=0.8)
            if w:
                a_.axvspan(dp - w / 2, dp + w / 2, color="red", alpha=0.08)
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


def _git_info():
    def git(*args):
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    try:
        status = git("status", "--porcelain", "--untracked-files=no")
        return dict(commit=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                    uncommitted_changes=bool(status) if status is not None else None)
    except Exception as e:   # git missing: record why instead of failing the run
        return dict(error=repr(e))


def _versions():
    v = dict(python=platform.python_version(), numpy=np.__version__)
    for mod in ("scipy", "matplotlib", "river"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    return v


def run_config(cfg, seeds=None, variants=None, plot=False, outdir=None, config_file=None, quiet=False):
    """Run every (variant, seed) of a config, save per-seed files and a per-variant summary.json."""
    seeds = list(seeds) if seeds else list(cfg["seeds"])
    root = Path(outdir or cfg["outdir"]) / cfg["name"]
    if "variants" in cfg:
        names = list(variants) if variants else list(cfg["variants"])
        todo = [(v, cfg["variants"][v]) for v in names]
    else:
        todo = [(None, cfg["model"])]
    git, versions = _git_info(), _versions()
    summaries = {}
    for vname, mparams in todo:
        vroot = root / vname if vname else root
        per_seed = []
        for s in seeds:
            started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            res = run_one(cfg, s, log=True, model_cfg=mparams)
            st, lg = res["stream"], res["logger"]
            arr = lg.arrays()
            m = analyze(arr, st.truth, st.drift_points, cfg["analysis"], cfg["checkpoint_every"], st.drift_widths)
            m.update(acc_final=res["acc_final"], f1_final=res["f1_final"], rules_final=int(res["n_rules"][-1]),
                     proc_time_s=res["proc_time_s"], log_time_s=res["log_time_s"])
            d = vroot / f"seed_{s}"
            lg.save(str(d))
            json.dump(m, open(d / "metrics.json", "w"), indent=2)
            json.dump(dict(config_name=cfg["name"], config_file=config_file, variant=vname, seed=s,
                           model_params=mparams, stream=st.meta, n=int(len(st.y)), d=int(st.X.shape[1]),
                           drift_points=st.drift_points, drift_widths=st.drift_widths,
                           rule_events_final=res["events"], started_utc=started,
                           proc_time_s=res["proc_time_s"], log_time_s=res["log_time_s"],
                           git=git, versions=versions),
                      open(d / "meta.json", "w"), indent=2)
            if plot:
                plot_run(arr, st, str(d / "trajectories.png"), cfg["checkpoint_every"])
            per_seed.append(m)
            if not quiet:
                tag = f"[{vname}] " if vname else ""
                print(f"{tag}seed {s}: acc={m['acc_final']:.4f} f1={m['f1_final']:.4f} "
                      f"rules={m['rules_final']} time={m['proc_time_s']:.1f}s")
        vroot.mkdir(parents=True, exist_ok=True)
        summ = aggregate(per_seed)
        json.dump(dict(config=cfg, variant=vname, seeds=seeds, summary=summ),
                  open(vroot / "summary.json", "w"), indent=2)
        summaries[vname] = summ
        if not quiet:
            for k, v in summ.items():
                if v["mean"] is not None:
                    print(f"{k:40s} {v['mean']:9.4f} ± {v['std']:.4f}  (valid {v['n_valid']}/{v['n_seeds']})")
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--variants", nargs="*", help="subset of the config's variants (default: all)")
    ap.add_argument("--outdir", help="override the config's outdir (e.g. one folder per workflow)")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    run_config(cfg, seeds=args.seeds, variants=args.variants, plot=args.plot,
               outdir=args.outdir, config_file=args.config)


if __name__ == "__main__":
    main()
