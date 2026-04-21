"""Plot the recomp-ratio sweep results.

Reads the sweep JSON (default ``standard_qa/inputs/hotpotqa_s_recomp_sweep_scores.json``)
and emits two PNGs:

* ``<stem>_recomp_sweep_ttft.png``      — TTFT vs recomp_ratio (3 curves)
* ``<stem>_recomp_sweep_accuracy.png``  — metric vs recomp_ratio (3 curves)

Usage::

    python analysis/plot_recomp_sweep.py \
        [--input  standard_qa/inputs/hotpotqa_s_recomp_sweep_scores.json] \
        [--outdir standard_qa/inputs]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _plot(payload: dict, outdir: Path, stem: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric = payload["metric_name"]
    per_ratio = payload["per_ratio"]
    xs_single = [r["recomp_ratio_single"] for r in per_ratio]
    xs_pair = [r["recomp_ratio_pair"] for r in per_ratio]

    ttft_full_mean = payload["mean_ttft_full"]
    score_full_mean = payload[f"mean_{metric}_full"]
    ttft_s = [r["mean_ttft_single"] for r in per_ratio]
    ttft_p = [r["mean_ttft_pair"] for r in per_ratio]
    sc_s = [r[f"mean_{metric}_single"] for r in per_ratio]
    sc_p = [r[f"mean_{metric}_pair"] for r in per_ratio]

    C_FULL = "#555555"
    C_SING = "#1f77b4"
    C_PAIR = "#d62728"

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=140)
    ax.axhline(ttft_full_mean, linestyle="--", color=C_FULL,
               label=f"Full Prefill  ({ttft_full_mean*1000:.1f} ms)")
    ax.plot(xs_single, ttft_s, marker="o", color=C_SING,
            label=f"CompCache (single-chunk) @ r")
    ax.plot(xs_single, ttft_p, marker="s", color=C_PAIR,
            label="CompCache (+Pairs) @ r/2")
    ax.set_xlabel("recomp_ratio (single-chunk method)")
    ax.set_ylabel("mean TTFT (s)")
    ax.set_title(f"TTFT vs recomp_ratio — HotpotQA ({payload['n_queries']} queries)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    for x, y in zip(xs_single, ttft_s):
        ax.annotate(f"{y*1000:.0f}", (x, y), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=7, color=C_SING)
    for x, y in zip(xs_single, ttft_p):
        ax.annotate(f"{y*1000:.0f}", (x, y), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=7, color=C_PAIR)
    fig.tight_layout()
    ttft_png = outdir / f"{stem}_recomp_sweep_ttft.png"
    fig.savefig(ttft_png)
    plt.close(fig)
    print(f"  wrote {ttft_png}")

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=140)
    ax.axhline(score_full_mean, linestyle="--", color=C_FULL,
               label=f"Full Prefill  ({score_full_mean:.3f})")
    ax.plot(xs_single, sc_s, marker="o", color=C_SING,
            label="CompCache (single-chunk) @ r")
    ax.plot(xs_single, sc_p, marker="s", color=C_PAIR,
            label="CompCache (+Pairs) @ r/2")
    ax.set_xlabel("recomp_ratio (single-chunk method)")
    ax.set_ylabel(f"mean {metric}")
    ax.set_title(
        f"{metric} vs recomp_ratio — HotpotQA ({payload['n_queries']} queries)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    for x, y in zip(xs_single, sc_s):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=7, color=C_SING)
    for x, y in zip(xs_single, sc_p):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=7, color=C_PAIR)
    fig.tight_layout()
    acc_png = outdir / f"{stem}_recomp_sweep_accuracy.png"
    fig.savefig(acc_png)
    plt.close(fig)
    print(f"  wrote {acc_png}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=140)
    ax = axes[0]
    ax.axhline(ttft_full_mean, linestyle="--", color=C_FULL, label="Full")
    ax.plot(xs_single, ttft_s, marker="o", color=C_SING, label="single @ r")
    ax.plot(xs_single, ttft_p, marker="s", color=C_PAIR, label="+Pairs @ r/2")
    ax.set_xlabel("recomp_ratio")
    ax.set_ylabel("mean TTFT (s)")
    ax.set_title("TTFT")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    ax.axhline(score_full_mean, linestyle="--", color=C_FULL, label="Full")
    ax.plot(xs_single, sc_s, marker="o", color=C_SING, label="single @ r")
    ax.plot(xs_single, sc_p, marker="s", color=C_PAIR, label="+Pairs @ r/2")
    ax.set_xlabel("recomp_ratio")
    ax.set_ylabel(f"mean {metric}")
    ax.set_title(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"HotpotQA recomp-ratio sweep  (n={payload['n_queries']}, pair ratio = single/2)",
        fontsize=11,
    )
    fig.tight_layout()
    combo_png = outdir / f"{stem}_recomp_sweep_combined.png"
    fig.savefig(combo_png)
    plt.close(fig)
    print(f"  wrote {combo_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="standard_qa/inputs/hotpotqa_s_recomp_sweep_scores.json",
    )
    ap.add_argument("--outdir", default=None,
                    help="Defaults to the input file's directory.")
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    payload = json.loads(in_path.read_text())
    outdir = Path(args.outdir).resolve() if args.outdir else in_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = in_path.stem
    if stem.endswith("_recomp_sweep_scores"):
        stem = stem[: -len("_recomp_sweep_scores")]
    elif stem.endswith("_scores"):
        stem = stem[: -len("_scores")]

    _plot(payload, outdir, stem)


if __name__ == "__main__":
    main()
