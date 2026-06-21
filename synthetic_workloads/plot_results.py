"""
Generate the three figures specified in `synthetic_workload_spec.md` §6.

  Fig 1 (cycling):     hit rate vs m/K
  Fig 2 (topic drift): rolling hit rate over time, vlines at phase boundaries
  Fig 3 (phase trans): (SOLAR - FIFO) hit rate vs K, annotate K*

Reads the per-result JSON files written by run_experiments.py.
Saves PNG files alongside the result directories.

Usage:
  python -m synthetic_workloads.plot_results \
      --results-dir synthetic_workloads/results
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_results(folder: Path) -> List[Dict]:
    out = []
    if not folder.exists():
        return out
    for fp in sorted(folder.glob("*.json")):
        if fp.name == "aggregate.json":
            continue
        with open(fp) as f:
            out.append(json.load(f))
    return out


# ---------------------------------------------------------------------------
# Fig 1: Cycling — hit rate vs m/K
# ---------------------------------------------------------------------------

def plot_cycling(folder: Path, fig_path: Path):
    results = _load_results(folder)
    if not results:
        print(f"[cycling] no results found in {folder}")
        return
    # group by (method, m/K)
    grouped = defaultdict(list)  # (method, mk) -> [hit_rates]
    for r in results:
        mk = r.get("m_over_k", r["workload_params"]["n_topics"] / r["capacity"])
        grouped[(r["method"], mk)].append(r["avg_hit_rate"])

    methods = sorted({k[0] for k in grouped})
    plt.figure(figsize=(7, 4.5))
    for m in methods:
        xs, mean, std = [], [], []
        for (mm, mk), vals in sorted(grouped.items()):
            if mm != m:
                continue
            xs.append(mk)
            mean.append(np.mean(vals))
            std.append(np.std(vals))
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        mean = np.asarray(mean)[order]
        std = np.asarray(std)[order]
        plt.errorbar(xs, mean, yerr=std, label=m, marker="o", capsize=3)
    plt.xlabel("m / K  (working-set / cache capacity ratio)")
    plt.ylabel("avg hit rate")
    plt.title("Cycling workload: FIFO thrashing vs m/K")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[cycling] saved {fig_path}")


# ---------------------------------------------------------------------------
# Fig 2: Topic drift — rolling hit rate
# ---------------------------------------------------------------------------

def plot_topic_drift(folder: Path, fig_path: Path, window: int = 20):
    results = _load_results(folder)
    if not results:
        print(f"[topic_drift] no results found in {folder}")
        return
    by_method = defaultdict(list)   # method -> [per_step_hits arrays]
    phase_boundaries = None
    for r in results:
        by_method[r["method"]].append(np.asarray(r["per_step_hits"], dtype=float))
        pb = r["workload_params"].get("phase_boundaries")
        if pb:
            phase_boundaries = pb

    plt.figure(figsize=(9, 4.5))
    for m, runs in sorted(by_method.items()):
        # Average across seeds (assume same length)
        runs = np.stack(runs, axis=0)  # (n_seeds, T)
        mean_per_step = runs.mean(axis=0)
        if len(mean_per_step) >= window:
            kernel = np.ones(window) / window
            rolling = np.convolve(mean_per_step, kernel, mode="valid")
            xs = np.arange(window - 1, len(mean_per_step))
        else:
            rolling = mean_per_step
            xs = np.arange(len(rolling))
        plt.plot(xs, rolling, label=m, lw=1.6)

    if phase_boundaries:
        for b in phase_boundaries:
            plt.axvline(b, color="grey", lw=0.7, ls="--", alpha=0.6)

    plt.xlabel("timestep")
    plt.ylabel(f"rolling hit rate (window={window})")
    plt.title("Topic-drift workload: adaptivity to non-stationary topics")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[topic_drift] saved {fig_path}")


# ---------------------------------------------------------------------------
# Fig 3: Phase transition — (SOLAR − FIFO) vs K, plus per-method curves
# ---------------------------------------------------------------------------

def plot_phase_transition(folder: Path, fig_path: Path):
    results = _load_results(folder)
    if not results:
        print(f"[working_set] no results found in {folder}")
        return

    # by_method[method][K] -> [hit_rates]
    by = defaultdict(lambda: defaultdict(list))
    for r in results:
        by[r["method"]][r["capacity"]].append(r["avg_hit_rate"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) per-method hit rate vs K
    ax = axes[0]
    for m, kdict in sorted(by.items()):
        Ks = sorted(kdict.keys())
        mean = np.array([np.mean(kdict[k]) for k in Ks])
        std  = np.array([np.std(kdict[k])  for k in Ks])
        ax.errorbar(Ks, mean, yerr=std, label=m, marker="o", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("cache capacity K (log)")
    ax.set_ylabel("avg hit rate")
    ax.set_title("Hit rate vs K (all methods)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    # (b) (SOLAR - FIFO) vs K with K* annotation
    ax = axes[1]
    if "solar" in by and "fifo" in by:
        Ks = sorted(set(by["solar"].keys()) & set(by["fifo"].keys()))
        diff_mean = np.array([
            np.mean(by["solar"][k]) - np.mean(by["fifo"][k]) for k in Ks
        ])
        ax.plot(Ks, diff_mean, "o-", color="tab:purple",
                label="SOLAR − FIFO", lw=2)
        ax.axhline(0, color="black", lw=0.8)
        # K* = first K (going from small → large) where diff crosses below zero
        kstar = None
        for i, k in enumerate(Ks):
            if i > 0 and diff_mean[i - 1] > 0 and diff_mean[i] <= 0:
                # Linear interp between Ks[i-1] and Ks[i]
                d0, d1 = diff_mean[i - 1], diff_mean[i]
                t = d0 / (d0 - d1) if d0 != d1 else 0.5
                kstar = Ks[i - 1] + t * (Ks[i] - Ks[i - 1])
                break
        if kstar is not None:
            ax.axvline(kstar, color="red", ls="--", alpha=0.8,
                       label=f"K* ≈ {kstar:.1f}")
        ax.set_xscale("log")
        ax.set_xlabel("cache capacity K (log)")
        ax.set_ylabel("Δ hit rate  (SOLAR − FIFO)")
        ax.set_title("Phase transition: where does FIFO win?")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[working_set] saved {fig_path}")


# ---------------------------------------------------------------------------
# Fig 4: Retrieval noise U-curve
# ---------------------------------------------------------------------------

def plot_retrieval_noise(folder: Path, fig_path: Path):
    results = _load_results(folder)
    if not results:
        print(f"[retrieval_noise] no results found in {folder}")
        return

    # Group by (method, intra, inter, K) -> [(hit, prec)]
    from collections import defaultdict
    grp = defaultdict(list)
    for r in results:
        # Use pool_size from result (workload param) rather than runner
        # capacity, because embedder_unlimited has cap=10**9 but its
        # effective pool size matches the workload's pool_size.
        K = int(r.get("pool_size", r["workload_params"].get("pool_size",
                                                            r["capacity"])))
        key = (
            r["method"],
            float(r.get("intra_topic_std", r["workload_params"]["intra_topic_std"])),
            float(r.get("inter_topic_sep", r["workload_params"]["inter_topic_sep"])),
            K,
        )
        grp[key].append((r["avg_hit_rate"], r["avg_precision_at_k"]))

    # Distinct (intra, inter) settings
    settings = sorted({(k[1], k[2]) for k in grp.keys()})
    methods = sorted({k[0] for k in grp.keys()})

    # One subplot per (intra, inter) setting; if only one setting, just one plot.
    n = len(settings)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.2 * rows),
                             squeeze=False)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for si, (intra, inter) in enumerate(settings):
        ax = axes[si // cols][si % cols]
        ax2 = ax.twinx()
        for mi, m in enumerate(methods):
            Ks, hit_mean, hit_std, prec_mean = [], [], [], []
            for (mm, ii, ee, K), vals in grp.items():
                if mm != m or ii != intra or ee != inter:
                    continue
                Ks.append(K)
                hits = np.array([v[0] for v in vals])
                precs = np.array([v[1] for v in vals])
                hit_mean.append(hits.mean())
                hit_std.append(hits.std())
                prec_mean.append(precs.mean())
            order = np.argsort(Ks)
            Ks = np.asarray(Ks)[order]
            hit_mean = np.asarray(hit_mean)[order]
            hit_std = np.asarray(hit_std)[order]
            prec_mean = np.asarray(prec_mean)[order]
            color = color_cycle[mi % len(color_cycle)]
            ax.errorbar(Ks, hit_mean, yerr=hit_std, marker="o", capsize=3,
                        color=color, label=f"{m} hit")
            ax2.plot(Ks, prec_mean, marker="x", linestyle="--",
                     color=color, alpha=0.6,
                     label=f"{m} prec@k")

            # Annotate peak K* (max hit rate)
            if len(Ks) > 0 and m == "fifo":
                pk_idx = int(np.argmax(hit_mean))
                ax.axvline(Ks[pk_idx], color=color, ls=":", alpha=0.6,
                           label=f"K*≈{Ks[pk_idx]} ({m})")

        ax.set_xscale("log")
        ax.set_xlabel("cache capacity K (log)")
        ax.set_ylabel("avg hit rate")
        ax2.set_ylabel("avg precision@k", color="grey")
        ax.set_title(f"intra_std={intra}, inter_sep={inter}")
        ax.grid(True, which="both", alpha=0.3)
        # Combine legends from both axes
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")

    # Hide unused subplots
    for si in range(len(settings), rows * cols):
        axes[si // cols][si % cols].axis("off")

    fig.suptitle("Retrieval-noise U-curve: hit & precision vs pool size",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[retrieval_noise] saved {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str,
                   default="synthetic_workloads/results")
    args = p.parse_args()

    rd = Path(args.results_dir)
    plot_cycling(rd / "cycling", rd / "cycling" / "figure_cycling.png")
    plot_topic_drift(rd / "topic_drift", rd / "topic_drift" / "figure_topic_drift.png")
    plot_phase_transition(rd / "working_set", rd / "working_set" / "figure_phase_transition.png")
    plot_retrieval_noise(rd / "retrieval_noise_static",
                         rd / "retrieval_noise_static" / "figure_retrieval_noise.png")
    plot_retrieval_noise(rd / "retrieval_noise_stream",
                         rd / "retrieval_noise_stream" / "figure_retrieval_noise.png")


if __name__ == "__main__":
    main()
