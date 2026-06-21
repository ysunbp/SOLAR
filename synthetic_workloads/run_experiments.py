"""
Main entrypoint: run the three workload experiments across seeds and K.

Implements the matrix from `synthetic_workload_spec.md` §6:
  6.1  Cycling — sweep n_topics (m/K ratio) at K=10
  6.2  Topic-drift — fixed K=20
  6.3  Working-set — sweep K

Usage:
  python -m synthetic_workloads.run_experiments \
      --out-dir synthetic_workloads/results \
      --experiments cycling drift working_set \
      --seeds 42 1337 2024

Per result is a JSON file:
  results/<exp>/<method>__<param_tag>__seed<seed>.json
And per-experiment aggregated CSV at results/<exp>/aggregate.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .runners import make_runner, ALL_METHODS
from .workloads import (
    CyclingWorkload,
    TopicDriftWorkload,
    WorkingSetSweep,
    RetrievalNoiseWorkload,
)
from .dataset_io import load_bundle, variant_dir


# ---------------------------------------------------------------------------
# Disk-or-online workload loader
# ---------------------------------------------------------------------------

def _load_or_generate(data_dir, exp_name: str, variant_tag: str,
                      seed: int, generator):
    """
    If `data_dir` is given AND the corresponding variant directory exists,
    load the bundle from disk; otherwise call `generator()` to build it.
    `generator` is a zero-arg callable that returns a fresh WorkloadBundle.
    """
    if data_dir is not None:
        vdir = variant_dir(data_dir, exp_name, variant_tag, seed)
        if (vdir / "meta.json").is_file():
            return load_bundle(vdir)
    return generator()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_result(out_dir: Path, exp: str, tag: str, method: str, seed: int, result: Dict):
    sub = out_dir / exp
    sub.mkdir(parents=True, exist_ok=True)
    fname = f"{method}__{tag}__seed{seed}.json"
    # per_step_hits / per_step_f1 can blow up file size — keep but rounded.
    result = dict(result)
    if "per_step_f1" in result:
        result["per_step_f1"] = [round(x, 4) for x in result["per_step_f1"]]
    with open(sub / fname, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    return sub / fname


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o)}")


def _aggregate_summary(results: List[Dict]) -> List[Dict]:
    """Drop the per-step arrays for compact aggregation."""
    rows = []
    for r in results:
        rr = {k: v for k, v in r.items() if k not in ("per_step_hits", "per_step_f1")}
        # Flatten workload_params into a string for CSV friendliness.
        rr["workload_params"] = json.dumps(r.get("workload_params", {}), sort_keys=True)
        rows.append(rr)
    return rows


def _save_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    import csv
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------------------------------------------------------------------------
# 6.1 Cycling
# ---------------------------------------------------------------------------

def run_cycling(
    out_dir: Path,
    seeds: List[int],
    methods: List[str],
    n_topics_list: Optional[List[int]] = None,
    capacity: int = 10,
    retrieve_k: int = 3,
    items_per_topic: int = 5,
    n_cycles: int = 10,
    dim: int = 128,
    data_dir: Optional[Path] = None,
) -> List[Dict]:
    if n_topics_list is None:
        n_topics_list = [10, 15, 20, 30, 50]   # m/K ∈ {1,1.5,2,3,5}

    all_results = []
    print(f"\n=== Cycling (K={capacity}, n_topics={n_topics_list}) ===")
    for n_topics in n_topics_list:
        tag = f"K{capacity}_m{n_topics}"
        dump_tag = f"m{n_topics}"   # matches dump_dataset.py layout
        for seed in seeds:
            wl = _load_or_generate(
                data_dir, "cycling", dump_tag, seed,
                lambda: CyclingWorkload(
                    n_topics=n_topics,
                    items_per_topic=items_per_topic,
                    dim=dim,
                    n_cycles=n_cycles,
                    seed=seed,
                ).generate(),
            )
            for method in methods:
                runner = make_runner(method, capacity=capacity,
                                     retrieve_k=retrieve_k, seed=seed)
                t0 = time.time()
                result = runner.run(wl)
                dt = time.time() - t0
                result["m_over_k"] = n_topics / capacity
                _save_result(out_dir, "cycling", tag, method, seed, result)
                all_results.append(result)
                print(f"  m={n_topics:3d}/K={capacity:2d}  m/K={n_topics/capacity:.1f}  "
                      f"{method:9s}  seed={seed}  hit={result['avg_hit_rate']:.3f}  "
                      f"f1={result['avg_f1']:.3f}  store={result['store_rate']:.2f}  "
                      f"({dt:.1f}s)")

    rows = _aggregate_summary(all_results)
    _save_csv(rows, out_dir / "cycling" / "aggregate.csv")
    return all_results


# ---------------------------------------------------------------------------
# 6.2 Topic Drift
# ---------------------------------------------------------------------------

def run_topic_drift(
    out_dir: Path,
    seeds: List[int],
    methods: List[str],
    capacity: int = 20,
    retrieve_k: int = 3,
    n_topics_total: int = 30,
    n_active_per_phase: int = 5,
    phase_length_list: Optional[List[int]] = None,
    n_phases: int = 6,
    overlap: int = 1,
    dim: int = 128,
    data_dir: Optional[Path] = None,
) -> List[Dict]:
    if phase_length_list is None:
        phase_length_list = [100]  # main config; spec allows {50,100,200,500}

    all_results = []
    print(f"\n=== Topic Drift (K={capacity}, n_phases={n_phases}, "
          f"phase_length={phase_length_list}) ===")
    for plen in phase_length_list:
        tag = f"K{capacity}_plen{plen}_ov{overlap}"
        dump_tag = f"plen{plen}"   # matches dump_dataset.py layout
        for seed in seeds:
            wl = _load_or_generate(
                data_dir, "topic_drift", dump_tag, seed,
                lambda: TopicDriftWorkload(
                    n_topics_total=n_topics_total,
                    n_active_per_phase=n_active_per_phase,
                    phase_length=plen,
                    n_phases=n_phases,
                    overlap=overlap,
                    dim=dim,
                    seed=seed,
                ).generate(),
            )
            for method in methods:
                runner = make_runner(method, capacity=capacity,
                                     retrieve_k=retrieve_k, seed=seed)
                t0 = time.time()
                result = runner.run(wl)
                dt = time.time() - t0
                _save_result(out_dir, "topic_drift", tag, method, seed, result)
                all_results.append(result)
                print(f"  plen={plen:3d}  {method:9s}  seed={seed}  "
                      f"hit={result['avg_hit_rate']:.3f}  "
                      f"f1={result['avg_f1']:.3f}  store={result['store_rate']:.2f}  "
                      f"({dt:.1f}s)")
    rows = _aggregate_summary(all_results)
    _save_csv(rows, out_dir / "topic_drift" / "aggregate.csv")
    return all_results


# ---------------------------------------------------------------------------
# 6.3 Working-set sweep (phase-transition)
# ---------------------------------------------------------------------------

def run_working_set(
    out_dir: Path,
    seeds: List[int],
    methods: List[str],
    capacity_list: Optional[List[int]] = None,
    retrieve_k: int = 3,
    n_topics: int = 15,
    items_per_topic: int = 10,
    query_length: int = 500,
    topic_distribution: str = "uniform",
    dim: int = 128,
    data_dir: Optional[Path] = None,
) -> List[Dict]:
    if capacity_list is None:
        capacity_list = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]

    all_results = []
    print(f"\n=== Working-set Sweep (n_topics={n_topics}, "
          f"items_per_topic={items_per_topic}, K={capacity_list}) ===")
    # Workload itself is K-independent in this experiment; cache the bundle
    # per seed to avoid re-generating it for every K.
    dump_tag = f"dist{topic_distribution}"
    bundle_cache: Dict[int, object] = {}
    for K in capacity_list:
        tag = f"K{K}_dist{topic_distribution}"
        for seed in seeds:
            if seed not in bundle_cache:
                bundle_cache[seed] = _load_or_generate(
                    data_dir, "working_set", dump_tag, seed,
                    lambda s=seed: WorkingSetSweep(
                        n_topics=n_topics,
                        items_per_topic=items_per_topic,
                        query_length=query_length,
                        topic_distribution=topic_distribution,
                        dim=dim,
                        seed=s,
                    ).generate(),
                )
            wl = bundle_cache[seed]
            for method in methods:
                runner = make_runner(method, capacity=K,
                                     retrieve_k=retrieve_k, seed=seed)
                t0 = time.time()
                result = runner.run(wl)
                dt = time.time() - t0
                _save_result(out_dir, "working_set", tag, method, seed, result)
                all_results.append(result)
                print(f"  K={K:3d}  {method:9s}  seed={seed}  "
                      f"hit={result['avg_hit_rate']:.3f}  "
                      f"f1={result['avg_f1']:.3f}  store={result['store_rate']:.2f}  "
                      f"({dt:.1f}s)")
    rows = _aggregate_summary(all_results)
    _save_csv(rows, out_dir / "working_set" / "aggregate.csv")
    return all_results


# ---------------------------------------------------------------------------
# 6.4 Retrieval-noise U-curve
# ---------------------------------------------------------------------------

def run_retrieval_noise(
    out_dir: Path,
    seeds: List[int],
    methods: List[str],
    capacity_list: Optional[List[int]] = None,
    retrieve_k: int = 3,
    n_topics: int = 50,
    items_per_topic: int = 100,
    query_length: int = 1000,
    relevant_per_query: int = 3,
    intra_topic_std_list: Optional[List[float]] = None,
    inter_topic_sep_list: Optional[List[float]] = None,
    dim: int = 128,
    mode: str = "static",
    data_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Two modes (selected by `mode`):

      * 'static' : pool of size K is built deterministically (gt + distractors),
        all methods get the same pool via prefill -> only fifo / embedder_unlimited
        produce different results.  Used to draw the canonical U-curve.

      * 'stream' : every item streams through the runner one by one, paired with
        a topic-aligned warmup query.  Each method uses its own admission /
        eviction policy to decide what to keep.  Eval queries fire after the
        warmup phase.  This compares fifo / lru / lfu / arc / thompson / lmu /
        lmu_ts under retrieval noise.
    """
    if capacity_list is None:
        capacity_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    if intra_topic_std_list is None:
        intra_topic_std_list = [0.15]
    if inter_topic_sep_list is None:
        inter_topic_sep_list = [0.3]

    all_results: List[Dict] = []
    print(f"\n=== Retrieval-noise U-curve [mode={mode}] "
          f"(n_topics={n_topics}, items_per_topic={items_per_topic}, "
          f"K={capacity_list}, intra={intra_topic_std_list}, "
          f"inter={inter_topic_sep_list}) ===")

    for intra in intra_topic_std_list:
        for inter in inter_topic_sep_list:
            tag_base = f"intra{intra:.2f}_inter{inter:.2f}"
            for seed in seeds:
                if mode == "stream":
                    # Stream mode: workload is independent of K (we stream
                    # the full item pool); generate once per seed.
                    stream_dump_tag = tag_base   # matches dump_dataset.py
                    wl_stream = _load_or_generate(
                        data_dir, "retrieval_noise_stream",
                        stream_dump_tag, seed,
                        lambda i=intra, e=inter, s=seed: RetrievalNoiseWorkload(
                            n_topics=n_topics,
                            items_per_topic=items_per_topic,
                            intra_topic_std=i,
                            inter_topic_sep=e,
                            query_length=query_length,
                            pool_size=0,             # unused in stream mode
                            relevant_per_query=relevant_per_query,
                            mode="stream",
                            dim=dim,
                            seed=s,
                        ).generate(),
                    )
                for K in capacity_list:
                    if mode == "static":
                        static_dump_tag = f"K{K}_{tag_base}"
                        wl = _load_or_generate(
                            data_dir, "retrieval_noise_static",
                            static_dump_tag, seed,
                            lambda K=K, i=intra, e=inter, s=seed: RetrievalNoiseWorkload(
                                n_topics=n_topics,
                                items_per_topic=items_per_topic,
                                intra_topic_std=i,
                                inter_topic_sep=e,
                                query_length=query_length,
                                pool_size=K,
                                relevant_per_query=relevant_per_query,
                                mode="static",
                                dim=dim,
                                seed=s,
                            ).generate(),
                        )
                    else:
                        wl = wl_stream

                    tag = f"K{K}_{tag_base}_{mode}"
                    for method in methods:
                        if mode == "static":
                            # Static: embedder_unlimited needs huge cap; fifo cap=K
                            cap = max(K, 10**6) if method == "embedder_unlimited" else K
                        else:
                            # Stream: every method uses K as its capacity;
                            # embedder_unlimited has K=infinity (5000) by design.
                            cap = max(K, 10**6) if method == "embedder_unlimited" else K
                        runner = make_runner(method, capacity=cap,
                                             retrieve_k=retrieve_k, seed=seed)
                        t0 = time.time()
                        result = runner.run(wl)
                        dt = time.time() - t0
                        result["intra_topic_std"] = intra
                        result["inter_topic_sep"] = inter
                        result["pool_size"] = K
                        result["rn_mode"] = mode
                        _save_result(out_dir,
                                     f"retrieval_noise_{mode}",
                                     tag, method, seed, result)
                        all_results.append(result)
                        print(f"  [{mode}] K={K:5d}  intra={intra:.2f} inter={inter:.2f}  "
                              f"{method:18s}  seed={seed}  "
                              f"hit={result['avg_hit_rate']:.3f}  "
                              f"prec@3={result['avg_precision_at_k']:.3f}  "
                              f"cache_avg={result['avg_cache_size']:.0f}  "
                              f"final_cache={result['final_cache_size']:5d}  "
                              f"store={result['store_rate']:.2f}  "
                              f"({dt:.1f}s)")
    rows = _aggregate_summary(all_results)
    _save_csv(rows, out_dir / f"retrieval_noise_{mode}" / "aggregate.csv")
    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str,
                   default="synthetic_workloads/results")
    p.add_argument("--data-dir", type=str, default=None,
                   help="If set, load workloads from this directory "
                        "(produced by `python -m synthetic_workloads.dump_dataset`) "
                        "instead of regenerating them online.")
    p.add_argument("--experiments", nargs="+",
                   default=["cycling", "topic_drift", "working_set"],
                   choices=["cycling", "topic_drift", "working_set", "retrieval_noise"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2024])
    p.add_argument("--methods", nargs="+", default=ALL_METHODS,
                   choices=ALL_METHODS)
    p.add_argument("--retrieve-k", type=int, default=3)
    # Cycling
    p.add_argument("--cycling-K", type=int, default=10)
    p.add_argument("--cycling-n-topics", nargs="+", type=int,
                   default=[10, 15, 20, 30, 50])
    p.add_argument("--cycling-cycles", type=int, default=10)
    # Topic drift
    p.add_argument("--drift-K", type=int, default=20)
    p.add_argument("--drift-phase-length", nargs="+", type=int, default=[100])
    p.add_argument("--drift-n-phases", type=int, default=6)
    p.add_argument("--drift-overlap", type=int, default=1)
    # Working-set
    p.add_argument("--ws-K-list", nargs="+", type=int,
                   default=[5, 10, 15, 20, 30, 50, 75, 100, 150, 200])
    p.add_argument("--ws-n-topics", type=int, default=15)
    p.add_argument("--ws-items-per-topic", type=int, default=10)
    p.add_argument("--ws-query-length", type=int, default=500)
    p.add_argument("--ws-distribution", type=str, default="uniform",
                   choices=["uniform", "zipf"])
    # Retrieval noise
    p.add_argument("--rn-K-list", nargs="+", type=int,
                   default=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
    p.add_argument("--rn-n-topics", type=int, default=50)
    p.add_argument("--rn-items-per-topic", type=int, default=100)
    p.add_argument("--rn-query-length", type=int, default=1000)
    p.add_argument("--rn-relevant-per-query", type=int, default=3)
    p.add_argument("--rn-intra-std", nargs="+", type=float, default=[0.15])
    p.add_argument("--rn-inter-sep", nargs="+", type=float, default=[0.3])
    p.add_argument("--rn-mode", nargs="+", default=["static", "stream"],
                   choices=["static", "stream"],
                   help="which retrieval-noise modes to run")
    p.add_argument("--rn-static-methods", nargs="+",
                   default=["fifo", "embedder_unlimited"])
    p.add_argument("--rn-stream-methods", nargs="+",
                   default=["fifo", "lru", "lfu", "arc", "solar_e",
                            "solar_a", "solar", "embedder_unlimited"])
    p.add_argument("--rn-methods", nargs="+",
                   default=None,
                   help="deprecated; if set, used for both modes")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir) if args.data_dir else None
    if data_dir is not None:
        if not data_dir.is_dir():
            raise SystemExit(f"--data-dir does not exist: {data_dir}")
        print(f"[data-dir] reading workloads from {data_dir}")

    summary = {}
    if "cycling" in args.experiments:
        rs = run_cycling(
            out_dir=out_dir,
            seeds=args.seeds,
            methods=args.methods,
            n_topics_list=args.cycling_n_topics,
            capacity=args.cycling_K,
            retrieve_k=args.retrieve_k,
            n_cycles=args.cycling_cycles,
            data_dir=data_dir,
        )
        summary["cycling"] = len(rs)
    if "topic_drift" in args.experiments:
        rs = run_topic_drift(
            out_dir=out_dir,
            seeds=args.seeds,
            methods=args.methods,
            capacity=args.drift_K,
            retrieve_k=args.retrieve_k,
            phase_length_list=args.drift_phase_length,
            n_phases=args.drift_n_phases,
            overlap=args.drift_overlap,
            data_dir=data_dir,
        )
        summary["topic_drift"] = len(rs)
    if "working_set" in args.experiments:
        rs = run_working_set(
            out_dir=out_dir,
            seeds=args.seeds,
            methods=args.methods,
            capacity_list=args.ws_K_list,
            retrieve_k=args.retrieve_k,
            n_topics=args.ws_n_topics,
            items_per_topic=args.ws_items_per_topic,
            query_length=args.ws_query_length,
            topic_distribution=args.ws_distribution,
            data_dir=data_dir,
        )
        summary["working_set"] = len(rs)
    if "retrieval_noise" in args.experiments:
        for mode in args.rn_mode:
            if args.rn_methods is not None:
                rn_methods = args.rn_methods
            else:
                rn_methods = (args.rn_static_methods if mode == "static"
                              else args.rn_stream_methods)
            rs = run_retrieval_noise(
                out_dir=out_dir,
                seeds=args.seeds,
                methods=rn_methods,
                capacity_list=args.rn_K_list,
                retrieve_k=args.retrieve_k,
                n_topics=args.rn_n_topics,
                items_per_topic=args.rn_items_per_topic,
                query_length=args.rn_query_length,
                relevant_per_query=args.rn_relevant_per_query,
                intra_topic_std_list=args.rn_intra_std,
                inter_topic_sep_list=args.rn_inter_sep,
                mode=mode,
                data_dir=data_dir,
            )
            summary[f"retrieval_noise_{mode}"] = len(rs)

    with open(out_dir / "run_summary.json", "w") as f:
        json.dump({
            "args": vars(args),
            "num_results_per_exp": summary,
        }, f, indent=2)
    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
