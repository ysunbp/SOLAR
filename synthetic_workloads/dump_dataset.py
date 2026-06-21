"""
Materialize the full synthetic workload matrix used by run_experiments.py
to disk, so that experiments can re-run from the same data without
re-generating it (and so the dataset can be inspected / open-sourced).

Layout:
    <out_dir>/
        cycling/
            m10/seed42/{items,queries,stream,embeddings,meta}.{jsonl,npz,json}
            m10/seed1337/...
            m15/seed42/...
            ...
        topic_drift/
            plen100/seed42/...
            ...
        working_set/
            disturibform/seed42/...           (K is set on the runner side)
            ...
        retrieval_noise_static/
            K10_intra0.15_inter0.30/seed42/...
            ...
        retrieval_noise_stream/
            intra0.15_inter0.30/seed42/...    (K-independent)
            ...
        INDEX.json

Usage examples:

    # Dump the entire experiment matrix at the default seeds.
    python -m synthetic_workloads.dump_dataset \
        --out-dir synthetic_workloads/datasets

    # Smaller subset
    python -m synthetic_workloads.dump_dataset \
        --workloads cycling working_set \
        --seeds 42 \
        --out-dir synthetic_workloads/datasets

    # Preview only (no files)
    python -m synthetic_workloads.dump_dataset --preview --workloads cycling
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

from .workloads import (
    CyclingWorkload,
    TopicDriftWorkload,
    WorkingSetSweep,
    RetrievalNoiseWorkload,
    WorkloadBundle,
)
from .dataset_io import save_bundle, variant_dir


# ---------------------------------------------------------------------------
# Default sweep matrix (must mirror run_experiments.py defaults!)
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = [42, 1337, 2024]

# Cycling: sweep n_topics ∈ {10, 15, 20, 30, 50}, K=10 fixed (K is runner-side)
DEFAULT_CYCLING_M = [10, 15, 20, 30, 50]
DEFAULT_CYCLING_CYCLES = 10
DEFAULT_CYCLING_ITEMS_PER_TOPIC = 5

# Topic drift: phase_length sweep
DEFAULT_DRIFT_PHASE_LENGTHS = [100]
DEFAULT_DRIFT_N_PHASES = 6
DEFAULT_DRIFT_OVERLAP = 1
DEFAULT_DRIFT_N_TOPICS_TOTAL = 30
DEFAULT_DRIFT_N_ACTIVE = 5

# Working set: K is runner-side, only one workload variant per seed
DEFAULT_WS_N_TOPICS = 15
DEFAULT_WS_ITEMS_PER_TOPIC = 10
DEFAULT_WS_QUERY_LENGTH = 500
DEFAULT_WS_DIST = "uniform"

# Retrieval noise (static): pool_size sweep
DEFAULT_RN_K_LIST = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
DEFAULT_RN_N_TOPICS = 50
DEFAULT_RN_ITEMS_PER_TOPIC = 100
DEFAULT_RN_QUERY_LENGTH = 1000
DEFAULT_RN_RELEVANT = 3
DEFAULT_RN_INTRA_STD = [0.15]
DEFAULT_RN_INTER_SEP = [0.3]


ALL_WORKLOADS = [
    "cycling",
    "topic_drift",
    "working_set",
    "retrieval_noise_static",
    "retrieval_noise_stream",
]


# ---------------------------------------------------------------------------
# Variant generators (yield (variant_tag, bundle))
# ---------------------------------------------------------------------------

def _gen_cycling(seed: int, m_list, items_per_topic, n_cycles, dim=128):
    for m in m_list:
        tag = f"m{m}"
        bundle = CyclingWorkload(
            n_topics=m, items_per_topic=items_per_topic, dim=dim,
            n_cycles=n_cycles, seed=seed,
        ).generate()
        yield tag, bundle


def _gen_topic_drift(seed: int, phase_length_list, n_phases, overlap,
                     n_topics_total, n_active, dim=128):
    for plen in phase_length_list:
        tag = f"plen{plen}"
        bundle = TopicDriftWorkload(
            n_topics_total=n_topics_total,
            n_active_per_phase=n_active,
            phase_length=plen,
            n_phases=n_phases,
            overlap=overlap,
            dim=dim,
            seed=seed,
        ).generate()
        yield tag, bundle


def _gen_working_set(seed: int, n_topics, items_per_topic, query_length,
                     distribution, dim=128):
    tag = f"dist{distribution}"
    bundle = WorkingSetSweep(
        n_topics=n_topics,
        items_per_topic=items_per_topic,
        query_length=query_length,
        topic_distribution=distribution,
        dim=dim,
        seed=seed,
    ).generate()
    yield tag, bundle


def _gen_retrieval_noise_static(seed: int, K_list, intra_list, inter_list,
                                n_topics, items_per_topic, query_length,
                                relevant_per_query, dim=128):
    for intra in intra_list:
        for inter in inter_list:
            for K in K_list:
                tag = f"K{K}_intra{intra:.2f}_inter{inter:.2f}"
                bundle = RetrievalNoiseWorkload(
                    n_topics=n_topics,
                    items_per_topic=items_per_topic,
                    intra_topic_std=intra,
                    inter_topic_sep=inter,
                    query_length=query_length,
                    pool_size=K,
                    relevant_per_query=relevant_per_query,
                    mode="static",
                    dim=dim,
                    seed=seed,
                ).generate()
                yield tag, bundle


def _gen_retrieval_noise_stream(seed: int, intra_list, inter_list,
                                n_topics, items_per_topic, query_length,
                                relevant_per_query, dim=128):
    # Stream-mode workload is K-independent: one variant per (intra, inter)
    for intra in intra_list:
        for inter in inter_list:
            tag = f"intra{intra:.2f}_inter{inter:.2f}"
            bundle = RetrievalNoiseWorkload(
                n_topics=n_topics,
                items_per_topic=items_per_topic,
                intra_topic_std=intra,
                inter_topic_sep=inter,
                query_length=query_length,
                pool_size=0,        # unused in stream mode
                relevant_per_query=relevant_per_query,
                mode="stream",
                dim=dim,
                seed=seed,
            ).generate()
            yield tag, bundle


# ---------------------------------------------------------------------------
# Preview (no IO)
# ---------------------------------------------------------------------------

def _preview(bundle: WorkloadBundle, name: str, tag: str, seed: int):
    print(f"\n========== {name}/{tag}/seed{seed} ==========")
    print(f"workload_type : {bundle.workload_type}")
    print(f"n_items_pool  : {len(bundle.items_pool)}")
    print(f"n_queries     : {len(bundle.queries)}")
    print(f"n_stream      : {len(bundle.item_stream)}")
    if bundle.items_pool:
        print(f"embedding_dim : {bundle.items_pool[0].embedding.shape[0]}")
    if hasattr(bundle, "prefill_items"):
        print(f"n_prefill     : {len(bundle.prefill_items)}")
    if hasattr(bundle, "n_warmup_steps"):
        print(f"n_warmup_steps: {bundle.n_warmup_steps}")

    print(f"\n  -- first 3 items --")
    for it in bundle.items_pool[:3]:
        print(f"    id={it.id:20s}  topic={it.topic_id:3d}  created_at={it.created_at}")
    print(f"  -- first 3 queries --")
    for q in bundle.queries[:3]:
        rel = q.relevant_item_ids[:3]
        suffix = " ..." if len(q.relevant_item_ids) > 3 else ""
        print(f"    id={q.id:18s}  t={q.timestep:4d}  topic={q.topic_id:3d}  "
              f"|gt|={len(q.relevant_item_ids):3d}  gt={rel}{suffix}")
    print(f"  -- first 5 stream steps --")
    for t in range(min(5, len(bundle.item_stream))):
        it = bundle.item_stream[t]
        sid = it.id if it is not None else "<none>"
        print(f"    t={t:4d}  streamed={sid}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workloads", nargs="+", default=ALL_WORKLOADS,
                   choices=ALL_WORKLOADS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--out-dir", type=str,
                   default="synthetic_workloads/datasets")
    p.add_argument("--preview", action="store_true",
                   help="Print bundle summaries to stdout, write nothing.")

    # Cycling
    p.add_argument("--cycling-m", nargs="+", type=int, default=DEFAULT_CYCLING_M)
    p.add_argument("--cycling-cycles", type=int, default=DEFAULT_CYCLING_CYCLES)
    # Topic drift
    p.add_argument("--drift-phase-length", nargs="+", type=int,
                   default=DEFAULT_DRIFT_PHASE_LENGTHS)
    # Working set
    p.add_argument("--ws-distribution", type=str, default=DEFAULT_WS_DIST,
                   choices=["uniform", "zipf"])
    # Retrieval noise
    p.add_argument("--rn-K-list", nargs="+", type=int, default=DEFAULT_RN_K_LIST)
    p.add_argument("--rn-intra-std", nargs="+", type=float, default=DEFAULT_RN_INTRA_STD)
    p.add_argument("--rn-inter-sep", nargs="+", type=float, default=DEFAULT_RN_INTER_SEP)
    args = p.parse_args()

    out_root = Path(args.out_dir)
    if not args.preview:
        out_root.mkdir(parents=True, exist_ok=True)

    summary: List[dict] = []
    t_start = time.time()
    for seed in args.seeds:
        for name in args.workloads:
            if name == "cycling":
                gen = _gen_cycling(
                    seed=seed,
                    m_list=args.cycling_m,
                    items_per_topic=DEFAULT_CYCLING_ITEMS_PER_TOPIC,
                    n_cycles=args.cycling_cycles,
                )
            elif name == "topic_drift":
                gen = _gen_topic_drift(
                    seed=seed,
                    phase_length_list=args.drift_phase_length,
                    n_phases=DEFAULT_DRIFT_N_PHASES,
                    overlap=DEFAULT_DRIFT_OVERLAP,
                    n_topics_total=DEFAULT_DRIFT_N_TOPICS_TOTAL,
                    n_active=DEFAULT_DRIFT_N_ACTIVE,
                )
            elif name == "working_set":
                gen = _gen_working_set(
                    seed=seed,
                    n_topics=DEFAULT_WS_N_TOPICS,
                    items_per_topic=DEFAULT_WS_ITEMS_PER_TOPIC,
                    query_length=DEFAULT_WS_QUERY_LENGTH,
                    distribution=args.ws_distribution,
                )
            elif name == "retrieval_noise_static":
                gen = _gen_retrieval_noise_static(
                    seed=seed,
                    K_list=args.rn_K_list,
                    intra_list=args.rn_intra_std,
                    inter_list=args.rn_inter_sep,
                    n_topics=DEFAULT_RN_N_TOPICS,
                    items_per_topic=DEFAULT_RN_ITEMS_PER_TOPIC,
                    query_length=DEFAULT_RN_QUERY_LENGTH,
                    relevant_per_query=DEFAULT_RN_RELEVANT,
                )
            elif name == "retrieval_noise_stream":
                gen = _gen_retrieval_noise_stream(
                    seed=seed,
                    intra_list=args.rn_intra_std,
                    inter_list=args.rn_inter_sep,
                    n_topics=DEFAULT_RN_N_TOPICS,
                    items_per_topic=DEFAULT_RN_ITEMS_PER_TOPIC,
                    query_length=DEFAULT_RN_QUERY_LENGTH,
                    relevant_per_query=DEFAULT_RN_RELEVANT,
                )
            else:
                raise ValueError(name)

            for tag, bundle in gen:
                if args.preview:
                    _preview(bundle, name, tag, seed)
                    continue
                vdir = variant_dir(out_root, name, tag, seed)
                t0 = time.time()
                meta = save_bundle(bundle, vdir)
                dt = time.time() - t0
                meta_entry = {
                    "exp": name, "tag": tag, "seed": seed,
                    "path": str(vdir.relative_to(out_root)),
                    **{k: meta[k] for k in ("n_items", "n_queries",
                                            "n_stream_steps",
                                            "embedding_dim",
                                            "has_prefill",
                                            "n_prefill_items",
                                            "n_warmup_steps")},
                }
                summary.append(meta_entry)
                print(f"  [{name}/{tag}/seed{seed}]  items={meta['n_items']:5d}  "
                      f"queries={meta['n_queries']:5d}  "
                      f"stream={meta['n_stream_steps']:5d}  "
                      f"prefill={meta_entry['n_prefill_items']:5d}  "
                      f"warmup={meta_entry['n_warmup_steps']}  ({dt:.1f}s)")

    if not args.preview:
        index_path = out_root / "INDEX.json"
        with open(index_path, "w") as f:
            json.dump({
                "seeds": args.seeds,
                "workloads": args.workloads,
                "n_variants": len(summary),
                "variants": summary,
            }, f, indent=2)
        print(f"\nDumped {len(summary)} variants to {out_root}")
        print(f"Index: {index_path}")
        print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
