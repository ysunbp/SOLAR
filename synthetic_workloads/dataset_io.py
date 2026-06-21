"""
Serialize / deserialize WorkloadBundle objects to disk.

Format (one workload variant = one directory):
    <out_dir>/
        items.jsonl          : id, topic_id, created_at  (one per item)
        queries.jsonl        : id, topic_id, timestep, relevant_item_ids
        stream.jsonl         : per-step view (t -> streamed_item_id, query_id)
        prefill.jsonl        : prefill items (retrieval-noise only; one per item)
        embeddings.npz       : item_ids, item_embs, query_ids, query_embs
        meta.json            : workload params + dynamic attrs (n_warmup_steps, ...)

This format is portable, human-inspectable and lossless: loading produces a
WorkloadBundle that is byte-identical to the freshly generated one (modulo
float32 round-trip).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .primitives import SyntheticItem, SyntheticQuery
from .workloads import WorkloadBundle


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_bundle(bundle: WorkloadBundle, out_dir: Path) -> dict:
    """Persist a WorkloadBundle to `out_dir` (created if missing)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) items_pool
    item_ids: List[str] = []
    item_embs: List[np.ndarray] = []
    with open(out_dir / "items.jsonl", "w") as f:
        for it in bundle.items_pool:
            item_ids.append(it.id)
            item_embs.append(it.embedding)
            f.write(json.dumps({
                "id": it.id,
                "topic_id": int(it.topic_id),
                "created_at": int(it.created_at),
            }) + "\n")
    item_embs_np = (np.stack(item_embs, axis=0).astype(np.float32)
                    if item_embs else np.zeros((0, 0), dtype=np.float32))

    # 2) queries
    query_ids: List[str] = []
    query_embs: List[np.ndarray] = []
    with open(out_dir / "queries.jsonl", "w") as f:
        for q in bundle.queries:
            query_ids.append(q.id)
            query_embs.append(q.embedding)
            f.write(json.dumps({
                "id": q.id,
                "topic_id": int(q.topic_id),
                "timestep": int(q.timestep),
                "relevant_item_ids": list(q.relevant_item_ids),
            }) + "\n")
    query_embs_np = (np.stack(query_embs, axis=0).astype(np.float32)
                     if query_embs else np.zeros((0, 0), dtype=np.float32))

    # 3) stream (per-step references; embeddings already in items)
    with open(out_dir / "stream.jsonl", "w") as f:
        for t in range(len(bundle.item_stream)):
            it = bundle.item_stream[t]
            f.write(json.dumps({
                "t": t,
                "streamed_item_id": (it.id if it is not None else None),
            }) + "\n")

    # 4) optional prefill items (for retrieval-noise static-mode)
    prefill = getattr(bundle, "prefill_items", None)
    has_prefill = prefill is not None
    if has_prefill:
        with open(out_dir / "prefill.jsonl", "w") as f:
            for it in prefill:
                f.write(json.dumps({
                    "id": (it.id if it is not None else None),
                }) + "\n")

    # 5) embeddings (numerical access)
    np.savez_compressed(
        out_dir / "embeddings.npz",
        item_ids=np.array(item_ids),
        item_embs=item_embs_np,
        query_ids=np.array(query_ids),
        query_embs=query_embs_np,
    )

    # 6) meta
    n_warmup_steps = getattr(bundle, "n_warmup_steps", None)
    meta = {
        "workload_type": bundle.workload_type,
        "params": bundle.params,
        "n_items": len(bundle.items_pool),
        "n_queries": len(bundle.queries),
        "n_stream_steps": len(bundle.item_stream),
        "embedding_dim": int(item_embs_np.shape[1]) if item_embs_np.size else 0,
        "has_prefill": bool(has_prefill),
        "n_prefill_items": (len(prefill) if has_prefill else 0),
        "n_warmup_steps": (int(n_warmup_steps) if n_warmup_steps is not None else None),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_bundle(in_dir: Path) -> WorkloadBundle:
    """Reconstruct a WorkloadBundle previously written via `save_bundle`."""
    in_dir = Path(in_dir)

    with open(in_dir / "meta.json") as f:
        meta = json.load(f)

    # Embeddings -> id-indexed lookup
    z = np.load(in_dir / "embeddings.npz")
    item_ids_arr = z["item_ids"]
    item_embs_arr = z["item_embs"]
    query_ids_arr = z["query_ids"]
    query_embs_arr = z["query_embs"]
    item_emb_by_id = {str(item_ids_arr[i]): item_embs_arr[i]
                      for i in range(len(item_ids_arr))}
    query_emb_by_id = {str(query_ids_arr[i]): query_embs_arr[i]
                       for i in range(len(query_ids_arr))}

    # 1) items_pool
    items_pool: List[SyntheticItem] = []
    items_by_id: dict = {}
    with open(in_dir / "items.jsonl") as f:
        for line in f:
            d = json.loads(line)
            it = SyntheticItem(
                id=d["id"],
                topic_id=int(d["topic_id"]),
                embedding=item_emb_by_id[d["id"]],
                created_at=int(d["created_at"]),
            )
            items_pool.append(it)
            items_by_id[it.id] = it

    # 2) queries
    queries: List[SyntheticQuery] = []
    with open(in_dir / "queries.jsonl") as f:
        for line in f:
            d = json.loads(line)
            queries.append(SyntheticQuery(
                id=d["id"],
                topic_id=int(d["topic_id"]),
                timestep=int(d["timestep"]),
                relevant_item_ids=list(d["relevant_item_ids"]),
                embedding=query_emb_by_id[d["id"]],
            ))

    # 3) stream
    item_stream: List[Optional[SyntheticItem]] = []
    with open(in_dir / "stream.jsonl") as f:
        for line in f:
            d = json.loads(line)
            sid = d.get("streamed_item_id")
            item_stream.append(items_by_id[sid] if sid is not None else None)

    bundle = WorkloadBundle(
        items_pool=items_pool,
        item_stream=item_stream,
        queries=queries,
        workload_type=meta["workload_type"],
        params=meta["params"],
    )

    # 4) optional prefill_items
    if meta.get("has_prefill"):
        prefill: List[Optional[SyntheticItem]] = []
        with open(in_dir / "prefill.jsonl") as f:
            for line in f:
                d = json.loads(line)
                pid = d.get("id")
                prefill.append(items_by_id[pid] if pid is not None else None)
        bundle.prefill_items = prefill   # type: ignore[attr-defined]

    # 5) optional n_warmup_steps (stream mode)
    if meta.get("n_warmup_steps") is not None:
        bundle.n_warmup_steps = int(meta["n_warmup_steps"])  # type: ignore[attr-defined]

    return bundle


# ---------------------------------------------------------------------------
# Variant naming convention
# ---------------------------------------------------------------------------

def variant_dir(root: Path, exp: str, variant_tag: str, seed: int) -> Path:
    """Standard layout: <root>/<exp>/<variant_tag>/seed<seed>/."""
    return Path(root) / exp / variant_tag / f"seed{seed}"
