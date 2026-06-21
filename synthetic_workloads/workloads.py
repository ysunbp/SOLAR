"""
Three workload generators:
  - CyclingWorkload         (Thm 4: FIFO thrashing)
  - TopicDriftWorkload      (non-stationary, adaptivity)
  - WorkingSetSweep         (phase transition)

All return (items_pool, query_sequence, item_stream) where
  items_pool : List[SyntheticItem]   — full pool (for ground-truth lookup)
  query_seq  : List[SyntheticQuery]  — per-step queries
  item_stream: List[SyntheticItem]   — items that arrive per step

`item_stream[t]` is the new item created at step t (the runner must `add` it
to the cache before retrieving for `query_seq[t]`).  All ground-truth
items used by queries are pre-injected before the test phase begins, OR they
are generated in `item_stream` *before* the query references them — see
each workload for the contract.

Item ids are globally unique strings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .primitives import (
    SyntheticItem,
    SyntheticQuery,
    generate_topic_centers,
    sample_item_embedding,
    sample_query_embedding,
)


# ---------------------------------------------------------------------------
# Common base class
# ---------------------------------------------------------------------------

@dataclass
class WorkloadBundle:
    """Container for a generated workload."""
    items_pool: List[SyntheticItem]
    item_stream: List[Optional[SyntheticItem]]  # one slot per step (may be None)
    queries: List[SyntheticQuery]
    workload_type: str
    params: Dict


# ---------------------------------------------------------------------------
# 1. Cycling workload (verify FIFO thrashing under Thm 4)
# ---------------------------------------------------------------------------

class CyclingWorkload:
    """
    `n_topics` topics cycle deterministically. With cache size K < n_topics,
    FIFO is forced into thrashing.

    Each step t:
      - topic_id = (t mod n_topics)
      - a fresh item of that topic is added to the stream
      - the query targets that topic; ground-truth = all items of that topic
        currently produced.
    """

    def __init__(
        self,
        n_topics: int = 20,
        items_per_topic: int = 5,
        dim: int = 128,
        n_cycles: int = 10,
        intra_topic_std: float = 0.1,
        query_noise: float = 0.05,
        min_sep: float = 0.5,
        seed: int = 42,
    ):
        self.n_topics = n_topics
        self.items_per_topic = items_per_topic
        self.dim = dim
        self.n_cycles = n_cycles
        self.intra_topic_std = intra_topic_std
        self.query_noise = query_noise
        self.min_sep = min_sep
        self.seed = seed

    def generate(self) -> WorkloadBundle:
        rng = np.random.default_rng(self.seed)

        centers = generate_topic_centers(
            self.n_topics, dim=self.dim, min_sep=self.min_sep, rng=rng
        )

        # 1) Pre-generate `items_per_topic` items for every topic (the "pool").
        items_pool: List[SyntheticItem] = []
        items_by_topic: Dict[int, List[SyntheticItem]] = {t: [] for t in range(self.n_topics)}
        for tid in range(self.n_topics):
            for j in range(self.items_per_topic):
                emb = sample_item_embedding(centers[tid], self.intra_topic_std, rng)
                item = SyntheticItem(
                    id=f"cy_t{tid}_i{j}",
                    topic_id=tid,
                    embedding=emb,
                    created_at=-1,  # set when actually streamed
                )
                items_pool.append(item)
                items_by_topic[tid].append(item)

        # 2) Build the cyclic query / item stream.
        n_steps = self.n_topics * self.n_cycles
        item_stream: List[Optional[SyntheticItem]] = [None] * n_steps
        queries: List[SyntheticQuery] = []

        # Round-robin pointer for picking which item (within topic) to stream.
        topic_ptr = {t: 0 for t in range(self.n_topics)}

        for t in range(n_steps):
            tid = t % self.n_topics
            # Stream a topic item (one fresh item per step). Round-robin within topic.
            item_idx = topic_ptr[tid] % self.items_per_topic
            topic_ptr[tid] += 1
            streamed = items_by_topic[tid][item_idx]
            streamed.created_at = t
            item_stream[t] = streamed

            # Ground truth = ALL items of this topic.  (Hit if any retrieved.)
            relevant_ids = [it.id for it in items_by_topic[tid]]
            q_emb = sample_query_embedding(centers[tid], self.query_noise, rng)
            queries.append(SyntheticQuery(
                id=f"cy_q{t}",
                embedding=q_emb,
                relevant_item_ids=relevant_ids,
                timestep=t,
                topic_id=tid,
            ))

        return WorkloadBundle(
            items_pool=items_pool,
            item_stream=item_stream,
            queries=queries,
            workload_type="cycling",
            params=dict(
                n_topics=self.n_topics,
                items_per_topic=self.items_per_topic,
                dim=self.dim,
                n_cycles=self.n_cycles,
                intra_topic_std=self.intra_topic_std,
                query_noise=self.query_noise,
                seed=self.seed,
            ),
        )


# ---------------------------------------------------------------------------
# 2. Topic-drift workload (non-stationary)
# ---------------------------------------------------------------------------

class TopicDriftWorkload:
    """
    `n_phases` phases, each `phase_length` steps long.  Each phase has
    `n_active_per_phase` topics active; consecutive phases share `overlap`
    topics (sliding window).

    Per step within a phase: a topic is sampled uniformly from the active set,
    a new item of that topic is streamed, and the query targets that topic.
    """

    def __init__(
        self,
        n_topics_total: int = 30,
        n_active_per_phase: int = 5,
        phase_length: int = 100,
        n_phases: int = 6,
        overlap: int = 1,
        items_per_topic: int = 8,
        dim: int = 128,
        intra_topic_std: float = 0.1,
        query_noise: float = 0.05,
        min_sep: float = 0.5,
        seed: int = 42,
    ):
        self.n_topics_total = n_topics_total
        self.n_active_per_phase = n_active_per_phase
        self.phase_length = phase_length
        self.n_phases = n_phases
        self.overlap = overlap
        self.items_per_topic = items_per_topic
        self.dim = dim
        self.intra_topic_std = intra_topic_std
        self.query_noise = query_noise
        self.min_sep = min_sep
        self.seed = seed

        # Sliding-window check: with `n_active_per_phase` and `overlap`,
        # we advance by (n_active - overlap) topics per phase.
        step = n_active_per_phase - overlap
        if step <= 0:
            raise ValueError("overlap must be < n_active_per_phase")
        required = (n_phases - 1) * step + n_active_per_phase
        if required > n_topics_total:
            raise ValueError(
                f"Need n_topics_total >= {required} for {n_phases} phases / "
                f"{n_active_per_phase} active / overlap={overlap}; "
                f"got {n_topics_total}."
            )

    def _phase_active_topics(self, phase: int) -> List[int]:
        step = self.n_active_per_phase - self.overlap
        start = phase * step
        return list(range(start, start + self.n_active_per_phase))

    def generate(self) -> WorkloadBundle:
        rng = np.random.default_rng(self.seed)
        centers = generate_topic_centers(
            self.n_topics_total, dim=self.dim, min_sep=self.min_sep, rng=rng,
        )

        # Pre-generate item pool: items_per_topic items for each topic
        items_pool: List[SyntheticItem] = []
        items_by_topic: Dict[int, List[SyntheticItem]] = {t: [] for t in range(self.n_topics_total)}
        for tid in range(self.n_topics_total):
            for j in range(self.items_per_topic):
                emb = sample_item_embedding(centers[tid], self.intra_topic_std, rng)
                item = SyntheticItem(
                    id=f"dr_t{tid}_i{j}",
                    topic_id=tid,
                    embedding=emb,
                    created_at=-1,
                )
                items_pool.append(item)
                items_by_topic[tid].append(item)

        topic_ptr = {t: 0 for t in range(self.n_topics_total)}

        n_steps = self.n_phases * self.phase_length
        item_stream: List[Optional[SyntheticItem]] = [None] * n_steps
        queries: List[SyntheticQuery] = []

        for phase in range(self.n_phases):
            active = self._phase_active_topics(phase)
            for j in range(self.phase_length):
                t = phase * self.phase_length + j
                tid = int(rng.choice(active))

                # Stream one item of topic `tid` (round-robin within topic).
                item_idx = topic_ptr[tid] % self.items_per_topic
                topic_ptr[tid] += 1
                streamed = items_by_topic[tid][item_idx]
                streamed.created_at = t
                item_stream[t] = streamed

                relevant_ids = [it.id for it in items_by_topic[tid]]
                q_emb = sample_query_embedding(centers[tid], self.query_noise, rng)
                queries.append(SyntheticQuery(
                    id=f"dr_q{t}",
                    embedding=q_emb,
                    relevant_item_ids=relevant_ids,
                    timestep=t,
                    topic_id=tid,
                ))

        return WorkloadBundle(
            items_pool=items_pool,
            item_stream=item_stream,
            queries=queries,
            workload_type="topic_drift",
            params=dict(
                n_topics_total=self.n_topics_total,
                n_active_per_phase=self.n_active_per_phase,
                phase_length=self.phase_length,
                n_phases=self.n_phases,
                overlap=self.overlap,
                items_per_topic=self.items_per_topic,
                dim=self.dim,
                intra_topic_std=self.intra_topic_std,
                query_noise=self.query_noise,
                seed=self.seed,
                phase_boundaries=[
                    p * self.phase_length for p in range(1, self.n_phases)
                ],
            ),
        )


# ---------------------------------------------------------------------------
# 3. Working-set sweep (phase transition)
# ---------------------------------------------------------------------------

class WorkingSetSweep:
    """
    Stationary workload of length `query_length` with topic distribution
    either uniform or zipf.  K is *not* set here — it is varied at the
    runner level.
    """

    def __init__(
        self,
        n_topics: int = 15,
        items_per_topic: int = 10,
        query_length: int = 500,
        topic_distribution: str = "uniform",   # "uniform" | "zipf"
        zipf_param: float = 1.2,
        dim: int = 128,
        intra_topic_std: float = 0.1,
        query_noise: float = 0.05,
        min_sep: float = 0.5,
        seed: int = 42,
    ):
        if topic_distribution not in ("uniform", "zipf"):
            raise ValueError(f"unknown topic_distribution: {topic_distribution}")
        self.n_topics = n_topics
        self.items_per_topic = items_per_topic
        self.query_length = query_length
        self.topic_distribution = topic_distribution
        self.zipf_param = zipf_param
        self.dim = dim
        self.intra_topic_std = intra_topic_std
        self.query_noise = query_noise
        self.min_sep = min_sep
        self.seed = seed

    def _sample_topic(self, rng: np.random.Generator) -> int:
        if self.topic_distribution == "uniform":
            return int(rng.integers(0, self.n_topics))
        # zipf: rank-based, restricted to [1, n_topics]
        # Use a truncated zipf via inverse-CDF style: weights ~ 1/r^a
        ranks = np.arange(1, self.n_topics + 1, dtype=float)
        w = 1.0 / np.power(ranks, self.zipf_param)
        w /= w.sum()
        return int(rng.choice(self.n_topics, p=w))

    def generate(self) -> WorkloadBundle:
        rng = np.random.default_rng(self.seed)
        centers = generate_topic_centers(
            self.n_topics, dim=self.dim, min_sep=self.min_sep, rng=rng,
        )

        items_pool: List[SyntheticItem] = []
        items_by_topic: Dict[int, List[SyntheticItem]] = {t: [] for t in range(self.n_topics)}
        for tid in range(self.n_topics):
            for j in range(self.items_per_topic):
                emb = sample_item_embedding(centers[tid], self.intra_topic_std, rng)
                item = SyntheticItem(
                    id=f"ws_t{tid}_i{j}",
                    topic_id=tid,
                    embedding=emb,
                    created_at=-1,
                )
                items_pool.append(item)
                items_by_topic[tid].append(item)

        topic_ptr = {t: 0 for t in range(self.n_topics)}

        item_stream: List[Optional[SyntheticItem]] = [None] * self.query_length
        queries: List[SyntheticQuery] = []
        for t in range(self.query_length):
            tid = self._sample_topic(rng)
            item_idx = topic_ptr[tid] % self.items_per_topic
            topic_ptr[tid] += 1
            streamed = items_by_topic[tid][item_idx]
            streamed.created_at = t
            item_stream[t] = streamed

            relevant_ids = [it.id for it in items_by_topic[tid]]
            q_emb = sample_query_embedding(centers[tid], self.query_noise, rng)
            queries.append(SyntheticQuery(
                id=f"ws_q{t}",
                embedding=q_emb,
                relevant_item_ids=relevant_ids,
                timestep=t,
                topic_id=tid,
            ))

        return WorkloadBundle(
            items_pool=items_pool,
            item_stream=item_stream,
            queries=queries,
            workload_type="working_set",
            params=dict(
                n_topics=self.n_topics,
                items_per_topic=self.items_per_topic,
                query_length=self.query_length,
                topic_distribution=self.topic_distribution,
                zipf_param=self.zipf_param,
                dim=self.dim,
                intra_topic_std=self.intra_topic_std,
                query_noise=self.query_noise,
                seed=self.seed,
            ),
        )


# ---------------------------------------------------------------------------
# 4. Retrieval-noise workload (U-curve: capacity-constraint justification)
# ---------------------------------------------------------------------------

class RetrievalNoiseWorkload:
    """
    Retrieval-noise workload (U-curve).

    Two modes:

    ─────────────────────────────────────────────────────────────────────
    mode = "static"  (default; spec §3.4 original design)
    ─────────────────────────────────────────────────────────────────────
      Tests *pure retrieval* under a controlled pool size, with no cache
      management dynamics:
        1. Pre-generate full item pool (n_topics × items_per_topic).
        2. Pre-generate `query_length` queries with ground-truth items.
        3. Build a static cache pool of size `pool_size`:
             - Always include every gt item any query needs.
             - Fill the rest with random distractors.
        4. Push the static pool to the runner as `prefill_items`, so
           after warmup the cache contains exactly that pool.
        5. Runtime item_stream is empty (None × query_length).

      All admission-based methods are equivalent here: they all see the
      same pool because `prefill_items` go through `_admit` which respects
      the runner's capacity.  Use this mode for the "fifo vs
      embedder_unlimited" U-curve plot.

    ─────────────────────────────────────────────────────────────────────
    mode = "stream"  (for comparing methods)
    ─────────────────────────────────────────────────────────────────────
      Tests *cache-management policies under retrieval noise*:
        1. Same item pool, same queries.
        2. Stream every item in `items_pool` (random permutation) at one
           item per step, paired with a "topic-aligned" warmup query to
           give admission policies retrieval feedback.  Each method
           builds up a cache of size ≤ K using its own admission/eviction
           policy.
        3. After the warmup stream, run the actual `query_length`
           evaluation queries (no new items: item_stream = None during
           this phase).

      Use this mode to show that selective admission (LMU/TS) keeps
      precision high even when the available pool is huge, while FIFO
      gets drowned by distractors.
    ─────────────────────────────────────────────────────────────────────

    `inter_topic_sep` controls topic-center distance — smaller sep ⇒
    harder to distinguish ⇒ stronger U-curve.
    """

    def __init__(
        self,
        n_topics: int = 50,
        items_per_topic: int = 100,
        intra_topic_std: float = 0.15,
        inter_topic_sep: float = 0.3,    # smaller = harder
        query_length: int = 1000,
        relevant_per_query: int = 3,
        pool_size: int = 500,            # K; static-mode pool size (only used in mode='static')
        mode: str = "static",            # 'static' | 'stream'
        warmup_queries_per_item: float = 1.0,  # stream-mode: queries per stream step
        dim: int = 128,
        query_noise: float = 0.05,
        seed: int = 42,
    ):
        if mode not in ("static", "stream"):
            raise ValueError(f"unknown mode: {mode!r}")
        self.n_topics = n_topics
        self.items_per_topic = items_per_topic
        self.intra_topic_std = intra_topic_std
        self.inter_topic_sep = inter_topic_sep
        self.query_length = query_length
        self.relevant_per_query = relevant_per_query
        self.pool_size = pool_size
        self.mode = mode
        self.warmup_queries_per_item = warmup_queries_per_item
        self.dim = dim
        self.query_noise = query_noise
        self.seed = seed

    # ----- Helpers ---------------------------------------------------------

    def _build_pool_and_eval_queries(self, rng):
        """Generate the full item pool and the evaluation queries.
        Returns (centers, items_pool, items_by_topic, eval_queries, gt_ids_set).
        """
        centers = generate_topic_centers(
            self.n_topics,
            dim=self.dim,
            min_sep=self.inter_topic_sep,
            rng=rng,
        )

        items_pool: List[SyntheticItem] = []
        items_by_topic: Dict[int, List[SyntheticItem]] = {t: [] for t in range(self.n_topics)}
        for tid in range(self.n_topics):
            for j in range(self.items_per_topic):
                emb = sample_item_embedding(centers[tid], self.intra_topic_std, rng)
                item = SyntheticItem(
                    id=f"rn_t{tid}_i{j}",
                    topic_id=tid,
                    embedding=emb,
                    created_at=-1,
                )
                items_pool.append(item)
                items_by_topic[tid].append(item)

        eval_queries: List[SyntheticQuery] = []
        gt_ids_set: set = set()
        for t in range(self.query_length):
            tid = int(rng.integers(0, self.n_topics))
            n_rel = min(self.relevant_per_query, self.items_per_topic)
            rel_choices = rng.choice(self.items_per_topic, size=n_rel, replace=False)
            relevant_ids = [items_by_topic[tid][int(j)].id for j in rel_choices]
            for rid in relevant_ids:
                gt_ids_set.add(rid)
            q_emb = sample_query_embedding(centers[tid], self.query_noise, rng)
            eval_queries.append(SyntheticQuery(
                id=f"rn_q{t}",
                embedding=q_emb,
                relevant_item_ids=relevant_ids,
                timestep=t,
                topic_id=tid,
            ))
        return centers, items_pool, items_by_topic, eval_queries, gt_ids_set

    # ----- Generators ------------------------------------------------------

    def _generate_static(self) -> WorkloadBundle:
        rng = np.random.default_rng(self.seed)
        (centers, items_pool, items_by_topic,
         eval_queries, gt_ids_set) = self._build_pool_and_eval_queries(rng)
        total_items = len(items_pool)

        gt_items = [it for it in items_pool if it.id in gt_ids_set]
        non_gt_items = [it for it in items_pool if it.id not in gt_ids_set]

        K = min(self.pool_size, total_items)
        if K >= len(gt_items):
            # Coverage regime: keep every gt + fill with distractors
            n_distractor = K - len(gt_items)
            chosen_distractors = list(rng.choice(
                np.arange(len(non_gt_items)),
                size=n_distractor, replace=False,
            )) if n_distractor > 0 else []
            distractors = [non_gt_items[int(i)] for i in chosen_distractors]
            pool_contents = gt_items + distractors
        else:
            # Tight regime: K too small; keep a random subset of gt items.
            chosen_gt = list(rng.choice(
                np.arange(len(gt_items)),
                size=K, replace=False,
            ))
            pool_contents = [gt_items[int(i)] for i in chosen_gt]

        rng.shuffle(pool_contents)
        for t, it in enumerate(pool_contents):
            it.created_at = -len(pool_contents) + t

        prefill_items: List[Optional[SyntheticItem]] = list(pool_contents)
        runtime_items: List[Optional[SyntheticItem]] = [None] * self.query_length

        bundle = WorkloadBundle(
            items_pool=items_pool,
            item_stream=runtime_items,
            queries=eval_queries,
            workload_type="retrieval_noise",
            params=dict(
                mode="static",
                n_topics=self.n_topics,
                items_per_topic=self.items_per_topic,
                intra_topic_std=self.intra_topic_std,
                inter_topic_sep=self.inter_topic_sep,
                query_length=self.query_length,
                relevant_per_query=self.relevant_per_query,
                pool_size=K,
                dim=self.dim,
                query_noise=self.query_noise,
                seed=self.seed,
                total_pool_size=total_items,
                num_unique_gt_items=len(gt_items),
                actual_pool_size=len(pool_contents),
            ),
        )
        bundle.prefill_items = prefill_items   # type: ignore[attr-defined]
        return bundle

    def _generate_stream(self) -> WorkloadBundle:
        rng = np.random.default_rng(self.seed)
        (centers, items_pool, items_by_topic,
         eval_queries, gt_ids_set) = self._build_pool_and_eval_queries(rng)
        total_items = len(items_pool)
        gt_items = [it for it in items_pool if it.id in gt_ids_set]

        # Warmup phase: stream every pool item with a topic-aligned query
        # so admission policies receive retrieval feedback.
        warmup_perm = rng.permutation(total_items)
        warmup_items: List[SyntheticItem] = [items_pool[int(i)] for i in warmup_perm]
        warmup_queries: List[SyntheticQuery] = []
        for t, it in enumerate(warmup_items):
            tid = it.topic_id
            n_rel = min(self.relevant_per_query, self.items_per_topic)
            rel_choices = rng.choice(self.items_per_topic, size=n_rel, replace=False)
            relevant_ids = [items_by_topic[tid][int(j)].id for j in rel_choices]
            q_emb = sample_query_embedding(centers[tid], self.query_noise, rng)
            warmup_queries.append(SyntheticQuery(
                id=f"rn_warm_q{t}",
                embedding=q_emb,
                relevant_item_ids=relevant_ids,
                timestep=t,
                topic_id=tid,
            ))
            it.created_at = t

        # Combined sequence: warmup (stream + warmup query) then evaluation
        # (no stream + eval query).  We still store warmup hits so the
        # runner skips scoring them via `n_warmup_steps` field.
        item_stream: List[Optional[SyntheticItem]] = list(warmup_items) + [None] * self.query_length
        all_queries: List[SyntheticQuery] = warmup_queries + eval_queries

        bundle = WorkloadBundle(
            items_pool=items_pool,
            item_stream=item_stream,
            queries=all_queries,
            workload_type="retrieval_noise_stream",
            params=dict(
                mode="stream",
                n_topics=self.n_topics,
                items_per_topic=self.items_per_topic,
                intra_topic_std=self.intra_topic_std,
                inter_topic_sep=self.inter_topic_sep,
                query_length=self.query_length,
                relevant_per_query=self.relevant_per_query,
                dim=self.dim,
                query_noise=self.query_noise,
                seed=self.seed,
                total_pool_size=total_items,
                num_unique_gt_items=len(gt_items),
                n_warmup_steps=len(warmup_items),
            ),
        )
        # Tell the runner to skip the first N steps when aggregating metrics.
        bundle.n_warmup_steps = len(warmup_items)  # type: ignore[attr-defined]
        return bundle

    def generate(self) -> WorkloadBundle:
        if self.mode == "static":
            return self._generate_static()
        return self._generate_stream()
