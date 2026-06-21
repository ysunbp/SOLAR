"""
SyntheticRunner family
======================

Cache simulators that mirror the admission / eviction logic of the
real-benchmark agents in `src/agent/`, but operate purely in numpy
embedding space.

Methods:
  - fifo
  - lru
  - lfu
  - arc
  - thompson
  - lmu
  - lmu_ts

For every method, a single class:

  runner = make_runner(method, capacity=K, retrieve_k=k, seed=...)
  result = runner.run(workload_bundle)

Result contract (matches `experience_eval_results.json`-style schema):

  {
    "workload_type": ...,
    "workload_params": {...},
    "method": ...,
    "capacity": K,
    "retrieve_k": k,
    "seed": ...,
    "num_steps": T,
    "avg_hit_rate": ...,
    "first_half_hit_rate": ...,
    "second_half_hit_rate": ...,
    "learning_slope": ...,
    "store_rate": ...,
    "avg_f1": ...,
    "per_step_hits": [0/1, ...],
    "per_step_f1": [...],
  }
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .primitives import SyntheticItem, SyntheticQuery
from .workloads import WorkloadBundle


# ---------------------------------------------------------------------------
# Cache entry book-keeping
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    item: SyntheticItem
    inserted_at: int           # step when admitted
    last_accessed: int         # step of last retrieval hit
    access_count: int = 0
    # Thompson-specific
    alpha: float = 2.0
    beta: float = 1.0


# ---------------------------------------------------------------------------
# Base runner: shared retrieval / metric logic
# ---------------------------------------------------------------------------

class BaseRunner:
    """
    Common machinery: retrieval (cosine top-k), per-step metric aggregation,
    final result packaging.  Sub-classes plug in admission / eviction hooks.

    Embeddings are unit-normalised (workloads guarantee this), so cosine
    similarity reduces to a dot-product → matrix multiply.
    """

    name: str = "base"

    def __init__(
        self,
        capacity: int,
        retrieve_k: int = 3,
        seed: int = 42,
    ):
        self.capacity = capacity
        self.retrieve_k = retrieve_k
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Cache: dict id -> CacheEntry (insertion-ordered)
        self.cache: "OrderedDict[str, CacheEntry]" = OrderedDict()

        # Stats
        self.num_admissions = 0
        self.num_evictions = 0
        self.num_steps = 0

    # ---- Hooks for subclasses ---------------------------------------------

    def should_admit(self, item: SyntheticItem, step: int) -> bool:
        """Default: admit everything (greedy)."""
        return True

    def evict_one(self, incoming: SyntheticItem, step: int) -> None:
        """Default: FIFO."""
        oldest_id = next(iter(self.cache))
        self.cache.pop(oldest_id)
        self.num_evictions += 1

    def on_retrieval(
        self,
        query: SyntheticQuery,
        retrieved_ids: List[str],
        step: int,
    ) -> None:
        """Hook: update access stats after retrieval (LRU/LFU/TS use this)."""
        for rid in retrieved_ids:
            entry = self.cache.get(rid)
            if entry is None:
                continue
            entry.last_accessed = step
            entry.access_count += 1

    # ---- Core step --------------------------------------------------------

    def _admit(self, item: SyntheticItem, step: int) -> None:
        if not self.should_admit(item, step):
            return
        if item.id in self.cache:
            return
        # Evict to make room
        while len(self.cache) >= self.capacity:
            self.evict_one(item, step)
        # Insert (preserve insertion order)
        self.cache[item.id] = CacheEntry(
            item=item,
            inserted_at=step,
            last_accessed=step,
            access_count=0,
        )
        self.num_admissions += 1

    def _retrieve(self, query: SyntheticQuery) -> List[str]:
        if not self.cache:
            return []
        ids = list(self.cache.keys())
        embs = np.stack([self.cache[i].item.embedding for i in ids], axis=0)
        sims = embs @ query.embedding  # shape (n,)
        k = min(self.retrieve_k, len(ids))
        # argpartition for top-k, then sort those k
        topk_idx = np.argpartition(-sims, k - 1)[:k]
        topk_idx = topk_idx[np.argsort(-sims[topk_idx])]
        return [ids[i] for i in topk_idx]

    def step(
        self,
        new_item: Optional[SyntheticItem],
        query: SyntheticQuery,
        t: int,
    ) -> Dict:
        """
        Online-cache semantics:
          1. Retrieve top-k for `query` from the *current* cache state
             (the new item arriving at this step is NOT yet in cache).
          2. Compute hit / F1 against ground truth.
          3. Update access stats / posteriors.
          4. Admit `new_item` (admission decisions may use the query's
             retrieval feedback indirectly via cache statistics).
        This means cycling/drift correctly probe the cache's *retention*
        ability rather than trivially hitting the just-inserted item.
        """
        retrieved = self._retrieve(query)

        gt_set = set(query.relevant_item_ids)
        ret_set = set(retrieved)
        hit = 1 if (gt_set & ret_set) else 0

        # Set-level F1 over IDs
        if not retrieved or not gt_set:
            f1 = 0.0
            precision_at_k = 0.0
            recall_at_k = 0.0
        else:
            tp = len(gt_set & ret_set)
            precision_at_k = tp / max(len(ret_set), 1)
            recall_at_k = tp / max(len(gt_set), 1)
            f1 = (
                0.0 if (precision_at_k + recall_at_k) == 0
                else 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
            )

        self.on_retrieval(query, retrieved, t)

        if new_item is not None:
            self._admit(new_item, t)

        self.num_steps += 1

        return {
            "hit": hit,
            "f1": f1,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "cache_size": len(self.cache),
        }

    # ---- Run --------------------------------------------------------------

    def run(self, bundle: WorkloadBundle) -> Dict:
        per_hits: List[int] = []
        per_f1: List[float] = []
        per_precision: List[float] = []
        per_recall: List[float] = []
        per_cache_size: List[int] = []

        # Optional prefill phase: items pushed into the cache BEFORE step 0
        # (used by the retrieval-noise workload to fill the cache up to K
        # before queries start being scored).  Prefill items go through
        # admission like any other item but are not paired with queries.
        prefill = getattr(bundle, "prefill_items", None)
        if prefill:
            for pf_t, item in enumerate(prefill):
                if item is not None:
                    # Use negative timesteps so on_retrieval bookkeeping
                    # remains monotonic; pass step=pf_t - len(prefill).
                    self._admit(item, pf_t - len(prefill))

        for t in range(len(bundle.queries)):
            new_item = bundle.item_stream[t] if t < len(bundle.item_stream) else None
            query = bundle.queries[t]
            r = self.step(new_item, query, t)
            per_hits.append(r["hit"])
            per_f1.append(r["f1"])
            per_precision.append(r["precision_at_k"])
            per_recall.append(r["recall_at_k"])
            per_cache_size.append(r["cache_size"])

        T = len(per_hits)
        if T == 0:
            return {}

        # Optional warmup-step skipping: the first `n_warmup_steps` queries
        # are not part of the evaluation horizon (used by stream-mode
        # retrieval-noise workload).  Aggregate metrics over [n_warmup:T).
        n_warm = int(getattr(bundle, "n_warmup_steps", 0) or 0)
        n_warm = min(max(n_warm, 0), T)
        eval_hits = per_hits[n_warm:]
        eval_f1 = per_f1[n_warm:]
        eval_precision = per_precision[n_warm:]
        eval_recall = per_recall[n_warm:]
        eval_cache_size = per_cache_size[n_warm:]
        T_eval = len(eval_hits)
        if T_eval == 0:
            # Fallback: nothing to evaluate (shouldn't happen in practice)
            eval_hits = per_hits
            eval_f1 = per_f1
            eval_precision = per_precision
            eval_recall = per_recall
            eval_cache_size = per_cache_size
            T_eval = T

        half = T_eval // 2
        avg_hit = float(np.mean(eval_hits))
        first_half = float(np.mean(eval_hits[:half])) if half > 0 else avg_hit
        second_half = float(np.mean(eval_hits[half:])) if T_eval - half > 0 else avg_hit

        # Linear regression slope of hit rate ~ time (rolling mean window)
        win = max(20, T_eval // 25)
        if T_eval >= win:
            rolling = np.convolve(eval_hits, np.ones(win) / win, mode="valid")
        else:
            rolling = np.asarray(eval_hits, dtype=float)
        if len(rolling) >= 2:
            x = np.arange(len(rolling), dtype=float)
            slope, _ = np.polyfit(x, rolling, 1)
        else:
            slope = 0.0

        # store_rate = admissions / candidates_seen
        # Number of candidates seen = number of non-None items in the stream up to T
        seen = sum(1 for it in bundle.item_stream[:T] if it is not None)
        if prefill:
            seen += sum(1 for it in prefill if it is not None)
        store_rate = (self.num_admissions / seen) if seen > 0 else 0.0

        return {
            "workload_type": bundle.workload_type,
            "workload_params": bundle.params,
            "method": self.name,
            "capacity": self.capacity,
            "retrieve_k": self.retrieve_k,
            "seed": self.seed,
            "num_steps": T_eval,
            "num_total_steps": T,
            "n_warmup_steps": n_warm,
            "avg_hit_rate": avg_hit,
            "first_half_hit_rate": first_half,
            "second_half_hit_rate": second_half,
            "learning_slope": float(slope),
            "store_rate": float(store_rate),
            "avg_f1": float(np.mean(eval_f1)),
            "num_admissions": self.num_admissions,
            "num_evictions": self.num_evictions,
            "final_cache_size": len(self.cache),
            "avg_precision_at_k": float(np.mean(eval_precision)) if len(eval_precision) else 0.0,
            "avg_recall_at_k": float(np.mean(eval_recall)) if len(eval_recall) else 0.0,
            "avg_cache_size": float(np.mean(eval_cache_size)) if len(eval_cache_size) else 0.0,
            "per_step_hits": eval_hits,
            "per_step_f1": [float(x) for x in eval_f1],
            "per_step_precision": [float(x) for x in eval_precision],
        }


# ---------------------------------------------------------------------------
# FIFO
# ---------------------------------------------------------------------------

class FIFORunner(BaseRunner):
    name = "fifo"
    # default `evict_one` is FIFO; default `should_admit` admits all.


# ---------------------------------------------------------------------------
# LRU
# ---------------------------------------------------------------------------

class LRURunner(BaseRunner):
    name = "lru"

    def evict_one(self, incoming, step):
        # Evict entry with smallest last_accessed
        victim = min(self.cache.items(), key=lambda kv: kv[1].last_accessed)[0]
        self.cache.pop(victim)
        self.num_evictions += 1


# ---------------------------------------------------------------------------
# LFU
# ---------------------------------------------------------------------------

class LFURunner(BaseRunner):
    name = "lfu"

    def evict_one(self, incoming, step):
        # Lowest access_count, ties → oldest insertion
        victim = min(
            self.cache.items(),
            key=lambda kv: (kv[1].access_count, kv[1].inserted_at),
        )[0]
        self.cache.pop(victim)
        self.num_evictions += 1


# ---------------------------------------------------------------------------
# ARC (simplified, capacity-bounded)
# ---------------------------------------------------------------------------

class ARCRunner(BaseRunner):
    """
    Adaptive Replacement Cache.
    Maintains four ordered sets: T1, T2 (active) and B1, B2 (ghosts).
    Adapts target T1 size `p` based on ghost-list hits.
    """
    name = "arc"

    def __init__(self, capacity: int, retrieve_k: int = 3, seed: int = 42):
        super().__init__(capacity, retrieve_k, seed)
        self._t1: "OrderedDict[str, bool]" = OrderedDict()
        self._t2: "OrderedDict[str, bool]" = OrderedDict()
        self._b1: "OrderedDict[str, bool]" = OrderedDict()
        self._b2: "OrderedDict[str, bool]" = OrderedDict()
        self._p: float = 0.0

    def _replace(self, in_b2: bool):
        c = self.capacity
        if self._t1 and (
            len(self._t1) > self._p
            or (in_b2 and len(self._t1) == int(self._p))
        ):
            victim, _ = self._t1.popitem(last=False)
            self._b1[victim] = True
            if len(self._b1) > c:
                self._b1.popitem(last=False)
        else:
            if self._t2:
                victim, _ = self._t2.popitem(last=False)
                self._b2[victim] = True
                if len(self._b2) > c:
                    self._b2.popitem(last=False)
            elif self._t1:
                victim, _ = self._t1.popitem(last=False)
                self._b1[victim] = True
                if len(self._b1) > c:
                    self._b1.popitem(last=False)
            else:
                return
        self.cache.pop(victim, None)
        self.num_evictions += 1

    def evict_one(self, incoming, step):
        # ARC handles eviction inside _replace; keep a simple wrapper.
        self._replace(in_b2=False)

    def _insert_active(self, item, step, into_t2: bool):
        """Insert into cache + T1/T2; assumes capacity already enforced."""
        self.cache[item.id] = CacheEntry(
            item=item, inserted_at=step, last_accessed=step,
        )
        self.num_admissions += 1
        if into_t2:
            self._t2[item.id] = True
        else:
            self._t1[item.id] = True

    def _admit(self, item, step):
        if item.id in self.cache:
            # Already present: treat like an access (T1→T2 promote)
            if item.id in self._t1:
                del self._t1[item.id]
                self._t2[item.id] = True
            elif item.id in self._t2:
                self._t2.move_to_end(item.id)
            return

        # Ghost-hit on insert? Remove ghost FIRST so _replace cannot
        # accidentally re-insert it.
        if item.id in self._b1:
            delta = max(1.0, len(self._b2) / max(len(self._b1), 1))
            self._p = min(self._p + delta, float(self.capacity))
            del self._b1[item.id]
            if len(self.cache) >= self.capacity:
                self._replace(in_b2=False)
            self._insert_active(item, step, into_t2=True)
            return
        if item.id in self._b2:
            delta = max(1.0, len(self._b1) / max(len(self._b2), 1))
            self._p = max(self._p - delta, 0.0)
            del self._b2[item.id]
            if len(self.cache) >= self.capacity:
                self._replace(in_b2=True)
            self._insert_active(item, step, into_t2=True)
            return

        # Plain new item — evict if needed, then insert into T1
        while len(self.cache) >= self.capacity:
            self._replace(in_b2=False)
        self._insert_active(item, step, into_t2=False)

    def on_retrieval(self, query, retrieved_ids, step):
        super().on_retrieval(query, retrieved_ids, step)
        for rid in retrieved_ids:
            if rid in self._t1:
                # Promote: T1 → T2
                del self._t1[rid]
                self._t2[rid] = True
            elif rid in self._t2:
                self._t2.move_to_end(rid)


# ---------------------------------------------------------------------------
# Thompson Sampling
# ---------------------------------------------------------------------------

class SolarERunner(BaseRunner):
    """
    Always admits; eviction uses Beta(α, β) Thompson sampling.

    - Optimistic prior α=2, β=1
    - on hit: α += 1
    - every 5 retrieve calls: β += 0.05 for non-retrieved entries
    - eviction score = beta_sample + novelty_bonus * novelty
        novelty = 1 - exp(-avg_dist_to_3nn / (2 * dim))
    """

    name = "solar_e"

    def __init__(
        self,
        capacity: int,
        retrieve_k: int = 3,
        seed: int = 42,
        optimistic_prior: bool = True,
        novelty_bonus: float = 0.3,
        beta_decay_every: int = 5,
        beta_decay_amount: float = 0.05,
    ):
        super().__init__(capacity, retrieve_k, seed)
        self.optimistic_prior = optimistic_prior
        self.novelty_bonus = novelty_bonus
        self.beta_decay_every = beta_decay_every
        self.beta_decay_amount = beta_decay_amount
        self._retrieve_step = 0

    def _init_entry(self, entry: CacheEntry):
        if self.optimistic_prior:
            entry.alpha, entry.beta = 2.0, 1.0
        else:
            entry.alpha, entry.beta = 1.0, 1.0

    def _admit(self, item, step):
        if item.id in self.cache:
            return
        while len(self.cache) >= self.capacity:
            self.evict_one(item, step)
        entry = CacheEntry(
            item=item, inserted_at=step, last_accessed=step,
        )
        self._init_entry(entry)
        self.cache[item.id] = entry
        self.num_admissions += 1

    def evict_one(self, incoming, step):
        if not self.cache:
            return
        ids = list(self.cache.keys())
        embs = np.stack([self.cache[i].item.embedding for i in ids], axis=0)

        # Novelty: avg cosine distance to 3 nearest neighbours in the cache
        # (cosine distance = 1 - cos_sim)
        n = len(ids)
        scores = []
        # Pre-compute pairwise sim
        sim = embs @ embs.T  # (n, n)
        np.fill_diagonal(sim, -np.inf)  # exclude self
        k_nn = min(3, n - 1) if n > 1 else 0
        for j, i in enumerate(ids):
            entry = self.cache[i]
            ts = float(self.rng.beta(entry.alpha, entry.beta))
            if k_nn > 0:
                topk_sim = np.partition(sim[j], -k_nn)[-k_nn:]
                avg_sim = float(np.mean(topk_sim))
                novelty = float(np.clip(1.0 - avg_sim, 0.0, 1.0))
            else:
                novelty = 0.5
            scores.append(ts + self.novelty_bonus * novelty)
        victim_idx = int(np.argmin(scores))
        self.cache.pop(ids[victim_idx])
        self.num_evictions += 1

    def on_retrieval(self, query, retrieved_ids, step):
        super().on_retrieval(query, retrieved_ids, step)
        retrieved_set = set(retrieved_ids)
        for rid in retrieved_set:
            entry = self.cache.get(rid)
            if entry is not None:
                entry.alpha += 1.0

        self._retrieve_step += 1
        if self._retrieve_step % self.beta_decay_every == 0:
            for cid, entry in self.cache.items():
                if cid not in retrieved_set:
                    entry.beta += self.beta_decay_amount


# ---------------------------------------------------------------------------
# LMU (admission via accumulated regret; smart eviction by recency+freq+div)
# ---------------------------------------------------------------------------

class SolarARunner(BaseRunner):
    """
    Admission: store iff accumulated_regret > τ (adaptive).
    Eviction: smart score = 0.4*recency + 0.3*freq + 0.3*diversity.
    """

    name = "solar_a"

    def __init__(
        self,
        capacity: int,
        retrieve_k: int = 3,
        seed: int = 42,
        lambda_cost: float = 5.0,
        lipschitz_L: float = 2.0,
        adaptive_window: int = 30,
        min_threshold: float = 0.1,
        max_threshold: float = 5.0,
        novelty_weight: float = 0.6,
        density_weight: float = 0.4,
    ):
        super().__init__(capacity, retrieve_k, seed)
        self.lambda_cost = lambda_cost
        self.lipschitz_L = lipschitz_L
        self.adaptive_window = adaptive_window
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.novelty_weight = novelty_weight
        self.density_weight = density_weight

        self.threshold = math.sqrt(2 * lambda_cost / lipschitz_L)
        self.accumulated_regret = 0.0
        self.novelty_history: List[float] = []
        self._L_hat_ema: Optional[float] = None
        self._candidates_seen = 0

    # ---- Admission helpers ------------------------------------------------

    def _adapt_threshold(self):
        if len(self.novelty_history) < self.adaptive_window:
            return
        recent = self.novelty_history[-self.adaptive_window:]
        new_L = max(float(np.std(recent)) * 2, 0.1)
        if self._L_hat_ema is None:
            self._L_hat_ema = new_L
        else:
            self._L_hat_ema = 0.7 * self._L_hat_ema + 0.3 * new_L
        new_thr = math.sqrt(2 * self.lambda_cost / self._L_hat_ema)
        self.threshold = float(np.clip(new_thr, self.min_threshold, self.max_threshold))

    def _info_gain(self, item_emb: np.ndarray) -> float:
        """
        Pure-cosine variant of LMU info-gain (synthetic space, unit-norm).
          novelty  = 1 - max_cos_sim     ∈ [0, 2]
          density  = max_cos_sim         ∈ [-1, 1]
        These are well-scaled (no dim-dependent dampening), so the regret
        accumulator builds up at a sensible rate against τ.
        """
        if not self.cache:
            return 1.0
        ids = list(self.cache.keys())
        embs = np.stack([self.cache[i].item.embedding for i in ids], axis=0)
        sims = embs @ item_emb            # cosine since unit-norm
        max_sim = float(np.max(sims))
        novelty = float(np.clip(1.0 - max_sim, 0.0, 1.0))

        # Density penalty: average top-k cosine sim (high sim → high density).
        k = min(5, len(ids))
        topk = np.partition(sims, -k)[-k:]
        density = float(np.clip(np.mean(topk), 0.0, 1.0))

        ig = self.novelty_weight * novelty - self.density_weight * density
        return max(ig, 0.0)

    def should_admit(self, item, step):
        self._candidates_seen += 1
        ig = self._info_gain(item.embedding)
        self.novelty_history.append(ig)
        self.accumulated_regret += ig
        self._adapt_threshold()

        bootstrap_n = min(max(self.capacity // 2, 5), 25)
        if len(self.cache) < bootstrap_n:
            self.accumulated_regret = 0.0
            return True
        if self.accumulated_regret > self.threshold:
            self.accumulated_regret = 0.0
            return True
        return False

    # ---- Eviction ---------------------------------------------------------

    def evict_one(self, incoming, step):
        if not self.cache:
            return
        ids = list(self.cache.keys())
        embs = np.stack([self.cache[i].item.embedding for i in ids], axis=0)
        sim = embs @ embs.T
        np.fill_diagonal(sim, -np.inf)
        n = len(ids)

        scores = []
        total_cands = max(self._candidates_seen, 1)
        for j, i in enumerate(ids):
            entry = self.cache[i]
            age = step - entry.last_accessed
            recency = math.exp(-age / 100.0)
            freq = entry.access_count / total_cands
            if n > 1:
                nn_sim = float(np.max(sim[j]))
                diversity = float(np.clip(1.0 - nn_sim, 0.0, 1.0))
            else:
                diversity = 1.0
            value = 0.4 * recency + 0.3 * freq + 0.3 * diversity
            scores.append(value)
        victim_idx = int(np.argmin(scores))
        self.cache.pop(ids[victim_idx])
        self.num_evictions += 1


# ---------------------------------------------------------------------------
# LMU + Thompson (combined)
# ---------------------------------------------------------------------------

class SolarRunner(SolarARunner):
    """
    Admission: LMU regret > τ.
    Eviction: Thompson sampling (with novelty bonus).
    """

    name = "solar"

    def __init__(
        self,
        capacity: int,
        retrieve_k: int = 3,
        seed: int = 42,
        # LMU params
        lambda_cost: float = 5.0,
        lipschitz_L: float = 2.0,
        adaptive_window: int = 30,
        min_threshold: float = 0.1,
        max_threshold: float = 5.0,
        novelty_weight: float = 0.6,
        density_weight: float = 0.4,
        # TS params
        optimistic_prior: bool = True,
        ts_novelty_bonus: float = 0.3,
        beta_decay_every: int = 5,
        beta_decay_amount: float = 0.05,
    ):
        super().__init__(
            capacity, retrieve_k, seed,
            lambda_cost=lambda_cost,
            lipschitz_L=lipschitz_L,
            adaptive_window=adaptive_window,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            novelty_weight=novelty_weight,
            density_weight=density_weight,
        )
        self.optimistic_prior = optimistic_prior
        self.ts_novelty_bonus = ts_novelty_bonus
        self.beta_decay_every = beta_decay_every
        self.beta_decay_amount = beta_decay_amount
        self._retrieve_step = 0

    def _init_entry(self, entry: CacheEntry):
        if self.optimistic_prior:
            entry.alpha, entry.beta = 2.0, 1.0
        else:
            entry.alpha, entry.beta = 1.0, 1.0

    def _admit(self, item, step):
        # Use LMU admission gate
        if not self.should_admit(item, step):
            return
        if item.id in self.cache:
            return
        while len(self.cache) >= self.capacity:
            self.evict_one(item, step)
        entry = CacheEntry(
            item=item, inserted_at=step, last_accessed=step,
        )
        self._init_entry(entry)
        self.cache[item.id] = entry
        self.num_admissions += 1

    def evict_one(self, incoming, step):
        if not self.cache:
            return
        ids = list(self.cache.keys())
        embs = np.stack([self.cache[i].item.embedding for i in ids], axis=0)
        sim = embs @ embs.T
        np.fill_diagonal(sim, -np.inf)
        n = len(ids)
        k_nn = min(3, n - 1) if n > 1 else 0

        scores = []
        for j, i in enumerate(ids):
            entry = self.cache[i]
            ts = float(self.rng.beta(entry.alpha, entry.beta))
            if k_nn > 0:
                topk_sim = np.partition(sim[j], -k_nn)[-k_nn:]
                avg_sim = float(np.mean(topk_sim))
                novelty = float(np.clip(1.0 - avg_sim, 0.0, 1.0))
            else:
                novelty = 0.5
            scores.append(ts + self.ts_novelty_bonus * novelty)
        victim_idx = int(np.argmin(scores))
        self.cache.pop(ids[victim_idx])
        self.num_evictions += 1

    def on_retrieval(self, query, retrieved_ids, step):
        # LMU/LRU-style access bookkeeping
        super().on_retrieval(query, retrieved_ids, step)
        retrieved_set = set(retrieved_ids)
        for rid in retrieved_set:
            entry = self.cache.get(rid)
            if entry is not None:
                entry.alpha += 1.0
        self._retrieve_step += 1
        if self._retrieve_step % self.beta_decay_every == 0:
            for cid, entry in self.cache.items():
                if cid not in retrieved_set:
                    entry.beta += self.beta_decay_amount


# ---------------------------------------------------------------------------
# EmbedderUnlimited (no eviction; for retrieval-noise comparison)
# ---------------------------------------------------------------------------

class EmbedderUnlimitedRunner(BaseRunner):
    """
    "Store everything, no eviction" baseline.  The cache size grows
    monotonically; capacity is treated as a soft cap (only enforced if
    the user passes a finite K).  Used as a control in the retrieval-noise
    experiment: it isolates the effect of pool size on retrieval precision
    independent of any eviction policy.
    """
    name = "embedder_unlimited"

    def __init__(self, capacity: int = 10**9, retrieve_k: int = 3, seed: int = 42):
        # Effectively no capacity bound by default.
        super().__init__(capacity=capacity, retrieve_k=retrieve_k, seed=seed)

    def evict_one(self, incoming, step):
        # Never evicts.  If cache is somehow over capacity, fall back to FIFO.
        if len(self.cache) >= self.capacity:
            oldest_id = next(iter(self.cache))
            self.cache.pop(oldest_id)
            self.num_evictions += 1


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_RUNNER_REGISTRY = {
    "fifo": FIFORunner,
    "lru": LRURunner,
    "lfu": LFURunner,
    "arc": ARCRunner,
    "solar_e": SolarERunner,
    "solar_a": SolarARunner,
    "solar": SolarRunner,
    "embedder_unlimited": EmbedderUnlimitedRunner,
}

ALL_METHODS = list(_RUNNER_REGISTRY.keys())


def make_runner(method: str, capacity: int, retrieve_k: int = 3, seed: int = 42) -> BaseRunner:
    if method not in _RUNNER_REGISTRY:
        raise ValueError(f"unknown method: {method!r}; valid: {ALL_METHODS}")
    return _RUNNER_REGISTRY[method](capacity=capacity, retrieve_k=retrieve_k, seed=seed)
