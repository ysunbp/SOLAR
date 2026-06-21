"""
SOLAR Agent

Combines regret-gated selective storage (only store when regret > threshold)
with Thompson Sampling's intelligent eviction (evict lowest-value entry).

This is the "best of both worlds":
- Regret-gated admission decides WHEN to store (reduces memory churn)
- Posterior-guided selection decides WHAT to evict (preserves high-value entries)

Key advantages:
- vs Greedy: doesn't store everything → less noise in memory
- vs FIFO: doesn't blindly evict oldest → preserves important early facts
- vs admission-only (SOLAR-A): smarter eviction when memory is full
- vs eviction-only (SOLAR-E): doesn't store trivial turns → better signal-to-noise
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field

from src.agent.embedder import EmbedderAgent, EmbedderAgentConfig


class SolarAgentConfig(EmbedderAgentConfig):
    """Configuration for SOLAR Agent."""
    # Admission (timing) parameters (when to store)
    lambda_cost: float = Field(
        default=5.0,
        description="Switching cost coefficient — higher means fewer updates"
    )
    lipschitz_L: float = Field(
        default=2.0,
        description="Estimated Lipschitz constant"
    )
    threshold_mode: str = Field(
        default="adaptive",
        description="Threshold mode: 'fixed' or 'adaptive'"
    )
    adaptive_window: int = Field(
        default=30,
        description="Window size for adaptive threshold estimation"
    )
    min_threshold: float = Field(
        default=0.1,
        description="Minimum threshold value"
    )
    max_threshold: float = Field(
        default=5.0,
        description="Maximum threshold value"
    )
    # Capacity and eviction parameters
    capacity: int = Field(
        default=500,
        description="Maximum number of memory entries"
    )
    novelty_weight: float = Field(
        default=0.6,
        description="Weight for novelty in info gain"
    )
    density_weight: float = Field(
        default=0.4,
        description="Weight for density penalty in info gain"
    )
    # Eviction (selection) parameters (what to evict)
    optimistic_prior: bool = Field(
        default=True,
        description="Use optimistic prior for Thompson Sampling"
    )
    ts_novelty_bonus: float = Field(
        default=0.3,
        description="Novelty bonus weight in eviction scoring"
    )


class SolarAgent(EmbedderAgent):
    """
    SOLAR Agent.

    - Storage decision: regret-gated admission (accumulated regret > threshold)
    - Eviction decision: Thompson Sampling (lowest sampled value)
    """

    def __init__(self, config: SolarAgentConfig = SolarAgentConfig()):
        super().__init__(config)
        self.solar_config = config

        # Admission state
        self.accumulated_regret: float = 0.0
        self.threshold: float = self._compute_initial_threshold()
        self.novelty_history: List[float] = []
        self.total_candidates: int = 0
        self.total_updates: int = 0
        self.total_evictions: int = 0
        self._active_count: int = 0  # maintained incrementally
        self._L_hat_ema: Optional[float] = None  # EMA-smoothed Lipschitz estimate

        # Eviction state
        self._alpha: Dict[int, float] = {}
        self._beta: Dict[int, float] = {}
        self._last_accessed: Dict[int, int] = {}
        self._retrieve_step: int = 0  # monotonic counter for batched beta decay

    # ------------------------------------------------------------------
    # Admission: threshold computation
    # ------------------------------------------------------------------

    def _compute_initial_threshold(self) -> float:
        return np.sqrt(2 * self.solar_config.lambda_cost / self.solar_config.lipschitz_L)

    def _adapt_threshold(self):
        window = self.solar_config.adaptive_window
        if len(self.novelty_history) < window:
            return
        recent = self.novelty_history[-window:]
        new_L_hat = max(np.std(recent) * 2, 0.1)
        # EMA smoothing to avoid threshold oscillation
        if self._L_hat_ema is None:
            self._L_hat_ema = new_L_hat
        else:
            self._L_hat_ema = 0.7 * self._L_hat_ema + 0.3 * new_L_hat
        new_threshold = np.sqrt(2 * self.solar_config.lambda_cost / self._L_hat_ema)
        self.threshold = float(np.clip(
            new_threshold,
            self.solar_config.min_threshold,
            self.solar_config.max_threshold,
        ))

    # ------------------------------------------------------------------
    # Admission: novelty / info gain
    # ------------------------------------------------------------------

    def _compute_novelty(self, candidate_vec: np.ndarray) -> float:
        if self.index.ntotal == 0:
            return 1.0
        candidate_vec = candidate_vec.reshape(1, -1)
        D, _ = self.index.search(candidate_vec, 1)
        l2_dist = float(D[0][0])
        novelty = 1.0 - np.exp(-l2_dist / (2 * self.solar_config.embedding_dim))
        return float(np.clip(novelty, 0.0, 1.0))

    def _compute_info_gain(self, candidate_vec: np.ndarray) -> float:
        novelty = self._compute_novelty(candidate_vec)
        density_penalty = 0.0
        if self.index.ntotal > 0:
            k = min(5, self.index.ntotal)
            candidate_vec_2d = candidate_vec.reshape(1, -1)
            D, _ = self.index.search(candidate_vec_2d, k)
            avg_dist = float(np.mean(D[0]))
            density_penalty = np.exp(-avg_dist / (2 * self.solar_config.embedding_dim))
        ig = (self.solar_config.novelty_weight * novelty
              - self.solar_config.density_weight * density_penalty)
        return max(ig, 0.0)

    # ------------------------------------------------------------------
    # LMU: should store decision
    # ------------------------------------------------------------------

    def _should_store(self, candidate_vec: np.ndarray) -> bool:
        self.total_candidates += 1
        ig = self._compute_info_gain(candidate_vec)
        self.novelty_history.append(ig)
        self.accumulated_regret += ig

        if self.solar_config.threshold_mode == "adaptive":
            self._adapt_threshold()

        # Bootstrap: always store while memory is still warming up.
        # Use _active_count (not index.ntotal) so compaction doesn't re-trigger.
        bootstrap_n = min(max(self.solar_config.capacity // 2, 5), 25)
        if self._active_count < bootstrap_n:
            self.accumulated_regret = 0.0
            self.total_updates += 1
            return True

        if self.accumulated_regret > self.threshold:
            self.accumulated_regret = 0.0
            self.total_updates += 1
            return True

        return False

    # ------------------------------------------------------------------
    # Thompson: eviction
    # ------------------------------------------------------------------

    def _get_active_indices(self) -> List[int]:
        return [i for i, m in enumerate(self.metadata) if not m.get("deleted", False)]

    def _get_active_count(self) -> int:
        """Return cached active count (O(1))."""
        return self._active_count

    def _init_entry_stats(self, idx: int):
        if self.solar_config.optimistic_prior:
            self._alpha[idx] = 2.0
            self._beta[idx] = 1.0
        else:
            self._alpha[idx] = 1.0
            self._beta[idx] = 1.0
        self._last_accessed[idx] = self.total_candidates

    def _sample_value(self, idx: int) -> float:
        alpha = self._alpha.get(idx, 1.0)
        beta = self._beta.get(idx, 1.0)
        return float(np.random.beta(alpha, beta))

    def _evict_thompson(self, new_vec: np.ndarray):
        """Thompson Sampling eviction with novelty bonus.
        Reads vectors from FAISS index directly (no re-embed)."""
        active_indices = self._get_active_indices()
        if not active_indices:
            return

        scores = []
        for idx in active_indices:
            ts_value = self._sample_value(idx)
            try:
                vec = self.index.reconstruct(idx).reshape(1, -1)
                k = min(3, self.index.ntotal)
                D, _ = self.index.search(vec, k)
                avg_dist = float(np.mean(D[0][1:])) if len(D[0]) > 1 else float(D[0][0])
                novelty = 1.0 - np.exp(-avg_dist / (2 * self.solar_config.embedding_dim))
            except Exception:
                novelty = 0.5
            combined = ts_value + self.solar_config.ts_novelty_bonus * novelty
            scores.append((idx, combined))

        scores.sort(key=lambda x: x[1])
        evict_idx = scores[0][0]
        self.metadata[evict_idx]["deleted"] = True
        self._active_count -= 1
        self.total_evictions += 1

        # Clean up
        self._alpha.pop(evict_idx, None)
        self._beta.pop(evict_idx, None)
        self._last_accessed.pop(evict_idx, None)

        # Periodic compaction to remove tombstones from FAISS index
        if self.total_evictions % 20 == 0:
            self._compact_index()

    def _compact_index(self):
        """Rebuild FAISS index and remap Thompson stats to new indices."""
        import faiss as _faiss
        old_to_new = {}
        valid_vectors = []
        valid_metadata = []
        new_alpha = {}
        new_beta = {}
        new_last_accessed = {}

        for old_idx, meta in enumerate(self.metadata):
            if not meta.get("deleted", False):
                new_idx = len(valid_metadata)
                old_to_new[old_idx] = new_idx
                try:
                    vec = self.index.reconstruct(old_idx)
                    valid_vectors.append(vec)
                except Exception:
                    valid_vectors.append(self._embed(meta["content"]))
                valid_metadata.append(meta)
                # Remap Thompson stats
                if old_idx in self._alpha:
                    new_alpha[new_idx] = self._alpha[old_idx]
                if old_idx in self._beta:
                    new_beta[new_idx] = self._beta[old_idx]
                if old_idx in self._last_accessed:
                    new_last_accessed[new_idx] = self._last_accessed[old_idx]

        # Rebuild
        if valid_vectors:
            self.index = _faiss.IndexFlatL2(self.config.embedding_dim)
            self.index.add(np.array(valid_vectors, dtype=np.float32))
        else:
            self.index = _faiss.IndexFlatL2(self.config.embedding_dim)

        self.metadata = valid_metadata
        self._alpha = new_alpha
        self._beta = new_beta
        self._last_accessed = new_last_accessed
        self._active_count = len(valid_metadata)

    def add_memory_solar(self, content: str, doc_id=None) -> bool:
        """
        LMU decides whether to store. If yes and memory is full,
        Thompson decides what to evict.
        """
        vector = self._embed(content)

        if not self._should_store(vector):
            return False

        # Evict if at capacity
        if self._active_count >= self.solar_config.capacity:
            self._evict_thompson(vector)

        # Store
        if doc_id is None:
            doc_id = f"lmu_ts_doc_{len(self.metadata)}"
        self.metadata.append({"doc_id": doc_id, "content": content})
        self.index.add(np.array([vector]))
        self._active_count += 1

        # Init Thompson stats
        new_idx = len(self.metadata) - 1
        self._init_entry_stats(new_idx)
        return True

    # ------------------------------------------------------------------
    # Experience-based LMU+TS gating (Exp 2: Experience Learning)
    # ------------------------------------------------------------------

    def add_memory_experience_gated(self, content: str, feedback_score: float, doc_id=None) -> bool:
        """
        Experience-based LMU+TS gating (Exp 2).
        
        Uses feedback score for regret accumulation (like LMU experience mode),
        but uses Thompson Sampling for eviction when memory is full.
        
        Args:
            content: The experience text
            feedback_score: Score in [0, 1]. Low = bad response = high regret.
            doc_id: Optional document ID.
        """
        self.total_candidates += 1
        
        # Cost = 1 - score: dislike → high cost → high regret
        cost = 1.0 - feedback_score
        self.accumulated_regret += cost
        self.novelty_history.append(cost)
        
        # Adapt threshold
        if self.solar_config.threshold_mode == "adaptive":
            self._adapt_threshold()
        
        # Bootstrap: always store while memory is still warming up.
        # Use _active_count to avoid re-triggering after compaction.
        bootstrap_n = min(max(self.solar_config.capacity // 2, 5), 25)
        if self._active_count < bootstrap_n:
            should_store = True
        else:
            should_store = self.accumulated_regret > self.threshold
        
        if not should_store:
            return False
        
        # Reset regret and store
        self.accumulated_regret = 0.0
        self.total_updates += 1
        
        vector = self._embed(content)
        
        # Evict if at capacity (using Thompson Sampling)
        if self._active_count >= self.solar_config.capacity:
            self._evict_thompson(vector)
        
        # Store
        if doc_id is None:
            doc_id = f"lmu_ts_exp_{len(self.metadata)}"
        self.metadata.append({"doc_id": doc_id, "content": content})
        self.index.add(np.array([vector]))
        self._active_count += 1
        
        # Init Thompson stats
        new_idx = len(self.metadata) - 1
        self._init_entry_stats(new_idx)
        return True

    # ------------------------------------------------------------------
    # Override: retrieve with Thompson update
    # ------------------------------------------------------------------

    def retrieve_memory(self, content: str, k=10) -> List[str]:
        """Retrieve + update Thompson distributions."""
        results = super().retrieve_memory(content, k)

        if results and self.index.ntotal > 0:
            query_vec = self._embed(content)
            query_vec_2d = query_vec.reshape(1, -1)
            search_k = min(k * 2, self.index.ntotal)
            D, I = self.index.search(query_vec_2d, search_k)

            retrieved_set = set()
            for idx in I[0]:
                if 0 <= idx < len(self.metadata) and not self.metadata[idx].get("deleted", False):
                    retrieved_set.add(idx)
                    self._alpha[idx] = self._alpha.get(idx, 1.0) + 1.0
                    self._last_accessed[idx] = self.total_candidates
                    if len(retrieved_set) >= k:
                        break

            # Batched beta decay: every 5 retrieve calls, smaller increment
            # to avoid penalising long-tail knowledge that is rarely queried.
            self._retrieve_step += 1
            if self._retrieve_step % 5 == 0:
                for idx in self._get_active_indices():
                    if idx not in retrieved_set:
                        self._beta[idx] = self._beta.get(idx, 1.0) + 0.05

        return results

    # ------------------------------------------------------------------
    # Override: add_conversation_to_memory
    # ------------------------------------------------------------------

    def add_conversation_to_memory(
        self,
        messages: List[Dict[str, str]],
        conversation_idx: Union[int, str] = 0,
    ):
        """Add conversation with LMU gating + Thompson eviction."""
        if isinstance(conversation_idx, int):
            conversation_idx = str(conversation_idx)
        stored = 0
        for msg_idx, msg in enumerate(messages):
            doc_id = f"conv_{conversation_idx}_{msg_idx}"
            content = f"Speaker {msg['role']} says: {msg['content']}"
            if self.add_memory_solar(content, doc_id):
                stored += 1
        return stored

    def get_lmu_ts_metrics(self) -> Dict:
        """Return combined metrics."""
        active = self._get_active_indices()
        return {
            "total_candidates": self.total_candidates,
            "total_updates": self.total_updates,
            "total_evictions": self.total_evictions,
            "update_rate": self.total_updates / max(self.total_candidates, 1),
            "active_memories": len(active),
            "capacity": self.solar_config.capacity,
            "current_threshold": self.threshold,
            "current_regret": self.accumulated_regret,
        }
