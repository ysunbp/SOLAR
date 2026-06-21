"""
SOLAR-A (admission-only) Memory Agent

Extends the Embedder-based memory system with a lazy update policy:
- Instead of storing every dialog turn, accumulates "regret" (information novelty).
- Only commits a memory write when accumulated regret exceeds a threshold τ.
- Threshold adapts based on observed embedding variance.
- When memory is full, evicts the entry with lowest estimated future value
  (combining recency, access frequency, and diversity contribution).

This reduces memory churn (switching cost) while retaining high-value information.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Literal, Union
from pydantic import BaseModel, Field

from src.agent.embedder import EmbedderAgent, EmbedderAgentConfig


class SolarAAgentConfig(EmbedderAgentConfig):
    """Configuration for SOLAR-A Agent, extends EmbedderAgentConfig."""
    # Admission-specific parameters
    lambda_cost: float = Field(
        default=5.0,
        description="Switching cost coefficient — higher means fewer updates"
    )
    lipschitz_L: float = Field(
        default=2.0,
        description="Estimated Lipschitz constant for cost function"
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
    capacity: int = Field(
        default=500,
        description="Maximum number of memory entries before eviction kicks in"
    )
    novelty_weight: float = Field(
        default=0.6,
        description="Weight for novelty in information gain calculation"
    )
    density_weight: float = Field(
        default=0.4,
        description="Weight for density penalty in information gain calculation"
    )


class SolarAAgent(EmbedderAgent):
    """
    Lazy Memory Update Agent.

    Inherits all embedding / FAISS / LLM machinery from EmbedderAgent.
    Overrides memory-write path so that each candidate turn is only stored
    when accumulated regret (novelty-based) exceeds an adaptive threshold.
    """

    def __init__(self, config: SolarAAgentConfig = SolarAAgentConfig()):
        super().__init__(config)
        self.solar_config = config

        # Admission internal state
        self.accumulated_regret: float = 0.0
        self.threshold: float = self._compute_initial_threshold()
        self.novelty_history: List[float] = []
        self.update_timestamps: List[int] = []
        self.total_candidates: int = 0
        self.total_updates: int = 0
        self.total_evictions: int = 0
        self._active_count: int = 0  # maintained incrementally
        self._L_hat_ema: Optional[float] = None  # EMA-smoothed Lipschitz estimate

        # For smart eviction: track access patterns
        self._access_counts: Dict[int, int] = {}   # metadata index -> access count
        self._last_accessed: Dict[int, int] = {}    # metadata index -> last access timestamp

    # ------------------------------------------------------------------
    # Threshold computation
    # ------------------------------------------------------------------

    def _compute_initial_threshold(self) -> float:
        """τ = √(2λ / L)"""
        return np.sqrt(
            2 * self.solar_config.lambda_cost / self.solar_config.lipschitz_L
        )

    def _adapt_threshold(self):
        """Adapt threshold based on observed novelty variance (EMA-smoothed)."""
        window = self.solar_config.adaptive_window
        if len(self.novelty_history) < window:
            return
        recent = self.novelty_history[-window:]
        new_L_hat = max(np.std(recent) * 2, 0.1)
        # EMA smoothing to avoid threshold oscillation from instantaneous std
        if self._L_hat_ema is None:
            self._L_hat_ema = new_L_hat
        else:
            self._L_hat_ema = 0.7 * self._L_hat_ema + 0.3 * new_L_hat
        new_threshold = np.sqrt(
            2 * self.solar_config.lambda_cost / self._L_hat_ema
        )
        self.threshold = float(np.clip(
            new_threshold,
            self.solar_config.min_threshold,
            self.solar_config.max_threshold,
        ))

    # ------------------------------------------------------------------
    # Novelty / information-gain estimation
    # ------------------------------------------------------------------

    def _compute_novelty(self, candidate_vec: np.ndarray) -> float:
        """
        Novelty = 1 − max cosine-similarity to existing memories.
        Returns value in [0, 1]; 1 = completely novel.
        """
        if self.index.ntotal == 0:
            return 1.0

        candidate_vec = candidate_vec.reshape(1, -1)
        # Use FAISS to find nearest neighbor (L2 distance)
        D, I = self.index.search(candidate_vec, 1)
        # Convert L2 distance to approximate cosine similarity
        # For normalized vectors: L2^2 = 2(1 - cos_sim)
        # Our vectors may not be normalized, so we use a heuristic
        l2_dist = float(D[0][0])
        # Heuristic: map L2 distance to novelty score
        # Small distance = low novelty, large distance = high novelty
        novelty = 1.0 - np.exp(-l2_dist / (2 * self.solar_config.embedding_dim))
        return float(np.clip(novelty, 0.0, 1.0))

    def _compute_info_gain(self, candidate_vec: np.ndarray) -> float:
        """
        Information gain combines novelty with a density penalty.
        High novelty + low local density = high info gain.
        """
        novelty = self._compute_novelty(candidate_vec)

        # Density penalty: how many existing memories are "close"?
        density_penalty = 0.0
        if self.index.ntotal > 0:
            k = min(5, self.index.ntotal)
            candidate_vec_2d = candidate_vec.reshape(1, -1)
            D, _ = self.index.search(candidate_vec_2d, k)
            # Average distance to k nearest neighbors
            avg_dist = float(np.mean(D[0]))
            # Low avg_dist = high density = high penalty
            density_penalty = np.exp(-avg_dist / (2 * self.solar_config.embedding_dim))

        ig = (self.solar_config.novelty_weight * novelty
              - self.solar_config.density_weight * density_penalty)
        return max(ig, 0.0)

    # ------------------------------------------------------------------
    # Admission decision: should we store this candidate?
    # ------------------------------------------------------------------

    def _should_store(self, candidate_vec: np.ndarray) -> bool:
        """
        Core admission decision.
        1. Compute info gain for candidate.
        2. Accumulate as regret.
        3. If accumulated regret > threshold → store and reset.
        """
        self.total_candidates += 1
        ig = self._compute_info_gain(candidate_vec)
        self.novelty_history.append(ig)
        self.accumulated_regret += ig

        # Adapt threshold
        if self.solar_config.threshold_mode == "adaptive":
            self._adapt_threshold()

        # Bootstrap: always store while memory is still warming up.
        # Use _active_count (not index.ntotal) so that compaction doesn't
        # accidentally re-trigger bootstrap mode.
        bootstrap_n = min(max(self.solar_config.capacity // 2, 5), 25)
        if self._active_count < bootstrap_n:
            self.accumulated_regret = 0.0
            self.total_updates += 1
            return True

        # Core decision
        if self.accumulated_regret > self.threshold:
            self.accumulated_regret = 0.0
            self.total_updates += 1
            self.update_timestamps.append(self.total_candidates)
            return True

        return False

    # ------------------------------------------------------------------
    # Smart eviction (when capacity is reached)
    # ------------------------------------------------------------------

    def _smart_evict(self, new_vec: np.ndarray):
        """
        Evict the memory entry with lowest estimated future value.
        Value = α·recency + β·frequency + γ·diversity_contribution

        Reads vectors from FAISS index directly (no re-embed).
        Only active (non-deleted) entries are considered.
        """
        n = len(self.metadata)
        if n == 0:
            return

        scores = []
        active_indices = []
        for i in range(n):
            if self.metadata[i].get("deleted", False):
                scores.append(float("inf"))  # never pick already-deleted entries
                continue

            # Recency (exponential decay)
            age = self.total_candidates - self._last_accessed.get(i, 0)
            recency = np.exp(-age / 100.0)

            # Access frequency
            freq = self._access_counts.get(i, 0) / max(self.total_candidates, 1)

            # Diversity: read vector from FAISS (no re-embed)
            try:
                vec = self.index.reconstruct(i).reshape(1, -1)
                D, _ = self.index.search(vec, 2)  # 2 because itself is included
                if len(D[0]) > 1:
                    diversity = 1.0 - np.exp(-float(D[0][1]) / (2 * self.solar_config.embedding_dim))
                else:
                    diversity = 1.0
            except Exception:
                diversity = 0.5

            value = 0.4 * recency + 0.3 * freq + 0.3 * diversity
            scores.append(value)
            active_indices.append(i)

        if not active_indices:
            return  # nothing active to evict

        # Evict the active entry with the lowest value
        evict_idx = int(np.argmin(scores))
        self.metadata[evict_idx]["deleted"] = True
        self._active_count -= 1
        self.total_evictions += 1

        # Periodic compaction to remove tombstones from FAISS index
        if self.total_evictions % 20 == 0:
            self._compact_index()

    def _compact_index(self):
        """Rebuild FAISS index and remap access stats to new indices."""
        import faiss as _faiss
        valid_vectors = []
        valid_metadata = []
        new_access_counts = {}
        new_last_accessed = {}

        for old_idx, meta in enumerate(self.metadata):
            if not meta.get("deleted", False):
                new_idx = len(valid_metadata)
                try:
                    vec = self.index.reconstruct(old_idx)
                    valid_vectors.append(vec)
                except Exception:
                    valid_vectors.append(self._embed(meta["content"]))
                valid_metadata.append(meta)
                # Remap access stats
                if old_idx in self._access_counts:
                    new_access_counts[new_idx] = self._access_counts[old_idx]
                if old_idx in self._last_accessed:
                    new_last_accessed[new_idx] = self._last_accessed[old_idx]

        # Rebuild
        if valid_vectors:
            self.index = _faiss.IndexFlatL2(self.config.embedding_dim)
            self.index.add(np.array(valid_vectors, dtype=np.float32))
        else:
            self.index = _faiss.IndexFlatL2(self.config.embedding_dim)

        self.metadata = valid_metadata
        self._access_counts = new_access_counts
        self._last_accessed = new_last_accessed
        self._active_count = len(valid_metadata)

    def add_memory_solar_a(self, content: str, doc_id=None) -> bool:
        """
        Admission-gated version of add_memory.
        Returns True if the memory was actually stored, False if skipped.
        """
        vector = self._embed(content)

        if not self._should_store(vector):
            return False

        # Check capacity and evict if needed (loop in case multiple slots are over)
        while self._active_count >= self.solar_config.capacity:
            before = self._active_count
            self._smart_evict(vector)
            if self._active_count >= before:
                break  # safety: avoid infinite loop if eviction failed

        # Store
        if doc_id is None:
            doc_id = f"lmu_doc_{len(self.metadata)}"
        self.metadata.append({"doc_id": doc_id, "content": content})
        self.index.add(np.array([vector]))
        self._active_count += 1

        # Track access
        idx = len(self.metadata) - 1
        self._access_counts[idx] = 0
        self._last_accessed[idx] = self.total_candidates
        return True

    # ------------------------------------------------------------------
    # Experience-based LMU gating (Exp 2: Experience Learning)
    # ------------------------------------------------------------------

    def add_memory_experience_gated(self, content: str, feedback_score: float, doc_id=None) -> bool:
        """
        Experience-based LMU gating (Exp 2).
        
        Uses feedback score instead of embedding novelty to decide storage.
        Low score (dislike) = high regret = more likely to store.
        This stores "lessons learned" — failed interactions that the agent
        should remember to avoid repeating mistakes.
        
        Args:
            content: The experience text (e.g., "Q: ... A: ... Feedback: dislike")
            feedback_score: Score in [0, 1]. Low = bad response = high regret.
            doc_id: Optional document ID.
            
        Returns:
            True if the experience was stored, False if skipped.
        """
        self.total_candidates += 1
        
        # Cost = 1 - score: dislike (low score) → high cost → high regret
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
        self.update_timestamps.append(self.total_candidates)
        
        vector = self._embed(content)
        
        # Check capacity and evict if needed (loop in case multiple slots are over)
        while self._active_count >= self.solar_config.capacity:
            before = self._active_count
            self._smart_evict(vector)
            if self._active_count >= before:
                break  # safety: avoid infinite loop if eviction failed
        
        # Store
        if doc_id is None:
            doc_id = f"lmu_exp_{len(self.metadata)}"
        self.metadata.append({"doc_id": doc_id, "content": content})
        self.index.add(np.array([vector]))
        self._active_count += 1
        
        # Track access
        idx = len(self.metadata) - 1
        self._access_counts[idx] = 0
        self._last_accessed[idx] = self.total_candidates
        return True

    # ------------------------------------------------------------------
    # Override: add_conversation_to_memory with LMU gating
    # ------------------------------------------------------------------

    def add_conversation_to_memory(
        self,
        messages: List[Dict[str, str]],
        conversation_idx: Union[int, str] = 0,
    ):
        """
        Add a conversation to memory, but each turn goes through LMU gating.
        Only turns with sufficient information gain are actually stored.
        """
        if isinstance(conversation_idx, int):
            conversation_idx = str(conversation_idx)
        stored = 0
        for msg_idx, msg in enumerate(messages):
            doc_id = f"conv_{conversation_idx}_{msg_idx}"
            content = f"Speaker {msg['role']} says: {msg['content']}"
            if self.add_memory_solar_a(content, doc_id):
                stored += 1
        return stored

    # ------------------------------------------------------------------
    # Override: retrieve_memory to track access patterns
    # ------------------------------------------------------------------

    def retrieve_memory(self, content: str, k=10) -> List[str]:
        """Retrieve + track access for eviction scoring."""
        results = super().retrieve_memory(content, k)

        # Update access tracking for retrieved entries
        if results:
            query_vec = self._embed(content)
            query_vec_2d = query_vec.reshape(1, -1)
            D, I = self.index.search(query_vec_2d, min(k * 2, max(self.index.ntotal, 1)))
            for idx in I[0]:
                if 0 <= idx < len(self.metadata) and not self.metadata[idx].get("deleted", False):
                    self._access_counts[idx] = self._access_counts.get(idx, 0) + 1
                    self._last_accessed[idx] = self.total_candidates

        return results

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_lmu_metrics(self) -> Dict:
        """Return LMU-specific metrics for analysis."""
        active_memories = sum(
            1 for m in self.metadata if not m.get("deleted", False)
        )
        return {
            "total_candidates": self.total_candidates,
            "total_updates": self.total_updates,
            "update_rate": self.total_updates / max(self.total_candidates, 1),
            "active_memories": active_memories,
            "current_threshold": self.threshold,
            "current_regret": self.accumulated_regret,
            "avg_novelty": float(np.mean(self.novelty_history)) if self.novelty_history else 0.0,
        }

    # ------------------------------------------------------------------
    # Persistence: save/load LMU state alongside FAISS index
    # ------------------------------------------------------------------

    def save_memories(self):
        """Save FAISS index + metadata + LMU state."""
        super().save_memories()
        lmu_state_path = os.path.join(self.config.memory_cache_dir, "lmu_state.json")
        state = {
            "accumulated_regret": self.accumulated_regret,
            "threshold": self.threshold,
            "total_candidates": self.total_candidates,
            "total_updates": self.total_updates,
            "novelty_history": self.novelty_history[-100:],  # keep last 100
            "update_timestamps": self.update_timestamps,
            "access_counts": {str(k): v for k, v in self._access_counts.items()},
            "last_accessed": {str(k): v for k, v in self._last_accessed.items()},
        }
        with open(lmu_state_path, "w") as f:
            json.dump(state, f, indent=2)

    def load_memories(self):
        """Load FAISS index + metadata + LMU state."""
        super().load_memories()
        lmu_state_path = os.path.join(self.config.memory_cache_dir, "lmu_state.json")
        if os.path.exists(lmu_state_path):
            with open(lmu_state_path, "r") as f:
                state = json.load(f)
            self.accumulated_regret = state.get("accumulated_regret", 0.0)
            self.threshold = state.get("threshold", self._compute_initial_threshold())
            self.total_candidates = state.get("total_candidates", 0)
            self.total_updates = state.get("total_updates", 0)
            self.novelty_history = state.get("novelty_history", [])
            self.update_timestamps = state.get("update_timestamps", [])
            self._access_counts = {int(k): v for k, v in state.get("access_counts", {}).items()}
            self._last_accessed = {int(k): v for k, v in state.get("last_accessed", {}).items()}
