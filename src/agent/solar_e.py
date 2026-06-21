"""
Thompson Sampling Memory Agent

Extends the Embedder-based memory system with Thompson Sampling for eviction.
When memory is full, uses Thompson Sampling to decide which entry to evict:
- Each entry has a Beta(α, β) distribution modeling its "value"
- Value is updated based on retrieval hits (accessed = reward)
- Eviction: sample from each entry's distribution, evict the one with lowest sample

Key advantage over FIFO:
- FIFO blindly evicts oldest → may remove important early facts
- Thompson explores: entries that haven't been accessed get uncertain distributions,
  giving them a chance to be kept (exploration) while frequently-accessed entries
  are confidently valued (exploitation)
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field

from src.agent.embedder import EmbedderAgent, EmbedderAgentConfig


class SolarEAgentConfig(EmbedderAgentConfig):
    """Configuration for Thompson Sampling Agent."""
    capacity: int = Field(
        default=500,
        description="Maximum number of memory entries before eviction kicks in"
    )
    warmup_fraction: float = Field(
        default=0.3,
        description="Fill memory to this fraction before starting Thompson eviction"
    )
    optimistic_prior: bool = Field(
        default=True,
        description="Use optimistic prior (α=2, β=1) to encourage exploration"
    )
    novelty_bonus: float = Field(
        default=0.3,
        description="Bonus for novel entries in eviction scoring"
    )


class SolarEAgent(EmbedderAgent):
    """
    Thompson Sampling Memory Agent.

    Stores every dialog turn. When capacity is reached, uses Thompson Sampling
    to decide which entry to evict based on estimated value distributions.
    """

    def __init__(self, config: SolarEAgentConfig = SolarEAgentConfig()):
        super().__init__(config)
        self.thompson_config = config
        self.total_candidates: int = 0
        self.total_evictions: int = 0
        self._active_count: int = 0  # maintained incrementally

        # Beta distribution parameters for each entry: (alpha, beta)
        # alpha = successes (retrieved/accessed), beta = failures (not accessed)
        self._alpha: Dict[int, float] = {}
        self._beta: Dict[int, float] = {}
        self._last_accessed: Dict[int, int] = {}
        self._retrieve_step: int = 0  # monotonic counter for batched beta decay

    def _get_active_count(self) -> int:
        """Return cached active count (O(1))."""
        return self._active_count

    def _get_active_indices(self) -> List[int]:
        """Get indices of non-deleted entries."""
        return [i for i, m in enumerate(self.metadata) if not m.get("deleted", False)]

    def _init_entry_stats(self, idx: int):
        """Initialize Beta distribution for a new entry."""
        if self.thompson_config.optimistic_prior:
            self._alpha[idx] = 2.0  # optimistic: assume it will be useful
            self._beta[idx] = 1.0
        else:
            self._alpha[idx] = 1.0  # uniform prior
            self._beta[idx] = 1.0
        self._last_accessed[idx] = self.total_candidates

    def _sample_value(self, idx: int) -> float:
        """Sample from Beta(α, β) distribution for this entry."""
        alpha = self._alpha.get(idx, 1.0)
        beta = self._beta.get(idx, 1.0)
        return float(np.random.beta(alpha, beta))

    def _evict_thompson(self, new_vec: np.ndarray):
        """
        Thompson Sampling eviction:
        1. Sample value from each entry's Beta distribution
        2. Add novelty bonus (diversity contribution) — read vectors from FAISS
        3. Evict entry with lowest combined score
        """
        active_indices = self._get_active_indices()
        if not active_indices:
            return

        scores = []
        for idx in active_indices:
            # Thompson sample
            ts_value = self._sample_value(idx)

            # Novelty bonus: read vector directly from FAISS (no re-embed)
            try:
                vec = self.index.reconstruct(idx).reshape(1, -1)
                k = min(3, self.index.ntotal)
                D, _ = self.index.search(vec, k)
                # Higher distance to neighbors = more unique = higher bonus
                avg_dist = float(np.mean(D[0][1:])) if len(D[0]) > 1 else float(D[0][0])
                novelty = 1.0 - np.exp(-avg_dist / (2 * self.thompson_config.embedding_dim))
            except Exception:
                novelty = 0.5

            combined = ts_value + self.thompson_config.novelty_bonus * novelty
            scores.append((idx, combined))

        # Evict entry with lowest score
        scores.sort(key=lambda x: x[1])
        evict_idx = scores[0][0]
        self.metadata[evict_idx]["deleted"] = True
        self._active_count -= 1
        self.total_evictions += 1

        # Clean up stats
        if evict_idx in self._alpha:
            del self._alpha[evict_idx]
        if evict_idx in self._beta:
            del self._beta[evict_idx]
        if evict_idx in self._last_accessed:
            del self._last_accessed[evict_idx]

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

    def add_memory_thompson(self, content: str, doc_id=None) -> bool:
        """
        Add memory with Thompson Sampling eviction when capacity is reached.
        Always stores (returns True), uses TS to decide eviction.
        """
        self.total_candidates += 1
        vector = self._embed(content)

        # Evict if at capacity
        if self._active_count >= self.thompson_config.capacity:
            self._evict_thompson(vector)

        # Store
        if doc_id is None:
            doc_id = f"thompson_doc_{len(self.metadata)}"
        self.metadata.append({"doc_id": doc_id, "content": content})
        self.index.add(np.array([vector]))
        self._active_count += 1

        # Initialize stats for new entry
        new_idx = len(self.metadata) - 1
        self._init_entry_stats(new_idx)
        return True

    def retrieve_memory(self, content: str, k=10) -> List[str]:
        """Retrieve + update Beta distributions based on access."""
        results = super().retrieve_memory(content, k)

        # Update stats: retrieved entries get a "success"
        if results and self.index.ntotal > 0:
            query_vec = self._embed(content)
            query_vec_2d = query_vec.reshape(1, -1)
            search_k = min(k * 2, self.index.ntotal)
            D, I = self.index.search(query_vec_2d, search_k)
            
            retrieved_set = set()
            for idx in I[0]:
                if 0 <= idx < len(self.metadata) and not self.metadata[idx].get("deleted", False):
                    retrieved_set.add(idx)
                    # Success: this entry was useful
                    self._alpha[idx] = self._alpha.get(idx, 1.0) + 1.0
                    self._last_accessed[idx] = self.total_candidates
                    if len(retrieved_set) >= k:
                        break

            # Non-retrieved active entries get a "failure" (batched decay)
            # Decay every 5 retrieve calls with a smaller increment to avoid
            # penalising long-tail knowledge that is rarely queried.
            self._retrieve_step += 1
            if self._retrieve_step % 5 == 0:
                for idx in self._get_active_indices():
                    if idx not in retrieved_set:
                        self._beta[idx] = self._beta.get(idx, 1.0) + 0.05

        return results

    def add_conversation_to_memory(
        self,
        messages: List[Dict[str, str]],
        conversation_idx: Union[int, str] = 0,
    ):
        """Add conversation with Thompson eviction."""
        if isinstance(conversation_idx, int):
            conversation_idx = str(conversation_idx)
        for msg_idx, msg in enumerate(messages):
            doc_id = f"conv_{conversation_idx}_{msg_idx}"
            content = f"Speaker {msg['role']} says: {msg['content']}"
            self.add_memory_thompson(content, doc_id)

    def get_thompson_metrics(self) -> Dict:
        """Return Thompson-specific metrics."""
        active = self._get_active_indices()
        avg_alpha = float(np.mean([self._alpha.get(i, 1.0) for i in active])) if active else 0
        avg_beta = float(np.mean([self._beta.get(i, 1.0) for i in active])) if active else 0
        return {
            "total_candidates": self.total_candidates,
            "total_evictions": self.total_evictions,
            "active_memories": len(active),
            "capacity": self.thompson_config.capacity,
            "avg_alpha": avg_alpha,
            "avg_beta": avg_beta,
        }
