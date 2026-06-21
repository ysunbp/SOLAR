"""
LFU (Least Frequently Used) Memory Agent

Extends the Embedder-based memory system with a capacity limit.
When memory is full, the least frequently accessed entry is evicted.

Key difference from LRU:
- LRU evicts by last access time (recency)
- LFU evicts by total access count (frequency)
- Tie-break: when counts are equal, evict the oldest entry (by insertion order)

This means entries that are retrieved many times survive longer,
regardless of when they were last accessed.
"""

import numpy as np
from typing import List, Dict, Union
from pydantic import Field

from src.agent.embedder import EmbedderAgent, EmbedderAgentConfig


class LFUAgentConfig(EmbedderAgentConfig):
    """Configuration for LFU Agent."""
    capacity: int = Field(
        default=500,
        description="Maximum number of memory entries. Least frequently used evicted when full."
    )


class LFUAgent(EmbedderAgent):
    """
    LFU Memory Agent.

    Stores every dialog turn. When capacity is reached, evicts the entry
    that has been retrieved the fewest times. Ties broken by oldest insertion.
    """

    def __init__(self, config: LFUAgentConfig = LFUAgentConfig()):
        super().__init__(config)
        self.lfu_config = config
        self.total_candidates: int = 0
        self.total_evictions: int = 0
        self._active_count: int = 0

        # Track retrieval count and insertion order for each entry
        self._retrieve_count: Dict[int, int] = {}
        self._insertion_order: Dict[int, int] = {}  # idx -> insertion step

    def _get_active_count(self) -> int:
        """Return cached active count (O(1))."""
        return self._active_count

    def _evict_lfu(self):
        """Evict the least frequently used non-deleted entry (tie-break: oldest)."""
        min_count = float('inf')
        min_insertion = float('inf')
        evict_idx = -1

        for i, meta in enumerate(self.metadata):
            if not meta.get("deleted", False):
                count = self._retrieve_count.get(i, 0)
                insertion = self._insertion_order.get(i, 0)
                # Evict: lowest count first, then oldest insertion as tie-break
                if count < min_count or (count == min_count and insertion < min_insertion):
                    min_count = count
                    min_insertion = insertion
                    evict_idx = i

        if evict_idx >= 0:
            self.metadata[evict_idx]["deleted"] = True
            self.total_evictions += 1
            self._active_count -= 1
            if evict_idx in self._retrieve_count:
                del self._retrieve_count[evict_idx]
            if evict_idx in self._insertion_order:
                del self._insertion_order[evict_idx]

    def add_memory_lfu(self, content: str, doc_id=None) -> bool:
        """
        Add memory with LFU eviction when capacity is reached.
        Always stores (returns True), evicts least frequently used if full.
        """
        self.total_candidates += 1

        # Evict LFU entry if at capacity
        if self._get_active_count() >= self.lfu_config.capacity:
            self._evict_lfu()

        # Store
        if doc_id is None:
            doc_id = f"lfu_doc_{len(self.metadata)}"
        vector = self._embed(content)
        self.metadata.append({"doc_id": doc_id, "content": content})
        self.index.add(np.array([vector]))
        self._active_count += 1

        # Initialize stats for new entry
        new_idx = len(self.metadata) - 1
        self._retrieve_count[new_idx] = 0
        self._insertion_order[new_idx] = self.total_candidates
        return True

    def retrieve_memory(self, content: str, k=10) -> List[str]:
        """
        Retrieve relevant documents and update retrieve_count for hits.
        """
        if self.index.ntotal == 0:
            return []
        vector = self._embed(content)
        D, I = self.index.search(np.array([vector]), min(k * 2, len(self.metadata)))
        rets = []
        for i in I[0]:
            if i < len(self.metadata) and not self.metadata[i].get("deleted", False):
                rets.append(self.metadata[i]["content"])
                # Increment retrieve count on hit
                self._retrieve_count[i] = self._retrieve_count.get(i, 0) + 1
                if len(rets) >= k:
                    break
        return rets

    def add_conversation_to_memory(
        self,
        messages: List[Dict[str, str]],
        conversation_idx: Union[int, str] = 0,
    ):
        """Add conversation with LFU eviction."""
        if isinstance(conversation_idx, int):
            conversation_idx = str(conversation_idx)
        for msg_idx, msg in enumerate(messages):
            doc_id = f"conv_{conversation_idx}_{msg_idx}"
            content = f"Speaker {msg['role']} says: {msg['content']}"
            self.add_memory_lfu(content, doc_id)

    def get_lfu_metrics(self) -> Dict:
        """Return LFU-specific metrics."""
        active_counts = [
            self._retrieve_count.get(i, 0)
            for i, m in enumerate(self.metadata)
            if not m.get("deleted", False)
        ]
        return {
            "total_candidates": self.total_candidates,
            "total_evictions": self.total_evictions,
            "active_memories": self._get_active_count(),
            "capacity": self.lfu_config.capacity,
            "avg_retrieve_count": float(np.mean(active_counts)) if active_counts else 0.0,
        }
