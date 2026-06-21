"""
LRU (Least Recently Used) Memory Agent

Extends the Embedder-based memory system with a capacity limit.
When memory is full, the least recently accessed entry is evicted.

Key difference from FIFO:
- FIFO evicts by insertion order (oldest stored)
- LRU evicts by last access time (least recently retrieved/hit)

This means frequently accessed entries survive longer, even if they
were stored early. An entry's "last accessed" timestamp is updated
every time it appears in a retrieval result.
"""

import numpy as np
from typing import List, Dict, Union
from pydantic import Field

from src.agent.embedder import EmbedderAgent, EmbedderAgentConfig


class LRUAgentConfig(EmbedderAgentConfig):
    """Configuration for LRU Agent."""
    capacity: int = Field(
        default=500,
        description="Maximum number of memory entries. Least recently used evicted when full."
    )


class LRUAgent(EmbedderAgent):
    """
    LRU Memory Agent.

    Stores every dialog turn. When capacity is reached, evicts the entry
    that was least recently accessed (retrieved). Uses the same
    embedding/retrieval as EmbedderAgent.
    """

    def __init__(self, config: LRUAgentConfig = LRUAgentConfig()):
        super().__init__(config)
        self.lru_config = config
        self.total_candidates: int = 0
        self.total_evictions: int = 0
        self._active_count: int = 0

        # Track last access step for each entry index
        # Initialized to insertion step; updated on retrieval hit
        self._last_accessed: Dict[int, int] = {}

    def _get_active_count(self) -> int:
        """Return cached active count (O(1))."""
        return self._active_count

    def _evict_lru(self):
        """Evict the least recently accessed non-deleted entry."""
        # Find active entry with smallest _last_accessed value
        min_step = float('inf')
        evict_idx = -1
        for i, meta in enumerate(self.metadata):
            if not meta.get("deleted", False):
                access_step = self._last_accessed.get(i, 0)
                if access_step < min_step:
                    min_step = access_step
                    evict_idx = i

        if evict_idx >= 0:
            self.metadata[evict_idx]["deleted"] = True
            self.total_evictions += 1
            self._active_count -= 1
            if evict_idx in self._last_accessed:
                del self._last_accessed[evict_idx]

    def add_memory_lru(self, content: str, doc_id=None) -> bool:
        """
        Add memory with LRU eviction when capacity is reached.
        Always stores (returns True), evicts least recently used if full.
        """
        self.total_candidates += 1

        # Evict LRU entry if at capacity
        if self._get_active_count() >= self.lru_config.capacity:
            self._evict_lru()

        # Store
        if doc_id is None:
            doc_id = f"lru_doc_{len(self.metadata)}"
        vector = self._embed(content)
        self.metadata.append({"doc_id": doc_id, "content": content})
        self.index.add(np.array([vector]))
        self._active_count += 1

        # Initialize last_accessed to current step
        new_idx = len(self.metadata) - 1
        self._last_accessed[new_idx] = self.total_candidates
        return True

    def retrieve_memory(self, content: str, k=10) -> List[str]:
        """
        Retrieve relevant documents and update last_accessed for hits.
        """
        if self.index.ntotal == 0:
            return []
        vector = self._embed(content)
        D, I = self.index.search(np.array([vector]), min(k * 2, len(self.metadata)))
        rets = []
        for i in I[0]:
            if i < len(self.metadata) and not self.metadata[i].get("deleted", False):
                rets.append(self.metadata[i]["content"])
                # Update last_accessed on retrieval hit
                self._last_accessed[i] = self.total_candidates
                if len(rets) >= k:
                    break
        return rets

    def add_conversation_to_memory(
        self,
        messages: List[Dict[str, str]],
        conversation_idx: Union[int, str] = 0,
    ):
        """Add conversation with LRU eviction."""
        if isinstance(conversation_idx, int):
            conversation_idx = str(conversation_idx)
        for msg_idx, msg in enumerate(messages):
            doc_id = f"conv_{conversation_idx}_{msg_idx}"
            content = f"Speaker {msg['role']} says: {msg['content']}"
            self.add_memory_lru(content, doc_id)

    def get_lru_metrics(self) -> Dict:
        """Return LRU-specific metrics."""
        return {
            "total_candidates": self.total_candidates,
            "total_evictions": self.total_evictions,
            "active_memories": self._get_active_count(),
            "capacity": self.lru_config.capacity,
        }
