"""
ARC (Adaptive Replacement Cache) Memory Agent

Implements a simplified ARC (Megiddo & Modha, 2003) eviction policy.
ARC adaptively balances between recency (LRU) and frequency (LFU) by
maintaining two LRU lists and two ghost lists:

  T1: entries accessed exactly once (recency-focused)
  T2: entries accessed more than once (frequency-focused)
  B1: ghost list for recently evicted T1 entries (metadata only)
  B2: ghost list for recently evicted T2 entries (metadata only)

The parameter `p` (target size for T1) is adapted online:
  - Hit in B1 → increase p (favor recency)
  - Hit in B2 → decrease p (favor frequency)

Eviction rule:
  - If |T1| > p: evict LRU of T1
  - Else: evict LRU of T2
"""

import numpy as np
from collections import OrderedDict
from typing import List, Dict, Union
from pydantic import Field

from src.agent.embedder import EmbedderAgent, EmbedderAgentConfig


class ARCAgentConfig(EmbedderAgentConfig):
    """Configuration for ARC Agent."""
    capacity: int = Field(
        default=500,
        description="Maximum number of memory entries (c = |T1| + |T2|)."
    )


class ARCAgent(EmbedderAgent):
    """
    ARC (Adaptive Replacement Cache) Memory Agent.

    Stores every dialog turn. When capacity is reached, uses the ARC
    algorithm to decide which entry to evict, adaptively balancing
    recency and frequency.
    """

    def __init__(self, config: ARCAgentConfig = ARCAgentConfig()):
        super().__init__(config)
        self.arc_config = config
        self.total_candidates: int = 0
        self.total_evictions: int = 0
        self._active_count: int = 0

        # ARC data structures
        # T1, T2: OrderedDict mapping idx -> True (order = LRU order, leftmost = LRU)
        self._t1: OrderedDict = OrderedDict()  # accessed once
        self._t2: OrderedDict = OrderedDict()  # accessed more than once
        # B1, B2: ghost lists (store doc_id for identification, no actual content)
        self._b1: OrderedDict = OrderedDict()  # ghost of T1 evictions
        self._b2: OrderedDict = OrderedDict()  # ghost of T2 evictions
        # Adaptive parameter p: target size for T1
        self._p: float = 0.0

    def _get_active_count(self) -> int:
        """Return cached active count (O(1))."""
        return self._active_count

    def _replace(self, in_b2: bool):
        """
        ARC REPLACE subroutine: evict one entry from T1 or T2.
        """
        if self._t1 and (
            len(self._t1) > self._p or
            (in_b2 and len(self._t1) == int(self._p))
        ):
            # Evict LRU of T1
            evict_idx, _ = self._t1.popitem(last=False)
            # Add to ghost B1
            doc_id = self.metadata[evict_idx].get("doc_id", "")
            self._b1[evict_idx] = doc_id
            # Cap ghost list size
            if len(self._b1) > self.arc_config.capacity:
                self._b1.popitem(last=False)
        else:
            # Evict LRU of T2
            if self._t2:
                evict_idx, _ = self._t2.popitem(last=False)
                doc_id = self.metadata[evict_idx].get("doc_id", "")
                self._b2[evict_idx] = doc_id
                if len(self._b2) > self.arc_config.capacity:
                    self._b2.popitem(last=False)
            elif self._t1:
                # Fallback: evict from T1 if T2 is empty
                evict_idx, _ = self._t1.popitem(last=False)
                doc_id = self.metadata[evict_idx].get("doc_id", "")
                self._b1[evict_idx] = doc_id
                if len(self._b1) > self.arc_config.capacity:
                    self._b1.popitem(last=False)
            else:
                return  # nothing to evict

        # Mark as deleted in metadata
        self.metadata[evict_idx]["deleted"] = True
        self.total_evictions += 1
        self._active_count -= 1

    def add_memory_arc(self, content: str, doc_id=None) -> bool:
        """
        Add memory with ARC eviction when capacity is reached.
        Always stores (returns True).
        """
        self.total_candidates += 1

        # Evict if at capacity
        if self._get_active_count() >= self.arc_config.capacity:
            self._replace(in_b2=False)

        # Store
        if doc_id is None:
            doc_id = f"arc_doc_{len(self.metadata)}"
        vector = self._embed(content)
        self.metadata.append({"doc_id": doc_id, "content": content})
        self.index.add(np.array([vector]))
        self._active_count += 1

        # New entry goes into T1 (accessed once)
        new_idx = len(self.metadata) - 1
        self._t1[new_idx] = True
        return True

    def retrieve_memory(self, content: str, k=10) -> List[str]:
        """
        Retrieve relevant documents and update ARC state on hits.

        On retrieval hit:
        - If entry is in T1: move to T2 (MRU position) — it's now "frequent"
        - If entry is in T2: move to MRU position of T2
        - If entry's ghost is in B1: adapt p upward (favor recency)
        - If entry's ghost is in B2: adapt p downward (favor frequency)
        """
        if self.index.ntotal == 0:
            return []
        vector = self._embed(content)
        D, I = self.index.search(np.array([vector]), min(k * 2, len(self.metadata)))
        rets = []
        for i in I[0]:
            if i < len(self.metadata) and not self.metadata[i].get("deleted", False):
                rets.append(self.metadata[i]["content"])
                # Update ARC state
                self._on_access(i)
                if len(rets) >= k:
                    break
        return rets

    def _on_access(self, idx: int):
        """Handle an access (retrieval hit) to entry idx."""
        c = self.arc_config.capacity

        if idx in self._t1:
            # Move from T1 to MRU of T2 (now "frequent")
            del self._t1[idx]
            self._t2[idx] = True
        elif idx in self._t2:
            # Move to MRU position of T2
            self._t2.move_to_end(idx)
        elif idx in self._b1:
            # Ghost hit in B1: adapt p upward (favor recency)
            delta = max(1.0, len(self._b2) / max(len(self._b1), 1))
            self._p = min(self._p + delta, float(c))
            del self._b1[idx]
            # This is a "miss" in the real cache but a ghost hit;
            # in full ARC the entry would be re-fetched. Here we just
            # note the adaptation (the entry is already active if it's
            # being retrieved from FAISS).
        elif idx in self._b2:
            # Ghost hit in B2: adapt p downward (favor frequency)
            delta = max(1.0, len(self._b1) / max(len(self._b2), 1))
            self._p = max(self._p - delta, 0.0)
            del self._b2[idx]

    def add_conversation_to_memory(
        self,
        messages: List[Dict[str, str]],
        conversation_idx: Union[int, str] = 0,
    ):
        """Add conversation with ARC eviction."""
        if isinstance(conversation_idx, int):
            conversation_idx = str(conversation_idx)
        for msg_idx, msg in enumerate(messages):
            doc_id = f"conv_{conversation_idx}_{msg_idx}"
            content = f"Speaker {msg['role']} says: {msg['content']}"
            self.add_memory_arc(content, doc_id)

    def get_arc_metrics(self) -> Dict:
        """Return ARC-specific metrics."""
        return {
            "total_candidates": self.total_candidates,
            "total_evictions": self.total_evictions,
            "active_memories": self._get_active_count(),
            "capacity": self.arc_config.capacity,
            "t1_size": len(self._t1),
            "t2_size": len(self._t2),
            "b1_size": len(self._b1),
            "b2_size": len(self._b2),
            "p": self._p,
        }
