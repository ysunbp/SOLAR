"""
ARC Solver — plugs the ARC Agent into the MemoryBench evaluation framework.

Same retrieval as Embedder, but with ARC (Adaptive Replacement Cache) eviction.
"""

from tqdm import tqdm
from typing import List, Dict

from src.agent.arc import ARCAgent, ARCAgentConfig
from src.solver.base import BaseSolver


class ARCSolver(BaseSolver):
    AGENT_CLASS = ARCAgent

    def __init__(self, config: ARCAgentConfig, memory_cache_dir: str):
        super().__init__(config, memory_cache_dir)
        self.method_name = "arc"
        self.current_conversation_memory_ids = []

    def create_or_load_memory(self, dialogs: List[Dict], dialogs_dir: str):
        return super()._create_or_load_memory(dialogs, dialogs_dir, can_thread=False)

    def memory_locomo_conversation(self, conversation, session_cnt: int):
        """Store Locomo conversation turns with ARC eviction."""
        pbar = tqdm(total=session_cnt, desc="[ARC] Adding conversation to memory")
        stored_total = 0
        session_idx = 1
        while f"session_{session_idx}" in conversation:
            session_date_time = conversation[f"session_{session_idx}_date_time"]
            session = conversation[f"session_{session_idx}"]
            for turn in session:
                turn_date_time = session_date_time + " Turn " + turn["dia_id"].split(":")[1]
                content = turn_date_time + "\n" + "Speaker " + turn["speaker"] + " says: " + turn["text"]
                self.agent.add_memory_arc(content, doc_id=turn_date_time)
                stored_total += 1
                self.current_conversation_memory_ids.append(turn_date_time)
            session_idx += 1
            pbar.update(1)
        pbar.close()
        metrics = self.agent.get_arc_metrics()
        print(f"  [ARC] Stored {stored_total} turns, "
              f"evictions: {metrics['total_evictions']}, "
              f"active: {metrics['active_memories']}/{metrics['capacity']}, "
              f"T1={metrics['t1_size']}, T2={metrics['t2_size']}, p={metrics['p']:.1f}")

    def memory_dialsim_conversation(self, conversation, session_cnt: int):
        return self.memory_locomo_conversation(conversation, session_cnt)

    def delete_conversation_memory(self):
        if len(self.current_conversation_memory_ids) > 0:
            for memory_id in self.current_conversation_memory_ids:
                self.agent.delete_memory(memory_id)
            self.agent.rebuild_index()
        self.current_conversation_memory_ids = []
