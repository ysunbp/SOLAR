"""
LMU Solver — plugs the LMU Agent into the MemoryBench evaluation framework.

Mirrors EmbedderSolver but uses SolarAAgent (with lazy update gating).
"""

from tqdm import tqdm
from typing import List, Dict

from src.agent.solar_a import SolarAAgent, SolarAAgentConfig
from src.solver.base import BaseSolver


class SolarASolver(BaseSolver):
    AGENT_CLASS = SolarAAgent

    def __init__(self, config: SolarAAgentConfig, memory_cache_dir: str):
        super().__init__(config, memory_cache_dir)
        self.method_name = "lmu"
        self.current_conversation_memory_ids = []

    def create_or_load_memory(self, dialogs: List[Dict], dialogs_dir: str):
        return super()._create_or_load_memory(dialogs, dialogs_dir, can_thread=False)

    def memory_locomo_conversation(self, conversation, session_cnt: int):
        """Store Locomo conversation turns through LMU gating."""
        pbar = tqdm(total=session_cnt, desc="[LMU] Adding conversation to memory")
        stored_total = 0
        candidate_total = 0
        session_idx = 1
        while f"session_{session_idx}" in conversation:
            session_date_time = conversation[f"session_{session_idx}_date_time"]
            session = conversation[f"session_{session_idx}"]
            for turn in session:
                candidate_total += 1
                turn_date_time = session_date_time + " Turn " + turn["dia_id"].split(":")[1]
                content = turn_date_time + "\n" + "Speaker " + turn["speaker"] + " says: " + turn["text"]
                if self.agent.add_memory_solar_a(content, doc_id=turn_date_time):
                    stored_total += 1
                    self.current_conversation_memory_ids.append(turn_date_time)
            session_idx += 1
            pbar.update(1)
        pbar.close()
        print(f"  [LMU] Stored {stored_total}/{candidate_total} turns "
              f"(update rate: {stored_total/max(candidate_total,1):.1%})")

    def memory_dialsim_conversation(self, conversation, session_cnt: int):
        return self.memory_locomo_conversation(conversation, session_cnt)

    def delete_conversation_memory(self):
        if len(self.current_conversation_memory_ids) > 0:
            for memory_id in self.current_conversation_memory_ids:
                self.agent.delete_memory(memory_id)
            self.agent.rebuild_index()
        self.current_conversation_memory_ids = []
