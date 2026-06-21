import importlib
from typing import Optional, Union, Dict

from src.solver.base import BaseAgentConfig
from src.solver.embedder import EmbedderAgentConfig

from src.solver.solar import SolarSolver
from src.agent.solar import SolarAgentConfig
from src.solver.solar_a import SolarASolver
from src.agent.solar_a import SolarAAgentConfig
from src.solver.solar_e import SolarESolver
from src.agent.solar_e import SolarEAgentConfig
from src.solver.fifo import FIFOSolver
from src.agent.fifo import FIFOAgentConfig
from src.solver.lru import LRUSolver
from src.agent.lru import LRUAgentConfig
from src.solver.lfu import LFUSolver
from src.agent.lfu import LFUAgentConfig
from src.solver.arc import ARCSolver
from src.agent.arc import ARCAgentConfig


def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

class SolverFactory:
    # SOLAR and its ablations / classic-heuristic baselines.
    #   solar    -> SOLAR (full framework: regret-gated timing + posterior-guided eviction)
    #   solar_a  -> SOLAR-A (admission only: regret-gated timing + heuristic eviction)
    #   solar_e  -> SOLAR-E (eviction only: always admit + Thompson-sampling eviction)
    #   fifo / lru / lfu / arc -> classic cache-replacement baselines
    #   embedder_message -> unlimited-capacity reference (no eviction)
    method_to_class = {
        "wo_memory": ("src.solver.base.BaseSolver", "src.solver.base.BaseAgentConfig"),
        "embedder_message": ("src.solver.embedder.EmbedderSolver", "src.solver.embedder.EmbedderAgentConfig"),
        "solar": ("src.solver.solar.SolarSolver", "src.agent.solar.SolarAgentConfig"),
        "solar_a": ("src.solver.solar_a.SolarASolver", "src.agent.solar_a.SolarAAgentConfig"),
        "solar_e": ("src.solver.solar_e.SolarESolver", "src.agent.solar_e.SolarEAgentConfig"),
        "fifo": ("src.solver.fifo.FIFOSolver", "src.agent.fifo.FIFOAgentConfig"),
        "lru": ("src.solver.lru.LRUSolver", "src.agent.lru.LRUAgentConfig"),
        "lfu": ("src.solver.lfu.LFUSolver", "src.agent.lfu.LFUAgentConfig"),
        "arc": ("src.solver.arc.ARCSolver", "src.agent.arc.ARCAgentConfig"),
    }

    @classmethod
    def create(cls, method_name: str, config: Dict, **kwargs):
        if method_name not in cls.method_to_class:
            raise ValueError(f"Unknown method name: {method_name}")
        
        class_type, config_class_type = cls.method_to_class[method_name]
        solver_class = load_class(class_type)
        config_class = load_class(config_class_type)

        memory_cache_dir = kwargs.get("memory_cache_dir", None)
        if memory_cache_dir is not None and "memory_cache_dir" in config_class.__init__.__code__.co_varnames:
            config["memory_cache_dir"] = memory_cache_dir
        for key, value in kwargs.items():
            if key in config_class.__init__.__code__.co_varnames:
                config[key] = value
        agent_config = config_class(**config)
        return solver_class(agent_config, memory_cache_dir=memory_cache_dir)
