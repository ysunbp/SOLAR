"""
Deep Dive: Regret Bounds — Improved IG-UCB with Thompson Sampling

Key improvements:
1. Better warmup strategy (optimistic initialization + initial exploration)
2. Thompson Sampling variant for faster exploration
3. Capacity-aware selection (considers memory fullness)
4. Verification of sublinear regret empirically
"""

import numpy as np
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory_state import MemoryState, MemoryEntry, Feedback, MemoryUpdatePolicy
from info_gain_memory import (
    InfoGainMemoryPolicy, BanditConfig, InformationGainEstimator, 
    UCBMemorySelector, RegretBoundAnalysis
)
from typing import Dict, List


class ThompsonSamplingMemoryPolicy(MemoryUpdatePolicy):
    """
    Thompson Sampling variant for memory selection.
    
    Instead of UCB's deterministic exploration bonus, 
    TS samples from posterior distributions over entry values.
    
    Advantages over UCB:
    - Faster convergence (no need for log(T) exploration)
    - Better empirical performance with limited data
    - Naturally balances exploration/exploitation
    
    Regret bound: O(√(T·K·log K)) — tighter than UCB by log factor
    """
    
    def __init__(self, config: BanditConfig):
        self.config = config
        self.ig_estimator = InformationGainEstimator(config.embedding_dim)
        
        # Thompson Sampling state: Beta distributions for each entry
        # Beta(α, β) where α = successes + 1, β = failures + 1
        self.alphas: Dict[int, float] = {}
        self.betas: Dict[int, float] = {}
        self._next_id = 0
        self._entry_id_map: Dict[int, int] = {}
        
        # Metrics
        self.total_steps = 0
        self.update_count = 0
        self.reject_count = 0
        self.cost_history: List[float] = []
        self.cumulative_regret = 0.0
        self.regret_history: List[float] = []
        
        # Warmup: add first N candidates unconditionally
        self.warmup_steps = config.capacity_K // 2
    
    def _register_entry(self) -> int:
        eid = self._next_id
        self._next_id += 1
        # Optimistic prior: Beta(2, 1) — biased toward success
        self.alphas[eid] = 2.0
        self.betas[eid] = 1.0
        return eid
    
    def _sample_value(self, entry_id: int) -> float:
        """Sample from Beta posterior."""
        a = self.alphas.get(entry_id, 2.0)
        b = self.betas.get(entry_id, 1.0)
        return np.random.beta(a, b)
    
    def _update_posterior(self, entry_id: int, reward: float):
        """Update Beta distribution with observed reward."""
        if entry_id not in self.alphas:
            self.alphas[entry_id] = 2.0
            self.betas[entry_id] = 1.0
        
        # Reward is [0, 1]: treat as Bernoulli-like
        self.alphas[entry_id] += reward
        self.betas[entry_id] += (1 - reward)
    
    def should_update(self, state: MemoryState, feedback: Feedback) -> bool:
        self.total_steps += 1
        
        # Update estimator
        self.ig_estimator.update(feedback.query, feedback.response_quality)
        
        # During warmup: always add
        if self.total_steps <= self.warmup_steps and not state.is_full:
            return True
        
        # Compute candidate info gain
        candidate = feedback.suggested_content if feedback.suggested_content is not None \
                    else feedback.query
        ig = self.ig_estimator.compute_info_gain(candidate, state, feedback.query)
        
        if not state.is_full:
            return ig > 0.05  # low threshold when not full
        
        # Thompson Sampling: sample value for weakest entry
        entry_ids = list(range(state.size))
        sampled_values = [(idx, self._sample_value(self._entry_id_map.get(idx, 0))) 
                         for idx in entry_ids]
        min_sampled = min(sampled_values, key=lambda x: x[1])
        
        # Estimate regret
        reward = (feedback.response_quality + 1) / 2
        best_possible = 1.0  # oracle
        instant_regret = best_possible - reward
        self.cumulative_regret += instant_regret
        self.regret_history.append(self.cumulative_regret)
        
        # Add if IG > weakest entry's sampled value
        should = ig > min_sampled[1] * 0.7
        if not should:
            self.reject_count += 1
        
        return should
    
    def compute_update(self, state: MemoryState, feedback: Feedback) -> MemoryState:
        new_state = state.copy()
        new_state.timestamp = feedback.timestamp
        
        candidate = feedback.suggested_content if feedback.suggested_content is not None \
                    else feedback.query / (np.linalg.norm(feedback.query) + 1e-10)
        
        new_entry = MemoryEntry(
            content=candidate,
            timestamp=feedback.timestamp,
            source='feedback',
            confidence=abs(feedback.response_quality)
        )
        
        if state.is_full:
            # Evict entry with lowest Thompson sample
            entry_ids = list(range(state.size))
            sampled_values = [(idx, self._sample_value(self._entry_id_map.get(idx, 0)))
                            for idx in entry_ids]
            evict_idx = min(sampled_values, key=lambda x: x[1])[0]
            new_state.entries.pop(evict_idx)
            
            # Update id map
            new_map = {}
            for idx, eid in self._entry_id_map.items():
                if idx > evict_idx:
                    new_map[idx - 1] = eid
                elif idx < evict_idx:
                    new_map[idx] = eid
            self._entry_id_map = new_map
        
        new_state.entries.append(new_entry)
        new_eid = self._register_entry()
        self._entry_id_map[len(new_state.entries) - 1] = new_eid
        
        # Update posteriors for retrieved entries
        results = state.retrieve(feedback.query, top_k=3)
        reward = (feedback.response_quality + 1) / 2
        for idx, sim in results:
            if idx in self._entry_id_map:
                self._update_posterior(self._entry_id_map[idx], reward * sim)
        
        self.update_count += 1
        return new_state
    
    def get_metrics(self) -> Dict[str, float]:
        return {
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'reject_count': self.reject_count,
            'update_rate': self.update_count / max(self.total_steps, 1),
            'cumulative_regret': self.cumulative_regret,
            'avg_regret_per_step': self.cumulative_regret / max(self.total_steps, 1),
        }


class AggressiveIGPolicy(MemoryUpdatePolicy):
    """
    More aggressive IG-UCB variant that updates more frequently.
    Fixes the warmup problem of the original by:
    1. Starting with optimistic UCB values
    2. Always exploring for first K entries
    3. Using lower threshold for eviction decisions
    """
    
    def __init__(self, config: BanditConfig):
        self.config = config
        self.ig_estimator = InformationGainEstimator(config.embedding_dim)
        self.total_steps = 0
        self.update_count = 0
        self.cost_history: List[float] = []
        self.cumulative_regret = 0.0
        self.regret_history: List[float] = []
        
        # Simple value tracking per entry
        self.entry_values: Dict[int, List[float]] = {}
        self._next_id = 0
        self._entry_id_map: Dict[int, int] = {}
    
    def _get_entry_value(self, entry_id: int) -> float:
        """Get mean value of an entry."""
        if entry_id not in self.entry_values or not self.entry_values[entry_id]:
            return 0.8  # optimistic
        return np.mean(self.entry_values[entry_id][-20:])  # recent mean
    
    def should_update(self, state: MemoryState, feedback: Feedback) -> bool:
        self.total_steps += 1
        self.ig_estimator.update(feedback.query, feedback.response_quality)
        
        # Always add if not full
        if not state.is_full:
            return True
        
        # Compute IG
        candidate = feedback.suggested_content if feedback.suggested_content is not None \
                    else feedback.query
        ig = self.ig_estimator.compute_info_gain(candidate, state, feedback.query)
        
        # Find weakest entry
        entry_ids = list(range(state.size))
        values = [(idx, self._get_entry_value(self._entry_id_map.get(idx, 0)))
                 for idx in entry_ids]
        min_val = min(values, key=lambda x: x[1])[1]
        
        # Track regret
        reward = (feedback.response_quality + 1) / 2
        self.cumulative_regret += (1 - reward)
        self.regret_history.append(self.cumulative_regret)
        
        # More aggressive: update if IG exceeds 60% of weakest (was 80%)
        return ig > min_val * 0.6
    
    def compute_update(self, state: MemoryState, feedback: Feedback) -> MemoryState:
        new_state = state.copy()
        new_state.timestamp = feedback.timestamp
        
        candidate = feedback.suggested_content if feedback.suggested_content is not None \
                    else feedback.query / (np.linalg.norm(feedback.query) + 1e-10)
        
        new_entry = MemoryEntry(
            content=candidate, timestamp=feedback.timestamp,
            source='feedback', confidence=abs(feedback.response_quality)
        )
        
        if state.is_full:
            entry_ids = list(range(state.size))
            values = [(idx, self._get_entry_value(self._entry_id_map.get(idx, 0)))
                     for idx in entry_ids]
            evict_idx = min(values, key=lambda x: x[1])[0]
            new_state.entries.pop(evict_idx)
            
            new_map = {}
            for idx, eid in self._entry_id_map.items():
                if idx > evict_idx:
                    new_map[idx - 1] = eid
                elif idx < evict_idx:
                    new_map[idx] = eid
            self._entry_id_map = new_map
        
        new_state.entries.append(new_entry)
        new_eid = self._next_id
        self._next_id += 1
        self._entry_id_map[len(new_state.entries) - 1] = new_eid
        self.entry_values[new_eid] = []
        
        # Update entry values based on feedback
        results = state.retrieve(feedback.query, top_k=3)
        reward = (feedback.response_quality + 1) / 2
        for idx, sim in results:
            if idx in self._entry_id_map:
                eid = self._entry_id_map[idx]
                if eid not in self.entry_values:
                    self.entry_values[eid] = []
                self.entry_values[eid].append(reward * sim)
        
        self.update_count += 1
        return new_state
    
    def get_metrics(self) -> Dict[str, float]:
        return {
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'update_rate': self.update_count / max(self.total_steps, 1),
            'cumulative_regret': self.cumulative_regret,
        }


def experiment_regret_over_time():
    """Track cumulative regret over time — verify sublinearity."""
    print("\n" + "="*70)
    print("EXPERIMENT: Regret Growth Over Time (Sublinearity Check)")
    print("="*70)
    
    from run_experiments import DomainSimulator, DomainConfig
    
    embedding_dim = 32
    capacity = 20
    T = 1000
    seed = 42
    
    domain_config = DomainConfig(
        name='test', embedding_dim=embedding_dim,
        mean_shift_rate=0.02, noise_level=0.2,
        num_topics=5, topic_switch_prob=0.1, adversarial_prob=0.05
    )
    
    policies = {
        'Greedy': GreedyMemoryUpdate(),
        'IG-UCB (original)': InfoGainMemoryPolicy(BanditConfig(capacity_K=capacity, embedding_dim=embedding_dim)),
        'IG-UCB (aggressive)': AggressiveIGPolicy(BanditConfig(capacity_K=capacity, embedding_dim=embedding_dim)),
        'Thompson Sampling': ThompsonSamplingMemoryPolicy(BanditConfig(capacity_K=capacity, embedding_dim=embedding_dim)),
    }
    
    results = {}
    
    for name, policy in policies.items():
        domain = DomainSimulator(domain_config, seed)
        state = MemoryState(capacity=capacity, embedding_dim=embedding_dim)
        
        cumulative_regret = 0.0
        regret_curve = []
        costs = []
        
        for t in range(T):
            query = domain.generate_query()
            feedback = domain.generate_feedback(state, query)
            
            should_update = policy.should_update(state, feedback)
            if should_update:
                state = policy.compute_update(state, feedback)
            
            cost = 1.0 - (feedback.response_quality + 1) / 2
            costs.append(cost)
            cumulative_regret += cost
            regret_curve.append(cumulative_regret)
            state.timestamp = t
        
        # Fit growth rate: regret ~ T^α
        # log(regret) ~ α·log(T)
        log_T = np.log(np.arange(1, T+1))
        log_R = np.log(np.array(regret_curve) + 1)
        alpha_fit = np.polyfit(log_T, log_R, 1)[0]
        
        results[name] = {
            'final_regret': cumulative_regret,
            'growth_exponent': alpha_fit,
            'is_sublinear': alpha_fit < 0.95,
            'avg_cost': np.mean(costs),
            'final_cost': np.mean(costs[-100:]),
            'num_updates': policy.update_count if hasattr(policy, 'update_count') else T,
        }
        
        print(f"\n  {name:25s}:")
        print(f"    Final regret: {cumulative_regret:.1f}")
        print(f"    Growth exponent α: {alpha_fit:.3f} ({'SUBLINEAR ✓' if alpha_fit < 0.95 else 'LINEAR ✗'})")
        print(f"    Avg cost: {np.mean(costs):.4f}, Final cost: {np.mean(costs[-100:]):.4f}")
        print(f"    Updates: {results[name]['num_updates']}")
    
    print(f"\n  Theory predicts: Greedy α=1.0, IG-UCB α≈0.5, TS α≈0.5")
    
    return results


def experiment_capacity_sensitivity():
    """How does performance scale with memory capacity K?"""
    print("\n" + "="*70)
    print("EXPERIMENT: Capacity Sensitivity — How Does K Affect Regret?")
    print("="*70)
    
    from run_experiments import DomainSimulator, DomainConfig
    
    embedding_dim = 32
    T = 500
    seed = 42
    
    K_values = [5, 10, 20, 50, 100]
    
    domain_config = DomainConfig(
        name='test', embedding_dim=embedding_dim,
        mean_shift_rate=0.02, noise_level=0.2,
        num_topics=8, topic_switch_prob=0.15, adversarial_prob=0.05
    )
    
    results = {'K': K_values, 'greedy_cost': [], 'ts_cost': [], 
               'greedy_final': [], 'ts_final': []}
    
    for K in K_values:
        # Greedy
        domain = DomainSimulator(domain_config, seed)
        greedy = GreedyMemoryUpdate()
        state = MemoryState(capacity=K, embedding_dim=embedding_dim)
        g_costs = []
        for t in range(T):
            query = domain.generate_query()
            feedback = domain.generate_feedback(state, query)
            if greedy.should_update(state, feedback):
                state = greedy.compute_update(state, feedback)
            g_costs.append(1.0 - (feedback.response_quality + 1) / 2)
            state.timestamp = t
        
        # Thompson Sampling
        domain = DomainSimulator(domain_config, seed)
        ts = ThompsonSamplingMemoryPolicy(BanditConfig(capacity_K=K, embedding_dim=embedding_dim))
        state = MemoryState(capacity=K, embedding_dim=embedding_dim)
        ts_costs = []
        for t in range(T):
            query = domain.generate_query()
            feedback = domain.generate_feedback(state, query)
            if ts.should_update(state, feedback):
                state = ts.compute_update(state, feedback)
            ts_costs.append(1.0 - (feedback.response_quality + 1) / 2)
            state.timestamp = t
        
        results['greedy_cost'].append(np.mean(g_costs))
        results['ts_cost'].append(np.mean(ts_costs))
        results['greedy_final'].append(np.mean(g_costs[-50:]))
        results['ts_final'].append(np.mean(ts_costs[-50:]))
        
        print(f"  K={K:3d}: Greedy avg={np.mean(g_costs):.4f} final={np.mean(g_costs[-50:]):.4f}, "
              f"TS avg={np.mean(ts_costs):.4f} final={np.mean(ts_costs[-50:]):.4f}")
    
    print(f"\n  Theory: Regret scales as O(√(T·K·log K))")
    print(f"  Empirical: TS cost ratio K=100/K=5 = {results['ts_cost'][-1]/results['ts_cost'][0]:.2f}")
    print(f"  Predicted ratio: √(100/5) = {np.sqrt(100/5):.2f}")
    
    return results


if __name__ == '__main__':
    from lazy_memory_update import GreedyMemoryUpdate
    
    r1 = experiment_regret_over_time()
    r2 = experiment_capacity_sensitivity()
    
    all_results = {
        'regret_over_time': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                                  for kk, vv in v.items()} for k, v in r1.items()},
        'capacity_sensitivity': {k: [float(x) for x in v] if isinstance(v, list) else v 
                                  for k, v in r2.items()},
    }
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regret_bounds_deep.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n\nResults saved to experiments/regret_bounds_deep.json")
