"""
Adversarial Feedback Sequence Design + Verification

This file implements the adversarial construction from Theorem 4 and verifies
empirically that Greedy indeed degrades to linear regret while Thompson 
Sampling maintains sublinear regret.

Key idea: The "Cycling Adversary" — topics cycle with period K+1 (exceeding 
memory capacity K), forcing greedy LRU eviction to always evict the entry 
that will be needed next.
"""

import numpy as np
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory_state import MemoryState, MemoryEntry, Feedback
from lazy_memory_update import (
    LazyMemoryUpdate, GreedyMemoryUpdate, SwitchingCostConfig
)
from deep_regret_bounds import ThompsonSamplingMemoryPolicy, AggressiveIGPolicy
from info_gain_memory import BanditConfig


class CyclingAdversary:
    """
    The Cycling Adversary from Theorem 4.
    
    Creates K+1 topic clusters and cycles through them.
    Memory capacity K means one topic's entry is always missing.
    
    Under LRU eviction (Greedy), the evicted entry is ALWAYS the one 
    needed in the next cycle → Greedy gets 0 reward after warmup.
    """
    
    def __init__(self, K: int, embedding_dim: int, noise: float = 0.05, seed: int = 42):
        self.K = K
        self.embedding_dim = embedding_dim
        self.noise = noise
        self.rng = np.random.RandomState(seed)
        self.num_topics = K + 1  # One more than capacity!
        
        # Generate orthogonal topic embeddings
        # Each topic gets a unique direction in embedding space
        self.topic_embeddings = np.zeros((self.num_topics, embedding_dim))
        for i in range(self.num_topics):
            vec = self.rng.randn(embedding_dim)
            # Gram-Schmidt against previous topics
            for j in range(i):
                vec = vec - np.dot(vec, self.topic_embeddings[j]) * self.topic_embeddings[j]
            self.topic_embeddings[i] = vec / (np.linalg.norm(vec) + 1e-10)
        
        self.timestep = 0
    
    def get_current_topic(self) -> int:
        """Cycling: topic index = timestep mod (K+1)."""
        return self.timestep % self.num_topics
    
    def generate_query(self) -> np.ndarray:
        """Generate query for current topic with slight noise."""
        topic_idx = self.get_current_topic()
        query = self.topic_embeddings[topic_idx].copy()
        query += self.rng.randn(self.embedding_dim) * self.noise
        query = query / (np.linalg.norm(query) + 1e-10)
        return query
    
    def generate_feedback(self, state: MemoryState, query: np.ndarray) -> Feedback:
        """
        Feedback: high reward if the correct topic entry is in memory, 
        low reward otherwise.
        """
        self.timestep += 1
        topic_idx = (self.timestep - 1) % self.num_topics
        correct_embedding = self.topic_embeddings[topic_idx]
        
        # Check if correct entry is retrievable
        if state.size > 0:
            results = state.retrieve(query, top_k=1)
            if results:
                _, sim, _ = results[0] if len(results[0]) == 3 else (results[0][0], results[0][1], None)
                # High similarity = correct entry is in memory
                reward = sim if sim > 0.7 else 0.0
            else:
                reward = 0.0
        else:
            reward = 0.0
        
        # Map to [-1, 1]
        feedback_score = 2 * reward - 1
        
        return Feedback(
            query=query,
            response_quality=feedback_score,
            feedback_type='explicit',
            suggested_content=correct_embedding.copy(),
            timestamp=self.timestep
        )


class BurstAdversary:
    """
    Burst Adversary: Topics come in bursts, then suddenly switch.
    
    This exploits Greedy's recency bias:
    - Phase 1 (length B): All queries about topic A → memory fills with A entries
    - Phase 2 (length B): All queries about topic B → memory fills with B entries
    - Phase 3 (length 1): Query about topic A → evicted! Reward = 0
    
    Greedy loses because it over-commits to recent topics.
    Thompson Sampling retains diverse entries (exploration prevents over-specialization).
    """
    
    def __init__(self, K: int, embedding_dim: int, burst_length: int = None, 
                 num_topics: int = 5, seed: int = 42):
        self.K = K
        self.embedding_dim = embedding_dim
        self.burst_length = burst_length or (K * 2)  # long enough to fill memory
        self.num_topics = num_topics
        self.rng = np.random.RandomState(seed)
        
        # Generate topic embeddings
        self.topic_embeddings = np.zeros((num_topics, embedding_dim))
        for i in range(num_topics):
            vec = self.rng.randn(embedding_dim)
            for j in range(i):
                vec = vec - np.dot(vec, self.topic_embeddings[j]) * self.topic_embeddings[j]
            self.topic_embeddings[i] = vec / (np.linalg.norm(vec) + 1e-10)
        
        self.timestep = 0
        self.current_topic = 0
        self._steps_in_burst = 0
    
    def get_current_topic(self) -> int:
        """Burst pattern: stay on topic for burst_length, then switch."""
        if self._steps_in_burst >= self.burst_length:
            self.current_topic = (self.current_topic + 1) % self.num_topics
            self._steps_in_burst = 0
        return self.current_topic
    
    def generate_query(self) -> np.ndarray:
        topic_idx = self.get_current_topic()
        self._steps_in_burst += 1
        
        query = self.topic_embeddings[topic_idx].copy()
        query += self.rng.randn(self.embedding_dim) * 0.05
        query = query / (np.linalg.norm(query) + 1e-10)
        return query
    
    def generate_feedback(self, state: MemoryState, query: np.ndarray) -> Feedback:
        self.timestep += 1
        topic_idx = self.current_topic
        correct_embedding = self.topic_embeddings[topic_idx]
        
        if state.size > 0:
            results = state.retrieve(query, top_k=1)
            if results:
                sim = results[0][1]
                reward = sim if sim > 0.5 else 0.0
            else:
                reward = 0.0
        else:
            reward = 0.0
        
        feedback_score = 2 * reward - 1
        
        return Feedback(
            query=query,
            response_quality=feedback_score,
            feedback_type='explicit',
            suggested_content=correct_embedding.copy(),
            timestamp=self.timestep
        )


class SwitchingCostAdversary:
    """
    Adversary specifically for Switching Costs (Theorem 1).
    
    Alternates between two memory configurations every step.
    Forces Greedy to switch every step → unbounded switching cost.
    """
    
    def __init__(self, embedding_dim: int, seed: int = 42):
        self.embedding_dim = embedding_dim
        self.rng = np.random.RandomState(seed)
        
        # Two orthogonal configurations
        self.config_A = self.rng.randn(embedding_dim)
        self.config_A = self.config_A / np.linalg.norm(self.config_A)
        self.config_B = self.rng.randn(embedding_dim)
        # Make orthogonal to A
        self.config_B = self.config_B - np.dot(self.config_B, self.config_A) * self.config_A
        self.config_B = self.config_B / np.linalg.norm(self.config_B)
        
        self.timestep = 0
    
    def generate_query(self) -> np.ndarray:
        """Alternate between config A and B every step."""
        if self.timestep % 2 == 0:
            query = self.config_A.copy()
        else:
            query = self.config_B.copy()
        query += self.rng.randn(self.embedding_dim) * 0.01
        query = query / (np.linalg.norm(query) + 1e-10)
        return query
    
    def generate_feedback(self, state: MemoryState, query: np.ndarray) -> Feedback:
        self.timestep += 1
        
        if (self.timestep - 1) % 2 == 0:
            correct = self.config_A
        else:
            correct = self.config_B
        
        if state.size > 0:
            results = state.retrieve(query, top_k=1)
            if results:
                sim = results[0][1]
                reward = max(sim, 0)
            else:
                reward = 0.0
        else:
            reward = 0.0
        
        return Feedback(
            query=query,
            response_quality=2*reward - 1,
            feedback_type='explicit',
            suggested_content=correct.copy(),
            timestamp=self.timestep
        )


def run_adversarial_experiment(adversary, policy, T: int, K: int, embedding_dim: int):
    """Run a single adversarial experiment."""
    state = MemoryState(capacity=K, embedding_dim=embedding_dim)
    
    rewards = []
    cumulative_regret = []
    cum_reg = 0.0
    update_count = 0
    
    for t in range(T):
        query = adversary.generate_query()
        feedback = adversary.generate_feedback(state, query)
        
        should_update = policy.should_update(state, feedback)
        if should_update:
            state = policy.compute_update(state, feedback)
            update_count += 1
        
        # Reward = (feedback + 1) / 2, in [0, 1]
        reward = (feedback.response_quality + 1) / 2
        rewards.append(reward)
        cum_reg += (1.0 - reward)  # regret against perfect oracle
        cumulative_regret.append(cum_reg)
        
        state.timestamp = t
    
    # Fit regret exponent
    if T > 20:
        log_T = np.log(np.arange(1, T+1))
        log_R = np.log(np.array(cumulative_regret) + 1)
        alpha = np.polyfit(log_T, log_R, 1)[0]
    else:
        alpha = 1.0
    
    return {
        'avg_reward': np.mean(rewards),
        'final_reward': np.mean(rewards[-T//5:]),
        'cumulative_regret': cum_reg,
        'regret_exponent': alpha,
        'is_sublinear': alpha < 0.95,
        'update_count': update_count,
        'rewards': rewards,
        'cumulative_regret_curve': cumulative_regret,
    }


def experiment_cycling_adversary():
    """
    Main experiment: Cycling adversary (Theorem 4 construction).
    Verifies Greedy has linear regret, TS has sublinear.
    """
    print("\n" + "="*70)
    print("ADVERSARIAL EXPERIMENT 1: Cycling Adversary (K+1 topics, capacity K)")
    print("="*70)
    
    K = 10
    embedding_dim = 32
    T = 1000
    
    print(f"  K={K} (capacity), topics={K+1}, T={T}")
    print(f"  Prediction: Greedy gets 0 reward after warmup (linear regret)")
    print(f"  Prediction: TS maintains diversity → sublinear regret")
    
    policies = {
        'Greedy (LRU)': lambda: GreedyMemoryUpdate(),
        'LMU (λ=5)': lambda: LazyMemoryUpdate(SwitchingCostConfig(lambda_cost=5.0, lipschitz_L=2.0)),
        'LMU (λ=20)': lambda: LazyMemoryUpdate(SwitchingCostConfig(lambda_cost=20.0, lipschitz_L=2.0)),
        'Thompson Sampling': lambda: ThompsonSamplingMemoryPolicy(BanditConfig(capacity_K=K, embedding_dim=embedding_dim)),
        'IG-UCB (aggressive)': lambda: AggressiveIGPolicy(BanditConfig(capacity_K=K, embedding_dim=embedding_dim)),
    }
    
    results = {}
    for name, make_policy in policies.items():
        adversary = CyclingAdversary(K, embedding_dim, seed=42)
        policy = make_policy()
        result = run_adversarial_experiment(adversary, policy, T, K, embedding_dim)
        results[name] = result
        
        status = "SUBLINEAR ✓" if result['is_sublinear'] else "LINEAR ✗"
        print(f"\n  {name:25s}:")
        print(f"    Avg reward: {result['avg_reward']:.4f}")
        print(f"    Final reward: {result['final_reward']:.4f}")
        print(f"    Regret exponent α: {result['regret_exponent']:.3f} [{status}]")
        print(f"    Updates: {result['update_count']}")
    
    return results


def experiment_burst_adversary():
    """
    Burst adversary: long runs of one topic, then sudden switch.
    Tests whether methods can retain diversity under exploitation pressure.
    """
    print("\n" + "="*70)
    print("ADVERSARIAL EXPERIMENT 2: Burst Adversary (topic bursts)")
    print("="*70)
    
    K = 15
    embedding_dim = 32
    T = 1000
    burst = K * 3  # burst = 45 steps per topic
    
    print(f"  K={K}, burst_length={burst}, T={T}")
    print(f"  Prediction: Greedy over-commits to current burst → fails on switch")
    
    policies = {
        'Greedy (LRU)': lambda: GreedyMemoryUpdate(),
        'LMU (λ=10)': lambda: LazyMemoryUpdate(SwitchingCostConfig(lambda_cost=10.0, lipschitz_L=2.0)),
        'Thompson Sampling': lambda: ThompsonSamplingMemoryPolicy(BanditConfig(capacity_K=K, embedding_dim=embedding_dim)),
    }
    
    results = {}
    for name, make_policy in policies.items():
        adversary = BurstAdversary(K, embedding_dim, burst_length=burst, num_topics=5, seed=42)
        policy = make_policy()
        result = run_adversarial_experiment(adversary, policy, T, K, embedding_dim)
        results[name] = result
        
        status = "SUBLINEAR ✓" if result['is_sublinear'] else "LINEAR ✗"
        print(f"\n  {name:25s}:")
        print(f"    Avg reward: {result['avg_reward']:.4f}")
        print(f"    Final reward: {result['final_reward']:.4f}")
        print(f"    α: {result['regret_exponent']:.3f} [{status}]")
    
    return results


def experiment_switching_cost_adversary():
    """
    Switching cost adversary: alternates every step.
    Greedy switches every step → massive switching cost.
    LMU batches updates → much lower total cost.
    """
    print("\n" + "="*70)
    print("ADVERSARIAL EXPERIMENT 3: Switching Cost Adversary (Theorem 1)")
    print("="*70)
    
    K = 10
    embedding_dim = 32
    T = 500
    lambda_cost = 5.0
    
    print(f"  K={K}, T={T}, λ={lambda_cost}")
    print(f"  Prediction: Greedy total cost → λ·T (switches every step)")
    print(f"  Prediction: LMU total cost → much lower (batches switches)")
    
    from lazy_memory_update import SwitchingCostCalculator
    switching_calc = SwitchingCostCalculator(SwitchingCostConfig(lambda_cost=lambda_cost))
    
    policies = {
        'Greedy': lambda: GreedyMemoryUpdate(),
        'LMU (λ=5)': lambda: LazyMemoryUpdate(SwitchingCostConfig(lambda_cost=5.0, lipschitz_L=2.0)),
        'LMU (λ=20)': lambda: LazyMemoryUpdate(SwitchingCostConfig(lambda_cost=20.0, lipschitz_L=2.0)),
    }
    
    results = {}
    for name, make_policy in policies.items():
        adversary = SwitchingCostAdversary(embedding_dim, seed=42)
        policy = make_policy()
        state = MemoryState(capacity=K, embedding_dim=embedding_dim)
        
        states_history = [state.copy()]
        task_costs = []
        
        for t in range(T):
            query = adversary.generate_query()
            feedback = adversary.generate_feedback(state, query)
            
            should_update = policy.should_update(state, feedback)
            if should_update:
                state = policy.compute_update(state, feedback)
            
            task_cost = 1.0 - (feedback.response_quality + 1) / 2
            task_costs.append(task_cost)
            states_history.append(state.copy())
            state.timestamp = t
        
        # Compute switching costs
        total_switching = 0.0
        for i in range(1, len(states_history)):
            sc = switching_calc.compute(states_history[i-1], states_history[i])
            total_switching += sc
        
        task_total = sum(task_costs)
        total_cost = task_total + lambda_cost * total_switching
        
        results[name] = {
            'task_cost': task_total,
            'switching_cost': total_switching,
            'total_cost': total_cost,
            'num_updates': policy.update_count if hasattr(policy, 'update_count') else T,
        }
        
        print(f"\n  {name:20s}:")
        print(f"    Task cost: {task_total:.2f}")
        print(f"    Switching cost: {total_switching:.4f}")
        print(f"    Total (task + λ·switch): {total_cost:.2f}")
        print(f"    Updates: {results[name]['num_updates']}")
    
    # Competitive ratios
    best_total = min(r['total_cost'] for r in results.values())
    print(f"\n  Competitive Ratios (vs best={best_total:.2f}):")
    for name, r in results.items():
        cr = r['total_cost'] / best_total
        print(f"    {name:20s}: CR = {cr:.3f}")
    
    return results


def main():
    """Run all adversarial experiments."""
    print("="*70)
    print("ADVERSARIAL SEQUENCE VERIFICATION")
    print("Proving Greedy degrades under adversarial feedback")
    print("="*70)
    
    r1 = experiment_cycling_adversary()
    r2 = experiment_burst_adversary()
    r3 = experiment_switching_cost_adversary()
    
    # Summary
    print("\n\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print("\n  Cycling Adversary (Theorem 4):")
    print(f"    Greedy α = {r1['Greedy (LRU)']['regret_exponent']:.3f} " + 
          ("✗ LINEAR" if not r1['Greedy (LRU)']['is_sublinear'] else "? not linear"))
    print(f"    TS α = {r1['Thompson Sampling']['regret_exponent']:.3f} " + 
          ("✓ SUBLINEAR" if r1['Thompson Sampling']['is_sublinear'] else "✗"))
    
    print("\n  Burst Adversary:")
    print(f"    Greedy α = {r2['Greedy (LRU)']['regret_exponent']:.3f}")
    print(f"    TS α = {r2['Thompson Sampling']['regret_exponent']:.3f}")
    
    print("\n  Switching Cost Adversary (Theorem 1):")
    best = min(r3.values(), key=lambda x: x['total_cost'])
    for name, r in r3.items():
        cr = r['total_cost'] / best['total_cost']
        print(f"    {name}: CR={cr:.2f} (total={r['total_cost']:.1f})")
    
    # Save
    all_results = {}
    for name, r in r1.items():
        all_results[f"cycling_{name}"] = {k: v for k, v in r.items() 
                                           if k not in ('rewards', 'cumulative_regret_curve')}
    for name, r in r2.items():
        all_results[f"burst_{name}"] = {k: v for k, v in r.items() 
                                         if k not in ('rewards', 'cumulative_regret_curve')}
    for name, r in r3.items():
        all_results[f"switching_{name}"] = r
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adversarial_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    
    print(f"\n\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
