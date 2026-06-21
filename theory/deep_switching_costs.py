"""
Deep Dive: Switching Costs Theory Validation

Experiments to validate the theoretical predictions:
1. CR(greedy) grows with T (unbounded)
2. CR(LMU) ≈ 1 + √(2λ/(LT)) (converges to 1)
3. Optimal threshold τ* = √(2λ/L) is indeed optimal
4. Sensitivity to λ: phase transition from greedy-better to LMU-better
"""

import numpy as np
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory_state import MemoryState, MemoryEntry, Feedback
from lazy_memory_update import (
    LazyMemoryUpdate, GreedyMemoryUpdate, NeverUpdatePolicy,
    SwitchingCostConfig, SwitchingCostCalculator
)


def compute_total_cost_with_switching(policy_name, states_history, costs, 
                                       lambda_cost, switching_calc):
    """
    Compute TRUE total cost = task costs + λ * switching costs.
    This is the actual objective being optimized.
    """
    task_total = sum(costs)
    switching_total = 0.0
    
    for i in range(1, len(states_history)):
        sc = switching_calc.compute(states_history[i-1], states_history[i])
        switching_total += sc
    
    return task_total + lambda_cost * switching_total, task_total, switching_total


def run_with_tracking(policy, domain_sim, T, embedding_dim, capacity):
    """Run policy and track full state history for competitive ratio computation."""
    from run_experiments import DomainSimulator
    
    state = MemoryState(capacity=capacity, embedding_dim=embedding_dim)
    states_history = [state.copy()]
    costs = []
    
    for t in range(T):
        query = domain_sim.generate_query()
        feedback = domain_sim.generate_feedback(state, query)
        
        should_update = policy.should_update(state, feedback)
        if should_update:
            state = policy.compute_update(state, feedback)
        
        # Track cost (retrieval quality)
        cost = 1.0 - (feedback.response_quality + 1) / 2
        costs.append(cost)
        states_history.append(state.copy())
        state.timestamp = t
    
    return states_history, costs


def experiment_1_cr_vs_T():
    """
    Experiment 1: How does competitive ratio grow with T?
    Prediction: CR(greedy) grows, CR(LMU) → 1
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Competitive Ratio vs Time Horizon T")
    print("="*70)
    
    from run_experiments import DomainSimulator, DomainConfig
    
    embedding_dim = 32
    capacity = 20
    lambda_cost = 2.0
    L = 2.0
    
    T_values = [50, 100, 200, 500, 1000]
    results = {'T': T_values, 'greedy_total': [], 'lmu_total': [], 
               'never_total': [], 'cr_greedy': [], 'cr_lmu': [],
               'theory_cr_lmu': []}
    
    switching_calc = SwitchingCostCalculator(SwitchingCostConfig(lambda_cost=lambda_cost))
    
    for T in T_values:
        seed = 42
        domain_config = DomainConfig(
            name='test', embedding_dim=embedding_dim,
            mean_shift_rate=0.02, noise_level=0.2,
            num_topics=5, topic_switch_prob=0.1, adversarial_prob=0.05
        )
        
        # Run greedy
        domain = DomainSimulator(domain_config, seed)
        greedy = GreedyMemoryUpdate()
        g_states, g_costs = run_with_tracking(greedy, domain, T, embedding_dim, capacity)
        g_total, g_task, g_switch = compute_total_cost_with_switching(
            'greedy', g_states, g_costs, lambda_cost, switching_calc)
        
        # Run LMU
        domain = DomainSimulator(domain_config, seed)
        lmu = LazyMemoryUpdate(SwitchingCostConfig(lambda_cost=lambda_cost, lipschitz_L=L))
        l_states, l_costs = run_with_tracking(lmu, domain, T, embedding_dim, capacity)
        l_total, l_task, l_switch = compute_total_cost_with_switching(
            'lmu', l_states, l_costs, lambda_cost, switching_calc)
        
        # Run never (as reference for "optimal offline" lower bound estimate)
        domain = DomainSimulator(domain_config, seed)
        never = NeverUpdatePolicy()
        n_states, n_costs = run_with_tracking(never, domain, T, embedding_dim, capacity)
        n_total, n_task, n_switch = compute_total_cost_with_switching(
            'never', n_states, n_costs, lambda_cost, switching_calc)
        
        # Estimate OPT (best offline) ≈ min(greedy_task_only, lmu_total)
        # A loose lower bound: optimal can't do better than min task cost with zero switching
        opt_estimate = min(g_task, l_total)  # optimistic
        
        cr_greedy = g_total / max(opt_estimate, 0.01)
        cr_lmu = l_total / max(opt_estimate, 0.01)
        theory_cr = 1.0 + np.sqrt(2 * lambda_cost / (L * T))
        
        results['greedy_total'].append(g_total)
        results['lmu_total'].append(l_total)
        results['never_total'].append(n_total)
        results['cr_greedy'].append(cr_greedy)
        results['cr_lmu'].append(cr_lmu)
        results['theory_cr_lmu'].append(theory_cr)
        
        print(f"\n  T={T:4d}: Greedy total={g_total:.2f} (task={g_task:.2f}, switch={g_switch:.2f})")
        print(f"         LMU total={l_total:.2f} (task={l_task:.2f}, switch={l_switch:.2f})")
        print(f"         CR(Greedy)={cr_greedy:.3f}, CR(LMU)={cr_lmu:.3f}, Theory={theory_cr:.3f}")
    
    print(f"\n  VERDICT: Greedy CR trend: {['↑' if results['cr_greedy'][i] < results['cr_greedy'][i+1] else '↓' for i in range(len(T_values)-1)]}")
    print(f"           LMU CR trend:    {['↑' if results['cr_lmu'][i] < results['cr_lmu'][i+1] else '↓' for i in range(len(T_values)-1)]}")
    
    return results


def experiment_2_lambda_sensitivity():
    """
    Experiment 2: At what λ does LMU become better than Greedy?
    The crossover point validates the theory.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: λ Sensitivity — When Does LMU Beat Greedy?")
    print("="*70)
    
    from run_experiments import DomainSimulator, DomainConfig
    
    embedding_dim = 32
    capacity = 20
    T = 300
    L = 2.0
    
    lambda_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    results = {'lambda': lambda_values, 'greedy_total': [], 'lmu_total': [],
               'greedy_wins': [], 'crossover_point': None}
    
    domain_config = DomainConfig(
        name='test', embedding_dim=embedding_dim,
        mean_shift_rate=0.02, noise_level=0.2,
        num_topics=5, topic_switch_prob=0.1, adversarial_prob=0.05
    )
    
    prev_winner = None
    
    for lam in lambda_values:
        switching_calc = SwitchingCostCalculator(SwitchingCostConfig(lambda_cost=lam))
        seed = 42
        
        # Greedy
        domain = DomainSimulator(domain_config, seed)
        greedy = GreedyMemoryUpdate()
        g_states, g_costs = run_with_tracking(greedy, domain, T, embedding_dim, capacity)
        g_total, _, _ = compute_total_cost_with_switching('greedy', g_states, g_costs, lam, switching_calc)
        
        # LMU
        domain = DomainSimulator(domain_config, seed)
        lmu = LazyMemoryUpdate(SwitchingCostConfig(lambda_cost=max(lam, 0.01), lipschitz_L=L))
        l_states, l_costs = run_with_tracking(lmu, domain, T, embedding_dim, capacity)
        l_total, _, _ = compute_total_cost_with_switching('lmu', l_states, l_costs, lam, switching_calc)
        
        greedy_wins = g_total < l_total
        results['greedy_total'].append(g_total)
        results['lmu_total'].append(l_total)
        results['greedy_wins'].append(greedy_wins)
        
        winner = 'Greedy' if greedy_wins else 'LMU'
        if prev_winner and winner != prev_winner and results['crossover_point'] is None:
            results['crossover_point'] = lam
        prev_winner = winner
        
        print(f"  λ={lam:5.1f}: Greedy={g_total:.2f}, LMU={l_total:.2f} → Winner: {winner}")
    
    if results['crossover_point']:
        print(f"\n  CROSSOVER at λ ≈ {results['crossover_point']}")
        print(f"  Theory predicts: LMU dominates when λ > task_cost_variance/T")
    else:
        # Check if LMU always wins or greedy always wins
        if all(results['greedy_wins']):
            print("\n  Greedy always wins (λ range too small to see crossover)")
        elif not any(results['greedy_wins']):
            print("\n  LMU always wins (even at λ=0, LMU's selective updating helps)")
    
    return results


def experiment_3_threshold_optimality():
    """
    Experiment 3: Is τ* = √(2λ/L) really optimal?
    Test different thresholds and find empirical optimum.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Threshold Optimality — Is τ* = √(2λ/L) Best?")
    print("="*70)
    
    from run_experiments import DomainSimulator, DomainConfig
    
    embedding_dim = 32
    capacity = 20
    T = 500
    lambda_cost = 2.0
    L = 2.0
    
    tau_star = np.sqrt(2 * lambda_cost / L)
    
    # Test thresholds around τ*
    multipliers = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    thresholds = [tau_star * m for m in multipliers]
    
    domain_config = DomainConfig(
        name='test', embedding_dim=embedding_dim,
        mean_shift_rate=0.02, noise_level=0.2,
        num_topics=5, topic_switch_prob=0.1, adversarial_prob=0.05
    )
    
    results = {'multiplier': multipliers, 'threshold': thresholds, 
               'total_cost': [], 'task_cost': [], 'switching_cost': [],
               'num_updates': []}
    
    switching_calc = SwitchingCostCalculator(SwitchingCostConfig(lambda_cost=lambda_cost))
    
    for mult, thresh in zip(multipliers, thresholds):
        seed = 42
        domain = DomainSimulator(domain_config, seed)
        lmu = LazyMemoryUpdate(SwitchingCostConfig(
            lambda_cost=lambda_cost, lipschitz_L=L,
            threshold_mode='fixed', fixed_threshold=thresh
        ))
        states, costs = run_with_tracking(lmu, domain, T, embedding_dim, capacity)
        total, task, switch = compute_total_cost_with_switching(
            'lmu', states, costs, lambda_cost, switching_calc)
        
        num_updates = lmu.update_count
        results['total_cost'].append(total)
        results['task_cost'].append(task)
        results['switching_cost'].append(switch)
        results['num_updates'].append(num_updates)
        
        marker = " ← τ*" if abs(mult - 1.0) < 0.01 else ""
        print(f"  τ = {mult:.2f}·τ* = {thresh:.3f}: "
              f"total={total:.2f} (task={task:.2f} + λ·switch={lambda_cost*switch:.2f}), "
              f"updates={num_updates}{marker}")
    
    # Find empirical best
    best_idx = np.argmin(results['total_cost'])
    print(f"\n  Empirical best: τ = {multipliers[best_idx]:.2f}·τ* (total={results['total_cost'][best_idx]:.2f})")
    print(f"  Theoretical optimum: τ = 1.00·τ* = {tau_star:.3f}")
    print(f"  Gap: {abs(multipliers[best_idx] - 1.0)*100:.0f}% from theory")
    
    return results


def experiment_4_domain_robustness():
    """
    Experiment 4: Does LMU's advantage hold across different domain types?
    This validates that the theory is domain-agnostic.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Domain Robustness — Does LMU Generalize?")
    print("="*70)
    
    from run_experiments import DomainSimulator, DomainConfig
    
    embedding_dim = 32
    capacity = 20
    T = 500
    lambda_cost = 3.0
    L = 2.0
    
    domains = {
        'very_stable': DomainConfig(name='Very Stable', embedding_dim=embedding_dim,
                                     mean_shift_rate=0.001, noise_level=0.05,
                                     num_topics=2, topic_switch_prob=0.02, adversarial_prob=0.01),
        'moderate': DomainConfig(name='Moderate', embedding_dim=embedding_dim,
                                  mean_shift_rate=0.01, noise_level=0.2,
                                  num_topics=5, topic_switch_prob=0.1, adversarial_prob=0.05),
        'highly_dynamic': DomainConfig(name='Highly Dynamic', embedding_dim=embedding_dim,
                                        mean_shift_rate=0.1, noise_level=0.4,
                                        num_topics=10, topic_switch_prob=0.3, adversarial_prob=0.1),
        'adversarial': DomainConfig(name='Adversarial', embedding_dim=embedding_dim,
                                     mean_shift_rate=0.02, noise_level=0.6,
                                     num_topics=5, topic_switch_prob=0.15, adversarial_prob=0.25),
    }
    
    switching_calc = SwitchingCostCalculator(SwitchingCostConfig(lambda_cost=lambda_cost))
    results = {}
    
    for domain_name, domain_config in domains.items():
        seed = 42
        
        # Greedy
        domain = DomainSimulator(domain_config, seed)
        greedy = GreedyMemoryUpdate()
        g_states, g_costs = run_with_tracking(greedy, domain, T, embedding_dim, capacity)
        g_total, g_task, g_switch = compute_total_cost_with_switching(
            'greedy', g_states, g_costs, lambda_cost, switching_calc)
        
        # LMU
        domain = DomainSimulator(domain_config, seed)
        lmu = LazyMemoryUpdate(SwitchingCostConfig(lambda_cost=lambda_cost, lipschitz_L=L))
        l_states, l_costs = run_with_tracking(lmu, domain, T, embedding_dim, capacity)
        l_total, l_task, l_switch = compute_total_cost_with_switching(
            'lmu', l_states, l_costs, lambda_cost, switching_calc)
        
        advantage = (g_total - l_total) / g_total * 100
        results[domain_name] = {
            'greedy': g_total, 'lmu': l_total, 
            'advantage_%': advantage,
            'greedy_updates': T, 'lmu_updates': lmu.update_count
        }
        
        winner = "LMU" if l_total < g_total else "Greedy"
        print(f"  {domain_name:15s}: Greedy={g_total:.1f}, LMU={l_total:.1f} "
              f"→ {winner} wins by {abs(advantage):.1f}% "
              f"(LMU uses {lmu.update_count}/{T} updates)")
    
    # Summary
    lmu_wins = sum(1 for v in results.values() if v['advantage_%'] > 0)
    print(f"\n  LMU wins {lmu_wins}/{len(domains)} domains")
    print(f"  Average advantage: {np.mean([v['advantage_%'] for v in results.values()]):.1f}%")
    
    return results


if __name__ == '__main__':
    r1 = experiment_1_cr_vs_T()
    r2 = experiment_2_lambda_sensitivity()
    r3 = experiment_3_threshold_optimality()
    r4 = experiment_4_domain_robustness()
    
    # Save all results
    all_results = {
        'experiment_1_cr_vs_T': {k: [float(x) if isinstance(x, (np.floating, float)) else x 
                                      for x in v] if isinstance(v, list) else v
                                  for k, v in r1.items()},
        'experiment_2_lambda': {k: [float(x) if isinstance(x, (np.floating, float)) else x 
                                     for x in v] if isinstance(v, list) else v
                                 for k, v in r2.items()},
        'experiment_3_threshold': {k: [float(x) if isinstance(x, (np.floating, float)) else x 
                                        for x in v] if isinstance(v, list) else v
                                    for k, v in r3.items()},
        'experiment_4_domains': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                                      for kk, vv in v.items()}
                                  for k, v in r4.items()},
    }
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'switching_costs_deep.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n\nResults saved to experiments/switching_costs_deep.json")
