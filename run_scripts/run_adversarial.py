"""
Experiment 3: Adversarial Verification

Runs the adversarial sequence experiments that prove:
  - FIFO/LRU incur linear regret under a cycling adversary
  - SOLAR-E (Thompson-sampling eviction) maintains sublinear regret
  - SOLAR's regret-gated admission gives bounded switching cost vs FIFO's unbounded cost

This is a pure numpy simulation — no LLM calls needed.

Usage:
  python run_scripts/run_adversarial.py
  python run_scripts/run_adversarial.py --output adversarial_results/
"""

import os
import sys

# Add project root and theory directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "theory"))

from theory.adversarial_verification import main as run_adversarial_main


if __name__ == "__main__":
    run_adversarial_main()
