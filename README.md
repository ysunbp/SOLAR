# SOLAR: Semantic Online Learning-Augmented Replacement

SOLAR is a learning-augmented cache-replacement policy for the **experience memory** of LLM agents. It treats an agent's retrieval buffer as an *online semantic cache* and decides, on the fly, **when** to modify the cache (regret-gated admission) and **what** to evict (Bayesian, posterior-guided selection). SOLAR needs no offline training and no extra LLM calls, adding `<1 ms` of overhead per step.

This repository contains the reference implementation of SOLAR, the classic cache-replacement baselines we compare against (FIFO, LRU, LFU, ARC), the on-policy evaluation pipeline on LoCoMo and DialSim, and the synthetic-workload / theory-verification experiments from the paper.

> **Method name mapping (paper ↔ code).**
>
> | Paper name | Code name (`--method`) | Config | Description |
> |------------|------------------------|--------|-------------|
> | **SOLAR**   | `solar`    | `solar.json`   | Full framework: regret-gated timing + posterior-guided eviction |
> | **SOLAR-A** | `solar_a`  | `solar_a.json` | Admission only (regret-gated timing + heuristic eviction) |
> | **SOLAR-E** | `solar_e`  | `solar_e.json` | Eviction only (always admit + Thompson-sampling eviction) |
> | FIFO        | `fifo`     | `fifo.json`    | First-in-first-out baseline |
> | LRU         | `lru`      | `lru.json`     | Least-recently-used baseline |
> | LFU         | `lfu`      | `lfu.json`     | Least-frequently-used baseline |
> | ARC         | `arc`      | `arc.json`     | Adaptive Replacement Cache baseline |
> | Embedder    | `embedder_message` | `embedder.json` | Unlimited-capacity reference (no eviction) |

## Repository Structure

```plain
configs/
    datasets/             # Dataset grouping configs (each / domain / task)
    memory_systems/       # Per-method hyper-parameter configs (solar.json = SOLAR, ...)
raw/
    Locomo/               # LoCoMo raw conversations (corpus source)
    DialSim/              # DialSim raw dialogues (corpus source)
run_scripts/
    run_experience_eval.py  # Main on-policy evaluation entry (LoCoMo / DialSim)
    run_adversarial.py      # Cycling-workload adversarial verification
src/
    agent/                # Cache-policy agents (solar, solar_a, solar_e, fifo, lru, lfu, arc, embedder)
    solver/               # Solver wrappers + SolverFactory
    dataset/              # Locomo / DialSim dataset loaders
    llms/                 # OpenAI-compatible LLM / embedder clients
    utils.py
synthetic_workloads/      # Synthetic data generators + controlled experiments
theory/                   # Theoretical verification (competitive ratio, regret, adversarial)
```

## Getting Started

### 1. Environment

```bash
conda create -n solar python=3.10
conda activate solar
pip install -r requirements.txt
```

### 2. Configure API Keys

SOLAR calls an **OpenAI-compatible** chat endpoint for two purposes: (1) the agent's response generation and (2) LLM-as-judge scoring of those responses. Any provider that speaks the OpenAI API format works — the official OpenAI API, Azure OpenAI, or a locally served model (e.g. vLLM / Ollama with an OpenAI-compatible server).

Copy the template and fill in your own credentials:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
# Judge / evaluation model (LLM-as-judge scoring)
EVALUATE_BASE_URL="https://api.openai.com/v1"
EVALUATE_MODEL="gpt-4o-mini"
EVALUATE_API_KEY="sk-..."        # <-- your key here

# Generation model (used by the memory agent)
OPENAI_BASE_URL="https://api.openai.com/v1"
OPENAI_API_KEY="sk-..."          # <-- your key here
```

- **Official OpenAI:** set `*_BASE_URL` to `https://api.openai.com/v1` and use a key starting with `sk-`.
- **Local model (vLLM/Ollama):** point `*_BASE_URL` at your server, e.g. `http://localhost:8000/v1`; the key can be any non-empty placeholder. To serve a model with vLLM:

  ```bash
  vllm serve Qwen/Qwen3-8B --port 8000 --chat-template qwen3_nonthinking.jinja
  ```

The default embedder is `sentence-transformers/all-MiniLM-L6-v2` (384-dim), downloaded automatically from HuggingFace on first run. You can change it in `configs/memory_systems/*.json`.

### 3. Datasets

We evaluate on two datasets from the **MemoryBench-Full** benchmark: **LoCoMo** and **DialSim**. MemoryBench is a third-party benchmark and is **not redistributed** in this repository; only the raw conversation corpora needed as input live under `raw/`. The QA splits are pulled automatically:

- **QA splits** are downloaded at runtime from HuggingFace [`THUIR/MemoryBench-Full`](https://huggingface.co/datasets/THUIR/MemoryBench-Full). Each session provides a `train` split (used as a warm-up stream to accumulate experience; **not scored**) and a `test` split (the reported F1). No manual download is needed — the `datasets` library fetches it on first use. Pass `--use-mb-lite` to use the smaller `THUIR/MemoryBench` for quick smoke tests.
- **Corpora** (the raw multi-session conversations) are read locally from `raw/Locomo/locomo10.json` and `raw/DialSim/`. These are included for convenience. If you need to refresh them, obtain the original LoCoMo and DialSim releases from their respective sources and place them under `raw/` in the same layout.

On first run the HuggingFace cache and embedder weights are downloaded automatically; subsequent runs are offline-capable.

## Running Experiments

### On-policy evaluation (LoCoMo / DialSim)

The protocol preloads the conversation corpus (shared by all methods), warms up each policy on the train split to fill the cache, then reports F1 on the held-out test split.

```bash
# Run SOLAR on a single LoCoMo session
python run_scripts/run_experience_eval.py --method solar --datasets Locomo-0

# Compare all methods on one session at capacity K=50
python run_scripts/run_experience_eval.py --all --datasets Locomo-0 --capacity 50

# Full sweep: all methods over capacities and seeds, all LoCoMo sessions
python run_scripts/run_experience_eval.py --all \
    --datasets Locomo-0 Locomo-1 Locomo-2 Locomo-3 Locomo-4 \
               Locomo-5 Locomo-6 Locomo-7 Locomo-8 Locomo-9 \
    --capacity-sweep 10 20 50 100 --seed-sweep 42 1337 2024

# Cross-dataset validation on DialSim
python run_scripts/run_experience_eval.py --all --all-dialsim \
    --capacity-sweep 10 20 50 100 --seed-sweep 42 1337 2024
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--method NAME` / `--methods N1 N2` | run one / several methods (see mapping table) |
| `--all` | run all 8 methods |
| `--datasets ...` / `--all-dialsim` | choose LoCoMo sessions / all DialSim shows |
| `--capacity K` / `--capacity-sweep ...` | cache size(s) `K` |
| `--seed K` / `--seed-sweep ...` | random seed(s); use ≥3 seeds for reportable numbers |
| `--lambda-cost FLOAT` | switching cost `λ` for SOLAR's admission threshold |
| `--use-mb-lite` | smaller QA split for fast smoke tests |
| `--output PATH` | results directory |

Key hyper-parameters for SOLAR live in `configs/memory_systems/solar.json` (e.g. `lambda_cost`, `capacity`, `threshold_mode`, `novelty_weight`).

### Synthetic workloads & theory verification

```bash
# Cycling-workload adversarial verification (FIFO thrashing, Theorem on FIFO regret)
python run_scripts/run_adversarial.py
```

The `synthetic_workloads/` directory contains the generators and controlled experiments (cycling workload, working-set sweep / phase transition, retrieval-noise U-curve); see `synthetic_workload_spec.md` for the full specification. The `theory/` directory holds the numerical verification of the competitive-ratio and regret bounds.

## Citation

If you find SOLAR useful, please cite our paper. (BibTeX to be added upon publication.)

## License

See [LICENSE](LICENSE).
