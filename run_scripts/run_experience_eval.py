"""
Experiment 2: Experience Learning Evaluation (On-Policy Protocol)

Aligned with MemoryBench official on-policy evaluation:

  Data source
  -----------
  - QA: HuggingFace `THUIR/MemoryBench-Full` (default) — the official
         extended release that covers the entire LoCoMo QA set (199 / session).
         Pass `--use-mb-lite` to fall back to the smaller `THUIR/MemoryBench`
         (25 / session) for fast smoke tests.
      * `train` split (160 QA / session in Full, 20 in Lite): used as a
        **warm-up** stream — each method runs on-policy through these QAs
        to *accumulate* experience. **F1 from this phase is NOT reported.**
      * `test`  split (39 QA / session in Full, 5 in Lite): used as the
        **reportable** evaluation stage. F1 metrics, slope, and improvement
        are computed on these held-out QAs only.
      * Lite’s 5 test QAs are a strict subset of Full’s 39 test QAs (verified
        on Locomo-0), so any prior Lite results remain comparable.
  - Corpus: original LoCoMo conversation from `raw/Locomo/locomo10.json`
         (MemoryBench HF dataset itself does not redistribute the raw dialogues).

  IMPORTANT — fields NOT used in this experiment
  ----------------------------------------------
  Each HF row also carries the pre-generated columns
    `dialog_<method>` and `implicit_feedback_<method>` for
    method ∈ {bm25, bm25_dialog, embedder, embedder_dialog, mem0, a_mem, memoryos}.
  Those columns are MemoryBench's *off-policy* material (replaying baselines'
  prior trajectories). They are intentionally **ignored** here, because we are
  evaluating an on-policy protocol where each method generates its own dialog
  and feedback in real time.

  Phase 1: Preload corpus (shared across all methods)
    - Load the full conversation history into a read-only FAISS index
    - Identical for all methods — no policy involved

  Phase 2a: Train-split warm-up (on-policy experience accumulation, NO scoring)
    For each QA in train split (ordered by `test_idx`):
      1. Retrieve from BOTH corpus memory AND experience memory
      2. LLM generates answer using combined context
      3. Score (silently — used only as the policy's feedback signal)
      4. Memory policy decides: store this experience or not?
    F1 statistics from this phase are written into a separate
    `warmup_*` field for diagnostics, but are NOT the headline number.

  Phase 2b: Test-split evaluation (reportable F1)
    Same loop on the 5 test QAs. These are the numbers we report.
    Experience memory carries over from Phase 2a — meaning whatever the
    policy chose to keep is what the model can use here.

  Methods: embedder / FIFO / LRU / LFU / ARC / SOLAR-A / SOLAR-E / SOLAR
    - Greedy (embedder_message): store every experience
    - FIFO: store every experience, evict oldest when full
    - SOLAR-A (solar_a): feedback-score-based regret-gated admission
    - SOLAR-E (solar_e): store every experience, Thompson-sampling eviction
    - SOLAR (solar): regret-gated admission + Thompson-sampling eviction

  Key insight:
    - Corpus is shared (all methods have the same factual knowledge)
    - The ONLY difference is how dialog history (QA experiences) is managed
    - The train-split warm-up gives the policy enough material (~140-194 QA)
      to actually fill capacity=50 and trigger evictions, so the
      switching-cost / capacity story is observable on the 5 test QAs.

  Multi-seed notes
  ----------------
  With Full (default) the test split is 39 QA / session, so single-seed
  numbers are far more stable than the 5-QA Lite split. Still, for any
  reportable number, run `--seed-sweep S1 S2 S3 ...` (≥3 seeds). The seed
  controls (a) the order in which train QAs are fed to the policy and
  (b) the order of test QAs. Test set composition itself is fixed by HF.

Usage:
  # Default (recommended): full on-policy protocol = train warm-up + test eval
  python run_scripts/run_experience_eval.py --datasets Locomo-0

  # Skip the warm-up (test-split-only mode, faster but less informative)
  python run_scripts/run_experience_eval.py --datasets Locomo-0 --no-train-warmup

  # Smoke-test on the smaller Lite split (THUIR/MemoryBench, 5 test QA)
  python run_scripts/run_experience_eval.py --datasets Locomo-0 --use-mb-lite

  # Run a single method on a specific session
  python run_scripts/run_experience_eval.py --method solar --datasets Locomo-0 Locomo-1

  # Run with custom experience capacity
  python run_scripts/run_experience_eval.py --capacity 50

  # Capacity sweep
  python run_scripts/run_experience_eval.py --capacity-sweep 20 50 100 200

  # Multi-seed sweep (recommended for any reportable number)
  python run_scripts/run_experience_eval.py --all --seed-sweep 42 1337 2024

  # Legacy mode: use the full 199 QA from raw/Locomo/locomo10.json instead of HF
  python run_scripts/run_experience_eval.py --legacy-full-locomo --datasets Locomo-0
"""

import os
import sys
import json
import copy
import time
import shutil
import random
import string
import faiss
import numpy as np
import regex
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from argparse import ArgumentParser
from collections import Counter
from tqdm import tqdm
from dotenv import load_dotenv
from nltk.stem import PorterStemmer

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.Locomo import Locomo_Dataset
# NOTE: SolverFactory is imported lazily inside build_solver() to keep
# `--help` and lightweight tooling fast and import-error-free.
from src.utils import get_memory_system_config_file, change_dialsim_conversation_to_locomo_form

# HuggingFace datasets (optional dependency; only needed for --use-hf, default ON)
try:
    from datasets import load_dataset as _hf_load_dataset
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False


# ============================================================
# Official LoCoMo Scoring (aligned with MemoryBench)
# ============================================================

_ps = PorterStemmer()


def normalize_answer(s: str) -> str:
    """Official LoCoMo answer normalization."""
    s = s.replace(',', '')
    s = regex.sub(r'\b(a|an|the|and)\b', ' ', s.lower())
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    return ' '.join(s.split())


def f1_score_official(prediction: str, ground_truth: str) -> float:
    """Official LoCoMo token-level F1 with Porter stemming."""
    prediction_tokens = [_ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [_ps.stem(w) for w in normalize_answer(ground_truth).split()]
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1_multi_hop(prediction: str, ground_truth: str) -> float:
    """Official LoCoMo multi-hop F1."""
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    return float(np.mean(
        [max([f1_score_official(pred, gt) for pred in predictions]) for gt in ground_truths]
    ))


def get_cat_5_answer(model_prediction: str, answer_key: dict) -> str:
    """Official LoCoMo adversarial answer extraction."""
    model_prediction = model_prediction.strip().lower()
    if len(model_prediction) == 1:
        return answer_key.get('a', model_prediction) if 'a' in model_prediction else answer_key.get('b', model_prediction)
    elif len(model_prediction) == 3:
        return answer_key.get('a', model_prediction) if '(a)' in model_prediction else answer_key.get('b', model_prediction)
    return model_prediction


def compute_locomo_f1(prediction: str, golden_answer, category: int) -> float:
    """Official LoCoMo F1, category-aware."""
    if category == 5:
        output = get_cat_5_answer(prediction, golden_answer) if isinstance(golden_answer, dict) else prediction.strip()
        if 'no information available' in output.lower() or 'not mentioned' in output.lower():
            return 1.0
        else:
            return 0.0

    output = prediction.strip()
    answer = str(golden_answer)

    if category == 2:
        answer = answer.split(';')[0].strip()
        return f1_score_official(output, answer)
    elif category == 3:
        answer = answer.split(';')[0].strip()
        return f1_score_official(output, answer)
    elif category == 1:
        return f1_multi_hop(output, answer)
    elif category == 4:
        return f1_score_official(output, answer)
    else:
        return f1_score_official(output, answer)


# ============================================================
# Configuration
# ============================================================

OUR_METHODS = [
    "embedder_message",  # Unlimited-capacity reference (no eviction)
    "fifo",              # FIFO eviction baseline
    "lru",               # LRU eviction baseline
    "lfu",               # LFU eviction baseline
    "arc",               # ARC (Adaptive Replacement Cache) baseline
    "solar_a",           # SOLAR-A: admission only (ours, ablation)
    "solar_e",           # SOLAR-E: eviction only via Thompson sampling (ours, ablation)
    "solar",             # SOLAR: full framework (ours)
]

LOCOMO_DATASETS = [f"Locomo-{i}" for i in range(10)]
DIALSIM_DATASETS = ["DialSim-friends", "DialSim-bigbang", "DialSim-theoffice"]
DEFAULT_LOCOMO_PATH = "./raw/Locomo/locomo10.json"  # source of corpus & legacy full-QA mode
DEFAULT_DIALSIM_PATH = "./raw/DialSim"  # source of DialSim corpus
# Default: full LoCoMo coverage from MemoryBench-Full (160 train + 39 test
# per session = the original 199 QA / session, just split into train/test).
# Lite mode (--use-mb-lite) falls back to MemoryBench (20 train + 5 test).
MEMORYBENCH_FULL_HF_NAME = "THUIR/MemoryBench-Full"
MEMORYBENCH_LITE_HF_NAME = "THUIR/MemoryBench"
MEMORYBENCH_HF_NAME = MEMORYBENCH_FULL_HF_NAME  # populated at CLI parse time
MEMORYBENCH_TEST_SPLIT = "test"   # 39 QA (Full) / 5 QA (Lite), reportable F1
MEMORYBENCH_TRAIN_SPLIT = "train"  # 160 QA (Full) / 20 QA (Lite), warm-up only

# Category names for reporting
CAT_NAMES = {1: "multi-hop", 2: "temporal", 3: "single-hop", 4: "open-domain", 5: "adversarial"}


# ============================================================
# QA loaders — option B: pull QA from THUIR/MemoryBench test split
# ============================================================

def _parse_info_field(info_raw):
    """`info` may arrive as a dict (HF Sequence) or a JSON-encoded string."""
    if isinstance(info_raw, dict):
        return info_raw
    if isinstance(info_raw, str):
        try:
            return json.loads(info_raw)
        except json.JSONDecodeError:
            return {}
    return {}


def load_qa_from_hf(dataset_name: str, split: str = MEMORYBENCH_TEST_SPLIT) -> List[Dict]:
    """
    Load QA samples for a single Locomo-X session from THUIR/MemoryBench[-Full].

    The exact HF repo is taken from the module-level `MEMORYBENCH_HF_NAME`,
    which the CLI sets to either `THUIR/MemoryBench-Full` (default) or
    `THUIR/MemoryBench` (when `--use-mb-lite` is passed).

    Args:
        dataset_name: e.g. "Locomo-0".
        split: "test" (39 QA in Full / 5 in Lite, reportable) or
               "train" (160 QA in Full / 20 in Lite, warm-up).

    Returns rows shaped like Locomo_Dataset.dataset entries so the rest of the
    pipeline can stay unchanged:
        {
            "test_idx": int,
            "origin_question": str,
            "info": {
                "golden_answer": ...,
                "category": int,
                "evidence": [...]
            },
            "lang": str,
        }

    NOTE: the pre-generated `dialog_<method>` / `implicit_feedback_<method>`
    columns from MemoryBench are deliberately discarded — they are off-policy
    artifacts, while this experiment is on-policy. We only keep the four
    "clean" columns above (no leakage between train and test).
    """
    if not _HF_AVAILABLE:
        raise RuntimeError(
            "`datasets` library not available — install it (`pip install datasets`) "
            "or use --legacy-full-locomo to fall back to raw/Locomo/locomo10.json."
        )

    ds = _hf_load_dataset(MEMORYBENCH_HF_NAME, dataset_name, split=split)
    is_dialsim = dataset_name.startswith("DialSim")
    samples = []
    for row in ds:
        info = _parse_info_field(row.get("info"))
        if not info or "golden_answer" not in info:
            continue

        # Extract question: DialSim uses input_prompt with [Question]...[Answer]
        # format; Locomo uses origin_question directly.
        if is_dialsim:
            prompt_text = row.get("input_prompt", "")
            q_match = regex.search(r'\[Question\]\s*(.*?)\s*\[Answer\]', prompt_text, regex.S)
            origin_question = q_match.group(1).strip() if q_match else prompt_text
        else:
            origin_question = row.get("origin_question", "")

        samples.append({
            "test_idx": int(row.get("test_idx", len(samples))),
            "origin_question": origin_question,
            "info": {
                "golden_answer": info.get("golden_answer"),
                "category": int(info.get("category", 0)),
                "evidence": info.get("evidence", []),
            },
            "lang": row.get("lang", "en"),
        })

    # Sort by test_idx for reproducibility before any seed-driven shuffle.
    samples.sort(key=lambda r: r["test_idx"])
    return samples


def load_corpus_only_from_locomo(data_path: str, dataset_name: str) -> Locomo_Dataset:
    """
    Load a Locomo_Dataset purely to access `.conversation` and `.conversation_cnt`
    for corpus preload. The `.dataset` field (199 QA) is *not* used in HF mode.
    """
    return Locomo_Dataset(
        data_path=data_path,
        dataset_name=dataset_name,
        test_metrics=["f1"],
        max_output_len=8192,
    )


class DialSimCorpusProxy:
    """
    Lightweight proxy that mimics the interface of Locomo_Dataset for corpus
    loading (`.conversation`, `.conversation_cnt`, `.dataset_name`) but loads
    from DialSim raw text files.
    """
    def __init__(self, data_path: str, dataset_name: str):
        self.dataset_name = dataset_name
        # e.g. "DialSim-friends" -> "friends"
        show_name = dataset_name.split("-")[-1]
        corpus_file = os.path.join(data_path, f"dialsim_corpus_{show_name}.txt")
        with open(corpus_file, "r", encoding="utf-8") as f:
            raw_text = f.read().strip()
        self.conversation, self.conversation_cnt = change_dialsim_conversation_to_locomo_form(raw_text)
        # DialSim has no separate QA in the corpus object; QA comes from HF.
        self.dataset = []


def load_corpus_for_dataset(ds_name: str):
    """
    Unified corpus loader: returns a corpus proxy (Locomo_Dataset or
    DialSimCorpusProxy) based on the dataset name.
    """
    if ds_name.startswith("DialSim"):
        return DialSimCorpusProxy(DEFAULT_DIALSIM_PATH, ds_name)
    else:
        return load_corpus_only_from_locomo(DEFAULT_LOCOMO_PATH, ds_name)


def compute_dialsim_accuracy(prediction: str, golden_answer: str) -> float:
    """
    DialSim scoring: exact-match accuracy (case-insensitive).
    Returns 1.0 if correct, 0.0 otherwise.
    """
    pred = prediction.strip().lower().rstrip(".")
    gold = golden_answer.strip().lower().rstrip(".")
    return 1.0 if pred == gold else 0.0


# ============================================================
# Corpus Memory: Shared read-only FAISS index for conversation history
# ============================================================

class CorpusMemory:
    """
    Read-only FAISS index holding the full conversation corpus.
    Shared across all methods — not managed by any policy.
    """

    def __init__(self, embedder_agent):
        """
        Initialize corpus memory using the same embedder as the agent.

        Args:
            embedder_agent: An EmbedderAgent instance (used for its _embed method).
        """
        self.agent = embedder_agent
        self.embedding_dim = embedder_agent.config.embedding_dim
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []  # list of {"doc_id": ..., "content": ...}

    def load_conversation(self, conversation: dict, session_cnt: int):
        """
        Load the full Locomo conversation corpus into the index.
        Same logic as solver.memory_locomo_conversation but stores directly.

        Args:
            conversation: The raw conversation dict from Locomo dataset.
            session_cnt: Number of sessions in the conversation.
        """
        print(f"    [Corpus] Loading {session_cnt} sessions...")
        vectors = []
        for session_idx in range(1, session_cnt + 1):
            session_key = f"session_{session_idx}"
            if session_key not in conversation:
                continue
            session_data = conversation[session_key]
            date_key = f"session_{session_idx}_date_time"
            date_str = conversation.get(date_key, f"Session {session_idx}")

            for msg_idx, msg in enumerate(session_data):
                speaker = msg.get("speaker", "Unknown")
                text = msg.get("text", "")
                content = f"[{date_str}] {speaker} said: {text}"
                doc_id = f"corpus_s{session_idx}_m{msg_idx}"

                vector = self.agent._embed(content)
                vectors.append(vector)
                self.metadata.append({"doc_id": doc_id, "content": content})

        if vectors:
            self.index.add(np.array(vectors, dtype=np.float32))
        print(f"    [Corpus] Loaded {self.index.ntotal} messages into corpus memory")

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieve top-k relevant corpus entries for a query.

        Args:
            query: The query text.
            k: Number of results to return.

        Returns:
            List of content strings.
        """
        if self.index.ntotal == 0:
            return []
        vector = self.agent._embed(query)
        actual_k = min(k, self.index.ntotal)
        D, I = self.index.search(np.array([vector]), actual_k)
        results = []
        for i in I[0]:
            if 0 <= i < len(self.metadata):
                results.append(self.metadata[i]["content"])
        return results


# ============================================================
# Experience Learning Evaluator
# ============================================================

class ExperienceLearningEvaluator:
    """
    Evaluates memory management methods in the on-policy experience learning setting.

    Protocol (aligned with MemoryBench official):
      - Phase 1: Preload corpus (shared, identical for all methods)
      - Phase 2: Sequential QA with experience accumulation through policy

    The key difference between methods is ONLY how they manage the
    experience memory (dialog history from QA interactions).
    """

    def __init__(self, capacity: int = 50, corpus_retrieve_k: int = 5,
                 experience_retrieve_k: int = 3, feedback_threshold: float = 0.5):
        self.capacity = capacity
        self.corpus_retrieve_k = corpus_retrieve_k
        self.experience_retrieve_k = experience_retrieve_k
        self.feedback_threshold = feedback_threshold  # score < this → "dislike"

    def build_solver(self, method_name: str, memory_cache_dir: str,
                     config_overrides: Optional[Dict] = None):
        """Build a solver instance for the given method (used for experience memory only)."""
        config_path = get_memory_system_config_file(method_name)
        with open(config_path, "r") as f:
            config = json.load(f)

        # Apply capacity override (this is the EXPERIENCE memory capacity)
        config["capacity"] = self.capacity
        config["retrieve_k"] = self.experience_retrieve_k

        if config_overrides:
            config.update(config_overrides)

        # Lazy import: only fail here (when a method is actually run),
        # not at module import time.
        from src.solver import SolverFactory

        solver = SolverFactory.create(
            method_name=method_name,
            config=config,
            memory_cache_dir=memory_cache_dir,
        )
        return solver

    def evaluate_session(
        self,
        method_name: str,
        dataset: Locomo_Dataset,
        output_dir: str,
        config_overrides: Optional[Dict] = None,
        force_rerun: bool = False,
        qa_samples: Optional[List[Dict]] = None,
        warmup_qa_samples: Optional[List[Dict]] = None,
        seed: int = 42,
    ) -> Dict:
        """
        Evaluate one method on one Locomo session.

        Args:
            dataset: Locomo_Dataset, used for `.conversation` / `.conversation_cnt`
                (corpus preload). When `qa_samples` is None, falls back to
                `dataset.dataset` (legacy mode, 199 QA from raw/Locomo/locomo10.json).
            qa_samples: Pre-loaded list of **reportable** QA dicts (HF test
                split, 5 per session). F1 from this stream is the headline.
            warmup_qa_samples: Optional list of **non-reportable** warm-up QA
                dicts (HF train split, ~140-194 / session). When provided, the
                evaluator first runs the policy through this stream so memory
                state actually fills capacity before the reportable test
                stream starts. F1 from this phase is logged as `warmup_*`
                only and is NOT the reportable number.
            seed: Per-trial random seed; controls QA ordering shuffle for
                BOTH the warm-up and the test stream.

        Protocol:
          Phase 1: Load full conversation corpus into shared read-only memory
          Phase 2a (optional): Train-split warm-up — same loop as 2b but F1
            is not reported. Memory state carries over into 2b.
          Phase 2b: Test-split evaluation — reportable F1.
            For each QA (in order):
              a. Retrieve from corpus memory (shared, read-only)
              b. Retrieve from experience memory (managed by policy)
              c. Combine context → LLM generates answer
              d. Score against golden answer (official F1)
              e. Construct experience text
              f. Policy decides: store or skip this experience
        """
        dataset_name = dataset.dataset_name
        result_file = os.path.join(output_dir, "experience_eval_results.json")

        if os.path.exists(result_file) and not force_rerun:
            print(f"  [SKIP] {method_name} on {dataset_name} — already evaluated")
            with open(result_file, "r") as f:
                return json.load(f)

        # Build solver for experience memory (starts EMPTY)
        memory_cache_dir = os.path.join(output_dir, "memory_cache")
        if os.path.exists(memory_cache_dir):
            shutil.rmtree(memory_cache_dir)

        solver = self.build_solver(method_name, memory_cache_dir, config_overrides)

        # ============================================================
        # Phase 1: Preload corpus (shared, not through policy)
        # ============================================================
        print(f"  [Phase 1] Loading corpus into shared memory...")
        t0 = time.time()
        corpus_memory = CorpusMemory(solver.agent)
        corpus_memory.load_conversation(dataset.conversation, dataset.conversation_cnt)
        corpus_load_time = time.time() - t0
        print(f"  [Phase 1] Corpus loaded in {corpus_load_time:.1f}s "
              f"({corpus_memory.index.ntotal} entries)")

        # ============================================================
        # Phase 2a (optional): Train-split warm-up — NOT scored externally
        # ============================================================
        warmup_summary = None
        if warmup_qa_samples is not None and len(warmup_qa_samples) > 0:
            warmup_data = list(warmup_qa_samples)
            if len(warmup_data) > 1 and seed is not None:
                rng_w = random.Random(seed)
                rng_w.shuffle(warmup_data)
            print(f"  [Phase 2a] Warm-up on HF train split "
                  f"(n={len(warmup_data)}, seed={seed}) — F1 NOT reported")
            (
                warm_scores, warm_feedbacks, warm_store_decisions,
                warm_step_details
            ) = self._run_qa_stream(
                solver=solver,
                method_name=method_name,
                corpus_memory=corpus_memory,
                qa_data=warmup_data,
                phase_label=f"[{method_name}] Exp2-warmup",
                step_offset=0,
                is_dialsim=dataset_name.startswith("DialSim"),
            )
            warmup_summary = {
                "num_samples": len(warm_scores),
                "avg_f1": float(np.mean(warm_scores)) if warm_scores else 0.0,
                "store_rate": (sum(warm_store_decisions) / max(len(warm_store_decisions), 1)),
                "final_experience_memory_size": (
                    warm_step_details[-1]['experience_memory_size']
                    if warm_step_details else 0
                ),
            }
            print(f"  [Phase 2a] Warm-up done: avg_f1={warmup_summary['avg_f1']:.4f}, "
                  f"store_rate={warmup_summary['store_rate']:.1%}, "
                  f"exp_mem={warmup_summary['final_experience_memory_size']}/{self.capacity}")

        # ============================================================
        # Phase 2b: Test-split evaluation — reportable F1
        # ============================================================
        if qa_samples is not None:
            # HF test split (sorted by test_idx in load_qa_from_hf).
            # Apply per-trial seed shuffle ONLY when more than 1 sample, so
            # that single-QA sessions degenerate to a fixed ordering.
            all_data = list(qa_samples)
            if len(all_data) > 1 and seed is not None:
                rng = random.Random(seed)
                rng.shuffle(all_data)
            qa_source = f"HF test split (n={len(all_data)}, seed={seed})"
        else:
            # Legacy fallback: full 199 QA from raw/Locomo/locomo10.json
            all_data = dataset.dataset  # list of dicts
            qa_source = f"legacy raw locomo10.json (n={len(all_data)})"
        n = len(all_data)

        phase_label_b = "[Phase 2b]" if warmup_qa_samples else "[Phase 2]"
        print(f"  {phase_label_b} Processing {n} QA items sequentially — {qa_source}")
        print(f"    Corpus retrieve_k={self.corpus_retrieve_k}, "
              f"Experience retrieve_k={self.experience_retrieve_k}, "
              f"Experience capacity={self.capacity}")

        # step_offset: keep step IDs unique across warm-up and test, so that
        # `add_memory_*(content, doc_id=f"exp_{step}")` cannot collide with
        # the warm-up doc IDs.
        step_offset_b = (warmup_summary["num_samples"] if warmup_summary else 0)
        scores, feedbacks, store_decisions, step_details = self._run_qa_stream(
            solver=solver,
            method_name=method_name,
            corpus_memory=corpus_memory,
            qa_data=all_data,
            phase_label=f"[{method_name}] Exp2",
            step_offset=step_offset_b,
            is_dialsim=dataset_name.startswith("DialSim"),
        )

        # Compute summary metrics (test split only — the reportable numbers)
        result = self._compute_summary(
            method_name, dataset_name, scores, feedbacks, store_decisions,
            step_details, corpus_load_time, corpus_memory.index.ntotal
        )
        if warmup_summary is not None:
            result["warmup"] = warmup_summary

        # Save
        os.makedirs(output_dir, exist_ok=True)
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        # Save step details for analysis (test stream only)
        details_file = os.path.join(output_dir, "step_details.json")
        with open(details_file, "w") as f:
            json.dump(step_details, f, indent=2)

        # Cleanup memory cache
        if os.path.exists(memory_cache_dir):
            shutil.rmtree(memory_cache_dir)

        return result

    def _run_qa_stream(
        self,
        solver,
        method_name: str,
        corpus_memory: "CorpusMemory",
        qa_data: List[Dict],
        phase_label: str,
        step_offset: int = 0,
        is_dialsim: bool = False,
    ) -> Tuple[List[float], List[str], List[bool], List[Dict]]:
        """
        Run one on-policy pass over a list of QA samples.

        Used for both Phase 2a (warm-up, not reported) and Phase 2b
        (test, reported). The two phases are identical from the policy's
        perspective; only the caller decides whether to report the
        resulting F1 numbers.

        Returns: (scores, feedbacks, store_decisions, step_details)
        """
        scores: List[float] = []
        feedbacks: List[str] = []
        store_decisions: List[bool] = []
        step_details: List[Dict] = []

        for local_t, data in enumerate(tqdm(qa_data, desc=phase_label)):
            t_start = time.time()
            t = local_t + step_offset
            question = data["origin_question"]
            info = data["info"]
            category = info["category"]
            golden_answer = info["golden_answer"]

            # 1a. Retrieve from corpus memory (shared, read-only)
            corpus_results = corpus_memory.retrieve(question, k=self.corpus_retrieve_k)

            # 1b. Retrieve from experience memory (managed by policy)
            experience_results = solver.agent.retrieve_memory(
                question, k=self.experience_retrieve_k
            )

            # 2. Combine context and generate answer
            corpus_context = "\n".join(corpus_results) if corpus_results else ""
            experience_context = "\n".join(experience_results) if experience_results else ""

            if corpus_context and experience_context:
                combined_context = (
                    f"=== Conversation History ===\n{corpus_context}\n\n"
                    f"=== Prior QA Experience ===\n{experience_context}"
                )
            elif corpus_context:
                combined_context = f"=== Conversation History ===\n{corpus_context}"
            elif experience_context:
                combined_context = f"=== Prior QA Experience ===\n{experience_context}"
            else:
                combined_context = "No relevant information available."

            if category == 5:
                prompt = f"""{combined_context}

Based on the above context, answer the following question.

Question: {question}
Short answer:"""
            else:
                prompt = f"""{combined_context}

Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question}
Short answer:"""

            messages = [{"role": "user", "content": prompt}]

            try:
                response = solver.agent.llm.generate_response(messages=messages)
            except Exception as e:
                print(f"  [ERROR] Step {t}: {e}")
                response = "Error generating response"

            # 3. Score (official LoCoMo F1 or DialSim accuracy)
            if is_dialsim:
                f1 = compute_dialsim_accuracy(response, str(golden_answer))
            else:
                f1 = compute_locomo_f1(response, golden_answer, category)
            feedback = "like" if f1 >= self.feedback_threshold else "dislike"
            scores.append(f1)
            feedbacks.append(feedback)

            # 4. Construct experience text
            # For dislike (wrong answer), only store Q + correct answer to avoid
            # polluting memory with the model's incorrect response.
            if category == 5:
                answer_str = str(golden_answer) if isinstance(golden_answer, dict) else golden_answer
            else:
                answer_str = str(golden_answer)

            if feedback == "dislike":
                experience_text = (
                    f"Q: {question}\n"
                    f"Correct: {answer_str}\n"
                    f"Feedback: {feedback} (F1={f1:.2f})"
                )
            else:
                experience_text = (
                    f"Q: {question}\n"
                    f"A: {response}\n"
                    f"Correct: {answer_str}\n"
                    f"Feedback: {feedback} (F1={f1:.2f})"
                )

            # 5. Policy decides: store or skip this experience
            stored = self._store_experience(solver, method_name, experience_text, f1, t)
            store_decisions.append(stored)

            # ExpMem must reflect the *active* (non-soft-deleted) experience
            # count, NOT FAISS `index.ntotal` (which is monotonically
            # increasing because FIFO / SOLAR-E / SOLAR-A / SOLAR all use
            # tombstone deletion). For Embedder/Greedy there is no `deleted`
            # field, so this falls back to len(metadata) == ntotal.
            active_exp = sum(
                1 for m in solver.agent.metadata
                if not m.get("deleted", False)
            )
            t_end = time.time()
            step_details.append({
                'step': t,
                'category': category,
                'f1': f1,
                'feedback': feedback,
                'stored': stored,
                'num_corpus_retrieved': len(corpus_results),
                'num_experience_retrieved': len(experience_results),
                'experience_memory_size': active_exp,
                'experience_total_candidates': solver.agent.index.ntotal,
                'latency_ms': (t_end - t_start) * 1000,
            })

        return scores, feedbacks, store_decisions, step_details

    def _store_experience(self, solver, method_name: str, content: str,
                          score: float, step: int) -> bool:
        """
        Store experience through the method's policy.

        - embedder_message: always store (unlimited reference)
        - FIFO: always store, evict oldest when full
        - SOLAR-A (solar_a): regret-gated admission on feedback score
        - SOLAR-E (solar_e): always store, Thompson-sampling eviction
        - SOLAR (solar): regret-gated admission + Thompson-sampling eviction
        """
        doc_id = f"exp_{step}"

        if method_name == "solar_a":
            # SOLAR-A: experience-based regret gating (admission only)
            return solver.agent.add_memory_experience_gated(content, score, doc_id)
        elif method_name == "solar":
            # SOLAR: regret-gated admission (full framework)
            if hasattr(solver.agent, 'add_memory_experience_gated'):
                return solver.agent.add_memory_experience_gated(content, score, doc_id)
            else:
                solver.agent.add_memory(content, doc_id)
                return True
        elif method_name == "fifo":
            solver.agent.add_memory_fifo(content, doc_id)
            return True
        elif method_name == "solar_e":
            solver.agent.add_memory_thompson(content, doc_id)
            return True
        elif method_name == "lru":
            solver.agent.add_memory_lru(content, doc_id)
            return True
        elif method_name == "lfu":
            solver.agent.add_memory_lfu(content, doc_id)
            return True
        elif method_name == "arc":
            solver.agent.add_memory_arc(content, doc_id)
            return True
        else:
            # Greedy: always store
            solver.agent.add_memory(content, doc_id)
            return True

    def _compute_summary(self, method_name: str, dataset_name: str,
                         scores: List[float], feedbacks: List[str],
                         store_decisions: List[bool],
                         step_details: List[Dict],
                         corpus_load_time: float,
                         corpus_size: int) -> Dict:
        """Compute summary metrics for one session."""
        n = len(scores)
        if n == 0:
            return {}

        # Overall metrics
        avg_f1 = float(np.mean(scores))
        std_f1 = float(np.std(scores))

        # First half vs second half (learning effect)
        mid = n // 2
        first_half_f1 = float(np.mean(scores[:mid])) if mid > 0 else 0
        second_half_f1 = float(np.mean(scores[mid:])) if mid < n else 0

        # Learning slope (linear regression on scores)
        if n > 5:
            x = np.arange(n)
            slope = float(np.polyfit(x, scores, 1)[0])
        else:
            slope = 0.0

        # Store rate
        total_stored = sum(store_decisions)
        store_rate = total_stored / max(n, 1)

        # Dislike store rate (how many dislikes were stored)
        dislike_indices = [i for i, f in enumerate(feedbacks) if f == "dislike"]
        dislike_stored = sum(1 for i in dislike_indices if store_decisions[i])
        dislike_store_rate = dislike_stored / max(len(dislike_indices), 1)

        # Per-category F1
        cat_scores = {}
        for d in step_details:
            cat = d['category']
            cat_name = CAT_NAMES.get(cat, f"cat_{cat}")
            if cat_name not in cat_scores:
                cat_scores[cat_name] = []
            cat_scores[cat_name].append(d['f1'])

        category_f1 = {k: {'avg': float(np.mean(v)), 'std': float(np.std(v)), 'n': len(v)}
                       for k, v in cat_scores.items()}

        # Final experience memory size
        final_exp_memory_size = step_details[-1]['experience_memory_size'] if step_details else 0

        # Average latency
        latencies = [d.get('latency_ms', 0) for d in step_details]
        avg_latency_ms = float(np.mean(latencies)) if latencies else 0.0

        return {
            'method': method_name,
            'dataset': dataset_name,
            'num_samples': n,
            'avg_f1': avg_f1,
            'std_f1': std_f1,
            'first_half_f1': first_half_f1,
            'second_half_f1': second_half_f1,
            'learning_slope': slope,
            'improvement': second_half_f1 - first_half_f1,
            'store_rate': store_rate,
            'total_stored': total_stored,
            'dislike_count': len(dislike_indices),
            'like_count': n - len(dislike_indices),
            'dislike_store_rate': dislike_store_rate,
            'final_experience_memory_size': final_exp_memory_size,
            'corpus_size': corpus_size,
            'experience_capacity': self.capacity,
            'corpus_load_time': corpus_load_time,
            'category_f1': category_f1,
            'avg_latency_ms': avg_latency_ms,
        }


# ============================================================
# Main orchestration
# ============================================================

def run_experience_evaluation(
    methods: List[str],
    dataset_names: List[str],
    output_base: str,
    capacity: int = 50,
    corpus_retrieve_k: int = 5,
    experience_retrieve_k: int = 3,
    config_overrides: Optional[Dict] = None,
    force_rerun: bool = False,
    use_hf: bool = True,
    use_train_warmup: bool = True,
    seeds: Optional[List[int]] = None,
    merge_train_test: bool = False,
) -> Dict:
    """Run Exp 2 evaluation for multiple methods on multiple datasets.

    Args:
        use_hf: If True (default), pull QA from THUIR/MemoryBench. If
            False, fall back to the full 199 QA from raw/Locomo/locomo10.json
            (legacy mode — incompatible with `use_train_warmup`).
        use_train_warmup: If True (default), additionally load the HF
            **train** split (~140-194 QA / session) and feed it through the
            policy as a non-reportable warm-up phase before evaluating on
            the test split. This is the official MemoryBench on-policy
            protocol. Has no effect when `use_hf=False`.
        seeds: List of seeds for multi-seed evaluation. Defaults to [42].
            Each seed reshuffles QA order WITHIN both warm-up and test
            streams; corpus and split composition stay identical.
        merge_train_test: If True, merge HF train and test splits into a
            single unified on-policy stream. All QA are scored and reported.
            Overrides `use_train_warmup` (no separate warm-up phase).
    """

    evaluator = ExperienceLearningEvaluator(
        capacity=capacity,
        corpus_retrieve_k=corpus_retrieve_k,
        experience_retrieve_k=experience_retrieve_k,
    )

    if seeds is None:
        seeds = [42]
    multi_seed = len(seeds) > 1

    # method -> dataset -> (single-seed dict OR aggregated multi-seed dict)
    all_results: Dict[str, Dict[str, Dict]] = {}

    for method in methods:
        all_results[method] = {}

        for ds_name in dataset_names:
            print(f"\n{'='*60}")
            print(f"  [Exp 2] {method} on {ds_name} "
                  f"(exp_capacity={capacity}, seeds={seeds}, source={'HF' if use_hf else 'legacy'})")
            print(f"{'='*60}")

            # Corpus: Locomo from raw/Locomo/locomo10.json, DialSim from raw/DialSim/
            try:
                dataset = load_corpus_for_dataset(ds_name)
            except Exception as e:
                print(f"  [ERROR] failed to load corpus for {ds_name}: {e}")
                all_results[method][ds_name] = {"error": f"corpus load: {e}"}
                continue

            # QA samples: HF test split + (optional) HF train split warm-up.
            qa_samples = None
            warmup_samples = None
            if use_hf:
                try:
                    qa_samples = load_qa_from_hf(ds_name, split=MEMORYBENCH_TEST_SPLIT)
                except Exception as e:
                    print(f"  [ERROR] failed to load HF test QA for {ds_name}: {e}")
                    all_results[method][ds_name] = {"error": f"hf load: {e}"}
                    continue
                if merge_train_test:
                    # Merge train + test into a single unified on-policy stream
                    try:
                        train_samples = load_qa_from_hf(ds_name, split=MEMORYBENCH_TRAIN_SPLIT)
                        qa_samples = train_samples + qa_samples  # train first, then test
                        print(f"  [MERGED] train ({len(train_samples)}) + test ({len(qa_samples) - len(train_samples)}) "
                              f"= {len(qa_samples)} QA — all scored and reported")
                    except Exception as e:
                        print(f"  [WARN] failed to load HF train split for merge: {e} "
                              f"— falling back to test-only mode")
                elif use_train_warmup:
                    try:
                        warmup_samples = load_qa_from_hf(ds_name, split=MEMORYBENCH_TRAIN_SPLIT)
                        print(f"  Loaded HF train warm-up: {len(warmup_samples)} QA, "
                              f"HF test: {len(qa_samples)} QA")
                    except Exception as e:
                        print(f"  [WARN] failed to load HF train warm-up for {ds_name}: {e} "
                              f"— falling back to test-only mode")
                        warmup_samples = None

            seed_results = []
            for seed in seeds:
                if multi_seed:
                    output_dir = os.path.join(output_base, ds_name, method, f"seed_{seed}")
                else:
                    output_dir = os.path.join(output_base, ds_name, method)
                try:
                    res = evaluator.evaluate_session(
                        method_name=method,
                        dataset=dataset,
                        output_dir=output_dir,
                        config_overrides=config_overrides,
                        force_rerun=force_rerun,
                        qa_samples=qa_samples,
                        warmup_qa_samples=warmup_samples,
                        seed=seed,
                    )
                    res["seed"] = seed
                    seed_results.append(res)
                    print(f"  seed={seed}: F1={res['avg_f1']:.4f}, "
                          f"slope={res['learning_slope']:.4f}, "
                          f"store_rate={res['store_rate']:.1%}, "
                          f"improvement={res['improvement']:+.4f}, "
                          f"exp_mem={res['final_experience_memory_size']}/{capacity}")
                except Exception as e:
                    print(f"  [ERROR] seed={seed}: {e}")
                    import traceback; traceback.print_exc()
                    seed_results.append({"seed": seed, "error": str(e)})

            # Aggregate seeds (mean / std)
            valid = [r for r in seed_results if "error" not in r]
            if not valid:
                all_results[method][ds_name] = {"error": "all seeds failed",
                                                 "per_seed": seed_results}
                continue

            agg_keys = ["avg_f1", "first_half_f1", "second_half_f1",
                        "improvement", "learning_slope", "store_rate",
                        "dislike_store_rate", "final_experience_memory_size",
                        "avg_latency_ms"]
            aggregated = {
                "per_seed": seed_results,
                "num_seeds": len(valid),
                "seeds": [r["seed"] for r in valid],
            }
            for k in agg_keys:
                vals = [r[k] for r in valid if k in r]
                if vals:
                    aggregated[k] = float(np.mean(vals))
                    aggregated[f"{k}_std"] = float(np.std(vals))
            # Carry over single-seed-only fields from the first seed for downstream
            # consumers (category_f1, num_samples, capacity, ...).
            for k in ("num_samples", "experience_capacity", "corpus_size",
                      "category_f1", "method", "dataset", "like_count",
                      "dislike_count"):
                if k in valid[0] and k not in aggregated:
                    aggregated[k] = valid[0][k]
            all_results[method][ds_name] = aggregated

            if multi_seed:
                print(f"  → {ds_name}/{method} aggregated over {len(valid)} seeds: "
                      f"F1={aggregated.get('avg_f1', 0):.4f}±{aggregated.get('avg_f1_std', 0):.4f}, "
                      f"store_rate={aggregated.get('store_rate', 0):.1%}±"
                      f"{aggregated.get('store_rate_std', 0):.1%}")

    return all_results


def print_comparison_table(all_results: Dict, output_base: str, capacity: int):
    """Print comparison table for Exp 2."""

    methods = list(all_results.keys())
    datasets = set()
    for m in methods:
        datasets.update(all_results[m].keys())
    datasets = sorted(datasets)

    # Aggregated results
    print(f"\n{'='*120}")
    print(f"EXP 2: EXPERIENCE LEARNING (On-Policy) — COMPARISON TABLE (exp_capacity={capacity})")
    print(f"{'='*120}")
    print(f"{'Method':<16} {'Avg F1↑':>8} {'1st Half':>9} {'2nd Half':>9} "
          f"{'Improve':>8} {'Slope↑':>7} {'StoreRate':>10} {'DislikeStore':>12} {'ExpMem':>7} {'Latency':>10}")
    print(f"{'-'*130}")

    method_agg = {}
    for method in methods:
        f1_scores = []
        first_halves = []
        second_halves = []
        slopes = []
        store_rates = []
        dislike_stores = []
        mem_sizes = []
        latencies = []

        for ds in datasets:
            r = all_results[method].get(ds, {})
            if "error" not in r and "avg_f1" in r:
                f1_scores.append(r['avg_f1'])
                first_halves.append(r['first_half_f1'])
                second_halves.append(r['second_half_f1'])
                slopes.append(r['learning_slope'])
                store_rates.append(r['store_rate'])
                dislike_stores.append(r['dislike_store_rate'])
                mem_sizes.append(r['final_experience_memory_size'])
                latencies.append(r.get('avg_latency_ms', 0.0))

        if f1_scores:
            agg = {
                'avg_f1': float(np.mean(f1_scores)),
                'first_half_f1': float(np.mean(first_halves)),
                'second_half_f1': float(np.mean(second_halves)),
                'improvement': float(np.mean(second_halves)) - float(np.mean(first_halves)),
                'slope': float(np.mean(slopes)),
                'store_rate': float(np.mean(store_rates)),
                'dislike_store_rate': float(np.mean(dislike_stores)),
                'exp_mem_size': float(np.mean(mem_sizes)),
                'avg_latency_ms': float(np.mean(latencies)),
            }
            method_agg[method] = agg

            print(f"{method:<16} {agg['avg_f1']:>8.4f} {agg['first_half_f1']:>9.4f} "
                  f"{agg['second_half_f1']:>9.4f} {agg['improvement']:>+8.4f} "
                  f"{agg['slope']:>7.4f} {agg['store_rate']:>10.1%} "
                  f"{agg['dislike_store_rate']:>12.1%} {agg['exp_mem_size']:>7.0f} "
                  f"{agg['avg_latency_ms']:>8.0f}ms")

    # Key insights
    if len(method_agg) > 1:
        print(f"\n{'='*80}")
        print("KEY INSIGHTS")
        print(f"{'='*80}")

        best = max(method_agg.items(), key=lambda x: x[1]['avg_f1'])
        worst = min(method_agg.items(), key=lambda x: x[1]['avg_f1'])
        print(f"  Best F1: {best[0]} ({best[1]['avg_f1']:.4f})")
        print(f"  Worst F1: {worst[0]} ({worst[1]['avg_f1']:.4f})")
        print(f"  Gap: {best[1]['avg_f1'] - worst[1]['avg_f1']:.4f}")

        best_improve = max(method_agg.items(), key=lambda x: x[1]['improvement'])
        print(f"  Best improvement (2nd-1st half): {best_improve[0]} ({best_improve[1]['improvement']:+.4f})")

        if 'solar' in method_agg and 'embedder_message' in method_agg:
            solar = method_agg['solar']
            ref = method_agg['embedder_message']
            print(f"\n  SOLAR vs Unlimited reference:")
            print(f"    F1: {solar['avg_f1']:.4f} vs {ref['avg_f1']:.4f} (Δ={solar['avg_f1']-ref['avg_f1']:+.4f})")
            print(f"    Store rate: {solar['store_rate']:.1%} vs {ref['store_rate']:.1%}")
            print(f"    Improvement: {solar['improvement']:+.4f} vs {ref['improvement']:+.4f}")
            print(f"    Exp memory: {solar['exp_mem_size']:.0f} vs {ref['exp_mem_size']:.0f}")

    # Per-category breakdown (aggregated across datasets)
    print(f"\n{'='*80}")
    print("PER-CATEGORY F1 BREAKDOWN")
    print(f"{'='*80}")

    cat_names_sorted = sorted(CAT_NAMES.values())
    header = f"{'Method':<16}" + "".join(f" {c:>12}" for c in cat_names_sorted)
    print(header)
    print("-" * len(header))

    for method in methods:
        cat_agg = {}
        for ds in datasets:
            r = all_results[method].get(ds, {})
            if "error" not in r and "category_f1" in r:
                for cat, stats in r['category_f1'].items():
                    if cat not in cat_agg:
                        cat_agg[cat] = []
                    cat_agg[cat].append(stats['avg'])

        row = f"{method:<16}"
        for cat in cat_names_sorted:
            if cat in cat_agg:
                row += f" {np.mean(cat_agg[cat]):>12.4f}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # ------------------------------------------------------------------
    # Warm-up diagnostics (only emitted when train-split warm-up was used)
    # ------------------------------------------------------------------
    has_warmup = any(
        ("warmup" in all_results[m].get(ds, {}))
        for m in methods for ds in datasets
    )
    if has_warmup:
        print(f"\n{'='*80}")
        print("WARM-UP DIAGNOSTICS (train split, F1 NOT reported)")
        print(f"{'='*80}")
        print(f"{'Method':<16} {'WarmN':>6} {'WarmF1':>8} {'WarmStore':>10} {'WarmExpMem':>11}")
        print(f"{'-'*60}")
        for method in methods:
            wn, wf1, wsr, wmem = [], [], [], []
            for ds in datasets:
                r = all_results[method].get(ds, {})
                w = r.get("warmup") if isinstance(r, dict) else None
                if w:
                    wn.append(w.get('num_samples', 0))
                    wf1.append(w.get('avg_f1', 0.0))
                    wsr.append(w.get('store_rate', 0.0))
                    wmem.append(w.get('final_experience_memory_size', 0))
            if wn:
                print(f"{method:<16} {int(np.mean(wn)):>6d} "
                      f"{np.mean(wf1):>8.4f} {np.mean(wsr):>10.1%} "
                      f"{np.mean(wmem):>11.0f}")

    # Save aggregated results
    agg_path = os.path.join(output_base, "exp2_aggregated_results.json")
    save_data = {
        'experiment': 'Exp 2: Experience Learning (On-Policy)',
        'protocol': (
            'Preload corpus (shared) + train-split warm-up + test-split eval'
            if has_warmup else
            'Preload corpus (shared) + selective experience accumulation (through policy)'
        ),
        'experience_capacity': capacity,
        'methods': method_agg,
        'per_method_dataset': {m: {ds: r for ds, r in ds_results.items()}
                               for m, ds_results in all_results.items()},
    }
    with open(agg_path, "w") as f:
        json.dump(save_data, f, indent=2, default=float)
    print(f"\nResults saved to: {agg_path}")


def main():
    parser = ArgumentParser(description="Exp 2: Experience Learning (On-Policy Protocol)")

    # What to evaluate
    parser.add_argument("--method", type=str, choices=OUR_METHODS,
                       help="Run only this method")
    parser.add_argument("--methods", type=str, nargs="+", choices=OUR_METHODS,
                       help="Run these methods")
    parser.add_argument("--datasets", type=str, nargs="+",
                       help="Specific datasets (e.g., Locomo-0 Locomo-1 DialSim-friends)")
    parser.add_argument("--all", action="store_true",
                       help="Run on all 10 Locomo datasets")
    parser.add_argument("--all-dialsim", action="store_true",
                       help="Run on all 3 DialSim datasets")
    parser.add_argument("--all-datasets", action="store_true",
                       help="Run on all Locomo + DialSim datasets")

    # Parameters
    parser.add_argument("--capacity", type=int, default=50,
                       help="Experience memory capacity (default: 50)")
    parser.add_argument("--corpus-retrieve-k", type=int, default=5,
                       help="Number of corpus entries to retrieve per query")
    parser.add_argument("--experience-retrieve-k", type=int, default=3,
                       help="Number of experiences to retrieve per query")
    parser.add_argument("--lambda-cost", type=float, default=None,
                       help="Override SOLAR lambda_cost")

    # Ablation sweeps
    parser.add_argument("--capacity-sweep", type=int, nargs="+",
                       help="Run capacity sweep (e.g., --capacity-sweep 20 50 100 200)")
    parser.add_argument("--seed-sweep", type=int, nargs="+",
                       help="Multi-seed evaluation (recommended >=3 seeds for HF mode "
                            "because each session only has 5 test QA). "
                            "E.g., --seed-sweep 42 1337 2024")

    # Data source
    parser.add_argument("--legacy-full-locomo", action="store_true",
                       help="Bypass HF and use the full 199 QA / session from "
                            "raw/Locomo/locomo10.json (NOT aligned with MemoryBench test split). "
                            "Implies --no-train-warmup.")
    parser.add_argument("--no-train-warmup", action="store_true",
                       help="Skip the HF train-split warm-up phase. By default the "
                            "evaluator runs each method through the train QAs "
                            "(F1 not reported) before scoring on the test QAs — this "
                            "is the official MemoryBench on-policy protocol. Use "
                            "this flag for a faster, less informative test-only run.")
    parser.add_argument("--use-mb-lite", action="store_true",
                       help="Use the smaller THUIR/MemoryBench (20 train + 5 test "
                            "per session) instead of the default "
                            "THUIR/MemoryBench-Full (160 train + 39 test). Useful "
                            "for fast smoke tests; reportable runs should keep the "
                            "default Full split.")
    parser.add_argument("--merge-train-test", action="store_true",
                       help="Merge HF train and test splits into a single unified "
                            "on-policy stream (e.g. 160+39=199 QA/session for Full). "
                            "All QA are scored and reported — no warm-up / test "
                            "distinction. This is a custom evaluation protocol "
                            "(not the MemoryBench default).")

    # Output
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--force", action="store_true",
                       help="Force re-run")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (used when --seed-sweep is not given)")

    args = parser.parse_args()

    # Resolve seed list
    if args.seed_sweep:
        seeds = list(args.seed_sweep)
    else:
        seeds = [args.seed]

    # Set seed for any global RNG used at startup time
    random.seed(seeds[0])
    np.random.seed(seeds[0])

    # Resolve data source
    use_hf = not args.legacy_full_locomo
    if use_hf and not _HF_AVAILABLE:
        print("[WARN] `datasets` not installed; falling back to --legacy-full-locomo.")
        use_hf = False
    merge_train_test = getattr(args, 'merge_train_test', False)
    use_train_warmup = use_hf and (not args.no_train_warmup) and (not merge_train_test)

    # Pick which HF repo to read from (Full = default, Lite = --use-mb-lite).
    # We mutate the module-level constant so that helpers built on top of
    # `MEMORYBENCH_HF_NAME` (load_qa_from_hf, banner, run_config) all stay
    # consistent without threading another argument everywhere.
    global MEMORYBENCH_HF_NAME
    MEMORYBENCH_HF_NAME = (
        MEMORYBENCH_LITE_HF_NAME if args.use_mb_lite else MEMORYBENCH_FULL_HF_NAME
    )

    # Determine methods
    if args.method:
        methods = [args.method]
    elif args.methods:
        methods = args.methods
    else:
        methods = OUR_METHODS

    # Determine datasets
    if args.all_datasets:
        dataset_names = LOCOMO_DATASETS + DIALSIM_DATASETS
    elif args.all_dialsim:
        dataset_names = DIALSIM_DATASETS
    elif args.all:
        dataset_names = LOCOMO_DATASETS
    elif args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = ["Locomo-0"]  # default: just one session

    # Config overrides
    config_overrides = {}
    if args.lambda_cost is not None:
        config_overrides["lambda_cost"] = args.lambda_cost

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_base = args.output
    else:
        output_base = f"exp2_results/{timestamp}_K{args.capacity}"
    os.makedirs(output_base, exist_ok=True)

    # Save run config
    run_config = {
        "experiment": "Exp 2: Experience Learning (On-Policy)",
        "protocol": (
            "Preload corpus (shared) + unified on-policy stream (train+test merged, all scored)"
            if merge_train_test else
            ("Preload corpus (shared) + train-split warm-up (policy accumulates "
             "experience, F1 not reported) + test-split evaluation (reportable F1)"
             if use_train_warmup else
             "Preload corpus (shared) + selective experience accumulation on test split only")
        ),
        "data_source": (
            f"{MEMORYBENCH_HF_NAME} "
            f"[{MEMORYBENCH_TRAIN_SPLIT}+{MEMORYBENCH_TEST_SPLIT} merged, all scored]"
            if merge_train_test else
            (f"{MEMORYBENCH_HF_NAME} "
             f"[warmup={MEMORYBENCH_TRAIN_SPLIT}, reportable={MEMORYBENCH_TEST_SPLIT}]"
             if use_train_warmup else
             (f"{MEMORYBENCH_HF_NAME} [{MEMORYBENCH_TEST_SPLIT} split, no warm-up]"
              if use_hf else f"legacy: {DEFAULT_LOCOMO_PATH} (199 QA / session)"))
        ),
        "corpus_source": DEFAULT_LOCOMO_PATH,
        "unused_hf_columns": [
            "dialog_<method>", "implicit_feedback_<method>",
            "# These are MemoryBench's off-policy artifacts. We discard them",
            "# because this experiment runs each method on-policy.",
        ] if use_hf else None,
        "methods": methods,
        "datasets": dataset_names,
        "experience_capacity": args.capacity,
        "corpus_retrieve_k": args.corpus_retrieve_k,
        "experience_retrieve_k": args.experience_retrieve_k,
        "lambda_cost": args.lambda_cost,
        "use_train_warmup": use_train_warmup,
        "seeds": seeds,
        "timestamp": timestamp,
    }
    with open(os.path.join(output_base, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\n{'#'*60}")
    print(f"EXPERIMENT 2: Experience Learning (On-Policy)")
    print(f"  Protocol: "
          + ("Preload corpus + unified on-policy stream (train+test merged, all scored)"
             if merge_train_test else
             ("Preload corpus + train warm-up + test eval"
              if use_train_warmup else
              "Preload corpus + test eval only (no warm-up)")))
    if merge_train_test:
        print(f"  Data source: HF {MEMORYBENCH_HF_NAME} (train+test merged)")
    else:
        print(f"  Data source: {'HF ' + MEMORYBENCH_HF_NAME + ' (test split)' if use_hf else 'legacy raw/Locomo/locomo10.json (199/session)'}")
    if use_train_warmup and not merge_train_test:
        print(f"  Train warm-up: HF train split (F1 NOT reported)")
    print(f"  Methods: {methods}")
    print(f"  Datasets: {dataset_names}")
    print(f"  Experience capacity: {args.capacity}")
    print(f"  Corpus retrieve_k: {args.corpus_retrieve_k}")
    print(f"  Experience retrieve_k: {args.experience_retrieve_k}")
    print(f"  Seeds: {seeds}")
    print(f"  Output: {output_base}")
    print(f"{'#'*60}\n")

    if args.capacity_sweep:
        # Capacity sweep
        print(f"\n{'='*60}")
        print(f"CAPACITY SWEEP: {args.capacity_sweep}")
        print(f"{'='*60}")

        for K in args.capacity_sweep:
            print(f"\n--- Experience Capacity K={K} ---")
            sweep_output = os.path.join(output_base, f"K_{K}")

            results = run_experience_evaluation(
                methods=methods,
                dataset_names=dataset_names,
                output_base=sweep_output,
                capacity=K,
                corpus_retrieve_k=args.corpus_retrieve_k,
                experience_retrieve_k=args.experience_retrieve_k,
                config_overrides=config_overrides if config_overrides else None,
                force_rerun=args.force,
                use_hf=use_hf,
                use_train_warmup=use_train_warmup,
                seeds=seeds,
                merge_train_test=merge_train_test,
            )
            print_comparison_table(results, sweep_output, K)
    else:
        # Standard evaluation
        results = run_experience_evaluation(
            methods=methods,
            dataset_names=dataset_names,
            output_base=output_base,
            capacity=args.capacity,
            corpus_retrieve_k=args.corpus_retrieve_k,
            experience_retrieve_k=args.experience_retrieve_k,
            config_overrides=config_overrides if config_overrides else None,
            force_rerun=args.force,
            use_hf=use_hf,
            use_train_warmup=use_train_warmup,
            seeds=seeds,
            merge_train_test=merge_train_test,
        )
        print_comparison_table(results, output_base, args.capacity)

    print(f"\n{'#'*60}")
    print(f"ALL DONE — Results in: {output_base}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
