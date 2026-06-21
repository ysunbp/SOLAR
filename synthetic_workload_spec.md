# 合成数据集生成指引 (Synthetic Workload Generator)

## 目标

生成可控的合成 workload，用于验证理论预测和做 sensitivity analysis。不需要 LLM 调用，纯 embedding 空间操作。

---

## 一、核心抽象

### 1.1 数据模型

```python
@dataclass
class SyntheticItem:
    id: str
    topic_id: int
    embedding: np.ndarray  # d-dimensional, unit normalized
    created_at: int        # timestep when this item was generated

@dataclass
class SyntheticQuery:
    id: str
    embedding: np.ndarray
    relevant_item_ids: List[str]  # ground truth
    timestep: int
```

### 1.2 评估方式

和 MemoryBench 一致：
- Cache 中 top-k retrieved items vs ground truth relevant items
- Metric: F1 (token-level 改为 set-level overlap，因为是合成的)
- 简化版: `hit = 1 if any(relevant_id in retrieved_ids) else 0`, 报 hit rate

---

## 二、Embedding 空间构造

### 2.1 Topic Cluster 生成

```python
def generate_topic_centers(n_topics: int, dim: int = 128, min_sep: float = 0.5):
    """
    在 unit hypersphere 上生成 n_topics 个 cluster center，
    保证两两之间 cosine distance >= min_sep
    
    方法: 随机采样 + rejection (dim=128 时很容易满足)
    """
    centers = []
    for _ in range(n_topics):
        while True:
            c = np.random.randn(dim)
            c = c / np.linalg.norm(c)
            if all(np.dot(c, existing) < (1 - min_sep) for existing in centers):
                centers.append(c)
                break
    return np.array(centers)
```

### 2.2 Topic 内 Item 生成

```python
def generate_items_for_topic(topic_center: np.ndarray, n_items: int, 
                              intra_topic_std: float = 0.1):
    """
    在 topic center 附近生成 items (高相似度，cosine ~0.9)
    intra_topic_std 控制 topic 内的 diversity
    """
    items = []
    for i in range(n_items):
        noise = np.random.randn(len(topic_center)) * intra_topic_std
        emb = topic_center + noise
        emb = emb / np.linalg.norm(emb)
        items.append(emb)
    return items
```

### 2.3 Query 生成

```python
def generate_query(topic_center: np.ndarray, query_noise: float = 0.05):
    """
    Query = topic center + small noise
    这样 query 和该 topic 的 items 有高 cosine similarity
    """
    noise = np.random.randn(len(topic_center)) * query_noise
    q = topic_center + noise
    return q / np.linalg.norm(q)
```

---

## 三、三种 Workload Pattern

### 3.1 Cycling Workload (验证 Thm 4)

**目的**: 证明 FIFO 在 cycling pattern 下 miss rate 趋近 100%

```python
class CyclingWorkload:
    """
    m 个 topics 循环出现, cache size K < m
    当 m/K > 1 时, FIFO 永远在驱逐即将被访问的 topic
    """
    def __init__(self, 
                 n_topics: int = 20,      # m, working set size
                 items_per_topic: int = 5,
                 dim: int = 128,
                 n_cycles: int = 10):
        pass
    
    def generate(self) -> Tuple[List[SyntheticItem], List[SyntheticQuery]]:
        """
        Query sequence: T1, T2, ..., T_m, T1, T2, ..., T_m, ... (repeat n_cycles times)
        每个 query 对应该 topic 的 1 个 random item 为 ground truth
        同时该 timestep 产生一个新 item (属于当前 topic)
        """
        pass

# 参数 sweep:
# m/K ratio ∈ {1.5, 2.0, 3.0, 5.0} (fixed K=10)
# 预期结果:
#   - FIFO hit rate → 0 when m/K >= 2 (thrashing)
#   - LMU+TS 学到 hot topics, hit rate 维持 > 0
```

### 3.2 Topic Drift Workload (验证 adaptivity)

**目的**: 展示 TS 能适应 non-stationary 环境，而 LRU/LFU 不行

```python
class TopicDriftWorkload:
    """
    分 phases，每个 phase 有不同的 active topic set
    模拟用户兴趣漂移
    """
    def __init__(self,
                 n_topics_total: int = 30,
                 n_active_per_phase: int = 5,   # 每个 phase 只有 5 个 topic 活跃
                 phase_length: int = 100,        # 每个 phase 持续 100 步
                 n_phases: int = 6,
                 overlap: int = 1):              # 相邻 phase 重叠 topic 数
        pass
    
    def generate(self) -> Tuple[List[SyntheticItem], List[SyntheticQuery]]:
        """
        Phase 1: topics {A,B,C,D,E} active, queries uniformly from these
        Phase 2: topics {E,F,G,H,I} active (E overlaps)
        Phase 3: topics {I,J,K,L,M} active
        ...
        
        每步产生 1 个新 item (属于当前 active topic 之一)
        """
        pass

# 参数 sweep:
# phase_length ∈ {50, 100, 200, 500}
# overlap ∈ {0, 1, 2}
# 预期结果:
#   - LFU: 保留旧 phase 的 high-freq items → fails after drift
#   - LRU: 勉强跟上但有 lag
#   - LMU+TS: TS posterior decay (β aging) 自动 forget old → adapts fast
```

### 3.3 Working Set Size Sweep (定位 phase transition)

**目的**: 精确找到 K* (phase transition point), 验证 coverage vs selectivity tradeoff

```python
class WorkingSetSweep:
    """
    Fixed workload pattern, sweep K to find transition point
    """
    def __init__(self,
                 n_topics: int = 15,
                 items_per_topic: int = 10,
                 query_length: int = 500,
                 topic_distribution: str = 'uniform'):  # or 'zipf'
        pass
    
    def generate(self) -> Tuple[List[SyntheticItem], List[SyntheticQuery]]:
        """
        Random topic queries (uniform or zipf)
        每步产生 1 个新 item
        """
        pass

# 参数 sweep:
# K ∈ {5, 10, 15, 20, 30, 50, 75, 100, 150, 200}
# 预期: 
#   - 存在 K* 使得 K < K* 时 LMU+TS > FIFO, K > K* 时 FIFO > LMU+TS
#   - K* ≈ n_topics * items_per_topic * admission_rate (理论预测)
```

### 3.4 Retrieval Noise U-Curve (论证 capacity constraint)

**目的**: 证明"即使存储免费，pool 变大后 retrieval precision 下降导致 downstream F1 先升后降"。这直接论证了 capacity constraint 的合理性：不是存不下，是检索噪声让你不该全存。

**为什么真实数据做不了**: LoCoMo ~200 items/session, DialSim ~300 items/dataset。在这个 scale 下 F1 单调递增（因为 retriever 在 200 items 里还能区分）。需要 scale 到数千 items 才能观测到 retrieval noise 的影响。

```python
class RetrievalNoiseWorkload:
    """
    大规模 item pool + 故意制造 inter-topic confusion
    让 retriever 在大 pool 中精度下降
    """
    def __init__(self,
                 n_topics: int = 50,
                 items_per_topic: int = 100,     # 总共 5000 items
                 intra_topic_std: float = 0.15,  # 比其他 workload 高 → topic 内更分散
                 inter_topic_sep: float = 0.3,   # 比其他 workload 低 → topic 间更近
                 query_length: int = 1000,
                 relevant_per_query: int = 3):    # 每个 query 有 3 个 ground truth items
        pass
    
    def generate(self) -> Tuple[List[SyntheticItem], List[SyntheticQuery]]:
        """
        关键设计:
        1. Topic centers 之间距离较近 (inter_topic_sep=0.3)
           → 当 pool 变大，来自 "近邻 topic" 的 items 会干扰检索
        2. 每个 topic 内的 items 较分散 (intra_topic_std=0.15)
           → 同 topic 的 items 之间 cosine 只有 ~0.7-0.8
        3. Query 对应 3 个 relevant items (都来自同一 topic)
        4. 评估: top-3 retrieved 中有几个 relevant
        
        物理直觉:
        - K=50:  pool 里只有 50 items, 来自 ~10 个 topics, noise 少
        - K=500: pool 里 500 items, 来自 ~30 个 topics, 有些 topics 彼此接近
        - K=5000: 全部 items 都在 pool, retriever 被 similar-but-wrong items 淹没
        """
        pass

# 实验设计:
# 不需要 eviction logic — 这个实验只测 FIFO (= 全存到满)
# 核心变量: K (pool size)
# 核心 metric: hit_rate AND retrieval_precision@3
#
# 参数:
# n_topics=50, items_per_topic=100 (total=5000)
# intra_topic_std ∈ {0.10, 0.15, 0.20} (sensitivity)
# inter_topic_sep ∈ {0.2, 0.3, 0.4} (sensitivity)
# K ∈ {10, 20, 50, 100, 200, 500, 1000, 2000, 5000}
#
# 预期结果:
#   - K 小: hit rate 低 (relevant items 没被存)
#   - K 中: hit rate 最高 (enough coverage, not too much noise)
#   - K 大: hit rate 下降 (retrieval precision 被 noise 拉低)
#   - Peak K* 取决于 inter_topic_sep (separation 越大, peak 越靠右)
#
# 如果看不到 U-curve:
#   - 增大 items_per_topic (更多 distractors)
#   - 减小 inter_topic_sep (让 topics 更难区分)
#   - 增大 intra_topic_std (让 same-topic items 更分散)
```

---

## 四、Runner 接口

### 4.1 和现有 pipeline 对齐

合成 workload 的 runner 需要和现有 `experience_eval` 兼容。核心接口：

```python
class SyntheticRunner:
    def __init__(self, workload, method: str, capacity: int, 
                 retrieve_k: int = 3, seed: int = 42):
        self.cache = []  # current cache state (list of SyntheticItem)
        self.capacity = capacity
        self.method = method  # 'fifo', 'lru', 'lfu', 'arc', 'thompson', 'lmu', 'lmu_ts'
    
    def step(self, new_item: SyntheticItem, query: SyntheticQuery) -> dict:
        """
        1. Retrieve: cosine similarity top-k from self.cache
        2. Evaluate: check overlap with query.relevant_item_ids
        3. Feedback: compute hit/miss signal
        4. Admission decision (method-specific)
        5. Eviction decision if needed (method-specific)
        6. Return: {hit: bool, f1: float, cache_size: int}
        """
        pass
    
    def run(self) -> dict:
        """Run full workload, return aggregated metrics"""
        pass
```

### 4.2 各 Method 实现

**关键**: 要和 MemoryBench 实验中的 method 实现逻辑一致，只是把 embedding retrieval 换成 cosine similarity in synthetic space。

各 method 的 admission/eviction 逻辑直接复用现有代码（在 `memory-research/` 目录中应该有）。

---

## 五、输出格式

和现有 `experience_eval_results.json` 对齐：

```json
{
  "workload_type": "cycling",
  "workload_params": {"n_topics": 20, "items_per_topic": 5, "n_cycles": 10},
  "method": "lmu_ts",
  "capacity": 10,
  "seed": 42,
  "num_steps": 200,
  "avg_hit_rate": 0.45,
  "first_half_hit_rate": 0.30,
  "second_half_hit_rate": 0.60,
  "learning_slope": 0.0015,
  "store_rate": 0.18,
  "per_step_hits": [0, 0, 1, 0, 1, ...]
}
```

---

## 六、实验矩阵

### 6.1 Cycling (Figure: hit rate vs m/K ratio)

```
Workload: CyclingWorkload
Sweep: n_topics ∈ {10, 15, 20, 30, 50}, K = 10 (so m/K = 1..5)
Methods: fifo, lru, lfu, arc, thompson, lmu, lmu_ts
Seeds: 42, 1337, 2024
Output: line plot, x=m/K, y=avg_hit_rate
```

### 6.2 Topic Drift (Figure: hit rate over time)

```
Workload: TopicDriftWorkload(n_phases=6, phase_length=100)
K = 20
Methods: all 7
Seeds: 42, 1337, 2024
Output: line plot, x=timestep, y=rolling_hit_rate (window=20)
         vertical lines at phase boundaries
```

### 6.3 Phase Transition (Figure: relative gain vs K)

```
Workload: WorkingSetSweep(n_topics=15, items_per_topic=10)
K ∈ {5, 10, 15, 20, 30, 50, 75, 100, 150, 200}
Methods: fifo, lmu_ts
Seeds: 42, 1337, 2024
Output: line plot, x=K, y=(lmu_ts_hit_rate - fifo_hit_rate)
         annotate crossover point K*
```

### 6.4 Retrieval Noise U-Curve (Figure: FIFO hit rate vs K at large scale)

**目的**: 论证 capacity constraint 的合理性。当 pool 变大时 retriever precision 下降，F1 先升后降。

**背景**: 在真实数据集 (LoCoMo ~200 items, DialSim ~300 items) 上无法观测到这个现象，因为 item 数量不够大。合成数据可以 scale 到数千 items。

```
Workload: RetrievalNoiseWorkload (见下方设计)
K ∈ {10, 20, 50, 100, 200, 500, 1000, 2000, 5000}
Methods: fifo (全存, 只控制 pool size), embedder_unlimited (作为对照)
Seeds: 42, 1337, 2024
Output: line plot, x=K (log scale), y=avg_hit_rate
         annotate: peak K* and the decline region
         secondary y-axis: retrieval precision@3
```

---

## 七、文件结构

```
synthetic_workloads/
├── workloads/
│   ├── cycling.py          # CyclingWorkload class
│   ├── topic_drift.py      # TopicDriftWorkload class
│   ├── working_set.py      # WorkingSetSweep class
│   └── retrieval_noise.py  # RetrievalNoiseWorkload class
├── methods/
│   ├── base.py             # SyntheticRunner base class + retrieval logic
│   ├── fifo.py
│   ├── lru.py
│   ├── lfu.py
│   ├── arc.py
│   ├── thompson.py
│   ├── lmu.py
│   └── lmu_ts.py
├── run_experiments.py       # Main entry: parse args, run sweep, save results
├── plot_results.py          # Generate figures
└── results/                 # Output JSON + figures
```

---

## 八、实现注意事项

1. **Retrieval 用 cosine similarity**, 不要用 faiss（数据量小，numpy 直接算）
2. **LMU admission 的 threshold τ**: 和 real benchmark 一致，用 EMA-smoothed adaptive τ
3. **Thompson 的 Beta posterior 更新**: hit → α += 1, aging → β += 0.05 every 5 steps (和现有实现一致)
4. **Seed 控制**: numpy random seed + topic generation seed 分开，确保同一 workload 不同 method 看到完全相同的 query sequence
5. **不需要 warmup**: 合成数据可以从 step 1 就开始 evaluate（或者 warmup 前 20 步不计分也行）
6. **dim=128** 足够（和 all-MiniLM-L6-v2 的 384 维相比，128 维已经能区分 topic）

---

## 九、验收标准

跑完后应该观察到：

| 实验 | 预期结果 | 如果不符说明 |
|------|---------|-------------|
| Cycling m/K=2 | FIFO hit≈0, LMU+TS hit>0.3 | admission 没有生效 |
| Topic Drift | LMU+TS 在 phase boundary 后快速恢复, LFU 持续低 | TS aging 参数不对 |
| Phase transition | 存在明确的 crossover K* | workload 太简单或太难 |
| Retrieval Noise | FIFO hit rate 先升后降, peak 在 K=100~500 之间 | inter_topic_sep 太大（topics 太容易区分）或 items 太少 |

如果结果不符合预期，首先检查 method 实现是否和 real benchmark 一致。

---

## 十、Paper 中的 Narrative

合成实验在 paper 中的角色和叙事：

### 10.1 在 paper 中的定位

```
6. Experiments
   6.1 Setup
   6.2 Real-World Benchmarks (MemoryBench + DialSim)  ← 证明 practical effectiveness
   6.3 Controlled Evaluation (Synthetic)              ← 验证理论 + justification
       6.3.1 Cycling workload validates Thm 4 (FIFO Ω(T) regret)
       6.3.2 Topic drift demonstrates TS adaptivity
       6.3.3 Working set sweep locates phase transition K*
       6.3.4 Retrieval noise justifies capacity constraint
   6.4 Ablation & Analysis
```

### 10.2 Retrieval Noise 实验的 Paper 叙事

**为什么做这个实验:**
> "A natural question is whether cache capacity constraints are meaningful when storage is cheap. We show that even with unlimited storage, retrieval quality degrades as pool size grows due to embedding space confusion among semantically similar items."

**写法示意:**
> "Figure X shows FIFO hit rate as a function of cache size K on our synthetic workload with 5000 items across 50 topics. Performance peaks at K ≈ [200-500] and declines thereafter, demonstrating that retrieval noise imposes an effective capacity constraint independent of storage cost. This inverted-U curve motivates our formulation: even absent hardware limits, there exists an optimal buffer size beyond which indiscriminate storage hurts downstream performance."

**连接到 contribution:**
> "This finding validates our problem formulation: cache management is necessary not because of storage scarcity, but because of retrieval noise. Selective admission (storing only high-value items) maintains retrieval precision, which is why LMU+TS outperforms FIFO even when FIFO has not yet reached its noise ceiling."
