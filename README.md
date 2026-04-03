# Prompt Optimizer Pipeline

基于 **LLM-as-a-Judge** 的自动化提示词评估与迭代系统。通过"运行测试 → 裁判打分 → 优化器生成新候选 → 择优保存"的闭环，自动将一个简陋的系统提示词迭代优化到高质量版本。

## 这个项目在做什么

你有一个**初始 Prompt**（比如"你是一个智能助手"）和一批**带标准答案的测试用例**，系统会自动跑多轮优化，最终输出一个在测试集上得分更高的 Prompt。

**工作流程：**

```
初始 Prompt
    ↓
[Step 0] 基准测试 — 用初始 Prompt 跑测试集，记录初始分数
    ↓
循环迭代（最多 max_iterations 轮）:
  [Step 1] TargetAgent 对采样的测试用例逐条生成回答
  [Step 2] EvaluatorJudge 对比标准答案打分（1-5分）并给出改进建议
  [Step 3] PromptOptimizer 根据反馈生成 3 个候选新 Prompt
  [Step 4] 对每个候选 Prompt 跑一遍测试集，选出得分最高的
  [Step 5] 若新 Prompt 分数更高则替换，否则保留原版
    ↓
达到迭代次数上限 or 分数超过阈值 → 保存最优 Prompt 到 results/best_prompt.txt
```

**四个核心角色：**

| 模块 | 职责 |
|---|---|
| `TargetAgent` | 被优化的对象，用当前 Prompt 回答测试问题 |
| `EvaluatorJudge` | 裁判，对比标准答案给回答打 1-5 分并给出改进建议 |
| `PromptOptimizer` | 元优化器，读取裁判反馈，生成 3 个候选新 Prompt |
| `DatasetHandler` | 管理测试集（问题 + 标准答案），每轮随机采样 N 条 |

所有模型调用走 DeepSeek API，Judge 和 Optimizer 默认用 `deepseek-reasoner`（R1），TargetAgent 默认用 `deepseek-chat`（V3）。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                   OptimizationPipeline                       │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │DatasetHandler│───>│  TargetAgent │───>│EvaluatorJudge │  │
│  │  (测试集管理)  │    │  (业务执行)   │    │  (LLM裁判)    │  │
│  └──────────────┘    └──────────────┘    └───────┬───────┘  │
│                                                  │ feedback  │
│                                         ┌────────▼────────┐  │
│                                         │ PromptOptimizer │  │
│                                         │ (元提示词引擎)   │  │
│                                         └────────┬────────┘  │
│                                                  │ 3 候选    │
│                                         ┌────────▼────────┐  │
│                                         │  择优 + 保存     │  │
│                                         └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**完整闭环流程：**

```
基准测试 (iteration=0)
    ↓
[Loop] for i in 1..max_iterations:
    Step 1: TargetAgent 对采样测试集逐条生成回答
    Step 2: EvaluatorJudge 逐条打分 + 汇总 feedback
    Step 3: PromptOptimizer 生成 3 个候选提示词
    Step 4: 对每个候选分别跑测试集，选出最高分
    Step 5: 若候选最高分 > 当前最优分，更新最优提示词
    Step 6: 达到分数阈值或迭代上限时终止
```

---

## 核心模块

### `DatasetHandler` — 测试集管理

- 支持 **JSON** 和 **CSV** 两种格式
- 随机采样（固定 seed，结果可复现）
- 字段完整性校验

```python
from src.dataset_handler import DatasetHandler

dataset = DatasetHandler("data/golden_dataset.json")
cases = dataset.sample(n=5, seed=42)   # 每次采样结果相同
```

数据格式（JSON）：

```json
[
  {
    "id": "001",
    "input": "用户的问题...",
    "expected_output": "标准参考答案..."
  }
]
```

---

### `TargetAgent` — 业务执行层

接收当前版本的 System Prompt 和用户输入，调用 Deepseek 生成回答。

```python
from src.target_agent import TargetAgent

agent = TargetAgent(model="deepseek-chat")
resp = agent.run(system_prompt="你是一个技术助手...", user_input="什么是 RAG？")
print(resp.output)          # 生成的回答
print(resp.input_tokens)    # 消耗 token 数
print(resp.latency_seconds) # 调用耗时
```

---

### `EvaluatorJudge` — LLM 裁判

用强模型（deepseek-reasoner）对 TargetAgent 的输出进行评分，强制返回结构化 JSON。

**返回结构：**

```json
{
  "score": 4,
  "improvement_feedback": "回答准确但缺少代码示例，建议补充...",
  "reasoning": "（裁判内部推理过程）"
}
```

- `score`：1-5 分（5 分最优）
- `improvement_feedback`：将直接传递给 PromptOptimizer 指导优化

```python
from src.evaluator_judge import EvaluatorJudge

judge = EvaluatorJudge()
record = judge.evaluate(
    user_input="什么是 RAG？",
    expected_output="标准答案...",
    agent_output="模型的实际回答...",
)
print(record.judge_result.score)               # 4
print(record.judge_result.improvement_feedback)
```

---

### `PromptOptimizer` — 元提示词引擎

接收旧 Prompt 和 feedback，使用 Meta-Prompt 引导模型生成 **3 个差异化的候选提示词**。

三个候选版本从不同角度优化（角色设定 / 约束条件 / 示例引导），供 Pipeline 择优选取。

```python
from src.prompt_optimizer import PromptOptimizer

optimizer = PromptOptimizer()
result = optimizer.optimize(
    current_prompt="你是一个助手，请回答问题。",
    improvement_feedback="[样本001 | 得分2/5] 建议：回答太笼统，需要更具体...",
)
print(result.candidates[0])  # 最优候选提示词
```

---

### `OptimizationPipeline` — 主控闭环

将以上模块串联，执行完整的自动化优化流程。

```python
from src.optimization_pipeline import OptimizationPipeline
from src.dataset_handler import DatasetHandler

pipeline = OptimizationPipeline(
    initial_prompt="你是一个助手，请回答问题。",
    dataset_handler=DatasetHandler("data/golden_dataset.json"),
    max_iterations=5,
    sample_size=10,
    score_threshold=4.5,
)
result = pipeline.run()

print(f"初始分数: {result.initial_score:.2f}/5.0")
print(f"最优分数: {result.best_score:.2f}/5.0")
print(f"最优提示词:\n{result.best_prompt}")
```

---

## 快速开始

### 1. 环境准备

```bash
# 使用已创建好的 conda 环境
conda activate prompt_optimize

# 或从配置文件创建
conda env create -f environment.yml
conda activate prompt_optimize
```

### 2. 配置 API Key

在项目根目录的 `.env` 文件中填写：

```
DEEPSEEK_API_KEY=your-api-key-here
```

### 3. 运行优化

```bash
cd /amax/tyut/user/wangchenyu/Prompt_optimize

# 默认配置（3轮迭代，每轮5个样本）
python main.py

# 自定义参数
python main.py \
  --iterations 5 \
  --sample-size 8 \
  --score-threshold 4.0

# 从质量更高的初始提示词开始（对比实验）
python main.py --use-better-prompt

# 查看全部参数
python main.py --help
```

### 4. 运行单元测试（无需 API Key）

```bash
python tests/test_modules.py
```

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dataset` | `data/golden_dataset.json` | 测试集路径（.json 或 .csv） |
| `--iterations` | `3` | 最大优化迭代次数 |
| `--sample-size` | `5` | 每轮采样的测试样本数 |
| `--score-threshold` | `4.5` | 提前停止的分数阈值（/5.0） |
| `--use-better-prompt` | `False` | 使用质量更高的初始提示词 |
| `--agent-model` | `deepseek-chat` | TargetAgent 模型 |
| `--judge-model` | `deepseek-reasoner` | EvaluatorJudge 模型 |
| `--optimizer-model` | `deepseek-reasoner` | PromptOptimizer 模型 |

---

## 日志与输出

每次运行自动生成独立的运行目录：

```
logs/
└── run_20260329_143022/
    ├── iterations.jsonl    # 每轮详情（JSONL 格式，每行一条记录）
    └── summary.json        # 汇总报告

results/
├── best_prompt.txt         # 当前最优提示词（实时更新）
└── history.csv             # 所有轮次的分数历史（可用于可视化）
```

**`summary.json` 示例：**

```json
{
  "run_id": "20260329_143022",
  "best_score": 4.2,
  "best_iteration": 2,
  "initial_score": 2.6,
  "score_improvement": 1.6,
  "iteration_scores": [
    {"iteration": 0, "score": 2.6},
    {"iteration": 1, "score": 3.4},
    {"iteration": 2, "score": 4.2}
  ]
}
```
---
## 项目结果展示

运行一个小demo进行演示

<img width="554" height="208" alt="image" src="https://github.com/user-attachments/assets/28b48055-b3f3-4ff0-8edb-b00525980fd6" />
<img width="554" height="729" alt="image" src="https://github.com/user-attachments/assets/7b0f24ff-b6c7-4e52-ac6b-778762d98423" />
<img width="554" height="182" alt="image" src="https://github.com/user-attachments/assets/4e56cb67-c22c-4dd1-b2fb-06a14f29f962" />


---

## 项目文件结构

```
Prompt_optimize/
├── main.py                      # 主入口（CLI）
├── requirements.txt             # pip 依赖
├── environment.yml              # conda 环境配置
├── .env                         # API Key 配置（DEEPSEEK_API_KEY）
├── data/
│   └── golden_dataset.json      # 12 条测试集（AI/编程/工具等领域）
├── src/
│   ├── __init__.py
│   ├── dataset_handler.py       # DatasetHandler
│   ├── target_agent.py          # TargetAgent
│   ├── evaluator_judge.py       # EvaluatorJudge + JudgeResult
│   ├── prompt_optimizer.py      # PromptOptimizer + OptimizedPromptSet
│   └── optimization_pipeline.py # OptimizationPipeline（主控）
├── tests/
│   └── test_modules.py          # 11 个单元测试（不调用 API）
├── logs/                        # 运行日志（自动生成）
└── results/                     # 最优提示词和分数历史（自动生成）
```

---

## 成本估算参考

每轮迭代的 API 调用次数（以 `sample_size=5` 为例）：

| 步骤 | 调用次数 | 说明 |
|---|---|---|
| TargetAgent 主测试 | 5 次 | 每个样本 1 次 |
| EvaluatorJudge | 5 次 | 每个样本 1 次（带 thinking） |
| PromptOptimizer | 1 次 | 生成 3 个候选（带 thinking） |
| 候选评估（3个） | 15 次 | 每个候选 × 5 个样本 |
| **每轮合计** | **~26 次** | |

建议先用 `--sample-size 3 --iterations 2` 小规模验证系统运行正常，再扩大参数。
