#!/usr/bin/env python3
"""
Prompt Optimizer Pipeline — 主入口

使用方式：
    # 直接运行（使用默认配置）
    python main.py

    # 指定参数
    python main.py --iterations 3 --sample-size 5

环境变量：
    DEEPSEEK_API_KEY: 必须设置，用于调用 DeepSeek API

目录结构：
    data/       测试集文件
    logs/       每次运行的详细日志
    results/    最优提示词和历史分数
    src/        核心模块
"""

import argparse
import sys
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

# 将项目根目录加入 Python 路径（确保 src 模块可以正确导入）
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_handler import DatasetHandler
from src.target_agent import TargetAgent
from src.evaluator_judge import EvaluatorJudge
from src.prompt_optimizer import PromptOptimizer
from src.optimization_pipeline import OptimizationPipeline


# ------------------------------------------------------------------ #
#  初始提示词（这是你要优化的起点）                                      #
# ------------------------------------------------------------------ #
INITIAL_SYSTEM_PROMPT = """你是一个智能助手，请回答用户的问题。"""

# 推荐的初始提示词（质量更高，可作为对比基准）
BETTER_INITIAL_PROMPT = """你是一个专业的技术助手，擅长 AI、机器学习、软件开发和数据分析领域。

请遵循以下原则回答问题：
1. 准确性优先：确保技术细节正确，不确定时明确说明
2. 结构清晰：使用标题、列表、代码块等格式组织内容
3. 实用为主：提供可直接应用的方案，附上具体示例
4. 适当简洁：避免冗余，聚焦用户真正需要的信息"""


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Prompt Optimizer Pipeline — 基于 LLM-as-a-Judge 的自动提示词优化系统"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/golden_dataset.json",
        help="测试集文件路径（支持 .json 或 .csv）",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="最大优化迭代次数（默认 3）",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="每轮从测试集采样的样本数（默认 5，减少可降低 API 成本）",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=4.5,
        help="提前停止的分数阈值（默认 4.5/5.0）",
    )
    parser.add_argument(
        "--use-better-prompt",
        action="store_true",
        help="使用质量更高的初始提示词（对比基准）",
    )
    parser.add_argument(
        "--agent-model",
        type=str,
        default="deepseek-chat",
        help="TargetAgent 使用的模型（deepseek-chat / deepseek-reasoner）",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="deepseek-reasoner",
        help="EvaluatorJudge 使用的模型（推荐 deepseek-reasoner）",
    )
    parser.add_argument(
        "--optimizer-model",
        type=str,
        default="deepseek-reasoner",
        help="PromptOptimizer 使用的模型（推荐 deepseek-reasoner）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 检查 API Key
    import os
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("[错误] 请设置环境变量 DEEPSEEK_API_KEY")
        print("  export DEEPSEEK_API_KEY='your-api-key-here'")
        sys.exit(1)

    # 选择初始提示词
    initial_prompt = (
        BETTER_INITIAL_PROMPT if args.use_better_prompt else INITIAL_SYSTEM_PROMPT
    )
    prompt_label = "质量提升版" if args.use_better_prompt else "最简版（待优化）"
    print(f"\n初始提示词类型: {prompt_label}")
    print(f"数据集路径: {args.dataset}")
    print(f"最大迭代次数: {args.iterations}")
    print(f"每轮采样数: {args.sample_size}")

    # 初始化各模块
    dataset = DatasetHandler(file_path=args.dataset)
    agent = TargetAgent(model=args.agent_model)
    judge = EvaluatorJudge(judge_model=args.judge_model)
    optimizer = PromptOptimizer(optimizer_model=args.optimizer_model)

    # 创建并运行 Pipeline
    pipeline = OptimizationPipeline(
        initial_prompt=initial_prompt,
        dataset_handler=dataset,
        target_agent=agent,
        evaluator_judge=judge,
        prompt_optimizer=optimizer,
        max_iterations=args.iterations,
        sample_size=args.sample_size,
        score_threshold=args.score_threshold,
        log_dir="logs",
        results_dir="results",
    )

    result = pipeline.run()

    # 打印最终最优提示词
    print("\n" + "="*60)
    print("最优提示词内容：")
    print("="*60)
    print(result.best_prompt)
    print("="*60)


if __name__ == "__main__":
    main()
