"""
OptimizationPipeline 模块
主控循环：将 DatasetHandler、TargetAgent、EvaluatorJudge、PromptOptimizer 串联，
实现完整的自动化 Prompt 优化闭环，并记录每次迭代的详细日志
"""

import json
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from src.dataset_handler import DatasetHandler, TestCase
from src.target_agent import TargetAgent, AgentResponse
from src.evaluator_judge import EvaluatorJudge, EvaluationRecord
from src.prompt_optimizer import PromptOptimizer, OptimizedPromptSet


# ------------------------------------------------------------------ #
#  数据结构                                                             #
# ------------------------------------------------------------------ #

@dataclass
class IterationResult:
    """单次迭代的完整结果快照"""
    iteration: int                      # 第几轮迭代（从 0 开始，0 为基准测试）
    prompt_version: str                 # 该轮使用的提示词
    average_score: float                # 该轮在测试集上的平均分
    evaluation_records: list[dict]      # 各测试样本的评估详情
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_input_tokens: int = 0         # 该轮 TargetAgent 消耗的总输入 token
    total_output_tokens: int = 0        # 该轮 TargetAgent 消耗的总输出 token
    wall_time_seconds: float = 0.0      # 该轮总耗时（秒）


@dataclass
class PipelineResult:
    """Pipeline 运行的最终汇总结果"""
    best_prompt: str                    # 最优提示词文本
    best_score: float                   # 最优平均分
    best_iteration: int                 # 最优提示词出现在第几轮
    initial_score: float                # 基准测试分数（iteration=0）
    score_improvement: float            # 相对于基准的分数提升
    all_iterations: list[IterationResult]  # 所有迭代的完整记录
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))


class OptimizationPipeline:
    """
    Prompt 自动优化主控流程。

    完整闭环：
    Step 0: 基准测试 ——> 记录初始分数
    Loop:
        Step 1: 运行测试集 ——> TargetAgent 对每条样本生成回答
        Step 2: 裁判打分  ——> EvaluatorJudge 对每条回答评分 + 汇总 feedback
        Step 3: 生成候选  ——> PromptOptimizer 基于 feedback 生成 3 个候选 Prompt
        Step 4: 择优测试  ——> 对每个候选 Prompt 运行测试集，选出得分最高的
        Step 5: 更新/保存 ——> 若候选最优分 > 当前最优分，更新最优 Prompt
    End Loop（达到最大迭代次数或分数足够高时停止）

    日志记录：
    - logs/run_{run_id}/iterations.jsonl  每轮详情（JSONL 格式）
    - logs/run_{run_id}/summary.json      汇总报告
    - results/best_prompt.txt             最优提示词文本
    - results/history.csv                 所有轮次的分数历史（便于可视化）
    """

    def __init__(
        self,
        initial_prompt: str,
        dataset_handler: DatasetHandler,
        target_agent: Optional[TargetAgent] = None,
        evaluator_judge: Optional[EvaluatorJudge] = None,
        prompt_optimizer: Optional[PromptOptimizer] = None,
        max_iterations: int = 5,
        sample_size: int = 10,
        sample_seed: int = 42,
        score_threshold: float = 4.5,
        log_dir: str = "logs",
        results_dir: str = "results",
    ) -> None:
        """
        Args:
            initial_prompt:    初始系统提示词（基准版本）
            dataset_handler:   测试集管理器
            target_agent:      业务 Agent（None 时使用默认配置）
            evaluator_judge:   裁判模块（None 时使用默认配置）
            prompt_optimizer:  优化器模块（None 时使用默认配置）
            max_iterations:    最大迭代轮数
            sample_size:       每轮从测试集采样的样本数（控制成本）
            sample_seed:       采样随机种子（保证可复现）
            score_threshold:   提前停止的分数阈值（平均分达到此值则停止）
            log_dir:           日志目录
            results_dir:       结果保存目录
        """
        self.initial_prompt = initial_prompt
        self.dataset = dataset_handler
        self.agent = target_agent or TargetAgent()
        self.judge = evaluator_judge or EvaluatorJudge()
        self.optimizer = prompt_optimizer or PromptOptimizer()
        self.max_iterations = max_iterations
        self.sample_size = sample_size
        self.sample_seed = sample_seed
        self.score_threshold = score_threshold

        # 日志目录，每次运行使用独立子目录
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_dir = Path(log_dir) / f"run_{self._run_id}"
        self._results_dir = Path(results_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._results_dir.mkdir(parents=True, exist_ok=True)

        # 内部状态
        self._all_iterations: list[IterationResult] = []
        self._best_prompt = initial_prompt
        self._best_score = 0.0
        self._best_iteration = 0

    # ------------------------------------------------------------------ #
    #  主入口                                                               #
    # ------------------------------------------------------------------ #

    def run(self) -> PipelineResult:
        """
        执行完整的优化闭环。

        Returns:
            PipelineResult 包含最优提示词和完整迭代历史
        """
        print(f"\n{'='*60}")
        print(f"  Prompt Optimizer Pipeline 启动")
        print(f"  运行 ID: {self._run_id}")
        print(f"  最大迭代次数: {self.max_iterations}")
        print(f"  每轮采样数量: {self.sample_size}")
        print(f"  提前停止阈值: {self.score_threshold}/5.0")
        print(f"{'='*60}\n")

        current_prompt = self.initial_prompt

        # ---- Step 0: 基准测试 ----
        print(">> [Step 0] 基准测试（使用初始提示词）")
        baseline_result = self._run_single_evaluation(
            prompt=current_prompt,
            iteration=0,
            label="基准测试",
        )
        self._record_iteration(baseline_result)
        self._best_score = baseline_result.average_score
        self._best_iteration = 0
        print(f"   基准平均分: {self._best_score:.2f}/5.0\n")

        if self._best_score >= self.score_threshold:
            print(f"[Pipeline] 初始提示词分数已达到阈值 {self.score_threshold}，无需优化。")
            return self._build_result()

        # ---- 主优化循环 ----
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'─'*60}")
            print(f">> [迭代 {iteration}/{self.max_iterations}]")

            # Step 1 & 2: 对当前最优 Prompt 运行测试集 + 裁判打分
            print(f"   当前最优分: {self._best_score:.2f}/5.0")
            print(f"   Step 1: 运行测试集...")
            eval_result = self._run_single_evaluation(
                prompt=self._best_prompt,
                iteration=iteration,
                label=f"迭代 {iteration} - 主测试",
            )

            # 收集 feedback 用于生成候选
            eval_records = self._reconstruct_eval_records(eval_result)
            combined_feedback = self.judge.collect_feedback(eval_records)
            print(f"   Step 2: 裁判评估完成，平均分 {eval_result.average_score:.2f}/5.0")

            # Step 3: 生成候选 Prompt
            print(f"   Step 3: 优化器生成候选提示词...")
            optimized_set: OptimizedPromptSet = self.optimizer.optimize(
                current_prompt=self._best_prompt,
                improvement_feedback=combined_feedback,
            )
            print(f"   生成了 {len(optimized_set.candidates)} 个候选提示词")

            # Step 4: 对每个候选 Prompt 运行测试集，选出最优
            print(f"   Step 4: 评估候选提示词...")
            best_candidate_score = -1.0
            best_candidate_prompt = self._best_prompt
            best_candidate_result: Optional[IterationResult] = None

            for ci, candidate in enumerate(optimized_set.candidates, 1):
                print(f"     候选 {ci}/{len(optimized_set.candidates)}...")
                cand_result = self._run_single_evaluation(
                    prompt=candidate,
                    iteration=iteration,
                    label=f"迭代 {iteration} - 候选 {ci}",
                )
                print(f"     候选 {ci} 平均分: {cand_result.average_score:.2f}/5.0")

                if cand_result.average_score > best_candidate_score:
                    best_candidate_score = cand_result.average_score
                    best_candidate_prompt = candidate
                    best_candidate_result = cand_result

            # Step 5: 更新最优 Prompt
            print(f"\n   Step 5: 候选最优分: {best_candidate_score:.2f}/5.0 "
                  f"| 当前最优分: {self._best_score:.2f}/5.0")

            if best_candidate_score > self._best_score:
                self._best_score = best_candidate_score
                self._best_prompt = best_candidate_prompt
                self._best_iteration = iteration
                print(f"   ✓ 发现更优提示词！分数提升至 {self._best_score:.2f}/5.0")
                # 记录最优候选迭代
                if best_candidate_result:
                    self._record_iteration(best_candidate_result)
            else:
                print(f"   ✗ 候选提示词未优于当前最优，保留原版本")
                self._record_iteration(eval_result)

            # 保存当前最优提示词到文件
            self._save_best_prompt()
            self._append_history_csv(iteration, best_candidate_score)

            # 提前停止检查
            if self._best_score >= self.score_threshold:
                print(f"\n[Pipeline] 达到提前停止阈值 {self.score_threshold}，终止优化。")
                break

        return self._build_result()

    # ------------------------------------------------------------------ #
    #  内部方法：评估                                                        #
    # ------------------------------------------------------------------ #

    def _run_single_evaluation(
        self,
        prompt: str,
        iteration: int,
        label: str,
    ) -> IterationResult:
        """
        对给定 Prompt 运行完整的测试集评估。

        1. 采样测试集
        2. TargetAgent 逐条生成回答
        3. EvaluatorJudge 逐条打分
        4. 汇总结果返回 IterationResult
        """
        start_time = time.perf_counter()
        test_cases: list[TestCase] = self.dataset.sample(
            n=self.sample_size,
            seed=self.sample_seed,
        )

        evaluation_records = []
        total_in_tokens = 0
        total_out_tokens = 0

        def _evaluate_one(tc: "TestCase") -> dict:
            agent_resp: AgentResponse = self.agent.run(
                system_prompt=prompt,
                user_input=tc.input,
            )
            eval_record: EvaluationRecord = self.judge.evaluate(
                user_input=tc.input,
                expected_output=tc.expected_output,
                agent_output=agent_resp.output,
                test_case_id=tc.id,
            )
            return {
                "test_case_id": tc.id,
                "user_input": tc.input,
                "expected_output": tc.expected_output,
                "agent_output": agent_resp.output,
                "score": eval_record.judge_result.score,
                "improvement_feedback": eval_record.judge_result.improvement_feedback,
                "reasoning": eval_record.judge_result.reasoning,
                "agent_latency_s": agent_resp.latency_seconds,
                "agent_input_tokens": agent_resp.input_tokens,
                "agent_output_tokens": agent_resp.output_tokens,
            }

        max_workers = min(len(test_cases), 5)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_evaluate_one, tc): tc for tc in test_cases}
            for future in as_completed(futures):
                record = future.result()
                evaluation_records.append(record)
                total_in_tokens += record["agent_input_tokens"]
                total_out_tokens += record["agent_output_tokens"]

        avg_score = (
            sum(r["score"] for r in evaluation_records) / len(evaluation_records)
            if evaluation_records else 0.0
        )
        wall_time = time.perf_counter() - start_time

        result = IterationResult(
            iteration=iteration,
            prompt_version=prompt,
            average_score=round(avg_score, 4),
            evaluation_records=evaluation_records,
            total_input_tokens=total_in_tokens,
            total_output_tokens=total_out_tokens,
            wall_time_seconds=round(wall_time, 2),
        )

        # 立即写入 JSONL 日志
        self._append_jsonl_log(result, label)
        return result

    def _reconstruct_eval_records(
        self, iter_result: IterationResult
    ) -> list[EvaluationRecord]:
        """
        从 IterationResult 的字典列表重建 EvaluationRecord，
        供 EvaluatorJudge.collect_feedback() 使用。
        """
        from src.evaluator_judge import JudgeResult

        records = []
        for r in iter_result.evaluation_records:
            jr = JudgeResult(
                score=r["score"],
                improvement_feedback=r["improvement_feedback"],
                reasoning=r.get("reasoning", ""),
            )
            er = EvaluationRecord(
                test_case_id=r["test_case_id"],
                user_input=r["user_input"],
                expected_output=r["expected_output"],
                agent_output=r["agent_output"],
                judge_result=jr,
            )
            records.append(er)
        return records

    # ------------------------------------------------------------------ #
    #  内部方法：日志与持久化                                                #
    # ------------------------------------------------------------------ #

    def _record_iteration(self, result: IterationResult) -> None:
        """将迭代结果追加到内存中的历史列表"""
        self._all_iterations.append(result)

    def _append_jsonl_log(self, result: IterationResult, label: str) -> None:
        """追加写入 JSONL 格式的详细日志（每行一个 JSON 对象）"""
        log_file = self._log_dir / "iterations.jsonl"
        entry = {
            "label": label,
            "iteration": result.iteration,
            "average_score": result.average_score,
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
            "wall_time_seconds": result.wall_time_seconds,
            "timestamp": result.timestamp,
            "prompt_preview": result.prompt_version[:200],
            "evaluation_records": result.evaluation_records,
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _save_best_prompt(self) -> None:
        """将当前最优提示词保存到文件"""
        best_file = self._results_dir / "best_prompt.txt"
        with open(best_file, "w", encoding="utf-8") as f:
            f.write(f"# 最优提示词\n")
            f.write(f"# 运行 ID: {self._run_id}\n")
            f.write(f"# 平均分: {self._best_score:.4f}/5.0\n")
            f.write(f"# 出现于迭代: {self._best_iteration}\n")
            f.write(f"# 更新时间: {datetime.now().isoformat()}\n\n")
            f.write(self._best_prompt)

    def _append_history_csv(self, iteration: int, score: float) -> None:
        """将每轮分数追加到 CSV 历史记录（便于后续可视化）"""
        history_file = self._results_dir / "history.csv"
        write_header = not history_file.exists()
        with open(history_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["run_id", "iteration", "average_score", "timestamp"])
            writer.writerow([
                self._run_id,
                iteration,
                round(score, 4),
                datetime.now().isoformat(),
            ])

    def _build_result(self) -> PipelineResult:
        """构建并保存最终汇总结果"""
        initial_score = (
            self._all_iterations[0].average_score
            if self._all_iterations else 0.0
        )
        result = PipelineResult(
            best_prompt=self._best_prompt,
            best_score=self._best_score,
            best_iteration=self._best_iteration,
            initial_score=initial_score,
            score_improvement=round(self._best_score - initial_score, 4),
            all_iterations=self._all_iterations,
            run_id=self._run_id,
        )

        # 保存 summary.json
        summary_file = self._log_dir / "summary.json"
        summary_data = {
            "run_id": result.run_id,
            "best_score": result.best_score,
            "best_iteration": result.best_iteration,
            "initial_score": result.initial_score,
            "score_improvement": result.score_improvement,
            "total_iterations": len(self._all_iterations),
            "best_prompt_preview": result.best_prompt[:500],
            "iteration_scores": [
                {"iteration": r.iteration, "score": r.average_score}
                for r in self._all_iterations
            ],
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*60}")
        print(f"  Pipeline 完成！")
        print(f"  初始分数: {result.initial_score:.2f}/5.0")
        print(f"  最优分数: {result.best_score:.2f}/5.0")
        print(f"  分数提升: +{result.score_improvement:.2f}")
        print(f"  最优提示词出现于: 第 {result.best_iteration} 轮")
        print(f"  日志目录: {self._log_dir}")
        print(f"  最优提示词: {self._results_dir / 'best_prompt.txt'}")
        print(f"{'='*60}\n")

        return result
