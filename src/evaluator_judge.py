"""
EvaluatorJudge 模块
LLM-as-a-Judge 核心裁判：对 TargetAgent 的输出进行评分并给出改进建议
强制返回结构化 JSON（score 1-5，improvement_feedback）
使用 DeepSeek API（兼容 OpenAI 协议）
"""

import json
import re
from openai import OpenAI
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator


# ------------------------------------------------------------------ #
#  结构化输出的 Pydantic 模型                                            #
# ------------------------------------------------------------------ #

class JudgeResult(BaseModel):
    """
    裁判模型必须返回的结构化结果

    score:                1-5 分（5 分最优）
    improvement_feedback: 针对 TargetAgent 当前输出的具体改进建议，
                          将直接用于指导 PromptOptimizer 生成新提示词
    reasoning:            裁判的内部推理过程（可选，用于日志调试）
    """
    score: int = Field(..., ge=1, le=5, description="质量评分 1-5，5 分最优")
    improvement_feedback: str = Field(
        ...,
        min_length=10,
        description="具体的改进建议，描述当前输出的不足和期望的改进方向"
    )
    reasoning: str = Field(default="", description="裁判推理过程（可选）")

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: int) -> int:
        if not 1 <= v <= 5:
            raise ValueError(f"score 必须在 1-5 之间，得到: {v}")
        return v


@dataclass
class EvaluationRecord:
    """单次评估的完整记录，用于日志写入"""
    test_case_id: str
    user_input: str
    expected_output: str
    agent_output: str
    judge_result: JudgeResult


# 裁判使用的元提示词模板
_JUDGE_SYSTEM_PROMPT = """你是一个极其严格、吹毛求疵的评估专家（LLM-as-a-Judge）。
你的任务是评估 AI 助手的回答质量，并与标准参考答案进行交叉对比。

【评分标准——必须严格执行】
- 5 分：回答在准确性、完整性、结构、深度上全面超越参考答案，提供了额外有价值的洞见
- 4 分：回答覆盖了参考答案的所有关键点，且表达更清晰或结构更优
- 3 分：回答基本准确，但遗漏了参考答案中 1-2 个重要信息点
- 2 分：回答存在明显遗漏或部分不准确，只覆盖了参考答案的一半内容
- 1 分：回答偏题、错误严重，或远不如参考答案

【重要原则】
- 仅仅"回答了问题"不足以得高分，必须与参考答案对标
- 参考答案是 3 分的基线，超越才能得 4-5 分
- 你应当主动寻找回答的不足：缺少代码示例？缺少对比分析？缺少具体数据？
- 严禁给出虚高评分，平均分应在 2-3 分区间，只有真正出色的回答才得 4-5 分

评分维度（综合考量）：
1. 准确性：回答是否与参考答案的核心事实一致（有错误直接扣分）
2. 完整性：是否覆盖了参考答案中的所有关键信息点（每遗漏一个关键点扣分）
3. 结构质量：是否有清晰的层次、列表、代码块等（参考答案有结构但回答没有则扣分）
4. 深度：是否给出了具体数据、公式、代码示例（泛泛而谈则扣分）
5. 超越性：是否提供了参考答案之外的有价值补充（这是得 5 分的必要条件）

你必须严格按照以下 JSON 格式返回结果，不得添加任何额外说明：
{
  "score": <1-5 之间的整数，参考答案对应 3 分基线>,
  "improvement_feedback": "<具体的改进建议，必须指出至少 2 处具体缺陷和改进方向>",
  "reasoning": "<你的评估推理过程，需列举对比参考答案的具体差异>"
}"""

_JUDGE_USER_TEMPLATE = """请评估以下 AI 助手的回答：

【用户原始问题】
{user_input}

【标准参考答案（Golden Answer）】
{expected_output}

【AI 助手的实际回答】
{agent_output}

请严格按照 JSON 格式返回评估结果。"""


class EvaluatorJudge:
    """
    LLM-as-a-Judge 评估模块。

    使用 DeepSeek API 对 TargetAgent 的输出进行交叉对比评分，
    强制返回结构化 JSON。

    推荐使用 deepseek-reasoner（R1）作为裁判，其内置推理链
    能提供更严谨的评估，且推理过程不占用输出 token。
    """

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        judge_model: str = "deepseek-reasoner",
        max_tokens: int = 8192,
    ) -> None:
        """
        Args:
            judge_model: 裁判模型 ID
                         - "deepseek-reasoner" 深度推理（R1，推荐作为裁判）
                         - "deepseek-chat"     通用对话（V3，速度更快）
            max_tokens:  最大输出 token 数
        """
        self.judge_model = judge_model
        self.max_tokens = max_tokens
        self._client = OpenAI(
            api_key=self._get_api_key(),
            base_url=self.DEEPSEEK_BASE_URL,
        )

    @staticmethod
    def _get_api_key() -> str:
        import os
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "未找到 DEEPSEEK_API_KEY 环境变量，请先执行：\n"
                "  export DEEPSEEK_API_KEY='your-api-key-here'"
            )
        return key

    def evaluate(
        self,
        user_input: str,
        expected_output: str,
        agent_output: str,
        test_case_id: str = "unknown",
    ) -> EvaluationRecord:
        """
        对 TargetAgent 的单次输出进行评分。

        Args:
            user_input:      原始用户输入
            expected_output: 黄金标准答案
            agent_output:    TargetAgent 的实际输出
            test_case_id:    测试样本 ID，用于日志追踪

        Returns:
            EvaluationRecord 包含完整评分信息
        """
        user_message = _JUDGE_USER_TEMPLATE.format(
            user_input=user_input,
            expected_output=expected_output,
            agent_output=agent_output,
        )

        response = self._client.chat.completions.create(
            model=self.judge_model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        # deepseek-reasoner 的最终回答在 content（推理过程在 reasoning_content，不计入输出）
        raw_text = response.choices[0].message.content or ""
        judge_result = self._parse_judge_output(raw_text)

        return EvaluationRecord(
            test_case_id=test_case_id,
            user_input=user_input,
            expected_output=expected_output,
            agent_output=agent_output,
            judge_result=judge_result,
        )

    def _parse_judge_output(self, raw_text: str) -> JudgeResult:
        """
        解析裁判模型的 JSON 输出，包含多层 fallback 机制。

        策略：
        1. 直接 json.loads 解析
        2. 用正则提取 JSON 代码块后再解析
        3. 两者均失败时返回默认的低分结果，避免 Pipeline 崩溃
        """
        # 尝试 1：直接解析
        try:
            data = json.loads(raw_text.strip())
            return JudgeResult(**data)
        except (json.JSONDecodeError, Exception):
            pass

        # 尝试 2：提取 ```json ... ``` 或 { ... } 代码块
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r"(\{.*?\})", raw_text, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return JudgeResult(**data)
            except Exception:
                pass

        # Fallback：解析完全失败，返回默认结果并记录错误
        print(f"[EvaluatorJudge] 警告：JSON 解析失败，原始输出:\n{raw_text[:300]}")
        return JudgeResult(
            score=1,
            improvement_feedback="裁判模型输出解析失败，无法提供具体建议。请检查模型输出格式。",
            reasoning=raw_text[:500],
        )

    def batch_evaluate(
        self,
        records: list[dict],
    ) -> list[EvaluationRecord]:
        """
        批量评估多条测试结果。

        Args:
            records: 包含 user_input, expected_output, agent_output, test_case_id 的字典列表

        Returns:
            EvaluationRecord 列表
        """
        results = []
        for r in records:
            result = self.evaluate(
                user_input=r["user_input"],
                expected_output=r["expected_output"],
                agent_output=r["agent_output"],
                test_case_id=r.get("test_case_id", "unknown"),
            )
            results.append(result)
        return results

    @staticmethod
    def compute_average_score(records: list[EvaluationRecord]) -> float:
        """计算一批评估记录的平均分"""
        if not records:
            return 0.0
        return sum(r.judge_result.score for r in records) / len(records)

    @staticmethod
    def collect_feedback(records: list[EvaluationRecord]) -> str:
        """
        汇总所有评估记录的改进建议，生成合并的 feedback 文本。
        该文本将传递给 PromptOptimizer。
        """
        feedback_lines = []
        for i, rec in enumerate(records, 1):
            feedback_lines.append(
                f"[样本 {rec.test_case_id} | 得分 {rec.judge_result.score}/5]\n"
                f"  问题: {rec.user_input[:80]}...\n"
                f"  建议: {rec.judge_result.improvement_feedback}"
            )
        return "\n\n".join(feedback_lines)
