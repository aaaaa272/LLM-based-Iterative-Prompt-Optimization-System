"""
PromptOptimizer 模块
元提示词引擎（Meta-Prompt Engine）：
接收 EvaluatorJudge 的反馈 + 旧 Prompt，引导大模型反思并输出 3 个候选新提示词
使用 DeepSeek API（兼容 OpenAI 协议）
"""

import json
import re
from openai import OpenAI
from dataclasses import dataclass


@dataclass
class OptimizedPromptSet:
    """优化后的提示词候选集合"""
    candidates: list[str]   # 3 个候选 Prompt（按从优到次排列）
    reasoning: str          # 优化器的推理过程
    raw_output: str         # 原始模型输出（用于调试）


# 元提示词（Meta-Prompt）：引导模型充当提示词工程师
_META_SYSTEM_PROMPT = """你是一个世界顶级的提示词工程师（Prompt Engineer）。
你的职责是分析 AI 助手当前提示词的缺陷，并基于评估反馈进行有针对性的优化。

优化原则：
1. 针对性：每个优化方向必须直接回应具体的评估反馈
2. 清晰性：提示词应明确定义角色、任务目标和输出格式
3. 多样性：3 个候选版本应探索不同的优化策略（如：角色设定、约束条件、示例引导）
4. 保留优点：在改进不足的同时，保留原提示词中有效的部分
5. 可测试性：优化后的提示词应该可以立即用于下一轮测试

你必须返回严格的 JSON 格式，不得添加任何额外说明：
{
  "reasoning": "<分析当前提示词问题和优化策略的推理过程>",
  "candidates": [
    "<候选提示词 1：最优改进方案>",
    "<候选提示词 2：第二优改进方案>",
    "<候选提示词 3：备选改进方案>"
  ]
}"""

_META_USER_TEMPLATE = """请基于以下信息，优化 AI 助手的系统提示词：

## 当前系统提示词（需要优化）
```
{current_prompt}
```

## 评估反馈汇总（来自 LLM-as-a-Judge）
```
{improvement_feedback}
```

## 优化要求
- 生成恰好 3 个不同策略的候选提示词
- 每个候选版本都必须完整（可以直接替换当前提示词）
- 候选版本应有明显的差异化（不同角度的优化策略）
- 优化幅度适中：不要过度修改，也不要保持原样

请严格按照 JSON 格式返回结果。"""


class PromptOptimizer:
    """
    元提示词优化引擎。

    工作流程：
    1. 接收旧 Prompt 和裁判汇总的 improvement_feedback
    2. 使用 Meta-Prompt 引导 DeepSeek 模型进行深度反思
    3. 生成 3 个具有差异化优化策略的候选 Prompt
    4. 返回结构化的候选集合供 Pipeline 选择

    推荐使用 deepseek-reasoner（R1）：其内置思维链会在输出前进行深度推理，
    reasoning_content 不计入输出 token，性价比极高。
    """

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        optimizer_model: str = "deepseek-reasoner",
        max_tokens: int = 4096,
        num_candidates: int = 3,
    ) -> None:
        """
        Args:
            optimizer_model: 优化器模型 ID
                             - "deepseek-reasoner" 深度推理（R1，推荐）
                             - "deepseek-chat"     通用对话（V3，速度更快）
            max_tokens:      最大输出 token 数（提示词可能较长）
            num_candidates:  生成候选提示词的数量
        """
        self.optimizer_model = optimizer_model
        self.max_tokens = max_tokens
        self.num_candidates = num_candidates
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

    def optimize(
        self,
        current_prompt: str,
        improvement_feedback: str,
    ) -> OptimizedPromptSet:
        """
        执行一轮提示词优化。

        Args:
            current_prompt:       当前版本的系统提示词（可能效果不佳）
            improvement_feedback: EvaluatorJudge 汇总的改进建议

        Returns:
            OptimizedPromptSet 包含 3 个候选提示词
        """
        user_message = _META_USER_TEMPLATE.format(
            current_prompt=current_prompt,
            improvement_feedback=improvement_feedback,
        )

        response = self._client.chat.completions.create(
            model=self.optimizer_model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": _META_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        # deepseek-reasoner 的最终回答在 content（推理过程在 reasoning_content）
        raw_text = response.choices[0].message.content or ""
        candidates, reasoning = self._parse_optimizer_output(raw_text, current_prompt)

        return OptimizedPromptSet(
            candidates=candidates,
            reasoning=reasoning,
            raw_output=raw_text,
        )

    def _parse_optimizer_output(
        self,
        raw_text: str,
        current_prompt: str,
    ) -> tuple[list[str], str]:
        """
        解析优化器输出的 JSON，包含多层 fallback 机制。

        Returns:
            (candidates_list, reasoning_str)
        """
        # 尝试 1：直接 JSON 解析
        try:
            data = json.loads(raw_text.strip())
            candidates = data.get("candidates", [])
            reasoning = data.get("reasoning", "")
            if len(candidates) >= self.num_candidates:
                return candidates[:self.num_candidates], reasoning
        except (json.JSONDecodeError, Exception):
            pass

        # 尝试 2：提取 JSON 代码块
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r"(\{.*?\})", raw_text, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group(1))
                candidates = data.get("candidates", [])
                reasoning = data.get("reasoning", "")
                if len(candidates) >= self.num_candidates:
                    return candidates[:self.num_candidates], reasoning
            except Exception:
                pass

        # 尝试 3：正则提取带编号的候选项
        numbered_pattern = re.findall(
            r"(?:候选\s*(?:提示词)?\s*[1-3]|Candidate\s*[1-3])[：:\s]*\n?(.*?)(?=(?:候选|Candidate|\Z))",
            raw_text,
            re.DOTALL | re.IGNORECASE,
        )
        if len(numbered_pattern) >= self.num_candidates:
            candidates = [c.strip() for c in numbered_pattern[:self.num_candidates]]
            return candidates, "（自动提取自非结构化输出）"

        # 终极 Fallback：复用当前 Prompt 并添加通用改进指令
        print(f"[PromptOptimizer] 警告：解析失败，使用 fallback 策略生成候选提示词")
        fallback_candidates = [
            current_prompt + "\n\n注意：请确保回答更加准确、完整且有针对性。",
            current_prompt + "\n\n注意：请用更清晰的结构组织回答，分点列出关键信息。",
            current_prompt + "\n\n注意：回答时请先理解问题核心，然后给出简洁实用的解决方案。",
        ]
        return fallback_candidates, "（解析失败，使用 fallback 候选提示词）"
