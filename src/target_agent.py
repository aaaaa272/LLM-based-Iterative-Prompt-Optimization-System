"""
TargetAgent 模块
执行具体的业务任务：接收 System Prompt + User Input，调用 LLM 生成回答
使用 DeepSeek API（兼容 OpenAI 协议）
"""

import time
from openai import OpenAI
from dataclasses import dataclass


@dataclass
class AgentResponse:
    """TargetAgent 单次调用的返回结果"""
    output: str             # 模型生成的文本
    input_tokens: int       # 消耗的输入 token 数
    output_tokens: int      # 消耗的输出 token 数
    latency_seconds: float  # 调用耗时（秒）


class TargetAgent:
    """
    业务 Agent，负责根据给定的 System Prompt 回答用户问题。

    使用 DeepSeek API（OpenAI 兼容格式），通过环境变量 DEEPSEEK_API_KEY 鉴权。

    使用示例:
        agent = TargetAgent(model="deepseek-chat")
        resp = agent.run(
            system_prompt="你是一个专业的技术助手...",
            user_input="什么是 RAG？"
        )
        print(resp.output)
    """

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        model: str = "deepseek-chat",
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ) -> None:
        """
        Args:
            model:       DeepSeek 模型 ID
                         - "deepseek-chat"     通用对话模型（V3，推荐）
                         - "deepseek-reasoner" 深度推理模型（R1）
            max_tokens:  最大输出 token 数
            temperature: 采样温度（0~1）
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        # DEEPSEEK_API_KEY 从环境变量读取
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

    def run(self, system_prompt: str, user_input: str) -> AgentResponse:
        """
        执行一次推理调用。

        Args:
            system_prompt: 当前版本的系统提示词
            user_input:    来自测试集的用户输入

        Returns:
            AgentResponse 包含输出文本、token 用量和耗时
        """
        start_time = time.perf_counter()

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
        )

        latency = time.perf_counter() - start_time
        choice = response.choices[0]

        # deepseek-reasoner 的最终回答在 content，推理过程在 reasoning_content
        output_text = choice.message.content or ""

        return AgentResponse(
            output=output_text,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_seconds=round(latency, 3),
        )
