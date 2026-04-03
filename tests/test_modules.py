"""
基础单元测试（不调用真实 API，仅验证模块导入和数据处理逻辑）
运行方式: python -m pytest tests/ -v
"""

import json
import sys
import tempfile
from pathlib import Path

# 项目根目录加入路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------------ #
#  DatasetHandler 测试                                                 #
# ------------------------------------------------------------------ #

def test_dataset_handler_load_json():
    """测试 JSON 格式测试集加载"""
    from src.dataset_handler import DatasetHandler

    sample_data = [
        {"id": "1", "input": "问题1", "expected_output": "答案1"},
        {"id": "2", "input": "问题2", "expected_output": "答案2"},
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(sample_data, f, ensure_ascii=False)
        tmp_path = f.name

    handler = DatasetHandler(file_path=tmp_path)
    assert len(handler) == 2
    cases = handler.get_all()
    assert cases[0].id == "1"
    assert cases[0].input == "问题1"
    assert cases[1].expected_output == "答案2"
    print("✓ DatasetHandler JSON 加载测试通过")


def test_dataset_handler_sample():
    """测试随机采样功能"""
    from src.dataset_handler import DatasetHandler

    sample_data = [{"id": str(i), "input": f"问题{i}", "expected_output": f"答案{i}"}
                   for i in range(20)]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(sample_data, f, ensure_ascii=False)
        tmp_path = f.name

    handler = DatasetHandler(file_path=tmp_path)

    # 采样 5 条，应该得到 5 条
    sampled = handler.sample(n=5, seed=42)
    assert len(sampled) == 5

    # 采样超过总数，应该返回全部
    sampled_all = handler.sample(n=100)
    assert len(sampled_all) == 20

    # 相同 seed 结果应该相同
    s1 = handler.sample(n=5, seed=99)
    s2 = handler.sample(n=5, seed=99)
    assert [tc.id for tc in s1] == [tc.id for tc in s2]
    print("✓ DatasetHandler 采样测试通过")


def test_dataset_handler_load_csv():
    """测试 CSV 格式测试集加载"""
    import csv
    from src.dataset_handler import DatasetHandler

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["id", "input", "expected_output"])
        writer.writerow(["1", "CSV问题1", "CSV答案1"])
        writer.writerow(["2", "CSV问题2", "CSV答案2"])
        tmp_path = f.name

    handler = DatasetHandler(file_path=tmp_path)
    assert len(handler) == 2
    assert handler.get_all()[0].input == "CSV问题1"
    print("✓ DatasetHandler CSV 加载测试通过")


# ------------------------------------------------------------------ #
#  EvaluatorJudge 测试（仅测试解析逻辑，不调用 API）                   #
# ------------------------------------------------------------------ #

def test_evaluator_judge_parse_valid_json():
    """测试 JSON 解析成功路径"""
    from src.evaluator_judge import EvaluatorJudge

    judge = EvaluatorJudge.__new__(EvaluatorJudge)  # 跳过 __init__（不初始化 API 客户端）

    valid_json = '{"score": 4, "improvement_feedback": "回答可以更加具体，建议补充实际案例", "reasoning": "整体较好"}'
    result = judge._parse_judge_output(valid_json)
    assert result.score == 4
    assert "具体" in result.improvement_feedback
    print("✓ EvaluatorJudge JSON 解析测试通过")


def test_evaluator_judge_parse_json_in_codeblock():
    """测试从 markdown 代码块中提取 JSON"""
    from src.evaluator_judge import EvaluatorJudge

    judge = EvaluatorJudge.__new__(EvaluatorJudge)

    text_with_codeblock = """
以下是我的评估结果：

```json
{"score": 3, "improvement_feedback": "需要补充示例代码，让用户更容易理解", "reasoning": "逻辑清晰但缺少示例"}
```
"""
    result = judge._parse_judge_output(text_with_codeblock)
    assert result.score == 3
    assert "示例" in result.improvement_feedback
    print("✓ EvaluatorJudge markdown 代码块解析测试通过")


def test_evaluator_judge_parse_fallback():
    """测试解析失败时的 fallback 机制"""
    from src.evaluator_judge import EvaluatorJudge

    judge = EvaluatorJudge.__new__(EvaluatorJudge)

    invalid_output = "这是一段无法解析的文本，没有 JSON 格式"
    result = judge._parse_judge_output(invalid_output)
    assert result.score == 1  # fallback 返回最低分
    assert len(result.improvement_feedback) > 0
    print("✓ EvaluatorJudge fallback 机制测试通过")


def test_judge_result_validation():
    """测试 JudgeResult Pydantic 校验"""
    from src.evaluator_judge import JudgeResult
    from pydantic import ValidationError

    # 有效数据
    jr = JudgeResult(score=5, improvement_feedback="非常好，整体表达清晰，无需进一步改进")
    assert jr.score == 5

    # 无效分数
    try:
        JudgeResult(score=6, improvement_feedback="测试反馈内容无效分数")
        assert False, "应该抛出 ValidationError"
    except ValidationError:
        pass

    # feedback 太短
    try:
        JudgeResult(score=3, improvement_feedback="短文本")
        assert False, "应该抛出 ValidationError"
    except ValidationError:
        pass

    print("✓ JudgeResult Pydantic 校验测试通过")


# ------------------------------------------------------------------ #
#  PromptOptimizer 测试（仅测试解析逻辑）                               #
# ------------------------------------------------------------------ #

def test_prompt_optimizer_parse_valid_json():
    """测试优化器输出 JSON 解析"""
    from src.prompt_optimizer import PromptOptimizer

    optimizer = PromptOptimizer.__new__(PromptOptimizer)
    optimizer.num_candidates = 3

    valid_output = json.dumps({
        "reasoning": "当前提示词过于简单",
        "candidates": [
            "候选提示词版本 A，更详细的描述...",
            "候选提示词版本 B，不同策略...",
            "候选提示词版本 C，备选方案...",
        ]
    }, ensure_ascii=False)

    candidates, reasoning = optimizer._parse_optimizer_output(
        valid_output, current_prompt="旧提示词"
    )
    assert len(candidates) == 3
    assert "简单" in reasoning
    print("✓ PromptOptimizer JSON 解析测试通过")


def test_prompt_optimizer_fallback():
    """测试优化器解析失败时的 fallback 机制"""
    from src.prompt_optimizer import PromptOptimizer

    optimizer = PromptOptimizer.__new__(PromptOptimizer)
    optimizer.num_candidates = 3

    candidates, reasoning = optimizer._parse_optimizer_output(
        raw_text="无效文本",
        current_prompt="原始提示词"
    )
    assert len(candidates) == 3
    assert all("原始提示词" in c for c in candidates)
    print("✓ PromptOptimizer fallback 机制测试通过")


# ------------------------------------------------------------------ #
#  EvaluatorJudge 辅助方法测试                                          #
# ------------------------------------------------------------------ #

def test_compute_average_score():
    """测试平均分计算"""
    from src.evaluator_judge import EvaluatorJudge, EvaluationRecord, JudgeResult

    records = []
    for score in [3, 4, 5]:
        jr = JudgeResult(score=score, improvement_feedback="测试反馈内容，请提供更多细节")
        records.append(EvaluationRecord(
            test_case_id=str(score),
            user_input="测试输入",
            expected_output="期望输出",
            agent_output="实际输出",
            judge_result=jr,
        ))

    avg = EvaluatorJudge.compute_average_score(records)
    assert abs(avg - 4.0) < 1e-6, f"期望 4.0，得到 {avg}"
    print("✓ 平均分计算测试通过")


def test_collect_feedback():
    """测试 feedback 汇总"""
    from src.evaluator_judge import EvaluatorJudge, EvaluationRecord, JudgeResult

    records = []
    for i in range(3):
        jr = JudgeResult(score=i + 3, improvement_feedback=f"改进建议 {i}：请提供更详细的说明")
        records.append(EvaluationRecord(
            test_case_id=str(i),
            user_input="用户输入内容示例",
            expected_output="期望输出内容",
            agent_output="实际输出内容",
            judge_result=jr,
        ))

    feedback = EvaluatorJudge.collect_feedback(records)
    assert "改进建议 0" in feedback
    assert "改进建议 2" in feedback
    print("✓ feedback 汇总测试通过")


if __name__ == "__main__":
    print("\n运行所有测试...\n")
    test_dataset_handler_load_json()
    test_dataset_handler_sample()
    test_dataset_handler_load_csv()
    test_evaluator_judge_parse_valid_json()
    test_evaluator_judge_parse_json_in_codeblock()
    test_evaluator_judge_parse_fallback()
    test_judge_result_validation()
    test_prompt_optimizer_parse_valid_json()
    test_prompt_optimizer_fallback()
    test_compute_average_score()
    test_collect_feedback()
    print("\n✅ 所有测试通过！")
