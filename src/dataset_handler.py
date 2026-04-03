"""
DatasetHandler 模块
负责加载和管理黄金测试集（Gold Standard Dataset）
支持 JSON 和 CSV 格式，包含 input 和 expected_output 字段
"""

import json
import csv
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TestCase:
    """单条测试样本的数据结构"""
    id: str
    input: str
    expected_output: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"TestCase(id={self.id!r}, input={self.input[:50]!r}...)"


class DatasetHandler:
    """
    黄金测试集管理器

    职责：
    - 从 JSON / CSV 文件加载测试样本
    - 提供随机采样以控制评估成本
    - 校验数据格式完整性

    JSON 格式示例:
        [{"id": "1", "input": "...", "expected_output": "..."}]

    CSV 格式示例（表头必须包含 id, input, expected_output）:
        id,input,expected_output
        1,"问题...","标准答案..."
    """

    def __init__(self, file_path: str) -> None:
        """
        Args:
            file_path: 测试集文件路径，支持 .json 或 .csv
        """
        self.file_path = Path(file_path)
        self._test_cases: list[TestCase] = []
        self._load()

    # ------------------------------------------------------------------ #
    #  私有方法：加载                                                       #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """根据文件扩展名分发到对应加载器"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.file_path}")

        suffix = self.file_path.suffix.lower()
        if suffix == ".json":
            self._load_json()
        elif suffix == ".csv":
            self._load_csv()
        else:
            raise ValueError(f"不支持的文件格式: {suffix}，请使用 .json 或 .csv")

        self._validate()
        print(f"[DatasetHandler] 成功加载 {len(self._test_cases)} 条测试样本，来源：{self.file_path}")

    def _load_json(self) -> None:
        with open(self.file_path, "r", encoding="utf-8") as f:
            raw_data: list[dict] = json.load(f)

        for idx, item in enumerate(raw_data):
            self._test_cases.append(TestCase(
                id=str(item.get("id", idx)),
                input=item["input"],
                expected_output=item["expected_output"],
                metadata={k: v for k, v in item.items()
                          if k not in ("id", "input", "expected_output")},
            ))

    def _load_csv(self) -> None:
        with open(self.file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                self._test_cases.append(TestCase(
                    id=str(row.get("id", idx)),
                    input=row["input"],
                    expected_output=row["expected_output"],
                    metadata={k: v for k, v in row.items()
                              if k not in ("id", "input", "expected_output")},
                ))

    def _validate(self) -> None:
        """简单完整性校验，确保核心字段非空"""
        for tc in self._test_cases:
            if not tc.input.strip():
                raise ValueError(f"测试样本 id={tc.id} 的 input 字段为空")
            if not tc.expected_output.strip():
                raise ValueError(f"测试样本 id={tc.id} 的 expected_output 字段为空")

    # ------------------------------------------------------------------ #
    #  公共方法                                                             #
    # ------------------------------------------------------------------ #

    def get_all(self) -> list[TestCase]:
        """返回全部测试样本"""
        return list(self._test_cases)

    def sample(self, n: int, seed: Optional[int] = None) -> list[TestCase]:
        """
        随机采样 n 条测试样本（用于快速评估，节省 API 成本）

        Args:
            n:    采样数量，若超过总数则返回全部
            seed: 随机种子，保证可复现

        Returns:
            采样后的 TestCase 列表
        """
        if n >= len(self._test_cases):
            return self.get_all()
        rng = random.Random(seed)
        return rng.sample(self._test_cases, n)

    def __len__(self) -> int:
        return len(self._test_cases)

    def __repr__(self) -> str:
        return f"DatasetHandler(file={self.file_path.name!r}, size={len(self)})"
