"""识别结果回归评测：改阈值后看全量影响，而不是只看单张图。

项目里的融合阈值注释写着「能恢复 483 的格子8，同时排除 486 的格子11」，
说明调参完全依赖肉眼逐张比对。这样既无法确认改动是否伤害了其他样本，
也无法在重构后确认行为未变。

本模块提供两种评测：

有真值时（``annotations/grid_annotations.json`` 由 GUI 标注产生）
    计算逐类、逐挑战类型的 Precision / Recall / F1 与完全匹配率。

无真值时
    做 A/B 对照：用两套阈值各跑一遍，报告有多少张图的选择发生了变化、
    具体变成了什么。这在「改一个阈值想知道影响面」时最有用。

两种模式都会把结果落盘，便于跨版本比较。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ..annotation_store import AnnotationStore
from ..grid.grid_engine import GridSpec, parse_grid, resolve_challenge_grid


@dataclass
class SampleCase:
    """一条待评测样本。"""

    path: Path
    challenge_type: str
    target_class: str
    spec: GridSpec
    truth: list[int] | None = None


@dataclass
class CaseOutcome:
    """单条样本的评测结果。"""

    path: str
    challenge_type: str
    target_class: str
    predicted: list[int]
    truth: list[int] | None = None

    @property
    def exact_match(self) -> bool | None:
        if self.truth is None:
            return None
        return sorted(self.predicted) == sorted(self.truth)

    @property
    def missed(self) -> list[int]:
        return sorted(set(self.truth or []) - set(self.predicted))

    @property
    def extra(self) -> list[int]:
        return sorted(set(self.predicted) - set(self.truth or []))


@dataclass
class RegressionReport:
    """整体评测报告。"""

    outcomes: list[CaseOutcome] = field(default_factory=list)

    @property
    def scored(self) -> list[CaseOutcome]:
        return [item for item in self.outcomes if item.truth is not None]

    def metrics(self) -> dict[str, Any]:
        """按整体、逐类、逐挑战类型统计 P/R/F1。"""
        scored = self.scored
        if not scored:
            return {}
        return {
            "整体": _score(scored),
            "逐类别": {
                name: _score([item for item in scored if item.target_class == name])
                for name in sorted({item.target_class for item in scored})
            },
            "逐挑战类型": {
                name: _score([item for item in scored if item.challenge_type == name])
                for name in sorted({item.challenge_type for item in scored})
            },
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "样本总数": len(self.outcomes),
            "含真值样本": len(self.scored),
            "指标": self.metrics(),
            "样本": [
                {
                    "图片": item.path,
                    "挑战类型": item.challenge_type,
                    "目标类别": item.target_class,
                    "预测格子": item.predicted,
                    "真实格子": item.truth,
                    "漏选": item.missed if item.truth is not None else None,
                    "误选": item.extra if item.truth is not None else None,
                }
                for item in self.outcomes
            ],
        }


def _score(items: list[CaseOutcome]) -> dict[str, float | int]:
    """按格子级 TP/FP/FN 汇总。"""
    true_positive = false_positive = false_negative = 0
    exact = 0
    for item in items:
        predicted = set(item.predicted)
        truth = set(item.truth or [])
        true_positive += len(predicted & truth)
        false_positive += len(predicted - truth)
        false_negative += len(truth - predicted)
        exact += int(item.exact_match is True)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "样本数": len(items),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "完全匹配率": round(exact / len(items), 4) if items else 0.0,
        "漏选格子数": false_negative,
        "误选格子数": false_positive,
    }


def load_cases(
    annotations_path: str | Path = "annotations/grid_annotations.json",
    *,
    limit: int | None = None,
) -> list[SampleCase]:
    """从 GUI 标注文件读取带真值的评测样本。"""
    store = AnnotationStore(annotations_path)
    cases: list[SampleCase] = []
    for image_path, record in store.data.items():
        path = Path(image_path)
        if not path.is_file():
            continue
        challenge_type = str(record.get("挑战类型", "dynamic"))
        grid_text = str(record.get("网格", ""))
        try:
            spec = parse_grid(grid_text) if grid_text else resolve_challenge_grid(challenge_type)
        except ValueError:
            spec = resolve_challenge_grid(challenge_type)
        cases.append(
            SampleCase(
                path=path,
                challenge_type=challenge_type,
                target_class=str(record.get("目标类别", "")),
                spec=spec,
                truth=[int(index) for index in record.get("真实格子", [])],
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases


def evaluate(
    cases: Iterable[SampleCase],
    recognize: Callable[[SampleCase], list[int]],
) -> RegressionReport:
    """对每条样本执行识别并汇总结果。

    ``recognize`` 接受一条样本、返回选中的格子索引，由调用方注入，
    因此本模块不依赖任何具体模型加载方式，便于在测试中替换。
    """
    report = RegressionReport()
    for case in cases:
        try:
            predicted = sorted(int(index) for index in recognize(case))
        except Exception as error:  # 单条失败不应中断整轮评测
            print(f"[回归评测] 跳过 {case.path.name}：{type(error).__name__}: {error}")
            continue
        report.outcomes.append(
            CaseOutcome(
                path=str(case.path),
                challenge_type=case.challenge_type,
                target_class=case.target_class,
                predicted=predicted,
                truth=case.truth,
            )
        )
    return report


@dataclass
class ComparisonEntry:
    """A/B 对照中发生变化的一条样本。"""

    path: str
    target_class: str
    challenge_type: str
    before: list[int]
    after: list[int]

    @property
    def added(self) -> list[int]:
        return sorted(set(self.after) - set(self.before))

    @property
    def removed(self) -> list[int]:
        return sorted(set(self.before) - set(self.after))


def compare(before: RegressionReport, after: RegressionReport) -> dict[str, Any]:
    """对照两次评测，报告选择发生变化的样本。

    这是无真值时最实用的信号：改一个阈值后，究竟影响了几张图、
    新增了哪些格子、丢掉了哪些格子。
    """
    baseline = {item.path: item for item in before.outcomes}
    changes: list[ComparisonEntry] = []
    for item in after.outcomes:
        previous = baseline.get(item.path)
        if previous is None or sorted(previous.predicted) == sorted(item.predicted):
            continue
        changes.append(
            ComparisonEntry(
                path=item.path,
                target_class=item.target_class,
                challenge_type=item.challenge_type,
                before=previous.predicted,
                after=item.predicted,
            )
        )
    total = len(after.outcomes)
    payload: dict[str, Any] = {
        "对照样本数": total,
        "结果变化样本数": len(changes),
        "变化比例": round(len(changes) / total, 4) if total else 0.0,
        "新增格子总数": sum(len(item.added) for item in changes),
        "移除格子总数": sum(len(item.removed) for item in changes),
        "变化明细": [
            {
                "图片": item.path,
                "目标类别": item.target_class,
                "挑战类型": item.challenge_type,
                "改前": item.before,
                "改后": item.after,
                "新增": item.added,
                "移除": item.removed,
            }
            for item in changes
        ],
    }
    if before.scored and after.scored:
        payload["指标对照"] = {
            "改前": before.metrics().get("整体", {}),
            "改后": after.metrics().get("整体", {}),
        }
    return payload


def format_report(report: RegressionReport) -> str:
    """生成可直接打印的中文评测报告。"""
    metrics = report.metrics()
    if not metrics:
        return (
            f"共评测 {len(report.outcomes)} 条样本，但没有任何真值标注。\n"
            "请先在 GUI 中标注真实格子，或改用 A/B 对照模式。"
        )
    overall = metrics["整体"]
    lines = [
        f"样本总数: {len(report.outcomes)}（含真值 {len(report.scored)}）",
        (
            f"整体: precision={overall['precision']:.4f} "
            f"recall={overall['recall']:.4f} f1={overall['f1']:.4f} "
            f"完全匹配率={overall['完全匹配率']:.4f}"
        ),
        f"漏选格子 {overall['漏选格子数']}，误选格子 {overall['误选格子数']}",
        "",
        f"{'目标类别':<16}{'样本':>6}{'precision':>11}{'recall':>9}{'f1':>9}{'完全匹配':>10}",
    ]
    for name, score in metrics["逐类别"].items():
        lines.append(
            f"{name:<16}{score['样本数']:>6}{score['precision']:>11.4f}"
            f"{score['recall']:>9.4f}{score['f1']:>9.4f}{score['完全匹配率']:>10.4f}"
        )
    lines.append("")
    lines.append(f"{'挑战类型':<16}{'样本':>6}{'precision':>11}{'recall':>9}{'f1':>9}")
    for name, score in metrics["逐挑战类型"].items():
        lines.append(
            f"{name:<16}{score['样本数']:>6}{score['precision']:>11.4f}"
            f"{score['recall']:>9.4f}{score['f1']:>9.4f}"
        )
    return "\n".join(lines)


def format_comparison(payload: dict[str, Any], *, max_rows: int = 20) -> str:
    """生成可直接打印的中文 A/B 对照报告。"""
    lines = [
        f"对照样本数: {payload['对照样本数']}",
        f"结果发生变化: {payload['结果变化样本数']} 张（{payload['变化比例']:.1%}）",
        f"新增格子 {payload['新增格子总数']}，移除格子 {payload['移除格子总数']}",
    ]
    details = payload.get("变化明细", [])
    if details:
        lines.append("")
        lines.append("变化明细:")
        for entry in details[:max_rows]:
            lines.append(
                f"  {Path(entry['图片']).name}｜{entry['目标类别']}｜{entry['挑战类型']}："
                f"{entry['改前']} → {entry['改后']}"
                f"（新增 {entry['新增']}，移除 {entry['移除']}）"
            )
        if len(details) > max_rows:
            lines.append(f"  …… 其余 {len(details) - max_rows} 条见 JSON 报告")
    return "\n".join(lines)


def save_report(report: RegressionReport, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return output
