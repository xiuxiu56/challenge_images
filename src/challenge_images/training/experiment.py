"""训练元数据和历史实验对比。"""

from __future__ import annotations

import csv
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import REPORTS_DIR, RUNS_DIR


def save_training_meta(run_dir: str | Path, *, model: str, config: dict[str, Any], data_dir: str | Path, class_names: dict[int, str] | None = None) -> Path:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        torch_version = torch.__version__
    except Exception:
        torch_version = "未知"
    try:
        import ultralytics
        ultralytics_version = ultralytics.__version__
    except Exception:
        ultralytics_version = "未知"
    meta = {"创建时间": datetime.now().isoformat(timespec="seconds"), "模型": model, "数据目录": str(data_dir), "训练参数": config, "类别顺序": class_names or {}, "Python版本": platform.python_version(), "PyTorch版本": torch_version, "Ultralytics版本": ultralytics_version}
    output = path / "training_meta.json"
    output.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def compare_runs(project_dir: str | Path = RUNS_DIR / "classify") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(Path(project_dir).glob("*/results.csv")):
        try:
            with csv_path.open(newline="", encoding="utf-8") as file:
                records = list(csv.DictReader(file))
            if not records: continue
            row: dict[str, Any] = {"实验": csv_path.parent.name, "结果文件": str(csv_path), "轮数": len(records)}
            for key, value in records[-1].items():
                if key.strip():
                    try: row[key.strip()] = float(value)
                    except (TypeError, ValueError): row[key.strip()] = value
            rows.append(row)
        except (OSError, csv.Error):
            continue
    return rows


def save_compare_report(path: str | Path = REPORTS_DIR / "experiment_compare.json") -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(compare_runs(), ensure_ascii=False, indent=2), encoding="utf-8")
    return output
