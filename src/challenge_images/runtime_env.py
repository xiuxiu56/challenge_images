"""运行环境检查与缓存目录管理。

本模块不导入 Ultralytics，菜单启动时可以安全调用。
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".runtime_cache"


def prepare_cache_dir() -> Path:
    """创建项目内缓存目录，避免使用不可写的用户缓存目录。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mpl_dir = CACHE_DIR / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_DIR / "ultralytics"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def mps_status() -> dict[str, Any]:
    """返回 MPS 构建状态与运行状态。"""
    result: dict[str, Any] = {
        "系统": platform.platform(),
        "架构": platform.machine(),
        "python": platform.python_version(),
        "torch": "未安装",
        "mps_built": False,
        "mps_available": False,
        "device": "cpu",
    }
    try:
        import torch

        result["torch"] = torch.__version__
        backend = getattr(torch.backends, "mps", None)
        result["mps_built"] = bool(backend and backend.is_built())
        result["mps_available"] = bool(backend and backend.is_available())
        result["device"] = "mps" if result["mps_available"] else "cpu"
    except Exception as exc:  # 诊断信息不能阻断菜单
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def choose_device(prefer: str | None = None) -> str:
    """规范化设备参数；MPS 不可用时自动回退 CPU。"""
    status = mps_status()
    value = (prefer or "").strip().lower()
    if value in {"mps", "metal"}:
        if status["mps_available"]:
            return "mps"
        print("[提示] 当前 PyTorch 的 MPS 运行态不可用，训练设备回退为 cpu。")
        return "cpu"
    if value in {"cpu", ""}:
        return "cpu" if value == "cpu" else str(status["device"])
    return prefer or str(status["device"])


def status_text() -> str:
    """生成中文环境摘要。"""
    info = mps_status()
    mps = "可用" if info["mps_available"] else "不可用"
    built = "已构建" if info["mps_built"] else "未构建"
    return (
        f"系统={info['系统']} | 架构={info['架构']} | "
        f"Python={info['python']} | torch={info['torch']} | "
        f"MPS={mps}（{built}）| 默认设备={info['device']}"
    )


def print_status() -> None:
    """打印完整环境检查结果。"""
    prepare_cache_dir()
    print("======== 运行环境 ========")
    print(status_text())
    print(f"项目缓存目录={CACHE_DIR}")
    if mps_status()["mps_available"]:
        print("建议：正式训练优先使用 device=mps。")
    else:
        print("建议：先确认 macOS、PyTorch arm64 与 Metal 环境，再运行正式训练。")


prepare_cache_dir()
