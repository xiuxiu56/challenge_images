"""挑战大图精确重复清理。

默认只生成报告；命令行传入 --delete 后才删除重复文件。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .sample_manager import scan_duplicates
from ..config import CHALLENGE_DIR, REPORTS_DIR


def deduplicate(root: str | Path, delete: bool = False, challenge_type: str | None = None) -> tuple[Path, int]:
    base = Path(root)
    groups = scan_duplicates(base, challenge_type=challenge_type)
    report = REPORTS_DIR / "duplicate_samples.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    removed: list[str] = []
    kept: list[str] = []
    for paths in groups.values():
        keep = paths[0]
        kept.append(keep)
        for duplicate in paths[1:]:
            # 未指定挑战类型时只做报告，避免跨 dynamic/imageselect/multicaptcha 误删。
            same_challenge = Path(keep).parts[-3] == Path(duplicate).parts[-3]
            if delete and challenge_type and same_challenge:
                path = Path(duplicate)
                if path.is_file():
                    path.unlink()
                    removed.append(duplicate)
            else:
                removed.append(duplicate)
    report.write_text(
        json.dumps({"根目录": str(base), "挑战类型": challenge_type or "全部（仅报告跨类型）", "重复组数": len(groups), "保留": kept, "待删除或已删除": removed, "已执行删除": delete}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report, len(removed)


def main() -> None:
    parser = argparse.ArgumentParser(description="挑战大图 SHA-256 精确去重")
    parser.add_argument("--root", default=str(CHALLENGE_DIR), help="挑战大图根目录")
    parser.add_argument("--delete", action="store_true", help="确认后删除重复文件")
    parser.add_argument("--challenge", choices=("dynamic", "imageselect", "multicaptcha"), help="只处理指定挑战类型")
    args = parser.parse_args()
    report, count = deduplicate(args.root, delete=args.delete, challenge_type=args.challenge)
    print(f"重复组处理完成：{count} 个重复文件")
    print(f"报告：{report}")


if __name__ == "__main__":
    main()
