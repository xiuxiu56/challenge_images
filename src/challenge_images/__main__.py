"""支持 ``python -m challenge_images`` 启动主菜单。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# main.py 位于项目根目录而非包内，必须先把根目录加入 sys.path 才能导入，
# 因此这条 import 无法上移到文件顶部。
from main import main  # noqa: E402


if __name__ == "__main__":
    main()
