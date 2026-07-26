"""支持 ``python -m challenge_images`` 启动主菜单。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from main import main


if __name__ == "__main__":
    main()
