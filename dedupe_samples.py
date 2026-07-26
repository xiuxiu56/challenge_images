"""项目根目录下的重复样本清理入口。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from challenge_images.data.dedupe_samples import main


if __name__ == "__main__":
    main()
