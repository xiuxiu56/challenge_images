"""在线挑战样本的响应解析、归档和 GUI 导入服务。

支持：
1. 人工从开发者工具导出的 reload/payload 文件导入
2. Playwright 自动捕获后的内存字节归档
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import threading
from typing import Any

from PIL import Image

from ..grid.grid_engine import resolve_challenge_grid
from .click_geometry import grid_from_pmeta
from ..category_map import (
    challenge_type_zh,
    class_to_mid,
    mid_to_dataset_class,
    mid_to_en,
    mid_to_zh,
    normalize_dataset_class,
)
from ..config import ONLINE_CAPTURE_DIR


DEMO_URL = "https://www.google.com/recaptcha/api2/demo"
IMAGE_CHALLENGE_TYPES = {"dynamic", "imageselect", "multicaptcha", "tileselect"}
ARCHIVE_FULL_CHALLENGE = "full_challenge"
ARCHIVE_REPLACEMENT_TILE = "replacement_tile"


@dataclass(frozen=True)
class OnlineSample:
    """一张已经归档、可直接交给 GUI 的在线样本。"""

    path: Path
    challenge_type: str
    raw_class: str
    target_class: str
    category_mid: str | None
    category_zh: str
    category_en: str
    sha256: str
    metadata_path: Path
    archive_kind: str = ARCHIVE_FULL_CHALLENGE
    source_tile_id: int | None = None
    source_tile_index: int | None = None

    def as_gui_sample(self) -> dict[str, Any]:
        """转换为 GUI 现有的样本字典结构。"""
        return {
            "path": self.path,
            "sha256": self.sha256,
            "raw_class": self.raw_class,
            "target_class": self.target_class,
        }


def _is_category_row(node: Any) -> bool:
    return (
        isinstance(node, list)
        and bool(node)
        and isinstance(node[0], str)
        and node[0].startswith("/m/")
    )


def _category_entry(row: list[Any]) -> dict[str, str | None]:
    label = row[6] if len(row) > 6 and isinstance(row[6], str) else None
    return {"id": str(row[0]), "label": label}


def parse_pmeta_categories(pmeta: Any) -> list[dict[str, str | None]]:
    """按 reload 的 pmeta 结构提取类别，支持单类和 multicaptcha。"""
    if not isinstance(pmeta, list) or not pmeta:
        return []
    if pmeta[0] != "pmeta":
        return [_category_entry(pmeta)] if _is_category_row(pmeta) else []
    if len(pmeta) > 1 and _is_category_row(pmeta[1]):
        return [_category_entry(pmeta[1])]
    rows: list[dict[str, str | None]] = []

    def collect(node: Any) -> None:
        if _is_category_row(node):
            rows.append(_category_entry(node))
        elif isinstance(node, list):
            for child in node:
                collect(child)

    if len(pmeta) > 5:
        collect(pmeta[5])
    unique: dict[str, dict[str, str | None]] = {}
    for row in rows:
        unique.setdefault(str(row["id"]), row)
    return list(unique.values())


def parse_reload_response(text: str) -> dict[str, Any]:
    """解析手工导出的 ``/reload`` 响应文本。"""
    raw = text.strip()
    if raw.startswith(")]}'"):
        newline = raw.find("\n")
        raw = raw[newline + 1 :] if newline >= 0 else raw[4:]
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("reload 响应必须是 JSON 数组")
    # 常规响应使用 [4]=pmeta、[5]=挑战类型；不同版本偶尔会插入字段，
    # 因此保留按标记扫描的兜底逻辑。
    pmeta = data[4] if len(data) > 4 and isinstance(data[4], list) and data[4] and data[4][0] == "pmeta" else None
    if pmeta is None:
        pmeta = next(
            (item for item in data if isinstance(item, list) and item and item[0] == "pmeta"),
            None,
        )
    raw_type = str(data[5]) if len(data) > 5 and isinstance(data[5], str) and data[5] else "unknown"
    if raw_type == "unknown":
        known_types = {"dynamic", "imageselect", "tileselect", "multicaptcha", "audio", "nocaptcha", "doscaptcha"}
        raw_type = next((item for item in data if isinstance(item, str) and item.lower() in known_types), "unknown")
    challenge_type, challenge_desc = challenge_type_zh(raw_type)
    categories = parse_pmeta_categories(pmeta)
    payload = data[9] if len(data) > 9 else None
    payload_hash = (
        hashlib.sha256(str(payload).encode("utf-8")).hexdigest()
        if payload
        else None
    )
    # 统一走 grid_from_pmeta：它会深度优先找分类行并按已验证的下标取行列。
    # 此处原本自己读 row[2]/row[3]，与旧版 grid_from_pmeta 犯同一个 off-by-one，
    # 导致全部 4×4 挑战解析失败。
    detected_grid = grid_from_pmeta(pmeta)
    resolved_grid = resolve_challenge_grid(challenge_type, detected_grid)
    grid = {"rows": resolved_grid.rows, "columns": resolved_grid.columns}
    return {
        "challenge_type": challenge_type,
        "challenge_type_desc": challenge_desc,
        "categories": categories,
        "pmeta": pmeta,
        "grid": grid,
        # 元数据只保存摘要，不保存完整 payload 令牌。
        "payload_token_sha256": payload_hash,
    }


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", value.strip())
    return cleaned.strip("_") or "未知类别"


class OnlineCaptureService:
    """管理在线样本归档，供 PySide6 GUI 调用。"""

    def __init__(self, root: str | Path = ONLINE_CAPTURE_DIR) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records_path = self.root / "records.json"
        self.replacements_root = self.root / "replacements"
        self.replacement_records_path = self.replacements_root / "records.json"
        self.legacy_records_path = self.root / "records.jsonl"
        self._archive_lock = threading.RLock()

    def import_sample(
        self,
        image_path: str | Path,
        reload_response_path: str | Path | None = None,
        challenge_type: str = "dynamic",
        category: str | None = None,
    ) -> OnlineSample:
        """归档一张 payload 图片，并可选解析对应 reload 响应。"""
        source = Path(image_path)
        if not source.is_file():
            raise FileNotFoundError(f"在线挑战图片不存在：{source}")
        # 导入时先让 Pillow 验证图片，避免隐藏文件或错误响应进入样本目录。
        with Image.open(source) as image:
            image.verify()

        reload_text: str | None = None
        reload_source: Path | None = None
        if reload_response_path:
            reload_source = Path(reload_response_path)
            if not reload_source.is_file():
                raise FileNotFoundError(f"reload 响应文件不存在：{reload_source}")
            reload_text = reload_source.read_text(encoding="utf-8")

        return self._archive_bytes(
            image_bytes=source.read_bytes(),
            source_image=str(source),
            suffix=source.suffix.lower(),
            reload_text=reload_text,
            challenge_type=challenge_type,
            category=category,
        )

    def import_bytes(
        self,
        image_bytes: bytes,
        reload_text: str | None = None,
        *,
        challenge_type: str = "dynamic",
        category: str | None = None,
        suffix: str = ".jpg",
        source_image: str = "memory://payload",
        archive_kind: str = ARCHIVE_FULL_CHALLENGE,
        source_tile_id: int | None = None,
        source_tile_index: int | None = None,
    ) -> OnlineSample:
        """直接归档内存中的 payload 与 reload 文本（在线自动捕获用）。"""
        if not image_bytes:
            raise ValueError("payload 图片字节为空")
        return self._archive_bytes(
            image_bytes=image_bytes,
            source_image=source_image,
            suffix=suffix,
            reload_text=reload_text,
            challenge_type=challenge_type,
            category=category,
            archive_kind=archive_kind,
            source_tile_id=source_tile_id,
            source_tile_index=source_tile_index,
        )

    def _archive_bytes(
        self,
        *,
        image_bytes: bytes,
        source_image: str,
        suffix: str,
        reload_text: str | None,
        challenge_type: str,
        category: str | None,
        archive_kind: str = ARCHIVE_FULL_CHALLENGE,
        source_tile_id: int | None = None,
        source_tile_index: int | None = None,
    ) -> OnlineSample:
        # 用临时校验保证不是 HTML/错误页
        from io import BytesIO

        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()

        parsed: dict[str, Any] = {}
        if reload_text:
            parsed = parse_reload_response(reload_text)

        parsed_type = str(parsed.get("challenge_type") or challenge_type).lower()
        if parsed_type not in IMAGE_CHALLENGE_TYPES:
            parsed_type = challenge_type if challenge_type in IMAGE_CHALLENGE_TYPES else "dynamic"
        categories = parsed.get("categories") or []
        primary = categories[0] if categories else {}
        mid = str(primary.get("id")) if primary.get("id") else class_to_mid(category)
        label = str(primary.get("label")) if primary.get("label") else (category or "")
        target_class = (
            mid_to_dataset_class(mid)
            or normalize_dataset_class(label)
            or normalize_dataset_class(category)
            or "Other"
        )
        category_zh = mid_to_zh(mid) if mid else (category or "未知类别")
        category_en = mid_to_en(mid) if mid else target_class
        raw_class = category_zh if category_zh != "未知类别" else target_class

        digest = hashlib.sha256(image_bytes).hexdigest()
        mid_name = _safe_name(mid.lstrip("/").replace("/", "_") if mid else target_class)
        clean_suffix = suffix.lower() if suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} else ".jpg"
        if archive_kind not in {ARCHIVE_FULL_CHALLENGE, ARCHIVE_REPLACEMENT_TILE}:
            raise ValueError(f"未知归档类型：{archive_kind}")

        if archive_kind == ARCHIVE_REPLACEMENT_TILE:
            target_dir = self.replacements_root / parsed_type / _safe_name(raw_class)
            tile_index = "unknown" if source_tile_index is None else str(source_tile_index)
            tile_id = "unknown" if source_tile_id is None else str(source_tile_id)
            image_basename = f"{mid_name}_tile{tile_index}_ds{tile_id}"
            records_path = self.replacement_records_path
        else:
            target_dir = self.root / parsed_type / _safe_name(raw_class)
            image_basename = mid_name
            records_path = self.records_path

        with self._archive_lock:
            target_dir.mkdir(parents=True, exist_ok=True)
            sequence = self._next_image_sequence(target_dir, image_basename)
            target = target_dir / f"{image_basename}_{sequence}{clean_suffix}"
            target.write_bytes(image_bytes)

            saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            challenge_desc = parsed.get("challenge_type_desc")
            if not challenge_desc:
                _, challenge_desc = challenge_type_zh(parsed_type)
            metadata_path = target.with_suffix(target.suffix + ".json")
            metadata = {
                "saved_at": saved_at,
                "archive_kind": archive_kind,
                "source_image": source_image,
                "image_path": str(target),
                "challenge_type": parsed_type,
                "challenge_type_desc": challenge_desc,
                "image_basename": image_basename,
                "image_name": target.name,
                "pmeta": parsed.get("pmeta"),
                "category_mid": mid,
                "category_zh": category_zh,
                "category_en": category_en,
                "target_class": target_class,
                "categories": categories,
                "sha256": digest,
                "payload_token_sha256": parsed.get("payload_token_sha256"),
                "grid": parsed.get("grid"),
                "source_tile_id": source_tile_id,
                "source_tile_index": source_tile_index,
            }
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            records = self._load_records(records_path)
            record = {
                "saved_at": saved_at,
                "id": self._next_record_id(records),
                "challenge_type": parsed_type,
                "challenge_type_desc": challenge_desc,
                "image_basename": image_basename,
                "image_name": target.name,
                # reload 响应数组下标 4，按原始数组结构完整保存。
                "pmeta": parsed.get("pmeta"),
            }
            if archive_kind == ARCHIVE_REPLACEMENT_TILE:
                record.update(
                    {
                        "archive_kind": archive_kind,
                        "source_tile_id": source_tile_id,
                        "source_tile_index": source_tile_index,
                    }
                )
            records.append(record)
            self._write_records(records_path, records)

        return OnlineSample(
            path=target,
            challenge_type=parsed_type,
            raw_class=raw_class,
            target_class=target_class,
            category_mid=mid,
            category_zh=category_zh,
            category_en=category_en,
            sha256=digest,
            metadata_path=metadata_path,
            archive_kind=archive_kind,
            source_tile_id=source_tile_id,
            source_tile_index=source_tile_index,
        )

    @staticmethod
    def _load_records(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return []
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            data = data["items"]
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    @staticmethod
    def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _next_record_id(records: list[dict[str, Any]]) -> int:
        ids = [item.get("id") for item in records]
        return max((value for value in ids if isinstance(value, int)), default=0) + 1

    @staticmethod
    def _next_image_sequence(target_dir: Path, basename: str) -> int:
        pattern = re.compile(rf"^{re.escape(basename)}_(\d+)\.", re.IGNORECASE)
        maximum = 0
        for path in target_dir.iterdir():
            match = pattern.match(path.name) if path.is_file() else None
            if match:
                maximum = max(maximum, int(match.group(1)))
        return maximum + 1

    def latest_sample(self) -> OnlineSample | None:
        """读取最后一条仍存在的在线样本。"""
        records = self._load_records(self.records_path)
        for item in reversed(records):
            path = self._resolve_record_image(item)
            if not path.is_file():
                continue
            return self._sample_from_record(item, path)
        return self._latest_legacy_sample()

    def _resolve_record_image(self, item: dict[str, Any]) -> Path:
        explicit = Path(str(item.get("image_path") or ""))
        if explicit.is_file():
            return explicit
        image_name = str(item.get("image_name") or "")
        challenge_type = _safe_name(str(item.get("challenge_type") or "dynamic"))
        if image_name:
            matches = list((self.root / challenge_type).glob(f"*/{image_name}"))
            if matches:
                return matches[0]
        return explicit

    def _sample_from_record(self, item: dict[str, Any], path: Path) -> OnlineSample:
        metadata_path = path.with_suffix(path.suffix + ".json")
        metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            try:
                loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata = loaded if isinstance(loaded, dict) else {}
            except (OSError, json.JSONDecodeError):
                metadata = {}
        pmeta = item.get("pmeta")
        categories = parse_pmeta_categories(pmeta)
        primary = categories[0] if categories else {}
        mid = metadata.get("category_mid") or primary.get("id")
        label = primary.get("label")
        target_class = (
            metadata.get("target_class")
            or mid_to_dataset_class(mid)
            or normalize_dataset_class(label)
            or "Other"
        )
        category_zh = metadata.get("category_zh") or (mid_to_zh(mid) if mid else "未知类别")
        category_en = metadata.get("category_en") or (mid_to_en(mid) if mid else target_class)
        return OnlineSample(
            path=path,
            challenge_type=str(item.get("challenge_type") or metadata.get("challenge_type") or "dynamic"),
            raw_class=str(category_zh or target_class),
            target_class=str(target_class),
            category_mid=str(mid) if mid else None,
            category_zh=str(category_zh),
            category_en=str(category_en),
            sha256=str(metadata.get("sha256") or hashlib.sha256(path.read_bytes()).hexdigest()),
            metadata_path=metadata_path,
            archive_kind=str(metadata.get("archive_kind") or ARCHIVE_FULL_CHALLENGE),
            source_tile_id=metadata.get("source_tile_id"),
            source_tile_index=metadata.get("source_tile_index"),
        )

    def _latest_legacy_sample(self) -> OnlineSample | None:
        """兼容旧版 records.jsonl；新归档不再写入该文件。"""
        if not self.legacy_records_path.is_file():
            return None
        lines = self.legacy_records_path.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            path = Path(str(item.get("image_path") or ""))
            if path.is_file():
                return self._sample_from_record(item, path)
        return None
