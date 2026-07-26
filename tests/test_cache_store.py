import time

from challenge_images.data.cache_store import (
    SHARD_LENGTH,
    cache_statistics,
    clear_cache,
    format_cache_report,
    iter_cache_files,
    migrate_flat_cache,
    prune_cache,
    sharded_path,
)

KEY = "a1b2c3d4e5f6" + "0" * 52  # 64 位十六进制


def _write_flat(root, keys):
    root.mkdir(parents=True, exist_ok=True)
    for key in keys:
        (root / f"{key}.json").write_text("{}", encoding="utf-8")


# ---------- 路径分片 ----------


def test_sharded_path_uses_key_prefix(tmp_path):
    path = sharded_path(tmp_path, KEY)
    assert path.parent.name == KEY[:SHARD_LENGTH]
    assert path.name == f"{KEY}.json"


def test_short_keys_fall_back_to_root(tmp_path):
    """键短于分片长度时不能产生非法路径。"""
    assert sharded_path(tmp_path, "a").parent == tmp_path
    assert sharded_path(tmp_path, "").parent == tmp_path


def test_sharding_distributes_keys(tmp_path):
    """两位十六进制前缀提供 256 个桶，真实 SHA-256 会均匀落入。"""
    import hashlib

    keys = [hashlib.sha256(str(index).encode()).hexdigest() for index in range(3000)]
    shards = {sharded_path(tmp_path, key).parent.name for key in keys}
    # 3000 个哈希应覆盖绝大多数桶；均匀性差会让分片失去意义。
    assert len(shards) > 240
    assert all(len(name) == SHARD_LENGTH for name in shards)

    # 上限确实是 256 个桶。
    exhaustive = {
        sharded_path(tmp_path, f"{index:02x}" + "0" * 62).parent.name
        for index in range(256)
    }
    assert len(exhaustive) == 256


# ---------- 迁移 ----------


def test_migrate_moves_flat_files(tmp_path):
    keys = [f"{index:02x}" + "f" * 62 for index in range(10)]
    _write_flat(tmp_path, keys)
    assert cache_statistics(tmp_path).flat_files == 10

    moved = migrate_flat_cache(tmp_path)
    stats = cache_statistics(tmp_path)
    assert moved == 10
    assert stats.flat_files == 0
    assert stats.files == 10  # 数量不变，只是换了位置
    assert stats.shards == 10


def test_migrate_is_idempotent(tmp_path):
    _write_flat(tmp_path, [KEY])
    assert migrate_flat_cache(tmp_path) == 1
    assert migrate_flat_cache(tmp_path) == 0
    assert cache_statistics(tmp_path).files == 1


def test_migrate_deduplicates_when_target_exists(tmp_path):
    """内容由键唯一决定，目标已存在时直接删源文件。"""
    target = sharded_path(tmp_path, KEY)
    target.parent.mkdir(parents=True)
    target.write_text('{"cached": true}', encoding="utf-8")
    (tmp_path / f"{KEY}.json").write_text("{}", encoding="utf-8")

    migrate_flat_cache(tmp_path)
    assert cache_statistics(tmp_path).files == 1
    # 保留的是分片里那份。
    assert target.read_text(encoding="utf-8") == '{"cached": true}'


def test_migrate_handles_missing_directory(tmp_path):
    assert migrate_flat_cache(tmp_path / "absent") == 0


# ---------- 统计 ----------


def test_statistics_counts_both_layouts(tmp_path):
    _write_flat(tmp_path, ["aa" + "0" * 62])
    sharded = sharded_path(tmp_path, "bb" + "1" * 62)
    sharded.parent.mkdir(parents=True)
    sharded.write_text("{}", encoding="utf-8")

    stats = cache_statistics(tmp_path)
    assert stats.files == 2
    assert stats.flat_files == 1
    assert stats.bytes_total > 0


def test_statistics_on_missing_directory(tmp_path):
    stats = cache_statistics(tmp_path / "absent")
    assert stats.files == 0
    assert "0" in format_cache_report(stats)


def test_report_mentions_unmigrated_files(tmp_path):
    _write_flat(tmp_path, [KEY])
    assert "未分片文件" in format_cache_report(cache_statistics(tmp_path))


# ---------- 清理 ----------


def test_prune_by_max_files_keeps_newest(tmp_path):
    keys = [f"{index:02x}" + "0" * 62 for index in range(5)]
    _write_flat(tmp_path, keys)
    migrate_flat_cache(tmp_path)
    # 人为拉开修改时间。
    for offset, key in enumerate(keys):
        path = sharded_path(tmp_path, key)
        stamp = time.time() - (len(keys) - offset) * 100
        import os

        os.utime(path, (stamp, stamp))

    removed = prune_cache(tmp_path, max_files=2)
    assert removed == 3
    remaining = {path.stem for path in iter_cache_files(tmp_path)}
    assert remaining == set(keys[-2:])


def test_prune_by_age(tmp_path):
    import os

    _write_flat(tmp_path, [KEY, "bb" + "0" * 62])
    migrate_flat_cache(tmp_path)
    stale = sharded_path(tmp_path, KEY)
    old = time.time() - 10 * 86400
    os.utime(stale, (old, old))

    removed = prune_cache(tmp_path, max_age_days=5)
    assert removed == 1
    assert not stale.exists()
    assert cache_statistics(tmp_path).files == 1


def test_prune_removes_empty_shards(tmp_path):
    _write_flat(tmp_path, [KEY])
    migrate_flat_cache(tmp_path)
    prune_cache(tmp_path, max_age_days=0)
    assert not (tmp_path / KEY[:SHARD_LENGTH]).exists()


def test_prune_without_limits_removes_nothing(tmp_path):
    _write_flat(tmp_path, [KEY])
    migrate_flat_cache(tmp_path)
    assert prune_cache(tmp_path) == 0
    assert cache_statistics(tmp_path).files == 1


def test_clear_cache_empties_directory(tmp_path):
    _write_flat(tmp_path, [KEY, "bb" + "0" * 62])
    migrate_flat_cache(tmp_path)
    assert clear_cache(tmp_path) == 2
    assert tmp_path.is_dir()
    assert cache_statistics(tmp_path).files == 0
