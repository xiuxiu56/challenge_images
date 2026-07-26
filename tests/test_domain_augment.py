import random

import numpy as np
from PIL import Image

from challenge_images.data.domain_augment import (
    BrightnessContrastJitter,
    DomainAugmentConfig,
    JpegRecompress,
    RandomDownscale,
    SlightBlur,
    build_domain_transforms,
    describe,
    inject_domain_augment,
)


def _tile(size=(112, 112)):
    """构造一张有纹理的图块；纯色图无法体现压缩与模糊的影响。"""
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def _difference(before: Image.Image, after: Image.Image) -> float:
    return float(
        np.abs(
            np.asarray(before, dtype=np.int16) - np.asarray(after, dtype=np.int16)
        ).mean()
    )


# ---------- 单项变换 ----------


def test_jpeg_recompress_changes_pixels_and_keeps_size():
    tile = _tile()
    out = JpegRecompress(1.0, 40, 40)(tile)
    assert out.size == tile.size
    assert _difference(tile, out) > 1.0


def test_downscale_restores_original_size():
    tile = _tile()
    out = RandomDownscale(1.0, 0.5)(tile)
    assert out.size == tile.size
    assert _difference(tile, out) > 1.0


def test_blur_changes_pixels():
    tile = _tile()
    out = SlightBlur(1.0, 0.7)(tile)
    assert out.size == tile.size
    assert _difference(tile, out) > 1.0


def test_jitter_changes_pixels():
    random.seed(1)
    tile = _tile()
    out = BrightnessContrastJitter(1.0, 0.2, 0.2)(tile)
    assert _difference(tile, out) > 0.5


def test_zero_probability_is_identity():
    """概率为 0 时必须原样返回，否则关闭增强也会污染数据。"""
    tile = _tile()
    for transform in (
        JpegRecompress(0.0, 40, 90),
        RandomDownscale(0.0, 0.5),
        SlightBlur(0.0, 0.7),
        BrightnessContrastJitter(0.0, 0.2, 0.2),
    ):
        assert _difference(tile, transform(tile)) == 0.0


def test_downscale_handles_tiny_images():
    """100px 图块缩到 60% 后仍需能还原，不能出现 0 尺寸。"""
    tiny = _tile((4, 4))
    out = RandomDownscale(1.0, 0.1)(tiny)
    assert out.size == (4, 4)


# ---------- 配置与组装 ----------


def test_disabled_config_produces_no_transforms():
    assert build_domain_transforms(DomainAugmentConfig.disabled()) == []
    assert "关闭" in describe(DomainAugmentConfig.disabled())


def test_default_config_includes_all_four():
    names = {type(t).__name__ for t in build_domain_transforms()}
    assert names == {
        "JpegRecompress",
        "RandomDownscale",
        "SlightBlur",
        "BrightnessContrastJitter",
    }


def test_jpeg_is_applied_last():
    """JPEG 模拟「最终交付浏览器」这一步，必须在其他退化之后。"""
    transforms = build_domain_transforms()
    assert type(transforms[-1]).__name__ == "JpegRecompress"


def test_describe_lists_strengths():
    text = describe()
    assert "JPEG重编码" in text
    assert "降采样" in text


# ---------- 注入位置 ----------


class _FakeToTensor:
    def __call__(self, image):
        return image


class _FakeCompose:
    def __init__(self, transforms):
        self.transforms = list(transforms)


def test_injection_happens_before_tensor_conversion():
    """域增强作用于 PIL 图像，插到 ToTensor 之后会直接报错。"""
    compose = _FakeCompose([object(), _FakeToTensor(), object()])
    compose.__class__.__name__ = "Compose"
    _FakeToTensor.__name__ = "ToTensor"

    inject_domain_augment(compose, DomainAugmentConfig())
    names = [type(t).__name__ for t in compose.transforms]
    tensor_index = names.index("ToTensor")
    injected = [i for i, n in enumerate(names) if n.endswith(("Recompress", "Downscale", "Blur", "Jitter"))]
    assert injected
    assert all(index < tensor_index for index in injected)


def test_injection_appends_when_no_tensor_step():
    compose = _FakeCompose([object()])
    inject_domain_augment(compose, DomainAugmentConfig())
    assert len(compose.transforms) == 5


def test_injection_is_noop_when_disabled():
    original = [object(), object()]
    compose = _FakeCompose(original)
    inject_domain_augment(compose, DomainAugmentConfig.disabled())
    assert compose.transforms == original


def test_injection_tolerates_unexpected_object():
    """拿到非 Compose 对象时安全返回，不应中断训练。"""
    sentinel = object()
    assert inject_domain_augment(sentinel, DomainAugmentConfig()) is sentinel


def test_full_pipeline_preserves_image_mode():
    random.seed(3)
    tile = _tile()
    image = tile
    for transform in build_domain_transforms():
        image = transform(image)
    assert image.mode == "RGB"
    assert image.size == tile.size
