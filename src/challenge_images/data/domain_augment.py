"""针对 reCAPTCHA 图块的域相关增强。

默认的 ``auto_augment=augmix`` 是通用 ImageNet 策略，模拟的是自然照片的
光照与几何变化。但本任务的域差异不在这里：

- 图块是从一张 JPEG 大图里切出来的，边缘带有压缩块效应
- 原生只有 100~112 像素，浏览器按 CSS 尺寸缩放后再显示
- 线上图片经过 Google 的二次编码，质量与训练集来源未必一致

因此真正该模拟的是**压缩与分辨率损失**，而不是再来一遍色彩几何变换。

另外注意：Ultralytics 在 ``auto_augment`` 开启时会设置
``disable_color_jitter = not force_color_jitter``，也就是说配置里的
``hsv_h/hsv_s/hsv_v`` 在 augmix 打开时**完全不生效**。本模块自带轻度
亮度对比度抖动，填补这个空缺。

全部变换工作在 PIL 图像上，需要插入到 ``ToTensor`` 之前。
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class DomainAugmentConfig:
    """域增强的强度配置。"""

    # JPEG 重编码质量范围；线上图块普遍在 60~90 之间。
    jpeg_probability: float = 0.5
    jpeg_quality_min: int = 45
    jpeg_quality_max: int = 95
    # 先缩小再放大，模拟分辨率损失。0.6 表示最多缩到 60% 再还原。
    downscale_probability: float = 0.3
    downscale_min: float = 0.6
    # 轻微高斯模糊，模拟浏览器缩放插值。
    blur_probability: float = 0.2
    blur_radius_max: float = 0.7
    # 亮度与对比度抖动，补上 augmix 打开后失效的 hsv 抖动。
    jitter_probability: float = 0.5
    brightness_delta: float = 0.20
    contrast_delta: float = 0.20

    @classmethod
    def disabled(cls) -> "DomainAugmentConfig":
        return cls(
            jpeg_probability=0.0,
            downscale_probability=0.0,
            blur_probability=0.0,
            jitter_probability=0.0,
        )


class JpegRecompress:
    """随机质量重编码，引入真实的 JPEG 块效应。"""

    def __init__(self, probability: float, quality_min: int, quality_max: int) -> None:
        self.probability = float(probability)
        self.quality_min = int(quality_min)
        self.quality_max = int(quality_max)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.probability <= 0 or random.random() >= self.probability:
            return image
        quality = random.randint(self.quality_min, self.quality_max)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomDownscale:
    """缩小再放回原尺寸，模拟分辨率损失。"""

    def __init__(self, probability: float, minimum_scale: float) -> None:
        self.probability = float(probability)
        self.minimum_scale = float(minimum_scale)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.probability <= 0 or random.random() >= self.probability:
            return image
        width, height = image.size
        scale = random.uniform(self.minimum_scale, 1.0)
        small = (max(1, int(width * scale)), max(1, int(height * scale)))
        if small == (width, height):
            return image
        shrunk = image.resize(small, Image.Resampling.BILINEAR)
        return shrunk.resize((width, height), Image.Resampling.BILINEAR)


class SlightBlur:
    """轻微高斯模糊，模拟浏览器缩放插值。"""

    def __init__(self, probability: float, radius_max: float) -> None:
        self.probability = float(probability)
        self.radius_max = float(radius_max)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.probability <= 0 or random.random() >= self.probability:
            return image
        radius = random.uniform(0.1, self.radius_max)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))


class BrightnessContrastJitter:
    """亮度与对比度抖动。

    Ultralytics 在 auto_augment 开启时会禁用自带的色彩抖动，
    这里补回一份强度温和的版本。
    """

    def __init__(self, probability: float, brightness: float, contrast: float) -> None:
        self.probability = float(probability)
        self.brightness = float(brightness)
        self.contrast = float(contrast)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.probability <= 0 or random.random() >= self.probability:
            return image
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)
        return image


def build_domain_transforms(config: DomainAugmentConfig | None = None) -> list[Any]:
    """按配置构造域增强变换列表；强度为 0 的项自动省略。"""
    settings = config or DomainAugmentConfig()
    transforms: list[Any] = []
    if settings.jitter_probability > 0:
        transforms.append(
            BrightnessContrastJitter(
                settings.jitter_probability,
                settings.brightness_delta,
                settings.contrast_delta,
            )
        )
    if settings.downscale_probability > 0:
        transforms.append(RandomDownscale(settings.downscale_probability, settings.downscale_min))
    if settings.blur_probability > 0:
        transforms.append(SlightBlur(settings.blur_probability, settings.blur_radius_max))
    # JPEG 放在最后：它模拟的是「最终交付给浏览器」这一步。
    if settings.jpeg_probability > 0:
        transforms.append(
            JpegRecompress(
                settings.jpeg_probability,
                settings.jpeg_quality_min,
                settings.jpeg_quality_max,
            )
        )
    return transforms


def _tensor_conversion_index(transforms: list[Any]) -> int:
    """找出 ToTensor 的位置；域增强必须插在它之前。"""
    for index, transform in enumerate(transforms):
        name = type(transform).__name__
        if name in {"ToTensor", "PILToTensor", "ToImage", "ToDtype"}:
            return index
    return len(transforms)


def inject_domain_augment(compose: Any, config: DomainAugmentConfig | None = None) -> Any:
    """把域增强插入已有的 torchvision Compose。

    返回同一个 Compose 对象（原地修改其 ``transforms`` 列表），
    以便调用方无需替换数据集上的引用。
    """
    extra = build_domain_transforms(config)
    if not extra:
        return compose
    transforms = getattr(compose, "transforms", None)
    if not isinstance(transforms, list):
        return compose
    position = _tensor_conversion_index(transforms)
    transforms[position:position] = extra
    return compose


def describe(config: DomainAugmentConfig | None = None) -> str:
    """生成可打印的中文增强说明。"""
    settings = config or DomainAugmentConfig()
    if not build_domain_transforms(settings):
        return "域增强: 关闭"
    return (
        "域增强: "
        f"JPEG重编码 p={settings.jpeg_probability:.2f} q={settings.jpeg_quality_min}~{settings.jpeg_quality_max}，"
        f"降采样 p={settings.downscale_probability:.2f} 最低{settings.downscale_min:.0%}，"
        f"模糊 p={settings.blur_probability:.2f} 半径≤{settings.blur_radius_max}，"
        f"亮度对比度抖动 p={settings.jitter_probability:.2f} ±{settings.brightness_delta:.0%}"
    )
