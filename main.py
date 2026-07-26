"""
challenge_images 入口菜单。
训练 / 验证 / 预测逻辑在其它模块；这里只做交互分发。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 保证从任意 cwd 启动都能 import 同目录模块
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from challenge_images.category_map import dump_mapping_table
from challenge_images.config import (
    ASSETS_DIR,
    DATA_DIR,
    NATIVE_TILE_PIXELS,
    REPORTS_DIR,
    DEFAULT_DEVICE,
    DEFAULT_SEGMENTATION_IMGSZ,
    DEFAULT_TRAIN,
    DEFAULT_TRAIN_IMGSZ,
    STRATIFIED_DATA_DIR,
    MODEL_CHOICES,
    RECOMMENDED_MODEL,
    EXPERIMENT_PRESETS,
    device_status,
    pick_device,
    next_available_run_name,
    resolve_default_weight,
    training_profile_for_model,
    DEFAULT_SEGMENTATION_TRAIN,
    RECOMMENDED_SEGMENTATION_MODEL,
    SEGMENTATION_MODEL_CHOICES,
    resolve_default_segmentation_weight,
)
from challenge_images.data.dataset_info import summarize
from challenge_images.training.train_cls import train_cls
from challenge_images.training.train_seg import train_seg
from challenge_images.training.val_cls import evaluate_directory, predict_cls, val_cls
from challenge_images.training.experiment import compare_runs, save_compare_report
from challenge_images.data.manifest import build_default_manifests
from challenge_images.data.dataset_audit import audit_dataset, format_audit_report, save_audit_report
from challenge_images.data.dataset_prepare import prepare_dataset
from challenge_images.data.stratified_split import build_stratified_dataset, format_split_report
from challenge_images.data.multilabel import (
    MANIFEST_FILENAME,
    build_manifest_from_folders,
    format_manifest_report,
)
from challenge_images.training.train_multilabel import train_multilabel
from challenge_images.data.hard_samples import build_m2_dataset, export_default_hard_samples
from challenge_images.runtime_env import print_status
from challenge_images.grid.grid_engine import draw_grid, grid_for_challenge
from challenge_images.segmentation.model_service import SegmentationModelService
from challenge_images.segmentation.result_fusion import format_fusion_report, fuse_predictions
from challenge_images.training.model_service import ModelService


def _input(prompt: str, default: str | None = None) -> str:
    if default is not None:
        raw = input(f"{prompt} [{default}]: ").strip()
        return raw if raw else default
    return input(f"{prompt}: ").strip()


def _pick_model() -> str:
    print("\n可选 YOLO26 分类模型:")
    for k, (name, desc) in MODEL_CHOICES.items():
        mark = " ← 推荐" if name == RECOMMENDED_MODEL else ""
        print(f"  {k}. {name:<18} {desc}{mark}")
    choice = _input("选择编号或输入权重路径（weights）", "3")
    if choice in MODEL_CHOICES:
        return MODEL_CHOICES[choice][0]
    return choice  # 自定义路径 / 文件名


def _pick_segmentation_model() -> str:
    print("\n可选 YOLO26 分割模型:")
    for key, (name, description) in SEGMENTATION_MODEL_CHOICES.items():
        mark = " ← 推荐" if name == RECOMMENDED_SEGMENTATION_MODEL else ""
        print(f"  {key}. {name:<18} {description}{mark}")
    choice = _input("选择编号或输入分割权重路径（weights）", "3")
    if choice in SEGMENTATION_MODEL_CHOICES:
        return SEGMENTATION_MODEL_CHOICES[choice][0]
    return choice


def menu_dataset_info() -> None:
    print("\n" + summarize())


def menu_environment() -> None:
    print_status()


def menu_audit() -> None:
    report = audit_dataset(DATA_DIR)
    print("\n" + format_audit_report(report))
    output = REPORTS_DIR / "dataset_audit.json"
    save_audit_report(output, DATA_DIR)
    print(f"审计 JSON 已保存：{output}")


def menu_prepare() -> None:
    output = _input("输出目录（output_dir）", str(DATA_DIR.parent / "dataset_cls_balanced"))
    balance = _input("训练集平衡方式（balance：none/sqrt）", "sqrt").lower()
    links = _input("优先使用符号链接（y/n）", "y").lower() in ("y", "yes", "1")
    result = prepare_dataset(output_dir=output, balance=balance, use_links=links)
    print(f"数据副本已生成：{result}")
    print("原始数据目录保持不变。")


def menu_stratified_split() -> None:
    """按类别分层重划 train/val，让每个类别都有足够的验证样本。"""
    source = Path(_input("源数据目录（source）", str(DATA_DIR)))
    if not source.is_dir():
        print(f"源数据目录不存在：{source}")
        return
    output = Path(_input("输出目录（output）", str(STRATIFIED_DATA_DIR)))
    try:
        val_ratio = float(_input("验证比例（val_ratio）", "0.15"))
        val_min = int(_input("每类验证下限（val_min）", "50"))
        val_max = int(_input("每类验证上限（val_max）", "300"))
    except ValueError:
        print("验证比例与上下限必须是数字，本次操作已取消。")
        return
    overwrite = _input("输出目录已存在时覆盖（y/n）", "n").lower() in ("y", "yes", "1")
    print("\n正在扫描并按内容哈希去重，大数据集需要几分钟……")
    try:
        report = build_stratified_dataset(
            source,
            output,
            val_ratio=val_ratio,
            val_min=val_min,
            val_max=val_max,
            overwrite=overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"[错误] {error}")
        return
    print("\n" + format_split_report(report))
    smallest_name, smallest_count = report.smallest_val_class
    print(f"\n分层报告已保存：{output / 'split_report.json'}")
    if smallest_count < 20:
        print(
            f"提示：{smallest_name} 仅有 {smallest_count} 张验证样本，"
            "该类指标仍然噪声较大，需要补充真实数据而不是继续调参。"
        )
    print("原始数据目录保持不变，输出全部为符号链接。")


def menu_build_multilabel_manifest() -> None:
    """从单标签目录推导多标签清单：同一张图出现在多个类别即为复合图块。"""
    root = Path(_input("数据集目录（root）", str(STRATIFIED_DATA_DIR)))
    if not root.is_dir():
        print(f"数据集目录不存在：{root}")
        return
    print("\n正在按内容哈希比对各类别目录……")
    manifest = build_manifest_from_folders(root)
    output = manifest.save(root / MANIFEST_FILENAME)
    print("\n" + format_manifest_report(manifest))
    print(f"\n多标签清单已保存：{output}")
    if not manifest.overrides:
        print("当前数据没有检测到复合图块；多标签训练会退化为单标签。")


def menu_train_multilabel() -> None:
    """训练多标签分类模型（sigmoid + BCE）。"""
    model = _pick_model()
    data = Path(_input("训练数据目录（data）", str(STRATIFIED_DATA_DIR)))
    if not data.is_dir():
        print(f"数据目录不存在：{data}")
        return
    if not (data / MANIFEST_FILENAME).is_file():
        print(f"未找到 {MANIFEST_FILENAME}；请先运行菜单 20 生成多标签清单。")
        if _input("仍然按单标签继续（y/n）", "n").lower() not in ("y", "yes", "1"):
            return
    try:
        epochs = int(_input("训练轮数（epochs）", str(DEFAULT_TRAIN["epochs"])))
        batch = int(_input("批次大小（batch）", str(DEFAULT_TRAIN["batch"])))
        imgsz = int(_input("输入尺寸（imgsz）", str(DEFAULT_TRAIN_IMGSZ)))
    except ValueError:
        print("训练轮数、批次大小和输入尺寸必须填写整数，本次训练已取消。")
        return
    device = pick_device(_input("训练设备（device：mps/cpu）", DEFAULT_DEVICE))
    name = _input("运行名称（name）", f"recaptcha_multilabel_{imgsz}")
    train_multilabel(
        model=model, data=data, epochs=epochs, batch=batch,
        imgsz=imgsz, device=device, name=name,
    )


def menu_train(smoke: bool = False) -> None:
    model = _pick_model() if not smoke else "yolo26n-cls.pt"
    profile = training_profile_for_model(model)
    if smoke:
        print(f"\n冒烟训练固定用 {model}")
        print("建议：保持默认参数，连续按回车即可开始 3 轮管线测试。")
    else:
        print(
            f"\n模型默认训练参数（可修改）：模型={model}，"
            f"训练轮数（epochs）={profile['epochs']}，"
            f"批次大小（batch）={profile['batch']}，"
            f"输入尺寸（imgsz）={profile['imgsz']}，"
            f"设备（device）={DEFAULT_DEVICE}。"
        )

    try:
        epochs = int(_input("训练轮数（epochs）", "3" if smoke else str(profile["epochs"])))
        batch = int(_input("批次大小（batch）", "32" if smoke else str(profile["batch"])))
        imgsz = int(_input("输入尺寸（imgsz）", "128" if smoke else str(profile["imgsz"])))
    except ValueError:
        print("训练轮数、批次大小和输入尺寸必须填写整数，本次训练已取消。")
        return
    # 数据目录不再随 imgsz 变化；运行名带上分辨率便于对照。
    selected_profile = profile if smoke else training_profile_for_model(model, imgsz=imgsz)
    recommended_data = DATA_DIR if smoke else Path(str(selected_profile["data"]))
    if not smoke:
        print(f"\n已根据输入尺寸 imgsz={imgsz} 生成本次训练配置：")
        print(f"  默认数据目录：{recommended_data}")
        print(f"  基础运行名称：{selected_profile['name']}")
        if imgsz > 2 * NATIVE_TILE_PIXELS:
            print(
                f"  提示：图块原生尺寸约 {NATIVE_TILE_PIXELS}px，"
                f"imgsz={imgsz} 属于大幅上采样，只会增加耗时而不增加信息量。"
            )
    data = Path(_input("训练数据目录（data）", str(recommended_data)))
    if not data.is_dir():
        print(f"数据目录不存在：{data}")
        return
    print(f"\n设备检测: {device_status()}")
    device = pick_device(_input("训练设备（device：mps/cpu）", DEFAULT_DEVICE))
    base_name = "smoke" if smoke else str(selected_profile["name"])
    suggested_name = next_available_run_name(base_name)
    if suggested_name != base_name:
        print(f"已发现同名实验，本次建议保存为：{suggested_name}")
    requested_name = _input("运行名称（name）", suggested_name)
    name = next_available_run_name(requested_name)
    if name != requested_name:
        print(f"运行名称已存在，已自动改为：{name}")

    # MPS 上 amp 默认可开；出 nan/异常再关
    amp_raw = _input("混合精度（amp：y/n）", "y").lower()
    amp = amp_raw in ("y", "yes", "1", "true")

    print("\n======== 最终训练配置 ========")
    print(f"模型（model）: {model}")
    print(f"数据（data）: {data}")
    print(f"运行名称（name）: {name}")
    print(f"训练轮数/批次大小/输入尺寸: {epochs}/{batch}/{imgsz}")
    print(
        f"优化器={DEFAULT_TRAIN['optimizer']}，"
        f"初始学习率={DEFAULT_TRAIN['lr0']}，"
        f"最终学习率比例={DEFAULT_TRAIN['lrf']}，"
        f"预热轮数={DEFAULT_TRAIN['warmup_epochs']}"
    )
    print(
        f"裁剪缩放={DEFAULT_TRAIN['scale']}，"
        f"自动增强={DEFAULT_TRAIN['auto_augment']}，"
        f"随机擦除={DEFAULT_TRAIN['erasing']}，"
        f"丢弃率={DEFAULT_TRAIN['dropout']}"
    )
    print(
        f"名义批次={DEFAULT_TRAIN['nbs']}，"
        f"确定性={DEFAULT_TRAIN['deterministic']}，"
        f"缓存={DEFAULT_TRAIN['cache']}，"
        f"混合精度={amp}"
    )

    train_cls(
        model=model,
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        name=name,
        smoke=smoke,
        amp=amp,
    )


def menu_val() -> None:
    default_w = resolve_default_weight("exp")
    weights = _input("权重路径（weights）", str(default_w) if default_w and default_w.is_file() else "")
    if not weights:
        print("未提供权重，取消。")
        return
    data = Path(_input("验证数据目录（data，可填写外部分类数据集）", str(DATA_DIR)))
    if not data.is_dir():
        print(f"验证数据目录不存在：{data}")
        return
    print(f"设备检测: {device_status()}")
    device = pick_device(_input("验证设备（device：mps/cpu）", DEFAULT_DEVICE))
    val_cls(weights=weights, data=data, device=device)
    if _input("是否生成逐类外部报告（y/n）", "y").lower() in ("y", "yes", "1"):
        report_data = Path(_input("逐类报告目录（data，类别/图片结构）", str(data)))
        print("提示：如果目录包含 val/，逐类报告将自动只评估 val；隐藏文件和非图片文件会被跳过。")
        evaluate_directory(
            weights=weights,
            data=report_data,
            imgsz=int(_input("外部报告输入尺寸（imgsz）", str(DEFAULT_TRAIN["imgsz"]))),
            device=device,
            report_path=REPORTS_DIR / "external_validation.json",
        )


def menu_predict() -> None:
    default_w = resolve_default_weight("exp")
    weights = _input("权重路径（weights）", str(default_w) if default_w and default_w.is_file() else "")
    source = _input("图片路径或目录（source）", "")
    if not weights or not source:
        print("权重或路径为空，取消。")
        return
    print(f"设备检测: {device_status()}")
    device = pick_device(_input("预测设备（device：mps/cpu）", DEFAULT_DEVICE))
    predict_cls(source=source, weights=weights, device=device)


def menu_mapping() -> None:
    print("\n" + dump_mapping_table())


def menu_manifest() -> None:
    for output in build_default_manifests():
        print(f"数据清单已生成：{output}")


def menu_compare_runs() -> None:
    rows = compare_runs()
    if not rows:
        print("暂无可比较的训练结果。")
        return
    print("\n======== 历史实验对比 ========")
    for row in rows:
        print(f"实验={row.get('实验')}，轮数={row.get('轮数')}，记录字段={len(row)}")
    print(f"实验对比报告已保存：{save_compare_report()}")


def menu_experiment_preset() -> None:
    names = list(EXPERIMENT_PRESETS)
    for index, name in enumerate(names, 1):
        print(f"  {index}. {name} → {EXPERIMENT_PRESETS[name]}")
    selected = int(_input("选择实验预设", "1")) - 1
    if selected < 0 or selected >= len(names):
        print("实验预设编号无效。")
        return
    cfg = EXPERIMENT_PRESETS[names[selected]]
    train_cls(device=DEFAULT_DEVICE, epochs=DEFAULT_TRAIN["epochs"], **cfg)


def menu_export_hard_samples() -> None:
    """导出已确认的困难图块，等待人工审核后再合并训练。"""
    result = export_default_hard_samples()
    print(f"困难样本已导出：{result['输出目录']}")
    print(f"已导出图块：{result['导出数量']} 张")
    print(f"暂缓项目：{result['暂缓数量']} 项（Boat 暂时只写清单）")
    print(f"审核清单：{result['清单路径']}")
    print("当前内置的 5 个非 Boat 图块已按你提供的标签标记为“已审核通过”。")
    if _input("现在生成 m2 训练数据链接副本（y/n）", "n").lower() in ("y", "yes", "1"):
        built = build_m2_dataset()
        print(f"m2 数据副本已生成：{built['输出目录']}")
        print(f"基础链接：{built['基础链接数量']}，困难样本：{built['审核通过并加入数量']}")
    else:
        print("尚未生成 m2 数据副本；完成审核后可再次进入此菜单生成。")


def menu_train_segmentation() -> None:
    """训练自定义 YOLO26 实例分割模型。"""
    model = _pick_segmentation_model()
    defaults = DEFAULT_SEGMENTATION_TRAIN
    data = Path(_input("分割数据配置（data.yaml）", str(defaults["data"])))
    if not data.is_file():
        print(f"分割数据配置不存在：{data}")
        print("请先准备 images/train、images/val、labels/train、labels/val 和 data.yaml。")
        return
    try:
        epochs = int(_input("训练轮数（epochs）", str(defaults["epochs"])))
        batch = int(_input("批次大小（batch）", str(defaults["batch"])))
        imgsz = int(_input("输入尺寸（imgsz）", str(defaults["imgsz"])))
    except ValueError:
        print("训练轮数、批次大小和输入尺寸必须填写整数，本次训练已取消。")
        return
    device = pick_device(_input("训练设备（device：mps/cpu）", DEFAULT_DEVICE))
    name = _input("运行名称（name）", str(defaults["name"]))
    train_seg(
        model=model,
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        name=name,
    )


def menu_segmentation_fusion() -> None:
    """对一张完整挑战图执行分类、分割和格子融合验证。"""
    default_cls = resolve_default_weight()
    classification_weights = _input(
        "分类权重路径（classification weights）",
        str(default_cls) if default_cls and default_cls.is_file() else "",
    )
    default_seg = resolve_default_segmentation_weight()
    segmentation_weights = _input(
        "分割权重路径（segmentation weights）",
        str(default_seg) if default_seg else RECOMMENDED_SEGMENTATION_MODEL,
    )
    source = Path(_input("完整挑战图片路径（source）", ""))
    if not classification_weights or not segmentation_weights or not source.is_file():
        print("分类权重、分割权重或完整挑战图片路径缺失，本次验证已取消。")
        return
    challenge_type = _input(
        "挑战类型（dynamic/imageselect/tileselect/multicaptcha）",
        "dynamic",
    ).lower()
    target_class = _input("目标类别（target）", "Car")
    mode = _input("融合策略（balanced/union/consensus）", "balanced").lower()
    device = pick_device(_input("推理设备（device：mps/cpu）", DEFAULT_DEVICE))
    spec = grid_for_challenge(challenge_type)

    from PIL import Image

    image = Image.open(source).convert("RGB")
    classification = ModelService(REPORTS_DIR / "fusion_classification_cache")
    segmentation = SegmentationModelService()
    classification.load(classification_weights, device)
    segmentation.load(segmentation_weights, device)
    # 分类分辨率跟随权重的训练分辨率，缺少元数据时回退全局默认。
    all_classification = classification.predict_grid(
        image,
        spec,
        threshold=0.0,
        target_class=None,
        imgsz=classification.training_imgsz or DEFAULT_TRAIN_IMGSZ,
        top_k=3,
        selected_only=False,
    )
    selected_classification = classification.select_target(
        all_classification,
        threshold=0.25,
        target_class=target_class,
        top_k=3,
        top1_threshold=0.80,
    )
    segmentation_result = segmentation.predict(
        image,
        spec,
        target_class,
        imgsz=DEFAULT_SEGMENTATION_IMGSZ,
        confidence=0.25,
    )
    fusion = fuse_predictions(
        all_classification,
        selected_classification,
        segmentation_result,
        target_class=target_class,
        grid_count=spec.count,
        mode=mode,
    )
    report = format_fusion_report(
        fusion,
        segmentation_result,
        target_class=target_class,
        spec=spec,
    )
    print("\n======== 分类 + 分割融合结果 ========")
    print(f"图片: {source}")
    print(f"挑战类型: {challenge_type}")
    print(report)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "segmentation_fusion_report.txt"
    report_path.write_text(
        f"图片: {source}\n挑战类型: {challenge_type}\n{report}",
        encoding="utf-8",
    )
    preview = draw_grid(
        segmentation_result.preview,
        spec,
        fusion.selected_indices,
        icon_path=ASSETS_DIR / "image.png",
    )
    preview_path = REPORTS_DIR / "segmentation_fusion_preview.png"
    preview.save(preview_path)
    print(f"融合报告已保存：{report_path}")
    print(f"mask 预览已保存：{preview_path}")


def menu_gui() -> None:
    from challenge_images.gui.gui_app import launch_gui
    try:
        launch_gui()
    except RuntimeError as exc:
        print(f"[GUI 环境提示] {exc}")


def menu_advice() -> None:
    print(
        """
========== 模型选择建议（本机 M4 Pro 24GB + 本数据集）==========

数据概况:
  - 约 5.6 万 train / 1.4k val / 14 类
  - 极不均衡: Tractor 仅 23 张；Bus/Bicycle/Hydrant 等 7k+
  - val 里 Mountain/Tractor 各只有 3 张 → 这两类指标噪声极大

模型（YOLO26 分类）:
  1. yolo26m-cls.pt   ★ 主推
     当前数据规模下优先比较精度；正式训练默认 batch=32。
  2. yolo26s-cls.pt   速度基线和部署对照
  3. yolo26n-cls.pt   冒烟 / 快速试超参（3~10 epoch）
  4. yolo26l/x-cls    24GB 一般能跑，但慢；收益未必值得

超参建议（分类）:
  - 输入尺寸（imgsz）=160
    实测原生尺寸：3×3 大图 300×300→每格 100×100，4×4 大图 450×450→每格 112×112。
    训练分辨率只需略高于原生尺寸；@320 相比 @224 多花 2.3 倍时间只换来 top1 +0.3%。
  - 批次大小（batch）=64（160 分辨率下显存充裕，可比 320 时期放大一倍）
  - 训练设备（device）=mps  ← Mac Apple Silicon 优先使用
  - 训练轮数（epochs）=50 + 早停轮数（patience）=12
  - 优化器（optimizer）=AdamW，初始学习率（lr0）=0.0005，余弦学习率（cos_lr）=True
  - warmup_epochs=3.0，momentum=0.9，nbs=64，lrf=0.05
  - dropout=0.10，weight_decay=0.001，erasing=0.02，scale=0.15
  - auto_augment=augmix，deterministic=True，cache=False
  - Boat 暂缓；船类样本不会加入当前 14 类训练集

MPS 注意:
  - 本机已自动检测 MPS；菜单里默认就是 mps
  - 没有 NVIDIA，不要填 cuda / 0（0 在 ultralytics 里常被当成 CUDA）
  - 统一内存 24GB：160 分辨率下 m + batch64 压力很小；OOM 先降 batch，再降 imgsz
  - amp 若 loss 变 nan / 报错 → 训练时改 amp=n
  - workers 保持 4 左右即可，macOS 多进程过大反而慢

训练策略:
  1) 菜单「6 冒烟训练」确认 MPS 管线
  2) 菜单「7 正式训练」用 yolo26m-cls 跑完整程
  3) 看 runs/classify/exp 下的 results / confusion matrix
  4) 重点盯 rare 类（Tractor/Chimney）是否几乎全错
  5) 若 s 整体 top1 已高但难类仍差 → 再试 m，或补数据/合并类

不建议一上来就 x 模型：时间成本高，且稀有类瓶颈在数据不在容量。
===============================================================
"""
    )


def main() -> None:
    actions = {
        "1": ("运行环境与 MPS 检查", menu_environment),
        "2": ("数据集统计 / 类别分布", menu_dataset_info),
        "3": ("查看类别映射表", menu_mapping),
        "4": ("数据质量审计（只读）", menu_audit),
        "5": ("生成训练副本 / 长尾平衡", menu_prepare),
        "6": ("冒烟训练（n 模型 / 3 epoch）", lambda: menu_train(smoke=True)),
        "7": ("正式训练（默认 m 模型）", lambda: menu_train(smoke=False)),
        "8": ("验证模型", menu_val),
        "9": ("单图/目录预测", menu_predict),
        "10": ("模型选择建议（说明）", menu_advice),
        "11": ("GUI 大图网格识别验证", menu_gui),
        "12": ("生成数据清单", menu_manifest),
        "13": ("比较历史训练实验", menu_compare_runs),
        "14": ("运行模型对比预设", menu_experiment_preset),
        "15": ("导出困难格子训练素材（待审核）", menu_export_hard_samples),
        "16": ("训练 YOLO26 实例分割模型", menu_train_segmentation),
        "17": ("分类 + 分割 mask 融合验证", menu_segmentation_fusion),
        "19": ("分层重划 train/val（每类保底验证样本）", menu_stratified_split),
        "20": ("生成多标签清单（识别复合图块）", menu_build_multilabel_manifest),
        "21": ("训练多标签分类模型（sigmoid+BCE）", menu_train_multilabel),
        "18": ("退出", None),
    }
    menu_groups = (
        ("环境与数据", ("1", "2", "3", "4", "5", "19", "20")),
        ("分类模型训练与验证", ("6", "7", "21", "8", "9", "10")),
        ("GUI 与实验管理", ("11", "12", "13", "14", "15")),
        ("分割模型与融合", ("16", "17")),
        ("其他", ("18",)),
    )

    while True:
        print("\n======== reCAPTCHA 分类与分割训练菜单 ========")
        print(f"数据: {DATA_DIR}")
        print(f"设备: {device_status()}")
        print(f"推荐模型: {RECOMMENDED_MODEL}")
        for group_name, keys in menu_groups:
            print(f"\n-------- {group_name} --------")
            for key in keys:
                title, _ = actions[key]
                print(f"  {key}. {title}")
        print(
            """
-------- 建议操作 --------
首次使用：1（检查 MPS）→ 2（确认数据）→ 6（冒烟训练）
正式训练：选择 7，默认使用 yolo26m-cls.pt + imgsz=160（贴近 112px 原生图块）
模型对比：选择 14，可运行 m@128、m@160、m@224、s@160
训练结果：runs/classify/<运行名称>/weights/best.pt
GUI 模型：训练结束后同步到 models/trained/<运行名称>/best.pt
困难样本：选择 15，导出已确认格子；Boat 只记录清单
分割训练：选择 16，使用 YOLO 多边形标签训练 yolo26m-seg.pt
融合验证：选择 17，按接口挑战类型确定3×3/4×4并融合分类 Top-K 与 mask
快速开始正式训练：直接输入 7
--------------------------"""
        )
        choice = _input("请选择", "1")
        if choice == "18":
            print("再见。")
            break
        item = actions.get(choice)
        if not item:
            print("无效选项。")
            continue
        _, fn = item
        try:
            fn()
        except FileNotFoundError as e:
            print(f"[错误] {e}")
        except KeyboardInterrupt:
            print("\n已中断当前操作，回到菜单。")
        except Exception as e:
            print(f"[异常] {type(e).__name__}: {e}")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n" + "=" * 100)
        print("\n【 ⛔ 】 用户停止了程序")
        print("=" * 100 + "\n")
