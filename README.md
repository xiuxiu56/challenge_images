# challenge_images

YOLO26 图像分类训练与本地挑战大图网格验证工具，适配 macOS Apple Silicon。

## 目录

```text
src/challenge_images/     Python 包源码
data/classification/      分类训练数据集
data/challenge/           dynamic、imageselect、multicaptcha 大图
data/archive/             原始压缩包
assets/                   GUI 资源图片
models/pretrained/        预训练权重
models/trained/           训练后权重
models/segmentation/      分割预训练与自定义权重
data/segmentation/        YOLO 多边形 mask 数据模板
runs/                     Ultralytics 训练产物
reports/                  审计、验证和 GUI 结果
tests/                    基础测试
```

## 安装

```bash
python3 -m pip install -r requirements.txt
```

也可以按包方式安装开发环境：

```bash
python3 -m pip install -e .
```

PySide6 是主 GUI；Tkinter 保留为无 PySide6 环境下的兼容备用界面。

训练好的模型按实验名称保存：

```text
models/trained/recaptcha_v2_n1/best.pt
models/trained/recaptcha_v2_n1/last.pt
models/trained/recaptcha_v2_n1/model_meta.json
```

GUI 只显示每个实验的 `best.pt`，显示名称由实验名和模型类型组成。

## 第二轮训练建议

正式训练会按所选模型自动应用独立配置。推荐的 m 模型默认使用：

```text
model=yolo26m-cls.pt
epochs=50
batch=32
imgsz=640
optimizer=AdamW
lr0=0.0005
lrf=0.05
cos_lr=True
warmup_epochs=3.0
momentum=0.9
nbs=64
patience=12
dropout=0.10
weight_decay=0.001
scale=0.15
auto_augment=augmix
erasing=0.02
amp=True
workers=4
cache=False
seed=0
deterministic=True
fraction=1.0
freeze=None
data=data/classification/dataset_cls_m2_640
name=recaptcha_v2_m2_640
```

n/s/m/l/x 的输入尺寸、批次大小、数据目录和运行名称会随模型选择联动，
菜单仍允许在开始训练前手动覆盖。
输入尺寸改为320时，菜单会自动改用 `dataset_cls_m2_320` 和对应运行名；
改回640时使用 `dataset_cls_m2_640`。320/640对照共用其余公共超参。

`translate` / `mixup` / `cutmix` 属于当前 Ultralytics 分类数据管线不使用的检测增强参数，
已从分类训练默认配置中移除。

训练前会同时检查 `runs/classify/` 和 `models/trained/`。运行名已存在时，
自动使用 `_v1`、`_v2`、`_v3` 递增名称，保留原训练记录与权重。

菜单 `15` 会把困难格子导出到 `data/classification/hard_samples_review/`。
用户已经明确确认的内置图块会标记为“已审核通过”；后续新增内容默认处于
“待人工审核”状态。只有审核通过的图块才会加入 m2 数据链接副本，Boat 类暂缓处理。

GUI 默认启用自动类别策略：Crosswalk 保留 224 多视角复核，Hydrant 使用 320，
Car 和 Bus 仅接受 Top-1。关闭“自动类别策略”后可以手动调整所有参数。

## 分类与分割 mask 融合

主菜单 `16` 用于训练自定义 YOLO26 实例分割模型，主菜单 `17` 用于对一张完整挑战图执行分类与分割融合验证。
GUI 的“分割与融合”标签可以独立选择分类权重和分割权重，并从当前离线页或在线页载入完整挑战图片。
融合页可直接开始识别、随机或顺序载入图片、扫描精确重复，
并把融合格子应用到当前在线网页。

融合数据流：

```text
/reload → 挑战类型与目标类别
挑战类型 → dynamic/imageselect/tileselect=3×3，multicaptcha=4×4
首次完整 /payload
├── 分类模型逐格输出 Top-K
└── 分割模型整图输出目标 mask，并计算 mask 对各格子的覆盖
    ↓
分类强证据复核整个 mask 实例
    ↓
同时满足格子覆盖率和 mask 占比后才保留格子
最终按平衡、并集或双证据策略生成融合格子
```

平衡融合默认使用 `0.80` 的实例分类复核阈值、`0.60` 的 mask
实例置信度阈值和 `0.10` 的 mask 格子占比。这会拒绝缺少分类
支持的整个误 mask，同时删掉跨越网格边界的少量泄漏像素。需要更高
召回率时仍可选择“并集（召回优先）”。

平衡融合还包含自适应漏检复核：常规分割完全没有目标时，
使用 `0.05` 只召回候选 mask，再由强分类证据决定是否接受。
同一目标被分成高、低置信两个半边 mask 时，只合并共享原始
覆盖格且有强分类支持的半边。摩托车实例会恢复同一 mask
中少量跨格的车轮和车身边缘，并压制明显更小的孤立背景实例。

分类模型仍保存在 `models/trained/`。分割模型使用独立目录：

```text
models/segmentation/
├── pretrained/            # yolo26m-seg.pt 等预训练权重
└── trained/<实验名>/       # best.pt、last.pt、segmentation_meta.json
```

分割训练数据模板位于 `data/segmentation/recaptcha_seg_v1/`。分类文件夹标签不能直接作为 mask 标签；需要为完整挑战图片准备 YOLO 多边形标签。
预训练分割模型未包含 Crosswalk、Bridge 等类别时，融合模块会明确显示“分割模型未覆盖”，并继续使用分类结果。

## 在线识别验证

GUI 中的“在线挑战采集”区域使用本机 Google Chrome：

1. 点击“开始在线会话”，程序打开本机 Google Chrome 并监听 `/reload`、`/payload`、`/replaceimage`。
2. 在 Chrome 中人工触发图片挑战。
3. 首次 `/payload` 完整图会自动保存并显示在 GUI；`/replaceimage` 请求提供 `ds`，紧随其后的 `/payload` 单格图按 `ds` 回填到指定格子。
4. 开启“在线识别验证（自动模型识别）”后，归档完成会自动运行当前模型。
5. “导入在线样本”保留为本地文件兜底入口。
6. 开启“自动刷新挑战（每3秒）”后，工作线程每3秒点击一次 Chrome 挑战刷新按钮，新图仍会自动保存和显示。
7. “自动点击并监控复选框”会持续检查图形挑战 iframe；挑战关闭且复选框未勾选时，5 秒后重新点击。
8. “每3分钟清理站点数据”会清理当前自动化上下文的站点存储、缓存以及第一方/第三方 Cookie，然后刷新页面。

首次完整 `/payload` 会归档到 `data/online_capture/<挑战类型>/<中文类别>/`，
文件名使用 `m_xxxxx_序号.jpg`。根目录 `records.json` 与采集参考脚本保持同一数组格式，
每条记录完整保留对应 `/reload` 响应数组下标 4 的 `pmeta`。

`replaceimage` 后续 `/payload` 是单格图片，单独归档到
`data/online_capture/replacements/<挑战类型>/<中文类别>/`，并在
`replacements/records.json` 记录请求参数 `ds` 的原始动态图块 ID 和固定 GUI 格子位置。
图片旁 JSON 保留 GUI 所需的类别、哈希、网格和归档来源信息。旧 `records.jsonl` 仅用于历史读取兼容，
新图片不再追加到该文件。

GUI 的“在线图片数据”标签用于只读查看在线归档：点击“刷新统计”后按 SHA-256 统计图片总数、唯一内容、
精确重复组和多余副本，并按完整挑战图/替换单格图、挑战类型、中文类别汇总。选择重复组可以查看完整哈希和组内全部项目相对路径；
该页面不会删除或移动图片。

归档目录：

```text
data/online_capture/
├── dynamic/<中文类别>/m_xxxxx_序号.jpg
├── imageselect/<中文类别>/m_xxxxx_序号.jpg
├── tileselect/<中文类别>/m_xxxxx_序号.jpg
├── multicaptcha/<中文类别>/m_xxxxx_序号.jpg
├── replacements/
│   ├── <挑战类型>/<中文类别>/单格图片
│   └── records.json
└── records.json
```

完整 payload 令牌不会写入业务元数据，只保存 SHA-256 摘要。复选框监控、模型识别、
网页图块点击、自动刷新和站点数据清理分别由 GUI 独立开关控制。

## 启动

```bash
python3 main.py
```

安装项目包后也可以使用：

```bash
python3 -m challenge_images
```

主菜单中的 `11` 启动 GUI 大图验证器。`dynamic` 和 `imageselect` 默认 3×3，`multicaptcha` 默认 4×4；GUI 可手动切换。

## 重复图片

默认只生成报告，不删除文件：

```bash
python3 dedupe_samples.py --root .
```

指定挑战类型后才执行删除：

```bash
python3 dedupe_samples.py --root . --challenge dynamic --delete
```

## 测试

```bash
PYTHONPATH=src python3 -m pytest tests -q
```
