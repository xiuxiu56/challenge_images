# challenge_images

YOLO26 图像分类训练与本地挑战大图网格验证工具，适配 macOS Apple Silicon。

## 目录

```text
src/challenge_images/     Python 包源码
config/                   阈值覆盖配置（thresholds.example.yaml 为模板）
data/classification/      分类训练数据集
data/challenge/           dynamic、imageselect、multicaptcha 大图
data/online_capture/      在线采集归档与解题反馈
data/archive/             原始压缩包
assets/                   GUI 资源图片
models/trained/           训练后权重
models/segmentation/      分割预训练与自定义权重
data/segmentation/        YOLO 多边形 mask 数据与模板
runs/                     Ultralytics 训练产物
reports/                  审计、验证和 GUI 结果
tests/                    测试（203 项）
```

> 预训练权重（`yolo26*.pt`）放在项目根目录，由 Ultralytics 按文件名解析，
> 因此请从项目根目录启动。

## 分辨率约定

实测原生尺寸：

| 对象 | 分辨率 |
|---|---|
| 3×3 dynamic / imageselect 大图 | 300×300 → 每格 100×100 |
| 4×4 multicaptcha 大图 | 450×450 → 每格 112×112 |
| 训练图块 | 88% 为 100×100 |

训练分辨率统一为 **160**（略高于原生，给骨干网络 stride 留余量）。
把 100px 图块上采样到 320/640 不增加任何信息量：实测 @320 相比 @224
多花 2.3 倍时间只换来 top1 +0.3%。

**推理分辨率自动跟随权重的训练分辨率**，从 `model_meta.json` 读取，
因此更换默认值不会让既有模型掉点。

## 数据质量注意事项

原始 `dataset_cls_full_57k` 存在三个问题，使用前请先运行菜单 `19` 重划：

1. **训练集含 18% 精确重复**。6 个类别被 `copy_NNNN_` 前缀人工复制到约 1500 张：
   Mountain 标称 1500 实际只有 30 张唯一图，Other 128，Chimney 275，Stair 383。
2. **验证集 42% 与训练集字节完全相同**（Chimney 达 100%）。基于该验证集得到的
   历史精度数字均被高估，不同实验之间的小幅差异不具可比性。
3. **45 张图被同时标注为多个类别**（Bus/Crosswalk 18 张、Car/Traffic Light 7 张），
   是单标签目录无法表达的复合图块。

菜单 `19` 会合并 train/val 后全局去重再分层划分，杜绝泄漏，并把标签冲突图
导出为多标签种子集。

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

## 训练

正式训练默认使用：

```text
model=yolo26m-cls.pt
imgsz=160          # 贴近 112px 原生图块
batch=64
epochs=50
optimizer=AdamW
lr0=0.0005  lrf=0.05  cos_lr=True  warmup_epochs=3.0
dropout=0.10  weight_decay=0.001  erasing=0.02  scale=0.15
auto_augment=augmix  patience=12  amp=True
data=data/classification/dataset_cls_m2_320
```

训练前会同时检查 `runs/classify/` 和 `models/trained/`，运行名已存在时
自动使用 `_v1`、`_v2` 递增名称，保留原记录与权重。

### best 权重的选择指标

Ultralytics 的 `best.pt` 按 `(top1 + top5) / 2` 选择，在极不均衡数据上
由大类主导（实测 top1 0.9227 而 macro-F1 仅 0.8595）。训练时会并行维护一份
`best_macro_f1.pt`，并逐轮记录 `macro_f1_history.json`（含逐类 P/R/F1）。
长尾类表现以后者为准。

### 多标签训练

reCAPTCHA 图块常同时含多个目标（车停在斑马线上）。softmax 强制概率和为 1，
目标会被主类别压制。菜单 `20` 从单标签目录推导多标签清单（按内容哈希识别
复合图块，零人工标注），菜单 `21` 用 sigmoid + BCE 训练：每类独立打分，
判定退化成一次阈值比较，不再需要多视角裁剪复核与类别对抑制阈值。

## 识别参数：类别 × 挑战类型

同一类别在两种题型下的图像性质相反，参数按二维组合：

| | 3×3 dynamic/imageselect/tileselect | 4×4 multicaptcha |
|---|---|---|
| 图像性质 | 每格是独立照片 | 一张连续照片切成 16 块 |
| 目标形态 | 完整占据整格 | 横跨多格，边缘格只有局部 |
| 策略 | 严格 Top-1（Car 0.85 / Bus 0.80） | 放宽候选到 0.35 并提 top_k |

`Car`/`Bus` 在 3×3 下 `candidate_threshold=1.0`（只认 Top-1），
若在 4×4 沿用会漏掉全部边缘格。调整层使用**上限**语义（取 `min`），
保证只放宽不收紧——基线本就宽松的 Motorcycle/Tractor 不受影响。

## 阈值配置与回归评测

融合阈值集中在 `challenge_images/thresholds.py`（21 个，分 7 组）。
需要调整时复制 `config/thresholds.example.yaml` 为 `config/thresholds.yaml`，
只写要改的键即可。

改完用菜单 `23` 跑回归评测：有真值时输出逐类、逐挑战类型的 P/R/F1 与
完全匹配率；无真值时做 A/B 对照，报告有多少张图结果变化、新增和移除了
哪些格子。**不要只看单张图判断阈值改动是否有效。**

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

### 预标注：把人工从「画」降级为「改」

菜单 `22` 用 COCO 预训练分割模型批量生成候选多边形。实测 229 张 multicaptcha
图耗时 11 秒，产出 755 个实例。注意它会标出图中**所有**可识别类别而非只标
本轮目标——只标目标会让画面里其他类别变成隐式负样本，反而污染训练。

COCO 覆盖 Car / Bus / Bicycle / Motorcycle / Traffic Light / Hydrant 六类；
Bridge、Chimney、Crosswalk、Mountain、Palm、Stair、Tractor 仍需人工标注，
报告中会显式列出。

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

### 解题反馈与自动标注

归档目录名是**本轮挑战的目标类别**，不是图里有什么，因此不能直接当训练标签。
真正可用的信号是解题结果：

> 挑战通过 ⇒ 点击的格子 = 确认正样本，未点击的格子 = 确认负样本

点击验证后会自动判定结果并写入 `data/online_capture/solve_outcomes.json`
（仅读 DOM，不发额外请求）。未通过的记录也会保存，但不参与标注——
无法区分「点错了」还是「漏点了」。

菜单 `24` 把通过的挑战切成带标签的图块。导出采用多标签格式：未点击的格子
标签为空列表，表示「不含本轮目标」而非「属于某个其他类别」——
单标签目录无法表达纯负样本，多标签下它就是全零多热向量。

## 启动

```bash
python3 main.py
```

安装项目包后也可以使用：

```bash
python3 -m challenge_images
```

主菜单中的 `11` 启动 GUI 大图验证器。`dynamic` 和 `imageselect` 默认 3×3，`multicaptcha` 默认 4×4；GUI 可手动切换。

### 新增菜单速查

| 编号 | 功能 |
|---|---|
| 19 | 分层重划 train/val（去重 + 杜绝泄漏 + 每类保底验证样本）|
| 20 | 生成多标签清单（按内容哈希识别复合图块）|
| 21 | 训练多标签分类模型（sigmoid + BCE）|
| 22 | 分割数据预标注（COCO 模型生成候选多边形）|
| 23 | 识别结果回归评测（阈值调优）|
| 24 | 导出在线解题真值标注 |

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
