"""本地挑战大图 GUI 验证器。

GUI 使用文件夹名称作为目标类别，使用 SHA-256 跳过重复样本；
成功/失败按钮用于记录整图验证结果，精确格子真值可在后续标注模式补充。
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
try:
    from tkinter import BooleanVar, DoubleVar, StringVar, Tk, filedialog, messagebox, ttk
    TK_AVAILABLE = True
except ModuleNotFoundError:
    # 无 Tkinter 环境下降级：PySide6 才是主 GUI，这里只是兼容备用界面。
    # mypy 无法表达「同一名字既是类型又可为 None」的可选导入模式。
    BooleanVar = DoubleVar = StringVar = Tk = filedialog = messagebox = ttk = None  # type: ignore[assignment,misc]
    TK_AVAILABLE = False

from PIL import Image
if TK_AVAILABLE:
    from PIL import ImageTk
else:
    ImageTk = None  # type: ignore[assignment]

from ..config import ANNOTATIONS_DIR, ASSETS_DIR, CHALLENGE_DIR, DEFAULT_DEVICE, REPORTS_DIR, ROOT, resolve_default_weight
from ..grid.grid_engine import GridSpec, draw_grid, grid_for_challenge, parse_grid
from ..training.model_service import ModelService, TilePrediction
from ..data.sample_manager import SampleManager, scan_duplicates, write_jsonl
from ..annotation_store import AnnotationStore


class ChallengeGUI:
    """Tkinter 主窗口。"""

    def __init__(self, root: Tk, project_root: Path = ROOT) -> None:
        self.root = root
        self.project_root = project_root
        self.root.title("YOLO26 大图网格识别验证器")
        self.root.geometry("1180x820")
        self.root.minsize(900, 650)
        self.service = ModelService()
        self.manager: SampleManager | None = None
        self.current: dict | None = None
        self.current_image: Image.Image | None = None
        self.current_predictions: list[TilePrediction] = []
        self.all_predictions: list[TilePrediction] = []
        self.tk_image: ImageTk.PhotoImage | None = None
        self.busy = False
        self.challenge_var = StringVar(value="dynamic")
        self.grid_var = StringVar(value=grid_for_challenge("dynamic").text)
        default_weight = resolve_default_weight()
        self.weights_var = StringVar(value=str(default_weight) if default_weight else "")
        self.data_var = StringVar(value=str(CHALLENGE_DIR))
        self.device_var = StringVar(value=DEFAULT_DEVICE)
        self.threshold_var = DoubleVar(value=0.25)
        self.target_var = StringVar(value="自动读取文件夹类别")
        self.status_var = StringVar(value="请先加载模型和样本")
        self.detail_var = StringVar(value="")
        self.success_var = StringVar(value="成功 0 / 失败 0 / 总计 0 / 成功率 0.00%")
        self.dedup_var = BooleanVar(value=True)
        self.status_filter_var = StringVar(value="全部")
        self.imgsz_var = StringVar(value="224")
        self.annotation_mode = False
        self.annotation_indices: set[int] = set()
        self.annotations = AnnotationStore(ANNOTATIONS_DIR / "grid_annotations.json")
        self._build()
        self._refresh_manager()

    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="挑战类型").grid(row=0, column=0, sticky="w")
        challenge = ttk.Combobox(top, textvariable=self.challenge_var, values=("dynamic", "imageselect", "multicaptcha"), state="readonly", width=14)
        challenge.grid(row=0, column=1, padx=5)
        challenge.bind("<<ComboboxSelected>>", lambda _event: self._on_challenge_changed())
        ttk.Label(top, text="网格").grid(row=0, column=2, sticky="w")
        ttk.Combobox(top, textvariable=self.grid_var, values=("3×3", "4×4"), state="readonly", width=8).grid(row=0, column=3, padx=5)
        ttk.Label(top, text="置信度").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(top, from_=0.0, to=1.0, increment=0.05, textvariable=self.threshold_var, width=7).grid(row=0, column=5, padx=5)
        ttk.Checkbutton(top, text="精确去重（跳过重复图片）", variable=self.dedup_var, command=self._refresh_manager).grid(row=0, column=6, padx=8)

        ttk.Label(top, text="模型权重").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.weights_var, width=62).grid(row=1, column=1, columnspan=5, sticky="ew", pady=(8, 0))
        ttk.Button(top, text="选择权重", command=self._choose_weights).grid(row=1, column=6, pady=(8, 0))
        ttk.Label(top, text="数据根目录").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.data_var, width=62).grid(row=2, column=1, columnspan=5, sticky="ew", pady=(8, 0))
        ttk.Button(top, text="选择目录", command=self._choose_data).grid(row=2, column=6, pady=(8, 0))
        ttk.Label(top, text="目标类别").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.target_var, width=22).grid(row=3, column=1, padx=5, sticky="w", pady=(8, 0))
        ttk.Label(top, text="multicaptcha 默认 4×4，其他默认 3×3；可手动切换").grid(row=3, column=2, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Label(top, text="输入尺寸（imgsz）").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(top, textvariable=self.imgsz_var, values=("224", "320", "640"), state="readonly", width=8).grid(row=4, column=1, sticky="w", pady=(8, 0))
        ttk.Label(top, text="状态筛选（status）").grid(row=4, column=2, sticky="w", pady=(8, 0))
        status_box = ttk.Combobox(top, textvariable=self.status_filter_var, values=("全部", "未处理", "成功", "失败"), state="readonly", width=10)
        status_box.grid(row=4, column=3, sticky="w", pady=(8, 0))
        status_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_manager())

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=8)
        left = ttk.Frame(body, padding=4)
        right = ttk.Frame(body, padding=8)
        body.add(left, weight=4)
        body.add(right, weight=2)
        self.canvas = ttk.Label(left, text="尚未加载图片", anchor="center")
        self.canvas.bind("<Button-1>", self._canvas_click)
        self.canvas.pack(fill="both", expand=True)
        ttk.Label(right, textvariable=self.detail_var, justify="left", anchor="nw").pack(fill="both", expand=True)

        buttons = ttk.Frame(self.root, padding=8)
        buttons.pack(fill="x")
        for text, command in (("加载模型", self._load_model), ("识别", self._recognize), ("标注模式", self._toggle_annotation), ("保存标注", self._save_annotation), ("成功", lambda: self._mark("success")), ("失败", lambda: self._mark("failed")), ("随机", self._random), ("下一个", self._next), ("扫描重复", self._scan_duplicates)):
            ttk.Button(buttons, text=text, command=command).pack(side="left", padx=4)
        ttk.Label(buttons, textvariable=self.status_var).pack(side="left", padx=16)
        ttk.Label(self.root, textvariable=self.success_var, padding=8).pack(fill="x")

    def _on_challenge_changed(self) -> None:
        self.grid_var.set(grid_for_challenge(self.challenge_var.get()).text)
        self._refresh_manager()

    def _refresh_manager(self) -> None:
        self.manager = SampleManager(self.data_var.get(), self.challenge_var.get(), self.dedup_var.get(), status_filter=self.status_filter_var.get())
        self.status_var.set(f"{self.challenge_var.get()}：可验证 {len(self.manager)} 张（精确重复已跳过={self.dedup_var.get()}）")

    def _choose_weights(self) -> None:
        path = filedialog.askopenfilename(title="选择 YOLO26 权重", filetypes=(("PyTorch 权重", "*.pt"), ("所有文件", "*.*")))
        if path:
            self.weights_var.set(path)

    def _choose_data(self) -> None:
        path = filedialog.askdirectory(title="选择项目根目录")
        if path:
            self.data_var.set(path)
            self._refresh_manager()

    def _load_model(self) -> None:
        try:
            info = self.service.load(self.weights_var.get(), self.device_var.get())
            self.status_var.set(f"模型已加载：{Path(info['weights']).name}，设备：{info['device']}，类别数：{len(info['classes'])}")
        except Exception as exc:
            messagebox.showerror("模型加载失败", str(exc))

    def _load_sample(self, sample: dict | None) -> None:
        if not sample:
            messagebox.showinfo("样本", "当前目录没有可用图片")
            return
        self.current = sample
        self.current_image = Image.open(sample["path"]).convert("RGB")
        self.current_predictions = []
        self.all_predictions = []
        self.annotation_indices = set((self.annotations.get(sample["path"]) or {}).get("真实格子", []))
        self.target_var.set(str(sample["target_class"]))
        self._show(self.current_image, [])
        self.detail_var.set(f"图片：{self._display_path(sample['path'])}\n目标类别：{sample['raw_class']}\nSHA-256：{sample['sha256'][:16]}…\n状态：待识别")

    def _display_path(self, path: str | Path) -> str:
        """界面使用项目相对路径，内部仍保留绝对路径。"""
        source = Path(path)
        try:
            return str(source.resolve().relative_to(self.project_root.resolve()))
        except ValueError:
            return source.name

    def _next(self) -> None:
        if self.manager is None:
            self._refresh_manager()
        self._load_sample(self.manager.next_sample() if self.manager else None)

    def _random(self) -> None:
        if self.manager is None:
            self._refresh_manager()
        self._load_sample(self.manager.random_sample() if self.manager else None)

    def _show(self, image: Image.Image, selected: list[int], low_confidence: list[int] | None = None) -> None:
        spec = parse_grid(self.grid_var.get())
        rendered = draw_grid(image, spec, selected or self.annotation_indices, low_confidence or [], ASSETS_DIR / "image.png")
        rendered.thumbnail((760, 620), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(rendered)
        self.canvas.configure(image=self.tk_image, text="")

    def _recognize(self) -> None:
        if self.current_image is None or self.current is None:
            self._next()
            if self.current_image is None:
                return
        if not self.service.loaded:
            messagebox.showinfo("模型", "请先点击“加载模型”")
            return
        if self.busy:
            return
        self.busy = True
        self.status_var.set("正在批量识别网格，请稍候……")
        image = self.current_image.copy()
        try:
            spec = parse_grid(self.grid_var.get())
        except ValueError as exc:
            self.busy = False
            messagebox.showerror("网格错误", str(exc))
            return
        target = self.target_var.get()
        if target == "自动读取文件夹类别":
            target = self.current["target_class"]
        threading.Thread(target=self._recognize_worker, args=(image, spec, target), daemon=True).start()

    def _recognize_worker(self, image: Image.Image, spec: GridSpec, target: str) -> None:
        try:
            all_predictions = self.service.predict_grid(image, spec, 0.0, None, int(self.imgsz_var.get()), selected_only=False, image_key=self.current["sha256"] if self.current else None)
            predictions = self.service.select_target(all_predictions, float(self.threshold_var.get()), target, 3)
            self.root.after(0, lambda: self._recognize_done(predictions, all_predictions))
        except Exception as exc:
            self.root.after(0, lambda error=exc: self._recognize_error(error))

    def _recognize_error(self, exc: Exception) -> None:
        self.busy = False
        messagebox.showerror("识别失败", str(exc))
        self.status_var.set("识别失败")

    def _recognize_done(self, predictions: list[TilePrediction], all_predictions: list[TilePrediction] | None = None) -> None:
        self.busy = False
        self.current_predictions = predictions
        self.all_predictions = all_predictions or predictions
        indices = [item.index for item in predictions]
        low = [item.index for item in predictions if item.confidence < 0.7]
        if self.current_image:
            self._show(self.current_image, indices, low)
        details = [f"识别到的格子：{indices}", f"识别类别：{self.target_var.get()}", "全部格子预测："]
        details.extend(f"  格子 {item.index}: {item.label}（{item.zh}）置信度={item.confidence:.4f}" for item in self.all_predictions)
        self.detail_var.set("\n".join(details))
        self.status_var.set(f"识别完成：{indices}")

    def _mark(self, status: str) -> None:
        if not self.current:
            messagebox.showinfo("样本", "请先加载并识别一张图片")
            return
        report = REPORTS_DIR / "gui_results.jsonl"
        if any(item.get("path") == str(self.current["path"]) and item.get("status") == status for item in self._read_results()):
            self.status_var.set("该样本已经记录过相同状态")
            return
        write_jsonl(report, {"time": datetime.now().isoformat(timespec="seconds"), "status": status, "challenge_type": self.challenge_var.get(), "grid": self.grid_var.get(), "path": str(self.current["path"]), "sha256": self.current["sha256"], "target_class": self.current["target_class"], "threshold": float(self.threshold_var.get()), "imgsz": int(self.imgsz_var.get()), "predicted_indices": [item.index for item in self.current_predictions], "all_predictions": [item.__dict__ for item in self.all_predictions], "真实格子": sorted(self.annotation_indices)})
        self._update_counts()
        self.status_var.set(f"已记录：{'成功' if status == 'success' else '失败'}")

    def _update_counts(self) -> None:
        path = REPORTS_DIR / "gui_results.jsonl"
        success = failed = 0
        if path.is_file():
            import json
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                    if item.get("status") == "success":
                        success += 1
                    elif item.get("status") == "failed":
                        failed += 1
                except json.JSONDecodeError:
                    continue
        total = success + failed
        rate = success / total * 100 if total else 0.0
        self.success_var.set(f"成功 {success} / 失败 {failed} / 总计 {total} / 成功率 {rate:.2f}%")

    def _read_results(self) -> list[dict]:
        import json
        path = REPORTS_DIR / "gui_results.jsonl"
        if not path.is_file():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return rows

    def _toggle_annotation(self) -> None:
        self.annotation_mode = not self.annotation_mode
        self.status_var.set("标注模式已开启：点击格子切换真实目标" if self.annotation_mode else "标注模式已关闭")

    def _canvas_click(self, event) -> None:
        if not self.annotation_mode or self.current_image is None:
            return
        spec = parse_grid(self.grid_var.get())
        # Label 控件可能缩放图片，使用显示区域比例映射到原图坐标。
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        col = min(spec.columns - 1, max(0, int(event.x / max(width, 1) * spec.columns)))
        row = min(spec.rows - 1, max(0, int(event.y / max(height, 1) * spec.rows)))
        index = row * spec.columns + col
        if index in self.annotation_indices:
            self.annotation_indices.remove(index)
        else:
            self.annotation_indices.add(index)
        self._show(self.current_image, list(self.annotation_indices))

    def _save_annotation(self) -> None:
        if not self.current:
            return
        self.annotations.set(self.current["path"], challenge_type=self.challenge_var.get(), grid=self.grid_var.get(), target_class=self.current["target_class"], indices=list(self.annotation_indices))
        self.status_var.set(f"已保存真实格子：{sorted(self.annotation_indices)}")

    def _scan_duplicates(self) -> None:
        groups = scan_duplicates(self.data_var.get())
        duplicate_count = sum(len(paths) - 1 for paths in groups.values())
        report = REPORTS_DIR / "duplicate_samples_gui.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        import json
        report.write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
        messagebox.showinfo("重复扫描", f"发现 {len(groups)} 个重复组，重复文件 {duplicate_count} 个。\n已生成报告：{report}\nGUI 默认只跳过重复，不直接删除。")


def launch_gui(project_root: str | Path = ROOT) -> None:
    """PySide6 主界面；Tkinter 仅在显式调用兼容入口时使用。"""
    from .qt_gui import launch_qt_gui
    launch_qt_gui(project_root)


def launch_tk_gui(project_root: str | Path = ROOT) -> None:
    """Tkinter 兼容界面入口。"""
    if not TK_AVAILABLE:
        raise RuntimeError("当前 Python 没有 Tk 组件，请使用 PySide6 主界面。")
    root = Tk()
    ChallengeGUI(root, Path(project_root))
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
