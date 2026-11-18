"""
EAIW 1.0 Main Desktop Workstation Window (PySide6).
Implements the 2008-2015 Pro Desktop Layout specified in Section VI:
- Left Panel: Toolbox, Historical Era Filter & 6 Laboratories Model Library
- Center Panel: Interactive Mask & Image Canvas + Original/Result Toggle + Multi-Model Comparison
- Right Panel: Native Model Parameter Console + 7-Axis Failure Simulation Sliders + Execution Mode Selector
- Bottom Dock: Task Queue, Iteration Timeline Strip, and SQLite AI Artifact Database Inspector
"""
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np

from core.engine import EAIWEngine
from core.image_io import ImageIO
from core.metadata import FAILURE_CATEGORIES

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QListWidget, QListWidgetItem, QGroupBox, QLabel, QPushButton,
        QComboBox, QSlider, QSpinBox, QDoubleSpinBox, QLineEdit,
        QTableWidget, QTableWidgetItem, QFileDialog, QSplitter, QTabWidget,
        QMessageBox
    )
    from PySide6.QtCore import Qt
    from ui.canvas_widget import MaskCanvasWidget
    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False


def launch_desktop_gui(base_dir: Path):
    if not HAS_PYSIDE6:
        print("PySide6 is not installed in this environment. Use 'python3 main.py --demo' for CLI execution.")
        return 1

    app = QApplication(sys.argv)
    qss_path = base_dir / "ui" / "theme_cs5.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    engine = EAIWEngine(base_dir)
    win = EAIWMainWindow(engine, base_dir)
    win.show()
    return app.exec()


if HAS_PYSIDE6:
    class EAIWMainWindow(QMainWindow):
        def __init__(self, engine: EAIWEngine, base_dir: Path):
            super().__init__()
            self.engine = engine
            self.base_dir = base_dir
            self.setWindowTitle("EAIW 1.0 — Early AI Visual Workstation (2014–2022 Historical Reproduction & Artifact Lab)")
            self.resize(1440, 900)

            self.current_input_path = str(base_dir / "datasets" / "testing" / "sample_street_scene.png")
            self.current_mask_path = str(base_dir / "datasets" / "testing" / "sample_street_mask.png")

            central = QWidget()
            self.setCentralWidget(central)
            main_layout = QVBoxLayout(central)

            top_splitter = QSplitter(Qt.Horizontal)
            main_layout.addWidget(top_splitter, stretch=4)

            # 1. Left Panel: Model Library & Laboratory Selector
            left_panel = QGroupBox("01-06 实验室与历史模型库 (Model Library)")
            left_layout = QVBoxLayout(left_panel)
            self.model_list = QListWidget()
            for m in self.engine.model_manager.list_models():
                item = QListWidgetItem(f"[{m['year']}] {m['name']}")
                item.setData(Qt.UserRole, m["model_id"])
                self.model_list.addItem(item)
            self.model_list.setCurrentRow(0)
            left_layout.addWidget(self.model_list)

            btn_load = QPushButton("导入图像 (Load Input Image)")
            btn_load.clicked.connect(self.on_load_image)
            left_layout.addWidget(btn_load)

            btn_clear_mask = QPushButton("重置遮罩 (Clear Mask)")
            btn_clear_mask.clicked.connect(lambda: self.canvas.clear_mask())
            left_layout.addWidget(btn_clear_mask)

            btn_compare = QPushButton("运行四代修复对照实验 (4-Model Compare)")
            btn_compare.clicked.connect(self.on_run_comparison)
            left_layout.addWidget(btn_compare)

            top_splitter.addWidget(left_panel)

            # 2. Center Panel: Interactive Canvas
            center_panel = QGroupBox("主视觉画布与遮罩编辑器 (Workstation Canvas & Mask Editor)")
            center_layout = QVBoxLayout(center_panel)
            self.canvas = MaskCanvasWidget()
            if Path(self.current_input_path).exists():
                self.canvas.load_rgb(ImageIO.load_image(self.current_input_path))
            center_layout.addWidget(self.canvas, stretch=1)

            canvas_bar = QHBoxLayout()
            btn_orig = QPushButton("显示原图+遮罩 (Original / Mask)")
            btn_orig.clicked.connect(self.show_original)
            btn_res = QPushButton("显示生成结果 (Result View)")
            btn_res.clicked.connect(self.show_result)
            canvas_bar.addWidget(btn_orig)
            canvas_bar.addWidget(btn_res)
            center_layout.addLayout(canvas_bar)
            top_splitter.addWidget(center_panel)

            # 3. Right Panel: Parameter & Failure Simulation Console
            right_panel = QGroupBox("原生参数与视觉故障控制台 (Parameter & Artifact Console)")
            right_layout = QVBoxLayout(right_panel)

            right_layout.addWidget(QLabel("运行模式 (Execution Mode):"))
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["Simulation", "Historical", "Modern Reproduction"])
            right_layout.addWidget(self.mode_combo)

            right_layout.addWidget(QLabel("随机种子 (Seed):"))
            self.seed_spin = QSpinBox()
            self.seed_spin.setRange(0, 999999)
            self.seed_spin.setValue(42)
            right_layout.addWidget(self.seed_spin)

            right_layout.addWidget(QLabel("迭代次数 / Steps:"))
            self.iter_spin = QSpinBox()
            self.iter_spin.setRange(1, 100)
            self.iter_spin.setValue(8)
            right_layout.addWidget(self.iter_spin)

            right_layout.addWidget(QLabel("文本提示词 / Prompt (CLIP/Diffusion):"))
            self.prompt_edit = QLineEdit("A surreal cybernetic cathedral painted by Beksinski")
            right_layout.addWidget(self.prompt_edit)

            # 7-Axis Failure Sliders
            fail_group = QGroupBox("Inpainting 七维故障模拟器 (Artifact Lab)")
            fail_layout = QVBoxLayout(fail_group)
            self.sliders: Dict[str, QSlider] = {}
            for key, label in [
                ("mask_accuracy", "Mask Accuracy (遮罩精度)"),
                ("context_awareness", "Context Awareness (上下文感知)"),
                ("semantic_consistency", "Semantic Consistency (语义一致性)"),
                ("spatial_consistency", "Spatial Consistency (空间透视关系)"),
                ("boundary_blending", "Boundary Blending (接缝融合度)"),
                ("resolution_quality", "Resolution Quality (局部细节分辨率)"),
            ]:
                row = QHBoxLayout()
                row.addWidget(QLabel(label))
                sl = QSlider(Qt.Horizontal)
                sl.setRange(10, 100)
                sl.setValue(100)
                self.sliders[key] = sl
                row.addWidget(sl)
                fail_layout.addLayout(row)
            right_layout.addWidget(fail_group)

            btn_run = QPushButton("▶ 执行当前模型推理 (Execute Model Node)")
            btn_run.setStyleSheet("background-color: #8c5a14; color: #ffffff; font-weight: bold; min-height: 32px;")
            btn_run.clicked.connect(self.on_execute_model)
            right_layout.addWidget(btn_run)

            top_splitter.addWidget(right_panel)
            top_splitter.setSizes([280, 680, 420])

            # 4. Bottom Dock: Iteration Timeline & Experiment / Failure Archive Table
            bottom_tabs = QTabWidget()
            self.exp_table = QTableWidget(0, 7)
            self.exp_table.setHorizontalHeaderLabels([
                "实验 ID", "实验室", "模型 ID", "运行模式", "PSNR / SSIM / LPIPS", "中间帧数", "输出路径"
            ])
            bottom_tabs.addTab(self.exp_table, "实验记录与迭代历史 (Experiment & Iteration Timeline)")

            self.fail_table = QTableWidget(0, 6)
            self.fail_table.setHorizontalHeaderLabels([
                "案例 ID", "故障分类 (8 Categories)", "关联技术", "模型版本", "可复现", "说明与人工标注"
            ])
            bottom_tabs.addTab(self.fail_table, "AI 视觉故障数据库 (AI Artifact Database)")

            main_layout.addWidget(bottom_tabs, stretch=2)
            self.refresh_tables()

        def show_original(self):
            self.canvas.show_result = False
            self.canvas.update()

        def show_result(self):
            if self.canvas.result_rgb is not None:
                self.canvas.show_result = True
                self.canvas.update()

        def on_load_image(self):
            path, _ = QFileDialog.getOpenFileName(self, "选择输入图像", str(self.base_dir), "Images (*.png *.jpg *.jpeg *.bmp)")
            if path:
                self.current_input_path = path
                self.canvas.load_rgb(ImageIO.load_image(path))

        def on_execute_model(self):
            item = self.model_list.currentItem()
            if not item:
                return
            model_id = item.data(Qt.UserRole)
            if self.canvas.mask is not None:
                ImageIO.save_mask(self.canvas.mask, self.current_mask_path)

            params: Dict[str, Any] = {
                "seed": self.seed_spin.value(),
                "iterations": self.iter_spin.value(),
                "steps": self.iter_spin.value(),
                "prompt": self.prompt_edit.text(),
            }
            for k, sl in self.sliders.items():
                params[k] = round(sl.value() / 100.0, 2)

            out_path = self.base_dir / "archive" / f"gui_run_{model_id}.png"
            res = self.engine.execute_request({
                "model_id": model_id,
                "execution_mode": self.mode_combo.currentText(),
                "input": self.current_input_path,
                "mask": self.current_mask_path,
                "parameters": params,
                "output": str(out_path),
            })
            self.canvas.set_result_rgb(ImageIO.load_image(res["output"]))
            self.refresh_tables()

        def on_run_comparison(self):
            if self.canvas.mask is not None:
                ImageIO.save_mask(self.canvas.mask, self.current_mask_path)
            comp_dir = self.base_dir / "archive" / "gui_comparison"
            self.engine.compare_inpainting_models(
                self.current_input_path,
                self.current_mask_path,
                comp_dir,
                execution_mode=self.mode_combo.currentText(),
            )
            self.refresh_tables()
            QMessageBox.information(self, "对照实验完成", f"已完成 PatchMatch / DeepFill v2 / LaMa / SD Inpainting 对照实验，结果保存在:\n{comp_dir}")

        def refresh_tables(self):
            exps = self.engine.db.list_experiments(limit=50)
            self.exp_table.setRowCount(len(exps))
            for r_idx, exp in enumerate(exps):
                m = exp.get("metrics", {})
                m_str = f"PSNR:{m.get('PSNR', '-')} | SSIM:{m.get('SSIM', '-')} | LPIPS:{m.get('LPIPS', '-')}"
                vals = [
                    exp["id"],
                    exp["laboratory"],
                    exp["model_id"],
                    exp["execution_mode"],
                    m_str,
                    str(len(exp.get("intermediate_paths", []))),
                    exp["output_path"],
                ]
                for c_idx, v in enumerate(vals):
                    self.exp_table.setItem(r_idx, c_idx, QTableWidgetItem(str(v)))

            fails = self.engine.db.list_artifact_failures()
            self.fail_table.setRowCount(len(fails))
            for r_idx, f in enumerate(fails):
                vals = [
                    f["case_id"],
                    f["failure_category"],
                    f["associated_tech"],
                    f"{f['model_id']} ({f['model_version']})",
                    "是 (Yes)" if f["reproducible"] else "否",
                    f["description"],
                ]
                for c_idx, v in enumerate(vals):
                    self.fail_table.setItem(r_idx, c_idx, QTableWidgetItem(str(v)))
