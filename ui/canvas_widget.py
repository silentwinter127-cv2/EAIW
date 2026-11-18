"""
Interactive Image & Mask Editor Canvas Widget for PySide6 Workstation.
Supports brush drawing on deletion masks, original/result split comparison, and zoom.
"""
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

try:
    from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSlider
    from PySide6.QtCore import Qt, QPoint
    from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False
    QWidget = object


if HAS_PYSIDE6:
    class MaskCanvasWidget(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMinimumSize(480, 360)
            self.img_rgb: Optional[np.ndarray] = None
            self.result_rgb: Optional[np.ndarray] = None
            self.mask: Optional[np.ndarray] = None
            self.brush_size: int = 22
            self.show_result: bool = False
            self.show_mask_overlay: bool = True
            self._drawing: bool = False
            self._last_pt: Optional[QPoint] = None

        def load_rgb(self, img_rgb: np.ndarray):
            self.img_rgb = img_rgb.copy()
            self.result_rgb = None
            self.show_result = False
            h, w, _ = img_rgb.shape
            if self.mask is None or self.mask.shape != (h, w):
                self.mask = np.zeros((h, w), dtype=np.uint8)
                # Default center object mask so users can immediately run Inpainting
                cv2.circle(self.mask, (w // 2, h // 2), min(h, w) // 6, 255, -1)
            self.update()

        def set_result_rgb(self, res_rgb: np.ndarray):
            self.result_rgb = res_rgb.copy()
            self.show_result = True
            self.update()

        def clear_mask(self):
            if self.mask is not None:
                self.mask.fill(0)
                self.update()

        def _widget_to_image_coords(self, pt: QPoint):
            if self.img_rgb is None:
                return None
            h, w, _ = self.img_rgb.shape
            x = int(pt.x() * w / max(1, self.width()))
            y = int(pt.y() * h / max(1, self.height()))
            return max(0, min(w - 1, x)), max(0, min(h - 1, y))

        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton and self.img_rgb is not None:
                self._drawing = True
                coords = self._widget_to_image_coords(event.position().toPoint())
                if coords and self.mask is not None:
                    cv2.circle(self.mask, coords, self.brush_size, 255, -1)
                    self._last_pt = QPoint(*coords)
                    self.update()

        def mouseMoveEvent(self, event):
            if self._drawing and self.img_rgb is not None and self.mask is not None:
                coords = self._widget_to_image_coords(event.position().toPoint())
                if coords:
                    cur = QPoint(*coords)
                    if self._last_pt is not None:
                        cv2.line(
                            self.mask,
                            (self._last_pt.x(), self._last_pt.y()),
                            coords,
                            255,
                            self.brush_size * 2,
                        )
                    self._last_pt = cur
                    self.update()

        def mouseReleaseEvent(self, event):
            if event.button() == Qt.LeftButton:
                self._drawing = False
                self._last_pt = None

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(24, 24, 24))
            if self.img_rgb is None:
                painter.setPen(QColor(150, 150, 150))
                painter.drawText(self.rect(), Qt.AlignCenter, "EAIW Canvas — Load or Select Test Image")
                return

            disp = self.result_rgb.copy() if (self.show_result and self.result_rgb is not None) else self.img_rgb.copy()
            if self.show_mask_overlay and not self.show_result and self.mask is not None:
                m_bool = self.mask > 127
                disp[m_bool, 0] = np.clip(disp[m_bool, 0] * 0.45 + 140, 0, 255)
                disp[m_bool, 1] = (disp[m_bool, 1] * 0.45).astype(np.uint8)
                disp[m_bool, 2] = (disp[m_bool, 2] * 0.45).astype(np.uint8)

            h, w, ch = disp.shape
            qimg = QImage(disp.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            ox = (self.width() - pix.width()) // 2
            oy = (self.height() - pix.height()) // 2
            painter.drawPixmap(ox, oy, pix)
