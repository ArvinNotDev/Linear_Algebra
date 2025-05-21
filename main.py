import sys
import string
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGridLayout, QComboBox, QListWidget, QMessageBox
)
from PyQt6.QtGui import QPalette, QColor, QLinearGradient, QBrush, QTransform, QPixmap, QPainter
from PyQt6.QtCore import Qt, QTimer
from datetime import datetime

from functions import gauss_jordan, invert_matrix, det_by_row_ops


class RotatingEmoji(QLabel):
    def __init__(self, emoji, parent=None):
        super().__init__(emoji, parent)
        self.angle = 0
        self.setFixedSize(40, 40)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.timer.start(50)

    def rotate(self):
        self.angle = (self.angle + 5) % 360
        pix = QPixmap(self.size())
        pix.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.translate(pix.width()/2, pix.height()/2)
        painter.rotate(self.angle)
        painter.translate(-pix.width()/2, -pix.height()/2)
        painter.setFont(self.font())
        painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, self.text())
        painter.end()
        self.setPixmap(pix)


class MatrixApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linear Algebra Project")
        self.setGeometry(100, 100, 1000, 600)

        self.dark_mode = True
        self.history = []
        self.max_history = 10

        self.mode = "Mode1"
        self.rows = 3
        self.cols = 3
        self.size = 3
        self.entries = []

        self.main_layout = QHBoxLayout(self)
        self.setLayout(self.main_layout)

        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        # Rotating corner emoji
        emoji = RotatingEmoji("😉")
        self.main_layout.addWidget(emoji, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # History Panel
        self.history_panel = QVBoxLayout()
        self.history_list = QListWidget()
        self.history_label = QLabel("🗂️ Matrix History")
        self.history_label.setStyleSheet("font: bold 14px; margin: 10px 0;")
        self.history_panel.addWidget(self.history_label)
        self.history_panel.addWidget(self.history_list)
        
        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("🔄 Load")
        self.delete_btn = QPushButton("🗑️ Delete")
        for btn in (self.load_btn, self.delete_btn):
            btn.setFixedSize(80, 40)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(
                "QPushButton { background: rgba(255,255,255,0.5); border:none; border-radius:12px; font-size:14px; }"
                "QPushButton:hover { background: rgba(255,255,255,0.8); }"
            )
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.delete_btn)
        self.load_btn.clicked.connect(self.load_history)
        self.delete_btn.clicked.connect(self.delete_history)
        self.history_panel.addLayout(btn_row)

        self.controls_panel = QVBoxLayout()
        mode_row = QHBoxLayout()
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["System of Equations", "Square Matrix"])
        self.mode_selector.currentIndexChanged.connect(self.switch_mode)
        mode_row.addWidget(QLabel("🔀 Mode:"))
        mode_row.addWidget(self.mode_selector)

        self.controls_panel.addLayout(mode_row)

        self.config_row = QHBoxLayout()
        self.controls_panel.addLayout(self.config_row)

        self.matrix_widget = QWidget()
        self.matrix_grid = QGridLayout()
        self.matrix_grid.setSpacing(10)
        self.matrix_widget.setLayout(self.matrix_grid)

        right_panel = QVBoxLayout()
        right_panel.addLayout(self.controls_panel)
        right_panel.addWidget(self.matrix_widget)

        self.main_layout.addLayout(self.history_panel, 2)
        self.main_layout.addLayout(right_panel, 8)

        self.switch_mode()

    def apply_theme(self):
        palette = QPalette()
        grad = QLinearGradient(0, 0, self.width(), self.height())
        if self.dark_mode:
            grad.setColorAt(0.0, QColor('#833ab4'))
            grad.setColorAt(0.5, QColor('#fd1d1d'))
            grad.setColorAt(1.0, QColor('#fcb045'))
        else:
            grad.setColorAt(0.0, QColor('#fcb045'))
            grad.setColorAt(0.5, QColor('#fd1d1d'))
            grad.setColorAt(1.0, QColor('#833ab4'))
        brush = QBrush(grad)
        palette.setBrush(QPalette.ColorRole.Window, brush)
        self.setAutoFillBackground(True)
        self.setPalette(palette)

        self.setStyleSheet(
            "QWidget { color: white; }"
            "QLabel { font: 14px 'Segoe UI'; }"
            "QLineEdit { border: none; border-radius: 10px; padding: 6px; background: rgba(255,255,255,0.7); color: #333; font-size:14px; }"
            "QComboBox { border-radius: 10px; padding: 6px; background: rgba(255,255,255,0.7); color: #333; font-size:14px; }"
            "QListWidget { border-radius: 10px; padding: 6px; background: rgba(255,255,255,0.7); color: #333; font-size:14px; }"
        )

    def switch_mode(self):
        self.mode = "Mode1" if self.mode_selector.currentIndex() == 0 else "Mode2"
        self.clear_layout(self.config_row)
        self.clear_layout(self.matrix_grid)
        self.entries.clear()

        if self.mode == "Mode1":
            self.build_equation_controls()
            self.build_equation_grid()
        else:
            self.build_square_controls()
            self.build_square_grid()

    def build_equation_controls(self):
        self.rows_input = QLineEdit(str(self.rows)); self.rows_input.setFixedSize(60, 40)
        self.cols_input = QLineEdit(str(self.cols)); self.cols_input.setFixedSize(60, 40)
        self.set_btn = QPushButton("Set Size"); self.solve_btn = QPushButton("Solve")
        for w in (self.rows_input, self.cols_input, self.set_btn, self.solve_btn):
            w.setCursor(Qt.CursorShape.PointingHandCursor)
            if isinstance(w, QPushButton):
                w.setStyleSheet(
                    "QPushButton { background: rgba(255,255,255,0.6); border:none; border-radius:20px; font-size:14px; height: 40px;}"
                    "QPushButton:hover { background: rgba(255,255,255,0.9); }"
                )
            self.config_row.addWidget(w)
        self.set_btn.clicked.connect(self.build_equation_grid)
        self.solve_btn.clicked.connect(self.solve_gauss_jordan)

    def build_square_controls(self):
        self.size_input = QLineEdit(str(self.size)); self.size_input.setFixedSize(60, 40)
        self.set_btn = QPushButton("Set n x n"); self.inv_btn = QPushButton("Inverse"); self.det_btn = QPushButton("Det")
        for w in (self.size_input, self.set_btn, self.inv_btn, self.det_btn):
            w.setCursor(Qt.CursorShape.PointingHandCursor)
            if isinstance(w, QPushButton):
                w.setStyleSheet(
                    "QPushButton { background: rgba(255,255,255,0.6); border:none; border-radius:20px; font-size:14px; height: 40px}"
                    "QPushButton:hover { background: rgba(255,255,255,0.9); }"
                )
            self.config_row.addWidget(w)
        self.set_btn.clicked.connect(self.build_square_grid)
        self.inv_btn.clicked.connect(self.matrix_inverse)
        self.det_btn.clicked.connect(self.matrix_determinant)

    def build_equation_grid(self):
        self.rows = min(int(self.rows_input.text()), 20)
        self.cols = min(int(self.cols_input.text()), 20)
        self.clear_layout(self.matrix_grid)
        self.entries = []
        var_names = list(string.ascii_lowercase[:self.cols])
        for i in range(self.rows):
            row_e = []
            for j in range(self.cols):
                e = QLineEdit("0"); e.setFixedSize(60, 40)
                self.matrix_grid.addWidget(e, i, j*3)
                row_e.append(e)
                lbl = QLabel(var_names[j]); self.matrix_grid.addWidget(lbl, i, j*3+1)
                plus_lbl = QLabel("+"); self.matrix_grid.addWidget(plus_lbl, i, j*3+2)
            eq_lbl = QLabel("="); rhs = QLineEdit("0"); rhs.setFixedSize(60, 40)
            self.matrix_grid.addWidget(eq_lbl, i, self.cols*3)
            self.matrix_grid.addWidget(rhs, i, self.cols*3+1)
            row_e.append(rhs)
            self.entries.append(row_e)

    def build_square_grid(self):
        self.size = min(int(self.size_input.text()), 20)
        self.clear_layout(self.matrix_grid)
        self.entries = []
        for i in range(self.size):
            row_e = []
            for j in range(self.size):
                e = QLineEdit("1" if i==j else "0"); e.setFixedSize(60, 40)
                self.matrix_grid.addWidget(e, i, j)
                row_e.append(e)
            self.entries.append(row_e)

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w: w.deleteLater()

    def add_to_history(self, matrix, mode):
        timestamp = datetime.now().strftime("%H:%M:%S")
        desc = f"{mode} {len(self.history)+1} ({timestamp})"
        self.history.insert(0, {"desc": desc, "matrix": matrix.tolist(), "mode": mode,
                               "rows": matrix.shape[0], "cols": matrix.shape[1] if mode=="Mode1" else matrix.shape[0]})
        if len(self.history) > self.max_history:
            self.history.pop()
        self.update_history_list()

    def update_history_list(self):
        self.history_list.clear()
        for item in self.history:
            self.history_list.addItem(item["desc"])

    def load_history(self):
        idx = self.history_list.currentRow()
        if idx < 0:
            QMessageBox.warning(self, "Error", "Select a matrix")
            return
        data = self.history[idx]
        self.mode_selector.setCurrentIndex(0 if data["mode"] == "Mode1" else 1)
        self.switch_mode()
        if data["mode"] == "Mode1":
            self.rows_input.setText(str(data["rows"]))
            self.cols_input.setText(str(data["cols"] - 1))
            self.build_equation_grid()
        else:
            self.size_input.setText(str(data["rows"]))
            self.build_square_grid()
        mat = np.array(data["matrix"])
        for i, row in enumerate(mat):
            for j, val in enumerate(row):
                self.entries[i][j].setText(str(val))

    def delete_history(self):
        idx = self.history_list.currentRow()
        if idx >= 0:
            del self.history[idx]
            self.update_history_list()

    def solve_gauss_jordan(self):
        try:
            mat = np.array([[float(e.text()) for e in row] for row in self.entries])
            self.add_to_history(mat, "Mode1")
            sol = gauss_jordan(mat)
            if isinstance(sol, str):
                QMessageBox.information(self, "Solution", "No unique solutions.")
            else:
                msg = "\n".join(f"{v} = {x:.4f}" for v, x in zip(string.ascii_lowercase, sol))
                QMessageBox.information(self, "Solution", msg)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def matrix_inverse(self):
        try:
            A = np.array([[float(e.text()) for e in row] for row in self.entries])
            self.add_to_history(A, "Mode2")
            inv = invert_matrix(A)
            msg = "\n".join("  ".join(f"{x:.4f}" for x in r) for r in inv)
            QMessageBox.information(self, "Inverse", msg)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def matrix_determinant(self):
        try:
            A = np.array([[float(e.text()) for e in row] for row in self.entries])
            self.add_to_history(A, "Mode2")
            d = det_by_row_ops(A)
            QMessageBox.information(self, "Determinant", f"det = {d}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MatrixApp()
    window.show()
    sys.exit(app.exec())
