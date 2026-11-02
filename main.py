import sys
from datetime import datetime

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIntValidator, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QComboBox,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from functions import gauss_jordan, invert_matrix, det_by_row_ops


class MatrixApp(QWidget):
    """A polished desktop interface for the repository's linear algebra tools."""

    MAX_DIMENSION = 20
    MAX_HISTORY = 12

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linear Algebra Studio")
        self.resize(1240, 860)
        self.setMinimumSize(960, 700)

        self.dark_mode = True
        self.history = []
        self.mode = "equations"
        self.rows = 3
        self.cols = 3
        self.size = 3
        self.entries = []

        self.build_ui()
        self.apply_theme()
        self.switch_mode()

    # ------------------------------------------------------------------
    # Interface construction
    # ------------------------------------------------------------------
    def build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 18)
        root.setSpacing(18)

        root.addLayout(self.build_header())

        body = QHBoxLayout()
        body.setSpacing(18)
        root.addLayout(body, 1)

        body.addWidget(self.build_sidebar())
        body.addWidget(self.build_workspace(), 1)

        footer = QHBoxLayout()
        footer.setContentsMargins(4, 0, 4, 0)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("muted")
        footer.addWidget(self.status_label)
        footer.addItem(QSpacerItem(10, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        footer.addWidget(QLabel("LINEAR ALGEBRA STUDIO  •  PYTHON + NUMPY"))
        root.addLayout(footer)

    def build_header(self):
        header = QHBoxLayout()
        header.setSpacing(14)

        brand_mark = QFrame()
        brand_mark.setObjectName("brandMark")
        brand_mark.setFixedSize(52, 52)
        mark_layout = QVBoxLayout(brand_mark)
        mark_layout.setContentsMargins(0, 0, 0, 0)
        mark = QLabel("∑")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setObjectName("brandGlyph")
        mark_layout.addWidget(mark)
        header.addWidget(brand_mark)

        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        title = QLabel("Linear Algebra Studio")
        title.setObjectName("appTitle")
        subtitle = QLabel("A focused workspace for matrices and linear systems")
        subtitle.setObjectName("muted")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block)
        header.addItem(QSpacerItem(20, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        self.theme_button = QPushButton("☼  Light appearance")
        self.theme_button.setObjectName("themeButton")
        self.theme_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.theme_button.clicked.connect(self.toggle_theme)
        header.addWidget(self.theme_button)
        return header

    def build_sidebar(self):
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setMinimumWidth(230)
        panel.setMaximumWidth(280)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 20, 18, 18)
        layout.setSpacing(12)

        heading = QLabel("Recent activity")
        heading.setObjectName("sectionTitle")
        description = QLabel("Your recently used matrices")
        description.setObjectName("muted")
        layout.addWidget(heading)
        layout.addWidget(description)

        self.history_list = QListWidget()
        self.history_list.setObjectName("historyList")
        self.history_list.setSpacing(5)
        self.history_list.itemDoubleClicked.connect(self.load_history)
        layout.addWidget(self.history_list, 1)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        load_button = QPushButton("Load")
        load_button.setObjectName("secondaryButton")
        load_button.clicked.connect(self.load_history)
        delete_button = QPushButton("Delete")
        delete_button.setObjectName("ghostButton")
        delete_button.clicked.connect(self.delete_history)
        actions.addWidget(load_button)
        actions.addWidget(delete_button)
        layout.addLayout(actions)

        clear_button = QPushButton("Clear history")
        clear_button.setObjectName("ghostButton")
        clear_button.clicked.connect(self.clear_history)
        layout.addWidget(clear_button)

        note = QLabel("History is kept for the current session and stores up to 12 matrices.")
        note.setObjectName("sidebarNote")
        note.setWordWrap(True)
        layout.addWidget(note)
        return panel

    def build_workspace(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        intro = QVBoxLayout()
        intro.setSpacing(3)
        heading = QLabel("Workspace")
        heading.setObjectName("pageTitle")
        description = QLabel("Choose a workflow, enter your values, and run an operation.")
        description.setObjectName("muted")
        intro.addWidget(heading)
        intro.addWidget(description)
        layout.addLayout(intro)

        self.toolbar = QFrame()
        self.toolbar.setObjectName("panel")
        toolbar_layout = QHBoxLayout(self.toolbar)
        toolbar_layout.setContentsMargins(16, 14, 16, 14)
        toolbar_layout.setSpacing(14)

        mode_block = QVBoxLayout()
        mode_block.setSpacing(5)
        mode_label = QLabel("WORKFLOW")
        mode_label.setObjectName("eyebrow")
        self.mode_selector = QComboBox()
        self.mode_selector.setMinimumWidth(210)
        self.mode_selector.addItems(["System of equations", "Square matrix"])
        self.mode_selector.currentIndexChanged.connect(self.switch_mode)
        mode_block.addWidget(mode_label)
        mode_block.addWidget(self.mode_selector)
        toolbar_layout.addLayout(mode_block)

        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFixedWidth(1)
        toolbar_layout.addWidget(divider)

        self.config_row = QHBoxLayout()
        self.config_row.setSpacing(9)
        toolbar_layout.addLayout(self.config_row, 1)
        layout.addWidget(self.toolbar)

        self.matrix_panel = QFrame()
        self.matrix_panel.setObjectName("panel")
        matrix_layout = QVBoxLayout(self.matrix_panel)
        matrix_layout.setContentsMargins(18, 16, 18, 16)
        matrix_layout.setSpacing(12)

        matrix_header = QHBoxLayout()
        matrix_heading_block = QVBoxLayout()
        matrix_heading_block.setSpacing(2)
        matrix_heading = QLabel("Matrix input")
        matrix_heading.setObjectName("sectionTitle")
        self.matrix_description = QLabel("Enter the coefficients and right-hand side values.")
        self.matrix_description.setObjectName("muted")
        matrix_heading_block.addWidget(matrix_heading)
        matrix_heading_block.addWidget(self.matrix_description)
        matrix_header.addLayout(matrix_heading_block)
        matrix_header.addItem(QSpacerItem(10, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        reset_button = QPushButton("Reset values")
        reset_button.setObjectName("ghostButton")
        reset_button.clicked.connect(self.reset_matrix)
        matrix_header.addWidget(reset_button)
        matrix_layout.addLayout(matrix_header)

        self.matrix_scroll = QScrollArea()
        self.matrix_scroll.setWidgetResizable(True)
        self.matrix_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.matrix_scroll.setMinimumHeight(280)
        self.matrix_widget = QWidget()
        self.matrix_grid = QGridLayout(self.matrix_widget)
        self.matrix_grid.setContentsMargins(6, 8, 6, 8)
        self.matrix_grid.setHorizontalSpacing(10)
        self.matrix_grid.setVerticalSpacing(10)
        self.matrix_grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.matrix_scroll.setWidget(self.matrix_widget)
        matrix_layout.addWidget(self.matrix_scroll, 1)
        layout.addWidget(self.matrix_panel, 1)

        self.result_panel = QFrame()
        self.result_panel.setObjectName("resultPanel")
        result_layout = QVBoxLayout(self.result_panel)
        result_layout.setContentsMargins(18, 14, 18, 14)
        result_layout.setSpacing(8)

        result_header = QHBoxLayout()
        self.result_title = QLabel("Result")
        self.result_title.setObjectName("sectionTitle")
        result_header.addWidget(self.result_title)
        result_header.addItem(QSpacerItem(10, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        self.copy_button = QPushButton("Copy")
        self.copy_button.setObjectName("ghostButton")
        self.copy_button.clicked.connect(self.copy_result)
        result_header.addWidget(self.copy_button)
        result_layout.addLayout(result_header)

        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        self.result_text.setObjectName("resultText")
        result_layout.addWidget(self.result_text)
        self.result_panel.hide()
        layout.addWidget(self.result_panel)

        return container

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()

    def apply_theme(self):
        if self.dark_mode:
            colors = {
                "background": "#0B1220",
                "panel": "#121D30",
                "panel_alt": "#17243A",
                "input": "#0D1728",
                "border": "#293A55",
                "text": "#E8EEF8",
                "muted": "#91A2BC",
                "accent": "#8293FF",
                "accent_hover": "#98A5FF",
                "accent_text": "#10172A",
                "selection": "#27385A",
                "result": "#102B32",
                "result_border": "#23545A",
                "scroll": "#34445F",
            }
            self.theme_button.setText("☼  Light appearance")
        else:
            colors = {
                "background": "#F2F5FB",
                "panel": "#FFFFFF",
                "panel_alt": "#F4F7FC",
                "input": "#F7F9FD",
                "border": "#DCE4F0",
                "text": "#1B2940",
                "muted": "#687A95",
                "accent": "#596BEA",
                "accent_hover": "#4658D5",
                "accent_text": "#FFFFFF",
                "selection": "#E6EBFF",
                "result": "#EDF8F6",
                "result_border": "#B9E2D9",
                "scroll": "#C6D0DF",
            }
            self.theme_button.setText("☾  Dark appearance")

        self.setStyleSheet(f"""
            QWidget {{
                color: {colors['text']};
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }}
            QWidget#appRoot {{
                background: {colors['background']};
            }}
            QFrame#panel {{
                background: {colors['panel']};
                border: 1px solid {colors['border']};
                border-radius: 16px;
            }}
            QFrame#brandMark {{
                background: {colors['panel_alt']};
                border: 1px solid {colors['border']};
                border-radius: 15px;
            }}
            QLabel#brandGlyph {{
                color: {colors['accent']};
                font-size: 28px;
                font-weight: 700;
            }}
            QLabel#appTitle {{
                font-size: 20px;
                font-weight: 700;
                letter-spacing: 0.2px;
            }}
            QLabel#pageTitle {{
                font-size: 22px;
                font-weight: 700;
            }}
            QLabel#sectionTitle {{
                font-size: 15px;
                font-weight: 650;
            }}
            QLabel#eyebrow {{
                color: {colors['muted']};
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1.1px;
            }}
            QLabel#muted {{
                color: {colors['muted']};
                font-size: 12px;
            }}
            QLabel#sidebarNote {{
                color: {colors['muted']};
                background: {colors['panel_alt']};
                border: 1px solid {colors['border']};
                border-radius: 10px;
                padding: 10px;
                font-size: 11px;
            }}
            QFrame#divider {{
                background: {colors['border']};
                border: none;
            }}
            QLineEdit {{
                background: {colors['input']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                border-radius: 9px;
                padding: 8px 10px;
                selection-background-color: {colors['selection']};
                min-height: 22px;
            }}
            QLineEdit:focus {{
                border: 1px solid {colors['accent']};
            }}
            QLineEdit#matrixCell {{
                min-width: 62px;
                max-width: 82px;
                min-height: 25px;
                padding: 7px 8px;
                font-family: 'Cascadia Code', Consolas, monospace;
                font-size: 13px;
            }}
            QLineEdit#dimensionInput {{
                min-width: 48px;
                max-width: 58px;
                min-height: 22px;
                font-weight: 600;
                text-align: center;
            }}
            QComboBox {{
                background: {colors['input']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                border-radius: 9px;
                padding: 8px 10px;
                min-height: 22px;
            }}
            QComboBox:focus {{
                border: 1px solid {colors['accent']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 26px;
            }}
            QComboBox QAbstractItemView {{
                background: {colors['panel']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                selection-background-color: {colors['selection']};
                outline: none;
                padding: 4px;
            }}
            QPushButton {{
                border: none;
                border-radius: 9px;
                padding: 9px 13px;
                font-weight: 600;
            }}
            QPushButton#primaryButton {{
                background: {colors['accent']};
                color: {colors['accent_text']};
            }}
            QPushButton#primaryButton:hover {{
                background: {colors['accent_hover']};
            }}
            QPushButton#secondaryButton {{
                background: {colors['panel_alt']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
            }}
            QPushButton#secondaryButton:hover, QPushButton#ghostButton:hover,
            QPushButton#themeButton:hover {{
                background: {colors['selection']};
            }}
            QPushButton#ghostButton, QPushButton#themeButton {{
                background: transparent;
                color: {colors['muted']};
                border: 1px solid {colors['border']};
            }}
            QPushButton#themeButton {{
                padding: 10px 14px;
            }}
            QPushButton:disabled {{
                color: {colors['muted']};
                background: {colors['panel_alt']};
            }}
            QListWidget#historyList {{
                background: transparent;
                border: none;
                outline: none;
                padding: 2px;
            }}
            QListWidget#historyList::item {{
                background: {colors['panel_alt']};
                border: 1px solid transparent;
                border-radius: 9px;
                padding: 11px 10px;
                margin: 2px 0;
            }}
            QListWidget#historyList::item:hover {{
                border: 1px solid {colors['border']};
            }}
            QListWidget#historyList::item:selected {{
                background: {colors['selection']};
                border: 1px solid {colors['accent']};
            }}
            QPlainTextEdit#resultText {{
                background: {colors['input']};
                color: {colors['text']};
                border: 1px solid {colors['result_border']};
                border-radius: 10px;
                padding: 8px;
                font-family: 'Cascadia Code', Consolas, monospace;
                font-size: 12px;
            }}
            QFrame#resultPanel {{
                background: {colors['result']};
                border: 1px solid {colors['result_border']};
                border-radius: 14px;
            }}
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 9px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical {{
                background: {colors['scroll']};
                border-radius: 4px;
                min-height: 24px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            QToolTip {{
                background: {colors['panel']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                padding: 5px;
            }}
        """)
        self.setObjectName("appRoot")

    # ------------------------------------------------------------------
    # Workflow and matrix editor
    # ------------------------------------------------------------------
    def switch_mode(self, *_):
        self.mode = "equations" if self.mode_selector.currentIndex() == 0 else "square"
        self.clear_layout(self.config_row)
        self.clear_layout(self.matrix_grid)
        self.entries = []
        self.result_panel.hide()

        if self.mode == "equations":
            self.matrix_description.setText("Build an augmented matrix for a system of linear equations.")
            self.build_equation_controls()
            self.build_equation_grid()
        else:
            self.matrix_description.setText("Enter a square matrix for inversion or determinant calculation.")
            self.build_square_controls()
            self.build_square_grid()

    def build_equation_controls(self):
        self.rows_input = self.make_dimension_input(self.rows)
        self.cols_input = self.make_dimension_input(self.cols)

        self.config_row.addWidget(self.make_field("EQUATIONS", self.rows_input))
        self.config_row.addWidget(self.make_field("VARIABLES", self.cols_input))

        apply_button = QPushButton("Apply size")
        apply_button.setObjectName("secondaryButton")
        apply_button.setCursor(Qt.CursorShape.PointingHandCursor)
        apply_button.clicked.connect(self.build_equation_grid)
        self.config_row.addWidget(apply_button)

        solve_button = QPushButton("Solve system  →")
        solve_button.setObjectName("primaryButton")
        solve_button.setCursor(Qt.CursorShape.PointingHandCursor)
        solve_button.clicked.connect(self.solve_gauss_jordan)
        self.config_row.addWidget(solve_button)

    def build_square_controls(self):
        self.size_input = self.make_dimension_input(self.size)
        self.config_row.addWidget(self.make_field("MATRIX SIZE", self.size_input))

        apply_button = QPushButton("Apply size")
        apply_button.setObjectName("secondaryButton")
        apply_button.setCursor(Qt.CursorShape.PointingHandCursor)
        apply_button.clicked.connect(self.build_square_grid)
        self.config_row.addWidget(apply_button)

        inverse_button = QPushButton("Inverse")
        inverse_button.setObjectName("primaryButton")
        inverse_button.setCursor(Qt.CursorShape.PointingHandCursor)
        inverse_button.clicked.connect(self.matrix_inverse)
        self.config_row.addWidget(inverse_button)

        determinant_button = QPushButton("Determinant")
        determinant_button.setObjectName("secondaryButton")
        determinant_button.setCursor(Qt.CursorShape.PointingHandCursor)
        determinant_button.clicked.connect(self.matrix_determinant)
        self.config_row.addWidget(determinant_button)

    def make_field(self, label_text, field):
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label = QLabel(label_text)
        label.setObjectName("eyebrow")
        layout.addWidget(label)
        layout.addWidget(field)
        return wrapper

    def make_dimension_input(self, value):
        field = QLineEdit(str(value))
        field.setObjectName("dimensionInput")
        field.setAlignment(Qt.AlignmentFlag.AlignCenter)
        field.setValidator(QIntValidator(1, self.MAX_DIMENSION, self))
        field.setFixedWidth(62)
        return field

    def make_matrix_input(self, value="0"):
        field = QLineEdit(value)
        field.setObjectName("matrixCell")
        field.setAlignment(Qt.AlignmentFlag.AlignCenter)
        field.setFixedHeight(42)
        field.setPlaceholderText("0")
        return field

    def build_equation_grid(self):
        dimensions = self.read_dimensions(self.rows_input, self.cols_input)
        if dimensions is None:
            return False
        self.rows, self.cols = dimensions
        self.clear_layout(self.matrix_grid)
        self.entries = []

        for i in range(self.rows):
            row_entries = []
            row_label = QLabel(f"E{i + 1}")
            row_label.setObjectName("eyebrow")
            row_label.setMinimumWidth(28)
            self.matrix_grid.addWidget(row_label, i, 0, alignment=Qt.AlignmentFlag.AlignVCenter)

            for j in range(self.cols):
                coefficient = self.make_matrix_input()
                self.matrix_grid.addWidget(coefficient, i, 1 + j * 3)
                row_entries.append(coefficient)

                variable = QLabel(f"x{j + 1}")
                variable.setObjectName("muted")
                self.matrix_grid.addWidget(variable, i, 2 + j * 3)

                if j < self.cols - 1:
                    plus = QLabel("+")
                    plus.setObjectName("muted")
                    self.matrix_grid.addWidget(plus, i, 3 + j * 3)

            equal = QLabel("=")
            equal.setObjectName("muted")
            self.matrix_grid.addWidget(equal, i, 1 + self.cols * 3)
            right_hand_side = self.make_matrix_input()
            self.matrix_grid.addWidget(right_hand_side, i, 2 + self.cols * 3)
            row_entries.append(right_hand_side)
            self.entries.append(row_entries)

        self.set_status(f"Equation editor ready · {self.rows} equations, {self.cols} variables")
        self.result_panel.hide()
        return True

    def build_square_grid(self):
        dimension = self.read_dimensions(self.size_input)
        if dimension is None:
            return False
        self.size = dimension[0]
        self.clear_layout(self.matrix_grid)
        self.entries = []

        corner = QLabel(" ")
        self.matrix_grid.addWidget(corner, 0, 0)
        for j in range(self.size):
            column_label = QLabel(f"C{j + 1}")
            column_label.setObjectName("eyebrow")
            column_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.matrix_grid.addWidget(column_label, 0, j + 1)

        for i in range(self.size):
            row_label = QLabel(f"R{i + 1}")
            row_label.setObjectName("eyebrow")
            self.matrix_grid.addWidget(row_label, i + 1, 0, alignment=Qt.AlignmentFlag.AlignVCenter)

            row_entries = []
            for j in range(self.size):
                value = "1" if i == j else "0"
                field = self.make_matrix_input(value)
                self.matrix_grid.addWidget(field, i + 1, j + 1)
                row_entries.append(field)
            self.entries.append(row_entries)

        self.set_status(f"Matrix editor ready · {self.size} × {self.size}")
        self.result_panel.hide()
        return True

    def read_dimensions(self, *fields):
        try:
            values = [int(field.text()) for field in fields]
        except ValueError:
            QMessageBox.warning(self, "Invalid dimensions", "Enter a whole number for every dimension.")
            return None

        if any(value < 1 or value > self.MAX_DIMENSION for value in values):
            QMessageBox.warning(
                self,
                "Dimensions out of range",
                f"Each dimension must be between 1 and {self.MAX_DIMENSION}.",
            )
            return None
        return values

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self.clear_layout(child_layout)

    def reset_matrix(self):
        for i, row in enumerate(self.entries):
            for j, field in enumerate(row):
                if self.mode == "square":
                    field.setText("1" if i == j else "0")
                else:
                    field.setText("0")
        self.result_panel.hide()
        self.set_status("Matrix values reset")

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------
    def add_to_history(self, matrix, mode):
        timestamp = datetime.now().strftime("%H:%M:%S")
        rows, columns = matrix.shape
        if mode == "equations":
            description = f"System · {rows} × {columns - 1} · {timestamp}"
        else:
            description = f"Matrix · {rows} × {columns} · {timestamp}"

        self.history.insert(
            0,
            {
                "desc": description,
                "matrix": matrix.tolist(),
                "mode": mode,
                "rows": rows,
                "cols": columns,
            },
        )
        self.history = self.history[: self.MAX_HISTORY]
        self.update_history_list()

    def update_history_list(self):
        self.history_list.clear()
        for item in self.history:
            self.history_list.addItem(item["desc"])

    def load_history(self, *_):
        index = self.history_list.currentRow()
        if index < 0:
            self.set_status("Select an item in history to load it")
            return

        data = self.history[index]
        selected_mode = 0 if data["mode"] == "equations" else 1
        self.mode_selector.setCurrentIndex(selected_mode)
        self.switch_mode()

        if data["mode"] == "equations":
            self.rows_input.setText(str(data["rows"]))
            self.cols_input.setText(str(data["cols"] - 1))
            if not self.build_equation_grid():
                return
        else:
            self.size_input.setText(str(data["rows"]))
            if not self.build_square_grid():
                return

        matrix = np.asarray(data["matrix"], dtype=float)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                self.entries[i][j].setText(f"{value:g}")

        self.set_status(f"Loaded {data['desc']}")

    def delete_history(self):
        index = self.history_list.currentRow()
        if index < 0:
            self.set_status("Select a history item to delete")
            return
        del self.history[index]
        self.update_history_list()
        self.set_status("History item deleted")

    def clear_history(self):
        if not self.history:
            self.set_status("History is already empty")
            return
        self.history.clear()
        self.update_history_list()
        self.set_status("History cleared")

    # ------------------------------------------------------------------
    # Operations and results
    # ------------------------------------------------------------------
    def read_matrix(self):
        if not self.entries:
            raise ValueError("The matrix editor is empty. Apply a size first.")

        values = []
        for row_index, row in enumerate(self.entries, start=1):
            current_row = []
            for column_index, field in enumerate(row, start=1):
                raw_value = field.text().strip()
                if not raw_value:
                    raise ValueError(
                        f"Cell at row {row_index}, column {column_index} is empty."
                    )
                try:
                    value = float(raw_value)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid number at row {row_index}, column {column_index}: {raw_value}"
                    ) from error
                if not np.isfinite(value):
                    raise ValueError("Matrix values must be finite numbers.")
                current_row.append(value)
            values.append(current_row)
        return np.asarray(values, dtype=float)

    def solve_gauss_jordan(self):
        try:
            matrix = self.read_matrix()
            if matrix.shape[0] != matrix.shape[1] - 1:
                raise ValueError(
                    "The current Gauss–Jordan implementation expects the same number "
                    "of equations and variables. Set a square coefficient matrix."
                )

            self.add_to_history(matrix, "equations")
            solution = gauss_jordan(matrix.copy())

            if isinstance(solution, str):
                if solution == "no solution":
                    self.show_result(
                        "No solution",
                        "The system is inconsistent, so no vector satisfies all equations.",
                    )
                    self.set_status("System is inconsistent")
                else:
                    self.show_result(
                        "Infinitely many solutions",
                        "The system does not have a unique solution.",
                    )
                    self.set_status("System has infinitely many solutions")
            else:
                lines = ["Unique solution", ""]
                lines.extend(
                    f"x{index + 1}  =  {value:.6g}"
                    for index, value in enumerate(solution)
                )
                self.show_result("Solution", "\n".join(lines))
                self.set_status("System solved successfully")
        except Exception as error:
            self.show_error(error)

    def matrix_inverse(self):
        try:
            matrix = self.read_matrix()
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Matrix inversion requires a square matrix.")

            self.add_to_history(matrix, "square")
            inverse = invert_matrix(matrix.copy())
            formatted = self.format_matrix(inverse)
            self.show_result("Inverse matrix", f"A⁻¹\n\n{formatted}")
            self.set_status("Inverse calculated successfully")
        except Exception as error:
            self.show_error(error)

    def matrix_determinant(self):
        try:
            matrix = self.read_matrix()
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("A determinant is defined only for square matrices.")

            self.add_to_history(matrix, "square")
            determinant = det_by_row_ops(matrix.copy())
            self.show_result("Determinant", f"det(A)  =  {determinant:.8g}")
            self.set_status("Determinant calculated successfully")
        except Exception as error:
            self.show_error(error)

    @staticmethod
    def format_matrix(matrix):
        return "\n".join(
            "  ".join(f"{value:10.4f}" for value in row)
            for row in matrix
        )

    def show_result(self, title, content):
        self.result_title.setText(title)
        self.result_text.setPlainText(content)
        self.result_panel.show()

    def copy_result(self):
        content = self.result_text.toPlainText().strip()
        if not content:
            self.set_status("There is no result to copy")
            return
        QApplication.clipboard().setText(content)
        self.set_status("Result copied to clipboard")

    def show_error(self, error):
        self.set_status("Operation could not be completed")
        QMessageBox.warning(self, "Unable to complete operation", str(error))

    def set_status(self, message):
        self.status_label.setText(message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Linear Algebra Studio")
    app.setFont(QFont("Segoe UI", 10))

    window = MatrixApp()
    window.show()
    sys.exit(app.exec())
