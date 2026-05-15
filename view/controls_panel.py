"""Controls panel for the Gradient Descent Visualizer."""

from typing import List, Tuple

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ControlsPanel(QWidget):
    """Panel with controls for function selection and optimization settings."""

    run_descent_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    animate_requested = pyqtSignal()
    find_critical_points_requested = pyqtSignal()
    function_changed = pyqtSignal(str)
    settings_changed = pyqtSignal()
    view_options_changed = pyqtSignal()

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel("Controls")
        header.setObjectName("headerLabel")
        layout.addWidget(header)

        self.function_group = QGroupBox("Function")
        function_layout = QVBoxLayout(self.function_group)
        function_layout.setSpacing(4)

        self.function_combo = QComboBox()
        self.expression_label = QLabel("")
        self.expression_label.setObjectName("expressionLabel")
        self.description_label = QLabel("")
        self.description_label.setWordWrap(True)

        function_layout.addWidget(self.function_combo)
        function_layout.addWidget(self.expression_label)
        function_layout.addWidget(self.description_label)

        layout.addWidget(self.function_group)

        self.domain_group = QGroupBox("Domain")
        domain_layout = QGridLayout(self.domain_group)

        self.x_min_spin = self._make_double_spin(-10.0, 10.0, 0.1, -3.0)
        self.x_max_spin = self._make_double_spin(-10.0, 10.0, 0.1, 3.0)
        self.y_min_spin = self._make_double_spin(-10.0, 10.0, 0.1, -3.0)
        self.y_max_spin = self._make_double_spin(-10.0, 10.0, 0.1, 3.0)

        domain_layout.addWidget(QLabel("X min"), 0, 0)
        domain_layout.addWidget(self.x_min_spin, 0, 1)
        domain_layout.addWidget(QLabel("X max"), 0, 2)
        domain_layout.addWidget(self.x_max_spin, 0, 3)
        domain_layout.addWidget(QLabel("Y min"), 1, 0)
        domain_layout.addWidget(self.y_min_spin, 1, 1)
        domain_layout.addWidget(QLabel("Y max"), 1, 2)
        domain_layout.addWidget(self.y_max_spin, 1, 3)

        layout.addWidget(self.domain_group)

        self.start_group = QGroupBox("Start Point")
        start_layout = QFormLayout(self.start_group)
        self.start_x_spin = self._make_double_spin(-10.0, 10.0, 0.1, 0.0)
        self.start_y_spin = self._make_double_spin(-10.0, 10.0, 0.1, 0.0)
        start_layout.addRow("X", self.start_x_spin)
        start_layout.addRow("Y", self.start_y_spin)
        layout.addWidget(self.start_group)

        self.optim_group = QGroupBox("Optimization")
        optim_layout = QGridLayout(self.optim_group)

        self.learning_rate_spin = self._make_double_spin(0.0001, 1.0, 0.001, 0.02)
        self.learning_rate_spin.setDecimals(4)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Standard", "Momentum", "AdaGrad", "Nesterov"])
        self.momentum_spin = self._make_double_spin(0.0, 0.99, 0.01, 0.9)
        self.max_iter_spin = self._make_int_spin(10, 10000, 500)
        self.tolerance_spin = self._make_double_spin(1e-10, 1e-2, 1e-6, 1e-6)
        self.tolerance_spin.setDecimals(8)
        self.resolution_spin = self._make_int_spin(20, 200, 80)

        optim_layout.addWidget(QLabel("Learning rate"), 0, 0)
        optim_layout.addWidget(self.learning_rate_spin, 0, 1)
        optim_layout.addWidget(QLabel("Optimizer"), 0, 2)
        optim_layout.addWidget(self.optimizer_combo, 0, 3)

        optim_layout.addWidget(QLabel("Momentum"), 1, 0)
        optim_layout.addWidget(self.momentum_spin, 1, 1)
        optim_layout.addWidget(QLabel("Max iterations"), 1, 2)
        optim_layout.addWidget(self.max_iter_spin, 1, 3)

        optim_layout.addWidget(QLabel("Tolerance"), 2, 0)
        optim_layout.addWidget(self.tolerance_spin, 2, 1)
        optim_layout.addWidget(QLabel("Grid resolution"), 2, 2)
        optim_layout.addWidget(self.resolution_spin, 2, 3)

        self.analytical_checkbox = QCheckBox("Use analytical derivatives")
        self.analytical_checkbox.setChecked(True)
        optim_layout.addWidget(self.analytical_checkbox, 3, 0, 1, 4)

        layout.addWidget(self.optim_group)

        self.view_group = QGroupBox("Visualization")
        view_layout = QVBoxLayout(self.view_group)
        self.show_tangent_checkbox = QCheckBox("Show tangent plane")
        self.show_gradient_checkbox = QCheckBox("Show gradient field")
        self.show_critical_checkbox = QCheckBox("Show critical points")
        self.show_descent_checkbox = QCheckBox("Show descent path")
        self.show_critical_checkbox.setChecked(True)
        self.show_descent_checkbox.setChecked(True)

        view_layout.addWidget(self.show_tangent_checkbox)
        view_layout.addWidget(self.show_gradient_checkbox)
        view_layout.addWidget(self.show_critical_checkbox)
        view_layout.addWidget(self.show_descent_checkbox)

        layout.addWidget(self.view_group)

        buttons_layout = QGridLayout()
        buttons_layout.setHorizontalSpacing(6)
        buttons_layout.setVerticalSpacing(6)
        self.run_button = QPushButton("Run Descent")
        self.run_button.setObjectName("runButton")
        self.animate_button = QPushButton("Animate")
        self.animate_button.setObjectName("animateButton")
        self.find_critical_button = QPushButton("Find Critical Points")
        self.find_critical_button.setObjectName("findCriticalButton")
        self.reset_button = QPushButton("Reset")
        self.reset_button.setObjectName("resetButton")

        buttons_layout.addWidget(self.run_button, 0, 0)
        buttons_layout.addWidget(self.animate_button, 0, 1)
        buttons_layout.addWidget(self.find_critical_button, 1, 0)
        buttons_layout.addWidget(self.reset_button, 1, 1)

        layout.addLayout(buttons_layout)
        layout.addStretch(1)

        self._connect_signals()

    def _connect_signals(self):
        self.function_combo.currentTextChanged.connect(self.function_changed.emit)

        for spin in [
            self.x_min_spin,
            self.x_max_spin,
            self.y_min_spin,
            self.y_max_spin,
            self.start_x_spin,
            self.start_y_spin,
            self.learning_rate_spin,
            self.momentum_spin,
            self.max_iter_spin,
            self.tolerance_spin,
            self.resolution_spin,
        ]:
            spin.valueChanged.connect(self.settings_changed.emit)

        self.optimizer_combo.currentTextChanged.connect(self.settings_changed.emit)
        self.analytical_checkbox.toggled.connect(self.settings_changed.emit)

        self.show_tangent_checkbox.toggled.connect(self.view_options_changed.emit)
        self.show_gradient_checkbox.toggled.connect(self.view_options_changed.emit)
        self.show_critical_checkbox.toggled.connect(self.view_options_changed.emit)
        self.show_descent_checkbox.toggled.connect(self.view_options_changed.emit)

        self.run_button.clicked.connect(self.run_descent_requested.emit)
        self.reset_button.clicked.connect(self.reset_requested.emit)
        self.animate_button.clicked.connect(self.animate_requested.emit)
        self.find_critical_button.clicked.connect(self.find_critical_points_requested.emit)

    def _make_double_spin(self, min_val: float, max_val: float, step: float, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setDecimals(4)
        spin.setValue(value)
        return spin

    def _make_int_spin(self, min_val: int, max_val: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(value)
        return spin

    def set_functions(self, names: List[str]):
        self.function_combo.blockSignals(True)
        self.function_combo.clear()
        self.function_combo.addItems(names)
        self.function_combo.blockSignals(False)

    def set_selected_function(self, name: str):
        self.function_combo.blockSignals(True)
        self.function_combo.setCurrentText(name)
        self.function_combo.blockSignals(False)

    def set_function_details(self, expression: str, description: str):
        self.expression_label.setText(expression)
        self.description_label.setText(description)

    def set_domain_range(self, x_min: float, x_max: float, y_min: float, y_max: float):
        self.x_min_spin.blockSignals(True)
        self.x_max_spin.blockSignals(True)
        self.y_min_spin.blockSignals(True)
        self.y_max_spin.blockSignals(True)

        self.x_min_spin.setValue(x_min)
        self.x_max_spin.setValue(x_max)
        self.y_min_spin.setValue(y_min)
        self.y_max_spin.setValue(y_max)

        self.x_min_spin.blockSignals(False)
        self.x_max_spin.blockSignals(False)
        self.y_min_spin.blockSignals(False)
        self.y_max_spin.blockSignals(False)

    def get_domain_range(self) -> Tuple[float, float, float, float]:
        return (
            self.x_min_spin.value(),
            self.x_max_spin.value(),
            self.y_min_spin.value(),
            self.y_max_spin.value(),
        )

    def set_start_point(self, x0: float, y0: float):
        self.start_x_spin.blockSignals(True)
        self.start_y_spin.blockSignals(True)

        self.start_x_spin.setValue(x0)
        self.start_y_spin.setValue(y0)

        self.start_x_spin.blockSignals(False)
        self.start_y_spin.blockSignals(False)

    def get_start_point(self) -> Tuple[float, float]:
        return (self.start_x_spin.value(), self.start_y_spin.value())

    def get_optimizer_settings(self) -> Tuple[float, str, float, int, float]:
        return (
            self.learning_rate_spin.value(),
            self.optimizer_combo.currentText(),
            self.momentum_spin.value(),
            self.max_iter_spin.value(),
            self.tolerance_spin.value(),
        )

    def get_resolution(self) -> int:
        return self.resolution_spin.value()

    def use_analytical(self) -> bool:
        return self.analytical_checkbox.isChecked()

    def get_view_options(self) -> Tuple[bool, bool, bool, bool]:
        return (
            self.show_tangent_checkbox.isChecked(),
            self.show_gradient_checkbox.isChecked(),
            self.show_critical_checkbox.isChecked(),
            self.show_descent_checkbox.isChecked(),
        )

    def set_view_options(
        self,
        show_tangent: bool,
        show_gradient: bool,
        show_critical: bool,
        show_descent: bool,
    ):
        self.show_tangent_checkbox.blockSignals(True)
        self.show_gradient_checkbox.blockSignals(True)
        self.show_critical_checkbox.blockSignals(True)
        self.show_descent_checkbox.blockSignals(True)

        self.show_tangent_checkbox.setChecked(show_tangent)
        self.show_gradient_checkbox.setChecked(show_gradient)
        self.show_critical_checkbox.setChecked(show_critical)
        self.show_descent_checkbox.setChecked(show_descent)

        self.show_tangent_checkbox.blockSignals(False)
        self.show_gradient_checkbox.blockSignals(False)
        self.show_critical_checkbox.blockSignals(False)
        self.show_descent_checkbox.blockSignals(False)

    def set_animation_active(self, active: bool):
        self.animate_button.setText("Stop" if active else "Animate")
