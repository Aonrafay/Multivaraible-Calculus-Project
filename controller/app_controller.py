"""Application controller wiring models and views for the visualizer."""

from typing import List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QObject, QTimer
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from model import CalculusEngine, CriticalPointAnalyzer, FunctionRegistry, GradientDescentOptimizer
from model.analysis import CriticalPoint
from model.functions import FunctionDefinition
from model.optimization import OptimizationResult, OptimizerType


_OPTIMIZER_MAP = {
    "Standard": OptimizerType.STANDARD,
    "Momentum": OptimizerType.MOMENTUM,
    "AdaGrad": OptimizerType.ADAGRAD,
    "Nesterov": OptimizerType.NESTEROV,
}


class AppController(QObject):
    """Controller that coordinates UI events with model computations."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.window = main_window

        self.registry = FunctionRegistry()
        self.calculus = CalculusEngine()
        self.optimizer = GradientDescentOptimizer(self.calculus)
        self.analyzer = CriticalPointAnalyzer(self.calculus)

        self.current_function: Optional[FunctionDefinition] = None
        self.current_result: Optional[OptimizationResult] = None
        self.critical_points: List[CriticalPoint] = []

        self.show_tangent_plane = False
        self.show_gradient_field = False
        self.show_critical_points = True
        self.show_descent_path = True

        self._grid_cache = {}

        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(120)
        self.animation_timer.timeout.connect(self._animate_step)
        self.animation_index = 0

        self._connect_signals()
        self._load_functions()

    def _connect_signals(self):
        panel = self.window.controls_panel
        panel.function_changed.connect(self.on_function_changed)
        panel.run_descent_requested.connect(self.on_run_descent)
        panel.reset_requested.connect(self.on_reset)
        panel.animate_requested.connect(self.on_animate_toggle)
        panel.find_critical_points_requested.connect(self.on_find_critical_points)
        panel.settings_changed.connect(self.on_settings_changed)
        panel.view_options_changed.connect(self.on_view_options_changed)

        self.window.action_reset.triggered.connect(self.on_reset)
        self.window.action_export_plot.triggered.connect(self.on_export_plot)
        self.window.action_export_contour.triggered.connect(self.on_export_contour)
        self.window.action_quit.triggered.connect(self.window.close)

        self.window.action_show_tangent.toggled.connect(self.on_menu_toggle_tangent)
        self.window.action_show_gradient_field.toggled.connect(self.on_menu_toggle_gradient)
        self.window.action_show_critical_points.toggled.connect(self.on_menu_toggle_critical)
        self.window.action_show_descent_path.toggled.connect(self.on_menu_toggle_descent)

        self.window.action_about.triggered.connect(self.on_about)
        self.window.action_math_help.triggered.connect(self.on_math_help)

        self.window.contour_widget.point_clicked.connect(self.on_contour_clicked)

    def _load_functions(self):
        names = self.registry.get_all_names()
        self.window.controls_panel.set_functions(names)
        if names:
            self.on_function_changed(names[0])
            self.window.controls_panel.set_selected_function(names[0])

    def on_function_changed(self, name: str):
        try:
            self.current_function = self.registry.get_function(name)
        except KeyError:
            self.current_function = None
            return

        func_def = self.current_function
        self.window.controls_panel.set_function_details(
            func_def.expression,
            func_def.description,
        )
        self.window.controls_panel.set_domain_range(
            func_def.default_x_range[0],
            func_def.default_x_range[1],
            func_def.default_y_range[0],
            func_def.default_y_range[1],
        )
        self.window.controls_panel.set_start_point(0.0, 0.0)

        self.current_result = None
        self.critical_points = []
        self.window.info_panel.set_function_info(func_def)
        self.window.info_panel.clear_details()
        self._invalidate_cache()
        self._update_plots()
        self.window.set_status("Ready")

    def on_settings_changed(self):
        if self.animation_timer.isActive():
            self._stop_animation()
        self._update_plots()

    def on_view_options_changed(self):
        (
            self.show_tangent_plane,
            self.show_gradient_field,
            self.show_critical_points,
            self.show_descent_path,
        ) = self.window.controls_panel.get_view_options()

        self._sync_menu_toggles()
        self._update_plots()

    def on_menu_toggle_tangent(self, checked: bool):
        self.show_tangent_plane = checked
        self._sync_controls_toggles()
        self._update_plots()

    def on_menu_toggle_gradient(self, checked: bool):
        self.show_gradient_field = checked
        self._sync_controls_toggles()
        self._update_plots()

    def on_menu_toggle_critical(self, checked: bool):
        self.show_critical_points = checked
        self._sync_controls_toggles()
        self._update_plots()

    def on_menu_toggle_descent(self, checked: bool):
        self.show_descent_path = checked
        self._sync_controls_toggles()
        self._update_plots()

    def on_run_descent(self):
        if not self.current_function:
            return

        if self.animation_timer.isActive():
            self._stop_animation()

        start_x, start_y = self.window.controls_panel.get_start_point()
        learning_rate, optimizer_label, momentum, max_iter, tolerance = (
            self.window.controls_panel.get_optimizer_settings()
        )
        use_analytical = self.window.controls_panel.use_analytical()

        self.optimizer.max_iterations = max_iter
        self.optimizer.tolerance = tolerance

        optimizer_type = _OPTIMIZER_MAP.get(optimizer_label, OptimizerType.STANDARD)
        self.current_result = self.optimizer.optimize(
            self.current_function,
            start_x,
            start_y,
            learning_rate,
            optimizer_type,
            momentum,
            use_analytical,
        )

        self._update_info_panel()
        self._update_plots()

        status = "Converged" if self.current_result.converged else "Did not converge"
        self.window.set_status(f"Optimization complete: {status}")

    def on_find_critical_points(self):
        if not self.current_function:
            return

        if self.animation_timer.isActive():
            self._stop_animation()

        x_range, y_range = self._get_domain_ranges()
        use_analytical = self.window.controls_panel.use_analytical()

        self.analyzer.grid_resolution = max(20, self.window.controls_panel.get_resolution())
        self.critical_points = self.analyzer.find_critical_points(
            self.current_function,
            x_range,
            y_range,
            use_analytical,
        )

        self._update_info_panel()
        self._update_plots()

        self.window.set_status(
            f"Found {len(self.critical_points)} critical points"
        )

    def on_reset(self):
        if not self.current_function:
            return

        if self.animation_timer.isActive():
            self._stop_animation()

        func_def = self.current_function
        self.window.controls_panel.set_domain_range(
            func_def.default_x_range[0],
            func_def.default_x_range[1],
            func_def.default_y_range[0],
            func_def.default_y_range[1],
        )
        self.window.controls_panel.set_start_point(0.0, 0.0)
        self.current_result = None
        self.critical_points = []
        self.window.info_panel.clear_details()
        self._invalidate_cache()
        self._update_plots()
        self.window.set_status("Reset to defaults")

    def on_animate_toggle(self):
        if not self.current_result or not self.current_result.path:
            self.window.set_status("Run descent first to animate")
            return

        if self.animation_timer.isActive():
            self._stop_animation()
            return

        self.animation_index = 1
        self.window.controls_panel.set_animation_active(True)
        self.window.set_status("Animating descent path")
        self.animation_timer.start()

    def _animate_step(self):
        if not self.current_result:
            self._stop_animation()
            return

        path = self.current_result.path
        if self.animation_index > len(path):
            self._stop_animation()
            return

        partial_path = path[: self.animation_index]
        self._update_plots(path_override=partial_path)
        self.animation_index += 1

    def _stop_animation(self):
        self.animation_timer.stop()
        self.window.controls_panel.set_animation_active(False)
        self.window.set_status("Animation stopped")

    def on_contour_clicked(self, x: float, y: float):
        self.window.controls_panel.set_start_point(x, y)
        self.window.set_status(f"Start point set to ({x:.2f}, {y:.2f})")
        self._update_plots()

    def on_export_plot(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self.window,
            "Export 3D Plot",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg);;PDF (*.pdf)",
        )
        if not file_path:
            return
        self.window.plot_widget.figure.savefig(file_path, dpi=200, bbox_inches="tight")
        self.window.set_status(f"Saved 3D plot to {file_path}")

    def on_export_contour(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self.window,
            "Export Contour Map",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg);;PDF (*.pdf)",
        )
        if not file_path:
            return
        self.window.contour_widget.figure.savefig(
            file_path, dpi=200, bbox_inches="tight"
        )
        self.window.set_status(f"Saved contour map to {file_path}")

    def on_about(self):
        QMessageBox.information(
            self.window,
            "About",
            "3D Gradient Descent Visualizer\nMVC CCP project",
        )

    def on_math_help(self):
        QMessageBox.information(
            self.window,
            "Math Concepts",
            "This visualizer shows gradient descent on multivariable functions.\n"
            "Use the contour map and 3D surface to compare paths, gradients,\n"
            "and critical points.",
        )

    def _update_info_panel(self):
        if not self.current_function:
            return

        if self.current_result:
            self.window.info_panel.set_optimization_info(
                self._format_optimization_summary(self.current_result)
            )
        else:
            self.window.info_panel.set_optimization_info("")

        if self.critical_points:
            self.window.info_panel.set_critical_points_info(
                self._format_critical_points_summary(self.critical_points)
            )
        else:
            self.window.info_panel.set_critical_points_info("")

    def _format_optimization_summary(self, result: OptimizationResult) -> str:
        lines = [
            f"Optimizer: {result.optimizer_type.value}",
            f"Learning rate: {result.learning_rate:.6f}",
            f"Iterations: {result.iterations}",
            f"Converged: {result.converged}",
            f"Final point: ({result.final_point[0]:.6f}, {result.final_point[1]:.6f})",
            f"Final value: {result.final_value:.6f}",
            "Final gradient:",
            f"  dfdx = {result.final_gradient[0]:.6f}",
            f"  dfdy = {result.final_gradient[1]:.6f}",
            f"Gradient magnitude: {result.final_gradient_magnitude:.8f}",
        ]
        return "\n".join(lines)

    def _format_critical_points_summary(self, points: List[CriticalPoint]) -> str:
        lines = []
        for cp in points:
            lines.append(
                f"{cp.point_type.value}: ({cp.x:.6f}, {cp.y:.6f})"
            )
            lines.append(f"  f(x,y) = {cp.z:.6f}")
            lines.append(f"  |grad| = {cp.gradient_magnitude:.8f}")
            lines.append(
                f"  Hessian det = {cp.determinant:.6f}, eigenvalues = ({cp.eigenvalues[0]:.6f}, {cp.eigenvalues[1]:.6f})"
            )
        return "\n".join(lines) if lines else "None found"

    def _get_domain_ranges(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        x_min, x_max, y_min, y_max = self.window.controls_panel.get_domain_range()
        x_min, x_max = self._normalize_range(x_min, x_max)
        y_min, y_max = self._normalize_range(y_min, y_max)

        self.window.controls_panel.set_domain_range(x_min, x_max, y_min, y_max)
        return (x_min, x_max), (y_min, y_max)

    def _normalize_range(self, min_val: float, max_val: float) -> Tuple[float, float]:
        if min_val == max_val:
            max_val = min_val + 1.0
        if min_val > max_val:
            min_val, max_val = max_val, min_val
        return min_val, max_val

    def _update_plots(self, path_override: Optional[List[Tuple[float, float, float]]] = None):
        if not self.current_function:
            return

        x_range, y_range = self._get_domain_ranges()
        resolution = self.window.controls_panel.get_resolution()
        use_analytical = self.window.controls_panel.use_analytical()

        X, Y, Z = self._get_grid(self.current_function, x_range, y_range, resolution)
        if X is None:
            return

        descent_path = None
        if self.current_result and self.show_descent_path:
            descent_path = self.current_result.path
        if path_override is not None:
            descent_path = path_override

        start_point = self.window.controls_panel.get_start_point()
        tangent_point = None
        if self.show_tangent_plane:
            if descent_path:
                tangent_point = (descent_path[-1][0], descent_path[-1][1])
            else:
                tangent_point = start_point

        gradient_U = gradient_V = None
        if self.show_gradient_field:
            gradient_U, gradient_V = self._get_gradient_field(
                self.current_function, X, Y, use_analytical
            )

        self.window.plot_widget.refresh_plot(
            self.current_function,
            x_range,
            y_range,
            descent_path=descent_path,
            critical_points=self.critical_points,
            tangent_point=tangent_point,
            show_gradient_field=self.show_gradient_field,
            show_tangent_plane=self.show_tangent_plane,
            show_descent_path=self.show_descent_path,
            show_critical_points=self.show_critical_points,
            calculus_engine=self.calculus,
            gradient_U=gradient_U,
            gradient_V=gradient_V,
            gradient_X=X,
            gradient_Y=Y,
            gradient_Z=Z,
        )

        self.window.contour_widget.refresh_plot(
            self.current_function,
            x_range,
            y_range,
            resolution=resolution,
            X=X,
            Y=Y,
            Z=Z,
            descent_path=descent_path,
            critical_points=self.critical_points,
            start_point=start_point,
            show_descent_path=self.show_descent_path,
            show_critical_points=self.show_critical_points,
            show_gradient_field=self.show_gradient_field,
            gradient_U=gradient_U,
            gradient_V=gradient_V,
        )

    def _get_grid(
        self,
        func_def: FunctionDefinition,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        resolution: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        cache_key = (func_def.name, x_range, y_range, resolution)
        if self._grid_cache.get("key") == cache_key:
            return (
                self._grid_cache.get("X"),
                self._grid_cache.get("Y"),
                self._grid_cache.get("Z"),
            )

        try:
            x = np.linspace(x_range[0], x_range[1], resolution)
            y = np.linspace(y_range[0], y_range[1], resolution)
            X, Y = np.meshgrid(x, y)
            Z = func_def.evaluate(X, Y)
            Z = np.where(np.isfinite(Z), Z, np.nan)
        except Exception:
            self.window.set_status("Failed to evaluate function for plotting")
            return None, None, None

        self._grid_cache = {"key": cache_key, "X": X, "Y": Y, "Z": Z}
        return X, Y, Z

    def _get_gradient_field(
        self,
        func_def: FunctionDefinition,
        X: np.ndarray,
        Y: np.ndarray,
        use_analytical: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        grad_key = (self._grid_cache.get("key"), use_analytical)
        if self._grid_cache.get("grad_key") == grad_key:
            U = self._grid_cache.get("U")
            V = self._grid_cache.get("V")
            if U is not None and V is not None:
                return U, V

        U, V = self.calculus.compute_gradient_field(func_def, X, Y, use_analytical)
        self._grid_cache["U"] = U
        self._grid_cache["V"] = V
        self._grid_cache["grad_key"] = grad_key
        return U, V

    def _invalidate_cache(self):
        self._grid_cache = {}

    def _sync_menu_toggles(self):
        self.window.action_show_tangent.blockSignals(True)
        self.window.action_show_gradient_field.blockSignals(True)
        self.window.action_show_critical_points.blockSignals(True)
        self.window.action_show_descent_path.blockSignals(True)

        self.window.action_show_tangent.setChecked(self.show_tangent_plane)
        self.window.action_show_gradient_field.setChecked(self.show_gradient_field)
        self.window.action_show_critical_points.setChecked(self.show_critical_points)
        self.window.action_show_descent_path.setChecked(self.show_descent_path)

        self.window.action_show_tangent.blockSignals(False)
        self.window.action_show_gradient_field.blockSignals(False)
        self.window.action_show_critical_points.blockSignals(False)
        self.window.action_show_descent_path.blockSignals(False)

    def _sync_controls_toggles(self):
        self.window.controls_panel.set_view_options(
            self.show_tangent_plane,
            self.show_gradient_field,
            self.show_critical_points,
            self.show_descent_path,
        )
