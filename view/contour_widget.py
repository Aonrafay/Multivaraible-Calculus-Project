"""2D contour plot widget for the Gradient Descent Visualizer."""

from typing import List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QSizePolicy, QWidget

from model.analysis import CriticalPoint, CriticalPointType
from model.functions import FunctionDefinition


class ContourWidget(QWidget):
    """Matplotlib-based 2D contour plot embedded in a PyQt5 widget."""

    point_clicked = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.figure = Figure(figsize=(4, 3), dpi=100, facecolor="#1e1e2e")
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self._style_axes()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self._contour = None
        self._path_artist = None
        self._critical_artists = []
        self._gradient_artists = []

        self.canvas.mpl_connect("button_press_event", self._on_click)

    def _style_axes(self):
        """Apply dark theme styling to the 2D axes."""
        self.ax.set_facecolor("#11111b")
        self.ax.tick_params(colors="#a6adc8", labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color("#313244")
        self.ax.grid(True, alpha=0.25, color="#585b70")

    def _on_click(self, event):
        """Emit a signal when the user clicks inside the axes."""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.point_clicked.emit(float(event.xdata), float(event.ydata))

    def _plot_contours(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
        """Draw filled contours and contour lines."""
        levels = 30
        self._contour = self.ax.contourf(
            X,
            Y,
            Z,
            levels=levels,
            cmap="viridis",
            alpha=0.85,
        )
        self.ax.contour(
            X,
            Y,
            Z,
            levels=levels,
            colors="#313244",
            linewidths=0.6,
            alpha=0.7,
        )

    def plot_descent_path(self, path: List[Tuple[float, float, float]]):
        """Plot the gradient descent path on the contour map."""
        if not path:
            return

        xs = [p[0] for p in path]
        ys = [p[1] for p in path]

        self.ax.plot(xs, ys, color="#a6e3a1", linewidth=2.2, zorder=5)
        self.ax.scatter(
            [xs[0]],
            [ys[0]],
            color="#f38ba8",
            s=60,
            zorder=6,
            edgecolors="white",
            linewidths=1.2,
            label="Start",
        )
        self.ax.scatter(
            [xs[-1]],
            [ys[-1]],
            color="#a6e3a1",
            s=70,
            marker="*",
            zorder=6,
            edgecolors="white",
            linewidths=1.2,
            label="End",
        )

    def plot_start_point(self, x0: float, y0: float):
        """Plot the starting point when no path is available."""
        self.ax.scatter(
            [x0],
            [y0],
            color="#cba6f7",
            s=60,
            marker="o",
            zorder=5,
            edgecolors="white",
            linewidths=1.0,
            label="Start",
        )

    def plot_critical_points(self, critical_points: List[CriticalPoint]):
        """Plot critical points on the contour map."""
        if not critical_points:
            return

        color_map = {
            CriticalPointType.LOCAL_MINIMUM: "#a6e3a1",
            CriticalPointType.LOCAL_MAXIMUM: "#f38ba8",
            CriticalPointType.SADDLE_POINT: "#f9e2af",
            CriticalPointType.DEGENERATE: "#cba6f7",
            CriticalPointType.UNKNOWN: "#585b70",
        }

        marker_map = {
            CriticalPointType.LOCAL_MINIMUM: "v",
            CriticalPointType.LOCAL_MAXIMUM: "^",
            CriticalPointType.SADDLE_POINT: "D",
            CriticalPointType.DEGENERATE: "o",
            CriticalPointType.UNKNOWN: "x",
        }

        for cp in critical_points:
            color = color_map.get(cp.point_type, "#585b70")
            marker = marker_map.get(cp.point_type, "x")
            self.ax.scatter(
                [cp.x],
                [cp.y],
                color=color,
                s=80,
                marker=marker,
                zorder=8,
                edgecolors="white",
                linewidths=1.2,
            )

    def plot_gradient_field(
        self,
        U: np.ndarray,
        V: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        scale: float = 0.3,
    ):
        """Plot gradient vector field arrows on the contour map."""
        stride = max(1, U.shape[0] // 15)
        Xs = X[::stride, ::stride]
        Ys = Y[::stride, ::stride]
        Us = U[::stride, ::stride] * scale
        Vs = V[::stride, ::stride] * scale

        self.ax.quiver(
            Xs,
            Ys,
            Us,
            Vs,
            color="#89b4fa",
            alpha=0.6,
            width=0.002,
        )

    def refresh_plot(
        self,
        func_def: FunctionDefinition,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        resolution: int = 80,
        X: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
        descent_path: Optional[List[Tuple[float, float, float]]] = None,
        critical_points: Optional[List[CriticalPoint]] = None,
        start_point: Optional[Tuple[float, float]] = None,
        show_descent_path: bool = True,
        show_critical_points: bool = True,
        show_gradient_field: bool = False,
        gradient_U: Optional[np.ndarray] = None,
        gradient_V: Optional[np.ndarray] = None,
    ):
        """Refresh the contour plot with overlays."""
        if X is None or Y is None or Z is None:
            x = np.linspace(x_range[0], x_range[1], resolution)
            y = np.linspace(y_range[0], y_range[1], resolution)
            X, Y = np.meshgrid(x, y)
            Z = func_def.evaluate(X, Y)

        self.ax.clear()
        self._style_axes()
        self._plot_contours(X, Y, Z)

        if show_gradient_field and gradient_U is not None and gradient_V is not None:
            self.plot_gradient_field(gradient_U, gradient_V, X, Y)

        if show_descent_path and descent_path:
            self.plot_descent_path(descent_path)
        elif start_point is not None:
            self.plot_start_point(start_point[0], start_point[1])

        if show_critical_points and critical_points:
            self.plot_critical_points(critical_points)

        self.ax.set_xlabel("X", fontsize=9, color="#cdd6f4")
        self.ax.set_ylabel("Y", fontsize=9, color="#cdd6f4")
        self.ax.set_title("Contour Map", fontsize=10, color="#f9e2af", pad=6)
        self.ax.set_aspect("auto")

        self.canvas.draw_idle()
