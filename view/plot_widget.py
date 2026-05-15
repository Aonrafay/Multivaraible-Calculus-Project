"""3D surface plot widget for the Gradient Descent Visualizer.

This module provides a matplotlib-based 3D surface plot embedded in a PyQt5
widget. It renders the function surface, tangent planes, gradient descent paths,
critical points, and gradient vector fields.
"""

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from typing import List, Tuple, Optional

from model.functions import FunctionDefinition
from model.analysis import CriticalPoint, CriticalPointType


class Plot3DWidget(QWidget):
    """Matplotlib 3D surface plot widget embedded in PyQt5.

    Displays:
    - 3D surface of the selected function
    - Tangent plane at a specified point
    - Gradient descent path (line + markers)
    - Critical points (colored markers)
    - Gradient vector field (optional arrows)
    """

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create matplotlib figure and 3D axes
        self.figure = Figure(figsize=(8, 6), dpi=100, facecolor='#1e1e2e')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # 3D axes
        self.ax = self.figure.add_subplot(111, projection='3d')
        self._style_axes()

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # State
        self._surface = None
        self._tangent_plane = None
        self._descent_path = None
        self._critical_points_artists = []
        self._gradient_field_artists = []

    def _style_axes(self):
        """Apply dark theme styling to the 3D axes."""
        self.ax.set_facecolor('#11111b')
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#313244')
        self.ax.yaxis.pane.set_edgecolor('#313244')
        self.ax.zaxis.pane.set_edgecolor('#313244')
        self.ax.tick_params(colors='#a6adc8', labelsize=8)
        self.ax.xaxis.label.set_color('#cdd6f4')
        self.ax.yaxis.label.set_color('#cdd6f4')
        self.ax.zaxis.label.set_color('#cdd6f4')
        self.ax.grid(True, alpha=0.2, color='#585b70')

    def plot_surface(
        self,
        func_def: FunctionDefinition,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        resolution: int = 60,
    ):
        """Plot the 3D surface of the function.

        Args:
            func_def: The function definition to plot.
            x_range: (x_min, x_max) domain.
            y_range: (y_min, y_max) domain.
            resolution: Number of grid points per axis.
        """
        self.ax.clear()
        self._style_axes()

        # Create mesh grid
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = func_def.evaluate(X, Y)

        # Plot surface with gradient coloring
        self._surface = self.ax.plot_surface(
            X, Y, Z,
            cmap='viridis',
            alpha=0.7,
            edgecolor='none',
            antialiased=True,
            rstride=2,
            cstride=2,
        )

        # Labels
        self.ax.set_xlabel('X', fontsize=10)
        self.ax.set_ylabel('Y', fontsize=10)
        self.ax.set_zlabel('f(x,y)', fontsize=10)
        self.ax.set_title(func_def.expression, color='#f9e2af', fontsize=12, pad=10)

        self.canvas.draw_idle()

    def plot_tangent_plane(
        self,
        func_def: FunctionDefinition,
        x0: float,
        y0: float,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        plane_size: float = 1.0,
        use_analytical: bool = True,
        calculus_engine=None,
    ):
        """Plot a tangent plane at the specified point on the surface.

        Args:
            func_def: The function definition.
            x0: X-coordinate of the tangent point.
            y0: Y-coordinate of the tangent point.
            x_range: Domain x range.
            y_range: Domain y range.
            plane_size: Size of the tangent plane visualization.
            use_analytical: Whether to use analytical gradient.
            calculus_engine: CalculusEngine instance for computing the plane.
        """
        if calculus_engine is None:
            return

        # Compute tangent plane
        x_plane = np.linspace(x0 - plane_size, x0 + plane_size, 10)
        y_plane = np.linspace(y0 - plane_size, y0 + plane_size, 10)
        X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
        Z_plane = calculus_engine.compute_tangent_plane(
            func_def, x0, y0, X_plane, Y_plane, use_analytical
        )

        # Plot tangent plane
        self._tangent_plane = self.ax.plot_surface(
            X_plane, Y_plane, Z_plane,
            alpha=0.4,
            color='#f38ba8',
            edgecolor='#f38ba8',
            linewidth=0.5,
        )

        # Mark the tangent point
        z0 = func_def.evaluate(np.array([x0]), np.array([y0]))[0]
        self.ax.scatter([x0], [y0], [z0], color='#f38ba8', s=80, zorder=5,
                       edgecolors='white', linewidths=1.5)

        self.canvas.draw_idle()

    def plot_descent_path(
        self,
        path: List[Tuple[float, float, float]],
        show_markers: bool = True,
        show_arrows: bool = True,
    ):
        """Plot the gradient descent path on the 3D surface.

        Args:
            path: List of (x, y, z) tuples representing the descent path.
            show_markers: Whether to show point markers along the path.
            show_arrows: Whether to show direction arrows along the path.
        """
        if not path:
            return

        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        zs = [p[2] for p in path]

        # Plot the path line
        self.ax.plot(xs, ys, zs, color='#a6e3a1', linewidth=2.5, zorder=10,
                    label='Descent Path')

        if show_markers:
            # Start point (large red marker)
            self.ax.scatter([xs[0]], [ys[0]], [zs[0]], color='#f38ba8', s=120,
                          zorder=15, marker='o', edgecolors='white', linewidths=2,
                          label='Start')

            # End point (large green marker)
            self.ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color='#a6e3a1', s=120,
                          zorder=15, marker='*', edgecolors='white', linewidths=2,
                          label='End')

            # Intermediate points (small markers)
            if len(path) > 2:
                step = max(1, len(path) // 15)  # Show ~15 intermediate markers
                mid_xs = xs[1:-1][::step]
                mid_ys = ys[1:-1][::step]
                mid_zs = zs[1:-1][::step]
                self.ax.scatter(mid_xs, mid_ys, mid_zs, color='#f9e2af', s=30,
                              zorder=12, marker='.', alpha=0.8)

        if show_arrows and len(path) > 1:
            # Show a few direction arrows along the path
            step = max(1, len(path) // 8)
            for i in range(0, len(path) - 1, step):
                dx = xs[i + 1] - xs[i]
                dy = ys[i + 1] - ys[i]
                dz = zs[i + 1] - zs[i]
                self.ax.quiver(xs[i], ys[i], zs[i], dx, dy, dz,
                              color='#94e2d5', arrow_length_ratio=0.3,
                              linewidth=1.5, zorder=11)

        self.ax.legend(loc='upper left', fontsize=9, facecolor='#1e1e2e',
                      edgecolor='#313244', labelcolor='#cdd6f4')

        self.canvas.draw_idle()

    def plot_critical_points(
        self,
        critical_points: List[CriticalPoint],
    ):
        """Plot critical points on the 3D surface with color-coded markers.

        Color coding:
        - Local Minimum: green (#a6e3a1)
        - Local Maximum: red (#f38ba8)
        - Saddle Point: yellow (#f9e2af)
        - Degenerate: purple (#cba6f7)

        Args:
            critical_points: List of CriticalPoint objects to display.
        """
        if not critical_points:
            return

        color_map = {
            CriticalPointType.LOCAL_MINIMUM: '#a6e3a1',
            CriticalPointType.LOCAL_MAXIMUM: '#f38ba8',
            CriticalPointType.SADDLE_POINT: '#f9e2af',
            CriticalPointType.DEGENERATE: '#cba6f7',
            CriticalPointType.UNKNOWN: '#585b70',
        }

        marker_map = {
            CriticalPointType.LOCAL_MINIMUM: 'v',  # Down triangle
            CriticalPointType.LOCAL_MAXIMUM: '^',  # Up triangle
            CriticalPointType.SADDLE_POINT: 'D',   # Diamond
            CriticalPointType.DEGENERATE: 'o',     # Circle
            CriticalPointType.UNKNOWN: 'x',
        }

        for cp in critical_points:
            color = color_map.get(cp.point_type, '#585b70')
            marker = marker_map.get(cp.point_type, 'x')
            self.ax.scatter(
                [cp.x], [cp.y], [cp.z],
                color=color, s=100, marker=marker,
                zorder=20, edgecolors='white', linewidths=1.5,
                label=f'{cp.point_type.value} ({cp.x:.2f}, {cp.y:.2f})',
            )

        self.canvas.draw_idle()

    def plot_gradient_field(
        self,
        func_def: FunctionDefinition,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        U: np.ndarray,
        V: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        scale: float = 0.3,
    ):
        """Plot gradient vector field arrows on the 3D surface.

        Args:
            func_def: The function definition.
            x_range: Domain x range.
            y_range: Domain y range.
            U: Gradient x-components (2D array).
            V: Gradient y-components (2D array).
            X: X grid coordinates.
            Y: Y grid coordinates.
            Z: Z grid (surface values).
            scale: Arrow scaling factor.
        """
        # Subsample for clarity
        stride = max(1, U.shape[0] // 12)

        Xs = X[::stride, ::stride]
        Ys = Y[::stride, ::stride]
        Zs = Z[::stride, ::stride]
        Us = U[::stride, ::stride] * scale
        Vs = V[::stride, ::stride] * scale

        # Compute z-component of arrows (zero — gradient is in xy-plane)
        Ws = np.zeros_like(Us)

        self.ax.quiver(
            Xs, Ys, Zs, Us, Vs, Ws,
            color='#89b4fa', alpha=0.6,
            arrow_length_ratio=0.3, linewidth=1.0,
        )

        self.canvas.draw_idle()

    def clear_tangent_plane(self):
        """Remove the tangent plane from the plot."""
        if self._tangent_plane is not None:
            self._tangent_plane.remove()
            self._tangent_plane = None
            self.canvas.draw_idle()

    def clear_descent_path(self):
        """Remove the descent path from the plot (requires full redraw)."""
        # Note: removing individual artists from 3D plots is tricky
        # We'll handle this via full redraws in the controller
        pass

    def clear_all(self):
        """Clear the entire plot."""
        self.ax.clear()
        self._style_axes()
        self._surface = None
        self._tangent_plane = None
        self._descent_path = None
        self._critical_points_artists = []
        self._gradient_field_artists = []
        self.canvas.draw_idle()

    def refresh_plot(
        self,
        func_def: FunctionDefinition,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        descent_path: Optional[List[Tuple[float, float, float]]] = None,
        critical_points: Optional[List[CriticalPoint]] = None,
        tangent_point: Optional[Tuple[float, float]] = None,
        show_gradient_field: bool = False,
        show_tangent_plane: bool = False,
        show_descent_path: bool = True,
        show_critical_points: bool = True,
        calculus_engine=None,
        gradient_U=None,
        gradient_V=None,
        gradient_X=None,
        gradient_Y=None,
        gradient_Z=None,
    ):
        """Full refresh of the 3D plot with all overlays.

        This is the primary method called by the controller to update
        the visualization after any state change.

        Args:
            func_def: The function definition to plot.
            x_range: Domain x range.
            y_range: Domain y range.
            descent_path: Optional descent path points.
            critical_points: Optional critical points to display.
            tangent_point: Optional (x, y) for tangent plane.
            show_gradient_field: Whether to show gradient arrows.
            show_tangent_plane: Whether to show tangent plane.
            show_descent_path: Whether to show descent path.
            show_critical_points: Whether to show critical points.
            calculus_engine: CalculusEngine for tangent plane computation.
            gradient_U, gradient_V, gradient_X, gradient_Y, gradient_Z:
                Gradient field arrays for vector field display.
        """
        # Clear and redraw surface
        self.plot_surface(func_def, x_range, y_range)

        # Add overlays
        if show_tangent_plane and tangent_point is not None and calculus_engine is not None:
            plane_size = min(x_range[1] - x_range[0], y_range[1] - y_range[0]) * 0.15
            self.plot_tangent_plane(
                func_def, tangent_point[0], tangent_point[1],
                x_range, y_range, plane_size, True, calculus_engine
            )

        if show_descent_path and descent_path is not None:
            self.plot_descent_path(descent_path)

        if show_critical_points and critical_points is not None:
            self.plot_critical_points(critical_points)

        if show_gradient_field and gradient_U is not None:
            self.plot_gradient_field(
                func_def, x_range, y_range,
                gradient_U, gradient_V, gradient_X, gradient_Y, gradient_Z,
            )

        self.canvas.draw_idle()

    def export_plot(self, filepath: str):
        """Export the current plot to an image file.

        Args:
            filepath: Path to save the image (e.g., .png, .pdf, .svg).
        """
        self.figure.savefig(filepath, dpi=150, bbox_inches='tight',
                           facecolor=self.figure.get_facecolor())