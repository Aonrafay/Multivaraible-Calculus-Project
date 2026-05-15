"""View layer for 3D Gradient Descent Visualizer.

Contains all GUI components using PyQt5:
- main_window: Top-level application window
- plot_widget: 3D surface plot with matplotlib
- contour_widget: 2D contour map visualization
- controls_panel: User input controls and settings
- info_panel: Information display for analysis results
"""

from view.main_window import MainWindow
from view.plot_widget import Plot3DWidget
from view.contour_widget import ContourWidget
from view.controls_panel import ControlsPanel
from view.info_panel import InfoPanel

__all__ = [
    "MainWindow",
    "Plot3DWidget",
    "ContourWidget",
    "ControlsPanel",
    "InfoPanel",
]