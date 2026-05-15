"""Main application window for the 3D Gradient Descent Visualizer.

This module defines the top-level QMainWindow that orchestrates the layout
of all sub-widgets: 3D plot, contour map, controls panel, and info panel.
"""

from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QStatusBar,
    QMenuBar,
    QMenu,
    QAction,
    QToolBar,
    QLabel,
    QSizePolicy,
    QScrollArea,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon

from view.plot_widget import Plot3DWidget
from view.contour_widget import ContourWidget
from view.controls_panel import ControlsPanel
from view.info_panel import InfoPanel


class MainWindow(QMainWindow):
    """Main application window for the 3D Gradient Descent Visualizer.

    Layout structure:
    ┌─────────────────────────────────────────────────────┐
    │  Menu Bar  │  Toolbar                               │
    ├──────────────────────┬──────────────────────────────┤
    │                      │  Controls Panel              │
    │  3D Surface Plot     │  ─────────────────           │
    │                      │  Contour Map                 │
    │                      │  ─────────────────           │
    │                      │  Info Panel                  │
    ├──────────────────────┴──────────────────────────────┤
    │  Status Bar                                         │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Gradient Descent Visualizer — MVC CCP")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Apply dark theme styling
        self._apply_style()

        # Initialize sub-widgets
        self.plot_widget = Plot3DWidget()
        self.contour_widget = ContourWidget()
        self.controls_panel = ControlsPanel()
        self.info_panel = InfoPanel()

        # Build the layout
        self._build_layout()

        # Build menu bar and toolbar
        self._build_menu_bar()
        self._build_toolbar()

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready — Select a function and click 'Run Descent'")
        self.status_bar.addWidget(self.status_label, 1)

    def _apply_style(self):
        """Apply a modern dark theme stylesheet to the application."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
            QMenuBar {
                background-color: #181825;
                color: #cdd6f4;
                border-bottom: 1px solid #313244;
                padding: 2px;
            }
            QMenuBar::item:selected {
                background-color: #45475a;
            }
            QMenu {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #313244;
            }
            QMenu::item:selected {
                background-color: #45475a;
            }
            QToolBar {
                background-color: #181825;
                border-bottom: 1px solid #313244;
                spacing: 5px;
                padding: 3px;
            }
            QStatusBar {
                background-color: #181825;
                color: #a6adc8;
                border-top: 1px solid #313244;
            }
            QScrollArea {
                border: none;
                background-color: #1e1e2e;
            }
            QSplitter::handle {
                background-color: #313244;
            }
            QSplitter::handle:hover {
                background-color: #585b70;
            }
            QGroupBox {
                border: 1px solid #313244;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 16px;
                font-weight: bold;
                color: #89b4fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #45475a;
                color: #cdd6f4;
                border: 1px solid #585b70;
                border-radius: 4px;
                padding: 6px 14px;
                font-weight: bold;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #585b70;
                border-color: #89b4fa;
            }
            QPushButton:pressed {
                background-color: #313244;
            }
            QPushButton#runButton {
                background-color: #a6e3a1;
                color: #1e1e2e;
                font-size: 13px;
                min-height: 34px;
            }
            QPushButton#runButton:hover {
                background-color: #94e2d5;
            }
            QPushButton#resetButton {
                background-color: #f38ba8;
                color: #1e1e2e;
            }
            QPushButton#resetButton:hover {
                background-color: #eba0ac;
            }
            QPushButton#animateButton {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QPushButton#animateButton:hover {
                background-color: #74c7ec;
            }
            QPushButton#findCriticalButton {
                background-color: #f9e2af;
                color: #1e1e2e;
            }
            QPushButton#findCriticalButton:hover {
                background-color: #fab387;
            }
            QComboBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 24px;
            }
            QComboBox:hover {
                border-color: #89b4fa;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #45475a;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e2e;
                color: #cdd6f4;
                selection-background-color: #45475a;
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 3px;
                min-height: 24px;
            }
            QDoubleSpinBox:hover, QSpinBox:hover {
                border-color: #89b4fa;
            }
            QCheckBox {
                color: #cdd6f4;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #585b70;
                background-color: #313244;
            }
            QCheckBox::indicator:checked {
                background-color: #a6e3a1;
                border-color: #a6e3a1;
            }
            QTextEdit {
                background-color: #11111b;
                color: #cdd6f4;
                border: 1px solid #313244;
                border-radius: 4px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                padding: 4px;
            }
            QLabel#headerLabel {
                color: #89b4fa;
                font-size: 14px;
                font-weight: bold;
            }
            QLabel#expressionLabel {
                color: #f9e2af;
                font-size: 13px;
                font-style: italic;
            }
        """)

    def _build_layout(self):
        """Build the main layout with splitters for resizable panels."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Left side: 3D plot (takes most space)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)
        left_layout.addWidget(self.plot_widget, stretch=1)

        # Right side: controls, contour, info (scrollable to avoid overlap)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)
        right_layout.addWidget(self.controls_panel, stretch=0)
        right_layout.addWidget(self.contour_widget, stretch=3)
        right_layout.addWidget(self.info_panel, stretch=2)

        # Set size policies
        left_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.contour_widget.setMinimumHeight(260)
        self.info_panel.setMinimumHeight(220)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setWidget(right_container)
        right_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        right_scroll.setMinimumWidth(380)
        right_scroll.setMaximumWidth(520)

        # Use splitter for resizable layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([900, 400])

        main_layout.addWidget(splitter)

    def _build_menu_bar(self):
        """Build the application menu bar."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")

        self.action_reset = QAction("Reset View", self)
        self.action_reset.setShortcut("Ctrl+R")
        file_menu.addAction(self.action_reset)

        self.action_export_plot = QAction("Export 3D Plot...", self)
        self.action_export_plot.setShortcut("Ctrl+E")
        file_menu.addAction(self.action_export_plot)

        self.action_export_contour = QAction("Export Contour Map...", self)
        file_menu.addAction(self.action_export_contour)

        file_menu.addSeparator()

        self.action_quit = QAction("Quit", self)
        self.action_quit.setShortcut("Ctrl+Q")
        file_menu.addAction(self.action_quit)

        # View menu
        view_menu = menu_bar.addMenu("View")

        self.action_show_tangent = QAction("Show Tangent Plane", self)
        self.action_show_tangent.setCheckable(True)
        self.action_show_tangent.setChecked(False)
        view_menu.addAction(self.action_show_tangent)

        self.action_show_gradient_field = QAction("Show Gradient Field", self)
        self.action_show_gradient_field.setCheckable(True)
        self.action_show_gradient_field.setChecked(False)
        view_menu.addAction(self.action_show_gradient_field)

        self.action_show_critical_points = QAction("Show Critical Points", self)
        self.action_show_critical_points.setCheckable(True)
        self.action_show_critical_points.setChecked(True)
        view_menu.addAction(self.action_show_critical_points)

        self.action_show_descent_path = QAction("Show Descent Path", self)
        self.action_show_descent_path.setCheckable(True)
        self.action_show_descent_path.setChecked(True)
        view_menu.addAction(self.action_show_descent_path)

        # Help menu
        help_menu = menu_bar.addMenu("Help")

        self.action_about = QAction("About", self)
        help_menu.addAction(self.action_about)

        self.action_math_help = QAction("Math Concepts", self)
        help_menu.addAction(self.action_math_help)

    def _build_toolbar(self):
        """Build the application toolbar with quick-access buttons."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Toolbar buttons (using text since we don't have icon files)
        toolbar.addAction(self.action_reset)
        toolbar.addSeparator()
        toolbar.addAction(self.action_export_plot)
        toolbar.addAction(self.action_export_contour)
        toolbar.addSeparator()
        toolbar.addAction(self.action_show_tangent)
        toolbar.addAction(self.action_show_gradient_field)
        toolbar.addAction(self.action_show_critical_points)
        toolbar.addAction(self.action_show_descent_path)

    def set_status(self, message: str):
        """Update the status bar message.

        Args:
            message: Status message to display.
        """
        self.status_label.setText(message)