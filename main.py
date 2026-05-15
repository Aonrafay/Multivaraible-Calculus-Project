"""Application entry point for the Gradient Descent Visualizer."""

import sys

from PyQt5.QtWidgets import QApplication

from controller.app_controller import AppController
from view.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    AppController(window)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
