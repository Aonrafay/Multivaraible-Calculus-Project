"""Information panel for analysis and optimization results."""

from PyQt5.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget

from model.functions import FunctionDefinition


class InfoPanel(QWidget):
    """Panel that displays function details and analysis results."""

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel("Info")
        header.setObjectName("headerLabel")
        layout.addWidget(header)

        self.function_name_label = QLabel("Function: -")
        self.expression_label = QLabel("")
        self.expression_label.setObjectName("expressionLabel")
        self.description_label = QLabel("")
        self.description_label.setWordWrap(True)

        layout.addWidget(self.function_name_label)
        layout.addWidget(self.expression_label)
        layout.addWidget(self.description_label)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setPlaceholderText("Run optimization or analysis to see details here.")
        layout.addWidget(self.details_text)

        self._optimization_text = ""
        self._critical_text = ""

    def set_function_info(self, func_def: FunctionDefinition):
        self.function_name_label.setText(f"Function: {func_def.name}")
        self.expression_label.setText(func_def.expression)
        self.description_label.setText(func_def.description)

    def set_optimization_info(self, text: str):
        self._optimization_text = text
        self._refresh_details()

    def set_critical_points_info(self, text: str):
        self._critical_text = text
        self._refresh_details()

    def clear_details(self):
        self._optimization_text = ""
        self._critical_text = ""
        self._refresh_details()

    def _refresh_details(self):
        sections = []
        if self._optimization_text:
            sections.append("Optimization\n" + self._optimization_text)
        if self._critical_text:
            sections.append("Critical Points\n" + self._critical_text)
        if sections:
            self.details_text.setPlainText("\n\n".join(sections))
        else:
            self.details_text.setPlainText("No results yet.")
