"""Model layer for 3D Gradient Descent Visualizer.

Contains mathematical computation modules:
- functions: Multivariable function definitions
- calculus: Gradient, partial derivatives, Hessian computation
- optimization: Gradient descent simulation
- analysis: Critical point detection and classification
"""

from model.functions import FunctionRegistry
from model.calculus import CalculusEngine
from model.optimization import GradientDescentOptimizer
from model.analysis import CriticalPointAnalyzer

__all__ = [
    "FunctionRegistry",
    "CalculusEngine",
    "GradientDescentOptimizer",
    "CriticalPointAnalyzer",
]