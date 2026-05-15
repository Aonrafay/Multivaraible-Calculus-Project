"""Multivariable function definitions for the 3D Gradient Descent Visualizer.

This module provides a registry of mathematical functions that can be
visualized and analyzed. Each function is defined with its symbolic expression,
evaluation callback, and metadata (name, description, default domain).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class FunctionDefinition:
    """Defines a multivariable function with all its metadata.

    Attributes:
        name: Display name of the function.
        expression: LaTeX-style symbolic expression string.
        description: Brief description of the function's properties.
        evaluate: Callable that takes (x, y) arrays and returns z values.
        default_x_range: Default x-axis range for plotting [min, max].
        default_y_range: Default y-axis range for plotting [min, max].
        has_analytical_gradient: Whether analytical gradient is available.
        analytical_gradient: Optional callable returning (df/dx, df/dy).
        has_analytical_hessian: Whether analytical Hessian is available.
        analytical_hessian: Optional callable returning Hessian matrix components.
    """

    name: str
    expression: str
    description: str
    evaluate: Callable[[np.ndarray, np.ndarray], np.ndarray]
    default_x_range: List[float] = field(default_factory=lambda: [-3.0, 3.0])
    default_y_range: List[float] = field(default_factory=lambda: [-3.0, 3.0])
    has_analytical_gradient: bool = False
    analytical_gradient: Optional[Callable] = None
    has_analytical_hessian: bool = False
    analytical_hessian: Optional[Callable] = None


# ─── Predefined Functions ────────────────────────────────────────────────────


def _paraboloid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """f(x,y) = x² + y² — Simple convex paraboloid with minimum at origin."""
    return x**2 + y**2


def _paraboloid_gradient(x: float, y: float) -> tuple:
    """Analytical gradient of paraboloid: (2x, 2y)."""
    return (2 * x, 2 * y)


def _paraboloid_hessian(x: float, y: float) -> tuple:
    """Analytical Hessian of paraboloid: [[2, 0], [0, 2]]."""
    return (2.0, 0.0, 0.0, 2.0)  # fxx, fxy, fyx, fyy


def _monkey_saddle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """f(x,y) = x³ - 3xy² — Monkey saddle with degenerate critical point."""
    return x**3 - 3 * x * y**2


def _monkey_saddle_gradient(x: float, y: float) -> tuple:
    """Analytical gradient of monkey saddle."""
    return (3 * x**2 - 3 * y**2, -6 * x * y)


def _monkey_saddle_hessian(x: float, y: float) -> tuple:
    """Analytical Hessian of monkey saddle."""
    return (6 * x, -6 * y, -6 * y, -6 * x)


def _rosenbrock(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """f(x,y) = (1-x)² + 100(y-x²)² — Rosenbrock's banana function."""
    return (1 - x)**2 + 100 * (y - x**2)**2


def _rosenbrock_gradient(x: float, y: float) -> tuple:
    """Analytical gradient of Rosenbrock function."""
    dfdx = -2 * (1 - x) - 400 * x * (y - x**2)
    dfdy = 200 * (y - x**2)
    return (dfdx, dfdy)


def _rosenbrock_hessian(x: float, y: float) -> tuple:
    """Analytical Hessian of Rosenbrock function."""
    fxx = 2 - 400 * y + 1200 * x**2
    fxy = -400 * x
    fyx = -400 * x
    fyy = 200
    return (fxx, fxy, fyx, fyy)


def _himmelblau(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """f(x,y) = (x²+y-11)² + (x+y²-7)² — Himmelblau's function with 4 minima."""
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def _himmelblau_gradient(x: float, y: float) -> tuple:
    """Analytical gradient of Himmelblau's function."""
    dfdx = 2 * (x**2 + y - 11) * 2 * x + 2 * (x + y**2 - 7)
    dfdy = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * 2 * y
    return (dfdx, dfdy)


def _himmelblau_hessian(x: float, y: float) -> tuple:
    """Analytical Hessian of Himmelblau's function."""
    fxx = 4 * (x**2 + y - 11) + 8 * x**2 + 2
    fxy = 4 * x + 4 * y
    fyx = 4 * x + 4 * y
    fyy = 4 * (y**2 + x - 7) + 8 * y**2 + 2
    return (fxx, fxy, fyx, fyy)


def _saddle_point(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """f(x,y) = x² - y² — Classic saddle point at origin."""
    return x**2 - y**2


def _saddle_point_gradient(x: float, y: float) -> tuple:
    """Analytical gradient of saddle point function."""
    return (2 * x, -2 * y)


def _saddle_point_hessian(x: float, y: float) -> tuple:
    """Analytical Hessian of saddle point function."""
    return (2.0, 0.0, 0.0, -2.0)


def _cubic_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """f(x,y) = x³ - 3xy + y³ — Function with multiple critical points."""
    return x**3 - 3 * x * y + y**3


def _cubic_function_gradient(x: float, y: float) -> tuple:
    """Analytical gradient of cubic function."""
    return (3 * x**2 - 3 * y, -3 * x + 3 * y**2)


def _cubic_function_hessian(x: float, y: float) -> tuple:
    """Analytical Hessian of cubic function."""
    return (6 * x, -3.0, -3.0, 6 * y)


def _wave_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """f(x,y) = sin(x) + cos(y) — Periodic wave function."""
    return np.sin(x) + np.cos(y)


def _wave_function_gradient(x: float, y: float) -> tuple:
    """Analytical gradient of wave function."""
    return (np.cos(x), -np.sin(y))


def _wave_function_hessian(x: float, y: float) -> tuple:
    """Analytical Hessian of wave function."""
    return (-np.sin(x), 0.0, 0.0, -np.cos(y))


def _gaussian(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """f(x,y) = -exp(-(x²+y²)/2) — Gaussian bump (negative for minimum)."""
    return -np.exp(-(x**2 + y**2) / 2)


def _gaussian_gradient(x: float, y: float) -> tuple:
    """Analytical gradient of Gaussian function."""
    exp_val = np.exp(-(x**2 + y**2) / 2)
    return (x * exp_val, y * exp_val)


def _gaussian_hessian(x: float, y: float) -> tuple:
    """Analytical Hessian of Gaussian function."""
    exp_val = np.exp(-(x**2 + y**2) / 2)
    fxx = exp_val * (x**2 - 1)
    fxy = exp_val * x * y
    fyx = exp_val * x * y
    fyy = exp_val * (y**2 - 1)
    return (fxx, fxy, fyx, fyy)


def _ackley(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """f(x,y) = -20·exp(-0.2·√(x²+y²)) - exp(cos(2πx)+cos(2πy)) + 20 + e
    Ackley function — many local minima, one global minimum at origin."""
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum1 = x**2 + y**2
    sum2 = np.cos(c * x) + np.cos(c * y)
    term1 = -a * np.exp(-b * np.sqrt(sum1 / 2))
    term2 = -np.exp(sum2 / 2)
    return term1 + term2 + a + np.e


# ─── Function Registry ───────────────────────────────────────────────────────


class FunctionRegistry:
    """Registry that manages all available function definitions.

    Provides methods to list, retrieve, and register functions.
    Acts as the data source for the Model layer.
    """

    def __init__(self):
        self._functions: Dict[str, FunctionDefinition] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register all predefined function definitions."""
        defaults = [
            FunctionDefinition(
                name="Paraboloid",
                expression="f(x,y) = x² + y²",
                description="Simple convex paraboloid with a single minimum at (0,0). "
                            "Classic test case for gradient descent convergence.",
                evaluate=_paraboloid,
                default_x_range=[-3.0, 3.0],
                default_y_range=[-3.0, 3.0],
                has_analytical_gradient=True,
                analytical_gradient=_paraboloid_gradient,
                has_analytical_hessian=True,
                analytical_hessian=_paraboloid_hessian,
            ),
            FunctionDefinition(
                name="Saddle Point",
                expression="f(x,y) = x² - y²",
                description="Classic saddle point at the origin. "
                            "The Hessian has both positive and negative eigenvalues.",
                evaluate=_saddle_point,
                default_x_range=[-3.0, 3.0],
                default_y_range=[-3.0, 3.0],
                has_analytical_gradient=True,
                analytical_gradient=_saddle_point_gradient,
                has_analytical_hessian=True,
                analytical_hessian=_saddle_point_hessian,
            ),
            FunctionDefinition(
                name="Monkey Saddle",
                expression="f(x,y) = x³ - 3xy²",
                description="Monkey saddle with a degenerate critical point at origin. "
                            "The Hessian is singular — neither min, max, nor regular saddle.",
                evaluate=_monkey_saddle,
                default_x_range=[-2.0, 2.0],
                default_y_range=[-2.0, 2.0],
                has_analytical_gradient=True,
                analytical_gradient=_monkey_saddle_gradient,
                has_analytical_hessian=True,
                analytical_hessian=_monkey_saddle_hessian,
            ),
            FunctionDefinition(
                name="Rosenbrock",
                expression="f(x,y) = (1-x)² + 100(y-x²)²",
                description="Rosenbrock's banana function. Narrow valley makes "
                            "gradient descent converge slowly. Minimum at (1,1).",
                evaluate=_rosenbrock,
                default_x_range=[-2.0, 2.0],
                default_y_range=[-1.0, 3.0],
                has_analytical_gradient=True,
                analytical_gradient=_rosenbrock_gradient,
                has_analytical_hessian=True,
                analytical_hessian=_rosenbrock_hessian,
            ),
            FunctionDefinition(
                name="Himmelblau",
                expression="f(x,y) = (x²+y-11)² + (x+y²-7)²",
                description="Himmelblau's function with four identical local minima "
                            "and one local maximum. Great for testing multi-modal optimization.",
                evaluate=_himmelblau,
                default_x_range=[-5.0, 5.0],
                default_y_range=[-5.0, 5.0],
                has_analytical_gradient=True,
                analytical_gradient=_himmelblau_gradient,
                has_analytical_hessian=True,
                analytical_hessian=_himmelblau_hessian,
            ),
            FunctionDefinition(
                name="Cubic",
                expression="f(x,y) = x³ - 3xy + y³",
                description="Cubic function with multiple critical points including "
                            "a local minimum, local maximum, and saddle point.",
                evaluate=_cubic_function,
                default_x_range=[-2.5, 2.5],
                default_y_range=[-2.5, 2.5],
                has_analytical_gradient=True,
                analytical_gradient=_cubic_function_gradient,
                has_analytical_hessian=True,
                analytical_hessian=_cubic_function_hessian,
            ),
            FunctionDefinition(
                name="Wave",
                expression="f(x,y) = sin(x) + cos(y)",
                description="Periodic wave function with infinitely many critical points. "
                            "Alternating between local minima and maxima.",
                evaluate=_wave_function,
                default_x_range=[-6.0, 6.0],
                default_y_range=[-6.0, 6.0],
                has_analytical_gradient=True,
                analytical_gradient=_wave_function_gradient,
                has_analytical_hessian=True,
                analytical_hessian=_wave_function_hessian,
            ),
            FunctionDefinition(
                name="Gaussian",
                expression="f(x,y) = -exp(-(x²+y²)/2)",
                description="Negative Gaussian bump. Single global minimum at origin, "
                            "smooth and radially symmetric.",
                evaluate=_gaussian,
                default_x_range=[-3.0, 3.0],
                default_y_range=[-3.0, 3.0],
                has_analytical_gradient=True,
                analytical_gradient=_gaussian_gradient,
                has_analytical_hessian=True,
                analytical_hessian=_gaussian_hessian,
            ),
            FunctionDefinition(
                name="Ackley",
                expression="f(x,y) = -20·exp(-0.2·√(x²+y²)) - exp(cos(2πx)+cos(2πy)) + 20 + e",
                description="Ackley function with many local minima surrounding "
                            "a single global minimum at origin. Tests escape from local optima.",
                evaluate=_ackley,
                default_x_range=[-5.0, 5.0],
                default_y_range=[-5.0, 5.0],
                has_analytical_gradient=False,
                analytical_gradient=None,
                has_analytical_hessian=False,
                analytical_hessian=None,
            ),
        ]
        for func_def in defaults:
            self._functions[func_def.name] = func_def

    def get_function(self, name: str) -> FunctionDefinition:
        """Retrieve a function definition by name.

        Args:
            name: The display name of the function.

        Returns:
            The corresponding FunctionDefinition.

        Raises:
            KeyError: If the function name is not registered.
        """
        if name not in self._functions:
            raise KeyError(f"Function '{name}' not found in registry.")
        return self._functions[name]

    def get_all_names(self) -> List[str]:
        """Return a sorted list of all registered function names."""
        return sorted(self._functions.keys())

    def get_all_functions(self) -> Dict[str, FunctionDefinition]:
        """Return the entire function dictionary."""
        return dict(self._functions)

    def register(self, func_def: FunctionDefinition):
        """Register a new function definition.

        Args:
            func_def: A FunctionDefinition instance to add to the registry.
        """
        self._functions[func_def.name] = func_def

    def unregister(self, name: str):
        """Remove a function from the registry.

        Args:
            name: The name of the function to remove.
        """
        if name in self._functions:
            del self._functions[name]