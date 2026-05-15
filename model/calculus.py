"""Calculus engine for computing gradients, partial derivatives, and Hessians.

This module provides numerical and analytical computation of:
- Partial derivatives (df/dx, df/dy)
- Gradient vectors
- Hessian matrices
- Tangent plane equations
"""

import numpy as np
from typing import Callable, Tuple, Optional
from model.functions import FunctionDefinition


class CalculusEngine:
    """Engine for computing calculus-related quantities for multivariable functions.

    Supports both analytical (when available) and numerical differentiation.
    Numerical differentiation uses central difference formulas for accuracy.
    """

    def __init__(self, epsilon: float = 1e-5):
        """Initialize the calculus engine.

        Args:
            epsilon: Step size for numerical differentiation. Default 1e-5.
        """
        self.epsilon = epsilon

    def compute_gradient(
        self,
        func_def: FunctionDefinition,
        x: float,
        y: float,
        use_analytical: bool = True,
    ) -> Tuple[float, float]:
        """Compute the gradient vector ∇f at point (x, y).

        Args:
            func_def: The function definition to compute gradient for.
            x: X-coordinate.
            y: Y-coordinate.
            use_analytical: Whether to use analytical gradient if available.

        Returns:
            Tuple (df/dx, df/dy) — the gradient vector components.
        """
        if use_analytical and func_def.has_analytical_gradient and func_def.analytical_gradient:
            return func_def.analytical_gradient(x, y)
        return self._numerical_gradient(func_def.evaluate, x, y)

    def compute_partial_derivative_x(
        self,
        func_def: FunctionDefinition,
        x: float,
        y: float,
        use_analytical: bool = True,
    ) -> float:
        """Compute the partial derivative ∂f/∂x at point (x, y).

        Args:
            func_def: The function definition.
            x: X-coordinate.
            y: Y-coordinate.
            use_analytical: Whether to use analytical derivative if available.

        Returns:
            The value of ∂f/∂x at (x, y).
        """
        gradient = self.compute_gradient(func_def, x, y, use_analytical)
        return gradient[0]

    def compute_partial_derivative_y(
        self,
        func_def: FunctionDefinition,
        x: float,
        y: float,
        use_analytical: bool = True,
    ) -> float:
        """Compute the partial derivative ∂f/∂y at point (x, y).

        Args:
            func_def: The function definition.
            x: X-coordinate.
            y: Y-coordinate.
            use_analytical: Whether to use analytical derivative if available.

        Returns:
            The value of ∂f/∂y at (x, y).
        """
        gradient = self.compute_gradient(func_def, x, y, use_analytical)
        return gradient[1]

    def compute_hessian(
        self,
        func_def: FunctionDefinition,
        x: float,
        y: float,
        use_analytical: bool = True,
    ) -> np.ndarray:
        """Compute the Hessian matrix H at point (x, y).

        The Hessian is the matrix of second-order partial derivatives:
            H = [[∂²f/∂x², ∂²f/∂x∂y],
                 [∂²f/∂y∂x, ∂²f/∂y²]]

        Args:
            func_def: The function definition.
            x: X-coordinate.
            y: Y-coordinate.
            use_analytical: Whether to use analytical Hessian if available.

        Returns:
            2x2 numpy array representing the Hessian matrix.
        """
        if use_analytical and func_def.has_analytical_hessian and func_def.analytical_hessian:
            fxx, fxy, fyx, fyy = func_def.analytical_hessian(x, y)
            return np.array([[fxx, fxy], [fyx, fyy]])
        return self._numerical_hessian(func_def.evaluate, x, y)

    def compute_tangent_plane(
        self,
        func_def: FunctionDefinition,
        x0: float,
        y0: float,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        use_analytical: bool = True,
    ) -> np.ndarray:
        """Compute the tangent plane to f at point (x0, y0) over a grid.

        The tangent plane equation is:
            z = f(x0, y0) + ∂f/∂x(x0,y0)·(x-x0) + ∂f/∂y(x0,y0)·(y-y0)

        Args:
            func_def: The function definition.
            x0: X-coordinate of the tangent point.
            y0: Y-coordinate of the tangent point.
            x_grid: X-coordinates grid for the plane evaluation.
            y_grid: Y-coordinates grid for the plane evaluation.
            use_analytical: Whether to use analytical gradient if available.

        Returns:
            Z values of the tangent plane over the grid.
        """
        z0 = func_def.evaluate(np.array([x0]), np.array([y0]))[0]
        dfdx, dfdy = self.compute_gradient(func_def, x0, y0, use_analytical)

        z_plane = z0 + dfdx * (x_grid - x0) + dfdy * (y_grid - y0)
        return z_plane

    def compute_gradient_magnitude(
        self,
        func_def: FunctionDefinition,
        x: float,
        y: float,
        use_analytical: bool = True,
    ) -> float:
        """Compute the magnitude (norm) of the gradient at (x, y).

        Args:
            func_def: The function definition.
            x: X-coordinate.
            y: Y-coordinate.
            use_analytical: Whether to use analytical gradient if available.

        Returns:
            ||∇f(x,y)|| — the Euclidean norm of the gradient.
        """
        dfdx, dfdy = self.compute_gradient(func_def, x, y, use_analytical)
        return np.sqrt(dfdx**2 + dfdy**2)

    def compute_gradient_field(
        self,
        func_def: FunctionDefinition,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        use_analytical: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the gradient vector field over a mesh grid.

        Args:
            func_def: The function definition.
            x_grid: 2D array of x-coordinates (from meshgrid).
            y_grid: 2D array of y-coordinates (from meshgrid).
            use_analytical: Whether to use analytical gradient if available.

        Returns:
            Tuple (U, V) — 2D arrays of gradient x and y components.
        """
        shape = x_grid.shape
        U = np.zeros(shape)
        V = np.zeros(shape)

        # Use vectorized computation if analytical gradient is available
        if use_analytical and func_def.has_analytical_gradient and func_def.analytical_gradient:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    U[i, j], V[i, j] = func_def.analytical_gradient(
                        x_grid[i, j], y_grid[i, j]
                    )
        else:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    U[i, j], V[i, j] = self._numerical_gradient(
                        func_def.evaluate, x_grid[i, j], y_grid[i, j]
                    )

        return U, V

    # ─── Numerical Differentiation Methods ────────────────────────────────────

    def _numerical_gradient(
        self,
        f: Callable[[np.ndarray, np.ndarray], np.ndarray],
        x: float,
        y: float,
    ) -> Tuple[float, float]:
        """Compute gradient numerically using central differences.

        Central difference formula:
            ∂f/∂x ≈ [f(x+h, y) - f(x-h, y)] / (2h)
            ∂f/∂y ≈ [f(x, y+h) - f(x, y-h)] / (2h)

        Args:
            f: The function evaluation callable.
            x: X-coordinate.
            y: Y-coordinate.

        Returns:
            Tuple (df/dx, df/dy) computed numerically.
        """
        h = self.epsilon

        # Partial derivative with respect to x
        dfdx = (
            f(np.array([x + h]), np.array([y]))[0]
            - f(np.array([x - h]), np.array([y]))[0]
        ) / (2 * h)

        # Partial derivative with respect to y
        dfdy = (
            f(np.array([x]), np.array([y + h]))[0]
            - f(np.array([x]), np.array([y - h]))[0]
        ) / (2 * h)

        return (dfdx, dfdy)

    def _numerical_hessian(
        self,
        f: Callable[[np.ndarray, np.ndarray], np.ndarray],
        x: float,
        y: float,
    ) -> np.ndarray:
        """Compute Hessian numerically using central differences.

        Second derivative formulas:
            ∂²f/∂x² ≈ [f(x+h,y) - 2f(x,y) + f(x-h,y)] / h²
            ∂²f/∂y² ≈ [f(x,y+h) - 2f(x,y) + f(x,y-h)] / h²
            ∂²f/∂x∂y ≈ [f(x+h,y+h) - f(x+h,y-h) - f(x-h,y+h) + f(x-h,y-h)] / (4h²)

        Args:
            f: The function evaluation callable.
            x: X-coordinate.
            y: Y-coordinate.

        Returns:
            2x2 numpy array representing the Hessian matrix.
        """
        h = self.epsilon
        h2 = h * h

        f_center = f(np.array([x]), np.array([y]))[0]

        # Second partial derivatives
        fxx = (
            f(np.array([x + h]), np.array([y]))[0]
            - 2 * f_center
            + f(np.array([x - h]), np.array([y]))[0]
        ) / h2

        fyy = (
            f(np.array([x]), np.array([y + h]))[0]
            - 2 * f_center
            + f(np.array([x]), np.array([y - h]))[0]
        ) / h2

        # Mixed partial derivative (cross derivative)
        fxy = (
            f(np.array([x + h]), np.array([y + h]))[0]
            - f(np.array([x + h]), np.array([y - h]))[0]
            - f(np.array([x - h]), np.array([y + h]))[0]
            + f(np.array([x - h]), np.array([y - h]))[0]
        ) / (4 * h2)

        return np.array([[fxx, fxy], [fxy, fyy]])