"""Gradient descent optimization engine for the 3D Gradient Descent Visualizer.

This module implements various gradient descent algorithms:
- Standard (vanilla) gradient descent
- Gradient descent with momentum
- Adaptive learning rate (AdaGrad-like)
- Nesterov accelerated gradient

Each optimizer records its path for visualization and animation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
from model.functions import FunctionDefinition
from model.calculus import CalculusEngine


class OptimizerType(Enum):
    """Available gradient descent optimizer types."""
    STANDARD = "Standard"
    MOMENTUM = "Momentum"
    ADAGRAD = "AdaGrad"
    NESTEROV = "Nesterov"


@dataclass
class OptimizationResult:
    """Result of a gradient descent optimization run.

    Attributes:
        path: List of (x, y, z) tuples representing the descent path.
        gradients: List of (dfdx, dfdy) tuples at each step.
        gradient_magnitudes: List of gradient magnitude at each step.
        converged: Whether the optimizer converged within tolerance.
        iterations: Number of iterations performed.
        final_point: The final (x, y) position.
        final_value: The final function value f(x, y).
        final_gradient: The final gradient (dfdx, dfdy).
        final_gradient_magnitude: The final gradient magnitude.
        optimizer_type: Which optimizer was used.
        learning_rate: The learning rate used.
    """

    path: List[Tuple[float, float, float]] = field(default_factory=list)
    gradients: List[Tuple[float, float]] = field(default_factory=list)
    gradient_magnitudes: List[float] = field(default_factory=list)
    converged: bool = False
    iterations: int = 0
    final_point: Tuple[float, float] = (0.0, 0.0)
    final_value: float = 0.0
    final_gradient: Tuple[float, float] = (0.0, 0.0)
    final_gradient_magnitude: float = 0.0
    optimizer_type: OptimizerType = OptimizerType.STANDARD
    learning_rate: float = 0.01


class GradientDescentOptimizer:
    """Gradient descent optimizer with multiple algorithm variants.

    Supports standard, momentum, AdaGrad, and Nesterov accelerated gradient
    descent. Records the full optimization path for visualization.
    """

    def __init__(
        self,
        calculus_engine: Optional[CalculusEngine] = None,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
    ):
        """Initialize the optimizer.

        Args:
            calculus_engine: A CalculusEngine instance for gradient computation.
                If None, a default one is created.
            max_iterations: Maximum number of descent iterations.
            tolerance: Convergence tolerance based on gradient magnitude.
        """
        self.calculus_engine = calculus_engine or CalculusEngine()
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(
        self,
        func_def: FunctionDefinition,
        start_x: float,
        start_y: float,
        learning_rate: float = 0.01,
        optimizer_type: OptimizerType = OptimizerType.STANDARD,
        momentum_coeff: float = 0.9,
        use_analytical: bool = True,
    ) -> OptimizationResult:
        """Run gradient descent optimization from a starting point.

        Args:
            func_def: The function to optimize (find minimum of).
            start_x: Starting x-coordinate.
            start_y: Starting y-coordinate.
            learning_rate: Step size for gradient descent.
            optimizer_type: Which optimization algorithm to use.
            momentum_coeff: Momentum coefficient (for momentum/Nesterov).
            use_analytical: Whether to use analytical gradient if available.

        Returns:
            OptimizationResult containing the full path and convergence info.
        """
        if optimizer_type == OptimizerType.STANDARD:
            return self._standard_descent(
                func_def, start_x, start_y, learning_rate, use_analytical
            )
        elif optimizer_type == OptimizerType.MOMENTUM:
            return self._momentum_descent(
                func_def, start_x, start_y, learning_rate, momentum_coeff, use_analytical
            )
        elif optimizer_type == OptimizerType.ADAGRAD:
            return self._adagrad_descent(
                func_def, start_x, start_y, learning_rate, use_analytical
            )
        elif optimizer_type == OptimizerType.NESTEROV:
            return self._nesterov_descent(
                func_def, start_x, start_y, learning_rate, momentum_coeff, use_analytical
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def _standard_descent(
        self,
        func_def: FunctionDefinition,
        start_x: float,
        start_y: float,
        learning_rate: float,
        use_analytical: bool,
    ) -> OptimizationResult:
        """Standard (vanilla) gradient descent.

        Update rule: x_new = x_old - lr * ∇f(x_old)
        """
        result = OptimizationResult(
            optimizer_type=OptimizerType.STANDARD,
            learning_rate=learning_rate,
        )

        x, y = start_x, start_y

        for i in range(self.max_iterations):
            z = func_def.evaluate(np.array([x]), np.array([y]))[0]
            dfdx, dfdy = self.compute_gradient(func_def, x, y, use_analytical)
            grad_mag = np.sqrt(dfdx**2 + dfdy**2)

            result.path.append((x, y, z))
            result.gradients.append((dfdx, dfdy))
            result.gradient_magnitudes.append(grad_mag)

            # Check convergence
            if grad_mag < self.tolerance:
                result.converged = True
                result.iterations = i + 1
                result.final_point = (x, y)
                result.final_value = z
                result.final_gradient = (dfdx, dfdy)
                result.final_gradient_magnitude = grad_mag
                return result

            # Update position
            x = x - learning_rate * dfdx
            y = y - learning_rate * dfdy

        # Did not converge
        z = func_def.evaluate(np.array([x]), np.array([y]))[0]
        dfdx, dfdy = self.compute_gradient(func_def, x, y, use_analytical)
        grad_mag = np.sqrt(dfdx**2 + dfdy**2)

        result.path.append((x, y, z))
        result.gradients.append((dfdx, dfdy))
        result.gradient_magnitudes.append(grad_mag)
        result.iterations = self.max_iterations
        result.final_point = (x, y)
        result.final_value = z
        result.final_gradient = (dfdx, dfdy)
        result.final_gradient_magnitude = grad_mag

        return result

    def _momentum_descent(
        self,
        func_def: FunctionDefinition,
        start_x: float,
        start_y: float,
        learning_rate: float,
        momentum_coeff: float,
        use_analytical: bool,
    ) -> OptimizationResult:
        """Gradient descent with momentum.

        Update rule:
            v_new = μ * v_old + lr * ∇f(x_old)
            x_new = x_old - v_new
        """
        result = OptimizationResult(
            optimizer_type=OptimizerType.MOMENTUM,
            learning_rate=learning_rate,
        )

        x, y = start_x, start_y
        vx, vy = 0.0, 0.0  # Velocity (momentum) terms

        for i in range(self.max_iterations):
            z = func_def.evaluate(np.array([x]), np.array([y]))[0]
            dfdx, dfdy = self.compute_gradient(func_def, x, y, use_analytical)
            grad_mag = np.sqrt(dfdx**2 + dfdy**2)

            result.path.append((x, y, z))
            result.gradients.append((dfdx, dfdy))
            result.gradient_magnitudes.append(grad_mag)

            if grad_mag < self.tolerance:
                result.converged = True
                result.iterations = i + 1
                result.final_point = (x, y)
                result.final_value = z
                result.final_gradient = (dfdx, dfdy)
                result.final_gradient_magnitude = grad_mag
                return result

            # Update velocity and position
            vx = momentum_coeff * vx + learning_rate * dfdx
            vy = momentum_coeff * vy + learning_rate * dfdy
            x = x - vx
            y = y - vy

        z = func_def.evaluate(np.array([x]), np.array([y]))[0]
        dfdx, dfdy = self.compute_gradient(func_def, x, y, use_analytical)
        grad_mag = np.sqrt(dfdx**2 + dfdy**2)

        result.path.append((x, y, z))
        result.gradients.append((dfdx, dfdy))
        result.gradient_magnitudes.append(grad_mag)
        result.iterations = self.max_iterations
        result.final_point = (x, y)
        result.final_value = z
        result.final_gradient = (dfdx, dfdy)
        result.final_gradient_magnitude = grad_mag

        return result

    def _adagrad_descent(
        self,
        func_def: FunctionDefinition,
        start_x: float,
        start_y: float,
        learning_rate: float,
        use_analytical: bool,
    ) -> OptimizationResult:
        """AdaGrad (adaptive learning rate) gradient descent.

        Update rule:
            G_new = G_old + ∇f(x_old)²
            x_new = x_old - lr / √(G_new + ε) * ∇f(x_old)
        """
        result = OptimizationResult(
            optimizer_type=OptimizerType.ADAGRAD,
            learning_rate=learning_rate,
        )

        x, y = start_x, start_y
        sum_grad_x_sq, sum_grad_y_sq = 0.0, 0.0  # Accumulated squared gradients
        eps = 1e-8  # Small constant to prevent division by zero

        for i in range(self.max_iterations):
            z = func_def.evaluate(np.array([x]), np.array([y]))[0]
            dfdx, dfdy = self.compute_gradient(func_def, x, y, use_analytical)
            grad_mag = np.sqrt(dfdx**2 + dfdy**2)

            result.path.append((x, y, z))
            result.gradients.append((dfdx, dfdy))
            result.gradient_magnitudes.append(grad_mag)

            if grad_mag < self.tolerance:
                result.converged = True
                result.iterations = i + 1
                result.final_point = (x, y)
                result.final_value = z
                result.final_gradient = (dfdx, dfdy)
                result.final_gradient_magnitude = grad_mag
                return result

            # Accumulate squared gradients
            sum_grad_x_sq += dfdx**2
            sum_grad_y_sq += dfdy**2

            # Adaptive learning rate
            adaptive_lr_x = learning_rate / (np.sqrt(sum_grad_x_sq) + eps)
            adaptive_lr_y = learning_rate / (np.sqrt(sum_grad_y_sq) + eps)

            # Update position
            x = x - adaptive_lr_x * dfdx
            y = y - adaptive_lr_y * dfdy

        z = func_def.evaluate(np.array([x]), np.array([y]))[0]
        dfdx, dfdy = self.compute_gradient(func_def, x, y, use_analytical)
        grad_mag = np.sqrt(dfdx**2 + dfdy**2)

        result.path.append((x, y, z))
        result.gradients.append((dfdx, dfdy))
        result.gradient_magnitudes.append(grad_mag)
        result.iterations = self.max_iterations
        result.final_point = (x, y)
        result.final_value = z
        result.final_gradient = (dfdx, dfdy)
        result.final_gradient_magnitude = grad_mag

        return result

    def _nesterov_descent(
        self,
        func_def: FunctionDefinition,
        start_x: float,
        start_y: float,
        learning_rate: float,
        momentum_coeff: float,
        use_analytical: bool,
    ) -> OptimizationResult:
        """Nesterov accelerated gradient descent.

        Update rule:
            x_lookahead = x_old - μ * v_old
            v_new = μ * v_old + lr * ∇f(x_lookahead)
            x_new = x_old - v_new
        """
        result = OptimizationResult(
            optimizer_type=OptimizerType.NESTEROV,
            learning_rate=learning_rate,
        )

        x, y = start_x, start_y
        vx, vy = 0.0, 0.0

        for i in range(self.max_iterations):
            z = func_def.evaluate(np.array([x]), np.array([y]))[0]
            dfdx, dfdy = self.compute_gradient(func_def, x, y, use_analytical)
            grad_mag = np.sqrt(dfdx**2 + dfdy**2)

            result.path.append((x, y, z))
            result.gradients.append((dfdx, dfdy))
            result.gradient_magnitudes.append(grad_mag)

            if grad_mag < self.tolerance:
                result.converged = True
                result.iterations = i + 1
                result.final_point = (x, y)
                result.final_value = z
                result.final_gradient = (dfdx, dfdy)
                result.final_gradient_magnitude = grad_mag
                return result

            # Lookahead point
            x_look = x - momentum_coeff * vx
            y_look = y - momentum_coeff * vy

            # Compute gradient at lookahead
            dfdx_look, dfdy_look = self.compute_gradient(
                func_def, x_look, y_look, use_analytical
            )

            # Update velocity and position
            vx = momentum_coeff * vx + learning_rate * dfdx_look
            vy = momentum_coeff * vy + learning_rate * dfdy_look
            x = x - vx
            y = y - vy

        z = func_def.evaluate(np.array([x]), np.array([y]))[0]
        dfdx, dfdy = self.compute_gradient(func_def, x, y, use_analytical)
        grad_mag = np.sqrt(dfdx**2 + dfdy**2)

        result.path.append((x, y, z))
        result.gradients.append((dfdx, dfdy))
        result.gradient_magnitudes.append(grad_mag)
        result.iterations = self.max_iterations
        result.final_point = (x, y)
        result.final_value = z
        result.final_gradient = (dfdx, dfdy)
        result.final_gradient_magnitude = grad_mag

        return result

    def compute_gradient(
        self,
        func_def: FunctionDefinition,
        x: float,
        y: float,
        use_analytical: bool = True,
    ) -> Tuple[float, float]:
        """Compute gradient using the calculus engine.

        Args:
            func_def: The function definition.
            x: X-coordinate.
            y: Y-coordinate.
            use_analytical: Whether to use analytical gradient if available.

        Returns:
            Tuple (dfdx, dfdy).
        """
        return self.calculus_engine.compute_gradient(func_def, x, y, use_analytical)