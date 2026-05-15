"""Critical point analysis for the 3D Gradient Descent Visualizer.

This module provides:
- Critical point detection (where gradient = 0)
- Classification of critical points (minimum, maximum, saddle, degenerate)
- Eigenvalue analysis of the Hessian matrix
- Grid-based search for critical points
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum
from model.functions import FunctionDefinition
from model.calculus import CalculusEngine


class CriticalPointType(Enum):
    """Classification of critical points based on Hessian eigenvalues."""
    LOCAL_MINIMUM = "Local Minimum"
    LOCAL_MAXIMUM = "Local Maximum"
    SADDLE_POINT = "Saddle Point"
    DEGENERATE = "Degenerate (Inconclusive)"
    UNKNOWN = "Unknown"


@dataclass
class CriticalPoint:
    """A detected and classified critical point.

    Attributes:
        x: X-coordinate of the critical point.
        y: Y-coordinate of the critical point.
        z: Function value at the critical point.
        point_type: Classification (min, max, saddle, degenerate).
        gradient: Gradient vector at the point (should be near-zero).
        gradient_magnitude: Magnitude of the gradient at the point.
        hessian: 2x2 Hessian matrix at the point.
        eigenvalues: Eigenvalues of the Hessian matrix.
        determinant: Determinant of the Hessian matrix.
    """

    x: float
    y: float
    z: float
    point_type: CriticalPointType
    gradient: Tuple[float, float]
    gradient_magnitude: float
    hessian: np.ndarray
    eigenvalues: Tuple[float, float]
    determinant: float


class CriticalPointAnalyzer:
    """Analyzer for detecting and classifying critical points of multivariable functions.

    Uses grid-based search to find approximate critical points (where ||∇f|| ≈ 0),
    then classifies them using the second derivative test (Hessian analysis).

    Second Derivative Test:
    - D = det(H) > 0 and f_xx > 0 → Local Minimum
    - D = det(H) > 0 and f_xx < 0 → Local Maximum
    - D = det(H) < 0 → Saddle Point
    - D = det(H) = 0 → Degenerate (Inconclusive)
    """

    def __init__(
        self,
        calculus_engine: Optional[CalculusEngine] = None,
        grid_resolution: int = 50,
        gradient_threshold: float = 0.1,
        refinement_iterations: int = 20,
    ):
        """Initialize the analyzer.

        Args:
            calculus_engine: A CalculusEngine instance for gradient/Hessian computation.
            grid_resolution: Number of grid points per axis for initial search.
            gradient_threshold: Threshold for considering a point as critical (||∇f|| < threshold).
            refinement_iterations: Number of Newton's method iterations to refine critical points.
        """
        self.calculus_engine = calculus_engine or CalculusEngine()
        self.grid_resolution = grid_resolution
        self.gradient_threshold = gradient_threshold
        self.refinement_iterations = refinement_iterations

    def find_critical_points(
        self,
        func_def: FunctionDefinition,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        use_analytical: bool = True,
    ) -> List[CriticalPoint]:
        """Find and classify critical points within the given domain.

        Performs a grid search to locate approximate critical points,
        refines them using Newton's method, then classifies using the
        second derivative test.

        Args:
            func_def: The function definition to analyze.
            x_range: (x_min, x_max) search domain.
            y_range: (y_min, y_max) search domain.
            use_analytical: Whether to use analytical derivatives if available.

        Returns:
            List of classified CriticalPoint objects.
        """
        # Step 1: Grid search for approximate critical points
        approximate_points = self._grid_search(func_def, x_range, y_range, use_analytical)

        # Step 2: Refine using Newton's method
        refined_points = []
        for (ax, ay) in approximate_points:
            rx, ry = self._refine_critical_point(func_def, ax, ay, use_analytical)
            # Check if refined point is still within domain
            if x_range[0] <= rx <= x_range[1] and y_range[0] <= ry <= y_range[1]:
                refined_points.append((rx, ry))

        # Step 3: Remove duplicate points (within tolerance)
        unique_points = self._remove_duplicates(refined_points, tolerance=0.05)

        # Step 4: Classify each unique critical point
        critical_points = []
        for (ux, uy) in unique_points:
            cp = self._classify_point(func_def, ux, uy, use_analytical)
            if cp.gradient_magnitude < self.gradient_threshold * 5:  # Allow some slack after refinement
                critical_points.append(cp)

        return critical_points

    def classify_point(
        self,
        func_def: FunctionDefinition,
        x: float,
        y: float,
        use_analytical: bool = True,
    ) -> CriticalPoint:
        """Classify a specific point as a critical point type.

        Args:
            func_def: The function definition.
            x: X-coordinate.
            y: Y-coordinate.
            use_analytical: Whether to use analytical derivatives if available.

        Returns:
            A classified CriticalPoint object.
        """
        return self._classify_point(func_def, x, y, use_analytical)

    def _grid_search(
        self,
        func_def: FunctionDefinition,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        use_analytical: bool,
    ) -> List[Tuple[float, float]]:
        """Search a grid for approximate critical points where ||∇f|| is small.

        Args:
            func_def: The function definition.
            x_range: (x_min, x_max) domain.
            y_range: (y_min, y_max) domain.
            use_analytical: Whether to use analytical gradient.

        Returns:
            List of approximate (x, y) critical point locations.
        """
        x_vals = np.linspace(x_range[0], x_range[1], self.grid_resolution)
        y_vals = np.linspace(y_range[0], y_range[1], self.grid_resolution)

        approximate_points = []
        threshold = self.gradient_threshold

        for xi in x_vals:
            for yi in y_vals:
                grad_mag = self.calculus_engine.compute_gradient_magnitude(
                    func_def, xi, yi, use_analytical
                )
                if grad_mag < threshold:
                    approximate_points.append((xi, yi))

        return approximate_points

    def _refine_critical_point(
        self,
        func_def: FunctionDefinition,
        x: float,
        y: float,
        use_analytical: bool,
    ) -> Tuple[float, float]:
        """Refine an approximate critical point using Newton's method.

        Newton's method for finding where ∇f = 0:
            [x_new, y_new] = [x, y] - H⁻¹ · ∇f

        This iteratively moves closer to the exact critical point.

        Args:
            func_def: The function definition.
            x: Initial x-coordinate (approximate).
            y: Initial y-coordinate (approximate).
            use_analytical: Whether to use analytical derivatives.

        Returns:
            Refined (x, y) coordinates closer to the true critical point.
        """
        for _ in range(self.refinement_iterations):
            gradient = self.calculus_engine.compute_gradient(func_def, x, y, use_analytical)
            grad_mag = np.sqrt(gradient[0]**2 + gradient[1]**2)

            if grad_mag < 1e-10:
                break  # Already at a critical point

            hessian = self.calculus_engine.compute_hessian(func_def, x, y, use_analytical)

            # Check if Hessian is invertible
            det = np.linalg.det(hessian)
            if abs(det) < 1e-12:
                # Hessian is singular; use gradient descent step instead
                step_size = 0.01
                x = x - step_size * gradient[0]
                y = y - step_size * gradient[1]
            else:
                # Newton step: [x,y] = [x,y] - H⁻¹ · ∇f
                hessian_inv = np.linalg.inv(hessian)
                grad_vec = np.array([gradient[0], gradient[1]])
                step = hessian_inv @ grad_vec
                x = x - step[0]
                y = y - step[1]

        return (x, y)

    def _classify_point(
        self,
        func_def: FunctionDefinition,
        x: float,
        y: float,
        use_analytical: bool,
    ) -> CriticalPoint:
        """Classify a critical point using the second derivative test.

        Args:
            func_def: The function definition.
            x: X-coordinate of the critical point.
            y: Y-coordinate of the critical point.
            use_analytical: Whether to use analytical derivatives.

        Returns:
            A fully classified CriticalPoint object.
        """
        z = func_def.evaluate(np.array([x]), np.array([y]))[0]
        gradient = self.calculus_engine.compute_gradient(func_def, x, y, use_analytical)
        grad_mag = np.sqrt(gradient[0]**2 + gradient[1]**2)
        hessian = self.calculus_engine.compute_hessian(func_def, x, y, use_analytical)

        # Compute eigenvalues and determinant
        eigenvalues = np.linalg.eigvals(hessian)
        det_h = np.linalg.det(hessian)
        fxx = hessian[0, 0]

        # Classify using the second derivative test
        if abs(det_h) < 1e-10:
            point_type = CriticalPointType.DEGENERATE
        elif det_h > 0:
            if fxx > 0:
                point_type = CriticalPointType.LOCAL_MINIMUM
            elif fxx < 0:
                point_type = CriticalPointType.LOCAL_MAXIMUM
            else:
                point_type = CriticalPointType.DEGENERATE
        elif det_h < 0:
            point_type = CriticalPointType.SADDLE_POINT
        else:
            point_type = CriticalPointType.UNKNOWN

        return CriticalPoint(
            x=x,
            y=y,
            z=z,
            point_type=point_type,
            gradient=gradient,
            gradient_magnitude=grad_mag,
            hessian=hessian,
            eigenvalues=(eigenvalues[0], eigenvalues[1]),
            determinant=det_h,
        )

    def _remove_duplicates(
        self,
        points: List[Tuple[float, float]],
        tolerance: float = 0.05,
    ) -> List[Tuple[float, float]]:
        """Remove duplicate critical points that are within tolerance distance.

        Args:
            points: List of (x, y) points to deduplicate.
            tolerance: Minimum distance between distinct points.

        Returns:
            Deduplicated list of (x, y) points.
        """
        if not points:
            return []

        unique = [points[0]]
        for (px, py) in points[1:]:
            is_duplicate = False
            for (ux, uy) in unique:
                dist = np.sqrt((px - ux)**2 + (py - uy)**2)
                if dist < tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append((px, py))

        return unique

    def format_critical_point_info(self, cp: CriticalPoint) -> str:
        """Format a critical point's information as a readable string.

        Args:
            cp: A CriticalPoint object.

        Returns:
            Formatted string with all critical point details.
        """
        lines = [
            f"━━━ {cp.point_type.value} ━━━",
            f"  Location: ({cp.x:.6f}, {cp.y:.6f})",
            f"  f(x,y) = {cp.z:.6f}",
            f"  ||∇f|| = {cp.gradient_magnitude:.8f}",
            f"  ∇f = ({cp.gradient[0]:.6f}, {cp.gradient[1]:.6f})",
            f"  Hessian:",
            f"    [[{cp.hessian[0,0]:.6f}, {cp.hessian[0,1]:.6f}],",
            f"     [{cp.hessian[1,0]:.6f}, {cp.hessian[1,1]:.6f}]]",
            f"  Eigenvalues: ({cp.eigenvalues[0]:.6f}, {cp.eigenvalues[1]:.6f})",
            f"  det(H) = {cp.determinant:.6f}",
        ]
        return "\n".join(lines)