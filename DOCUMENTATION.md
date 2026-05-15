# MVC Gradient Descent Visualizer Documentation

## Overview
This project is a PyQt5 desktop application that visualizes gradient descent on 2D functions using both a 3D surface plot and a 2D contour plot. It follows a clean MVC (Model-View-Controller) architecture to keep math/logic, UI rendering, and user interaction separate.

Key features:
- Multiple built-in mathematical functions
- Gradient descent with multiple optimizers
- Critical point detection and classification
- 3D surface plot, 2D contour plot, tangent plane, and gradient field
- Export plots and animate optimization paths

## Requirements and Setup
- Python 3.9+ (recommended)
- PyQt5, numpy, matplotlib

Install dependencies:
```
pip install -r requirements.txt
```

Run the app:
```
python main.py
```

## Project Structure
```
main.py
requirements.txt
controller/
  app_controller.py
model/
  __init__.py
  analysis.py
  calculus.py
  functions.py
  optimization.py
view/
  __init__.py
  main_window.py
  plot_widget.py
  contour_widget.py
  controls_panel.py
  info_panel.py
```

## MVC Architecture (How it Works)

### Model (Math and Algorithms)
The model layer contains all math and computation. It has no UI code.

1) model/functions.py
- Defines FunctionDefinition and FunctionRegistry
- Each function includes:
  - name, expression, description
  - evaluate(x, y) -> z
  - optional analytical gradient and Hessian
- Registry provides a list of functions to show in the UI

2) model/calculus.py
- CalculusEngine computes derivatives and Hessians
- Uses analytical formulas if available; otherwise uses numerical differentiation
- Provides:
  - compute_gradient
  - compute_partial_derivative_x / y
  - compute_hessian
  - compute_tangent_plane
  - compute_gradient_field

3) model/optimization.py
- GradientDescentOptimizer runs optimization
- Supports Standard, Momentum, AdaGrad, and Nesterov
- Returns OptimizationResult containing:
  - path of points
  - gradient history
  - convergence and final state

4) model/analysis.py
- CriticalPointAnalyzer finds and classifies critical points
- Uses grid search + Newton refinement + Hessian analysis
- Returns CriticalPoint objects with type and details

### View (Frontend UI)
The view layer renders the interface and plots. It does not perform math.

1) view/main_window.py
- The main QMainWindow container
- Sets the global stylesheet (dark theme)
- Lays out the UI with a left plot and a right sidebar
- The right panel is scrollable to avoid overlapping on smaller windows

2) view/plot_widget.py
- 3D Matplotlib widget (surface plot)
- Can display:
  - surface
  - tangent plane
  - gradient descent path
  - critical points
  - gradient field arrows

3) view/contour_widget.py
- 2D contour map of the function
- Shows the descent path and critical points
- Emits a signal when the user clicks the plot (sets start point)

4) view/controls_panel.py
- User input controls for:
  - function selection
  - domain range
  - start point
  - optimizer configuration
  - visualization toggles
- Action buttons: Run, Animate, Find Critical Points, Reset
- Buttons are arranged in a 2x2 grid to avoid overlapping

5) view/info_panel.py
- Displays the selected function info
- Shows optimization and critical point summaries

### Controller (App Logic and Wiring)
The controller connects the model and view and handles user actions.

controller/app_controller.py responsibilities:
- Initializes model services (registry, calculus, optimizer, analyzer)
- Listens to UI events (button clicks, toggles, selection changes)
- Updates plots and info panel
- Manages animation and plot export
- Maintains view state (show/hide tangent plane, gradient field, etc.)

Key controller data flow:
1) User changes a setting or presses a button
2) Controller recomputes data using model components
3) Controller updates both 3D and 2D plots
4) Controller updates info panel text and status bar

## Frontend Details (UI/UX)

### Layout
- Left: 3D surface plot
- Right: Scrollable sidebar
  - Controls panel
  - Contour map
  - Info panel

### Styling
- Global dark theme using Qt stylesheet in main_window.py
- Consistent colors and fonts for labels, buttons, inputs, and text areas
- Group boxes are bordered and labeled for clarity

### Interactions
- Function selection updates expression, description, and plot
- Domain controls update the plot range
- Clicking the contour map sets the start point
- Run Descent computes a path and renders it
- Animate shows the path step-by-step
- Find Critical Points runs analysis and displays markers and info
- Export options save the plots to PNG/JPG/PDF

### Overlap Prevention
- Right panel is inside a QScrollArea
- Contour and info panels have minimum heights
- Action buttons use a grid layout instead of a single row

## Key UI Workflows

### Run Gradient Descent
1) Select a function
2) Set domain and start point
3) Choose optimizer settings
4) Click Run Descent
5) View the path in both plots and check results in the Info panel

### Find Critical Points
1) Select a function and domain
2) Click Find Critical Points
3) Critical points are plotted and listed in the Info panel

### Animate Path
1) Run descent first
2) Click Animate
3) The path is drawn over time
4) Click Animate again to stop

## Extending the App

### Add a New Function
1) Add a new FunctionDefinition in model/functions.py
2) Provide expression, evaluate function, and optional derivatives
3) The function will appear automatically in the UI

### Add a New Optimizer
1) Implement a new method in model/optimization.py
2) Add it to the OptimizerType enum
3) Map the label in controller/app_controller.py
4) Add the option label in view/controls_panel.py

### Add a New Visualization Overlay
1) Add a new toggle in ControlsPanel and menu
2) Handle the toggle in AppController
3) Update Plot3DWidget/ContourWidget to render it

## Troubleshooting
- Blank plot: check domain range; ensure min < max
- Slow performance: lower the grid resolution
- Non-finite values (NaN/Inf): use smaller domains for aggressive functions
- No convergence: reduce learning rate or increase max iterations

## Notes
- The controller keeps a cache of the plot grid and gradient field to reduce recomputation.
- The app is designed to stay responsive for typical grid resolutions (50-120).
