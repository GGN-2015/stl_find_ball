# stl_find_ball
find ball structure in a certain .stl model.

## Installation

```bash
pip install stl_find_ball
```

## Usage

```python

# input your stl filepath
STL_FILE = "BONE-1.stl"

# the max search range of balls
# the best ALPHA is 2 * (radius of ball)
ALPHA = 0.01

# detected_spheres is a list of tuple
# tuple: ((center_x, center_y, center_z), radius)
detected_spheres = locate_sphere_in_stl(
    stl_path=STL_FILE,
    alpha=ALPHA
)

# visualize the balls with stl model with VTK
vtk_visualization(
    stl_path=STL_FILE, 
    spheres=detected_spheres
)
```
