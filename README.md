# stl_find_ball
find ball structure in a certain .stl model.

## Installation

```bash
pip install stl_find_ball
```

## Usage

```python
from stl_find_ball import locate_sphere_in_stl, vtk_visualization
import json

# input your stl filepath
STL_FILE = "./test_data/Bone-1.new.stl"

# detected_spheres is a list of tuple
# tuple: ((center_x, center_y, center_z), radius, point_per_area)
detected_spheres = locate_sphere_in_stl(stl_path=STL_FILE, max_ball_cnt=4)

# visualize the balls with stl model with VTK
vtk_visualization(
    stl_path=STL_FILE, 
    spheres=detected_spheres
)

# Output json form data
detected_spheres = [item[:2] for item in detected_spheres]
with open(STL_FILE[:-4] + ".json", "w") as fp:
    fp.write(json.dumps(detected_spheres, indent=4))
```
