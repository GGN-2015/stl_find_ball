import numpy as np
from scipy.spatial import KDTree
import trimesh
from tqdm import tqdm
import vtk

def locate_sphere_in_stl(
        stl_path:str, 
        alpha:float, 
        distance_tolerance=None) -> list[tuple[tuple[float, ...], float]]:
    """
    Detect spheres in STL model and visualize using VTK
    """
    if distance_tolerance is None:
        distance_tolerance = alpha / 20
        
    # Load STL model
    print("===== Loading STL Model =====")
    mesh = trimesh.load(stl_path)
    vertices = mesh.vertices # type:ignore
    unique_vertices = np.unique(vertices, axis=0)

    print(f"Total original vertices: {len(vertices)}")
    print(f"Total unique vertices: {len(unique_vertices)}")

    # Calculate coordinate ranges
    x_min, y_min, z_min = np.min(unique_vertices, axis=0)
    x_max, y_max, z_max = np.max(unique_vertices, axis=0)
    print(f"\n===== Coordinate Ranges =====")
    print(f"X: [{x_min:.2f}, {x_max:.2f}]")
    print(f"Y: [{y_min:.2f}, {y_max:.2f}]")
    print(f"Z: [{z_min:.2f}, {z_max:.2f}]")

    # Build KD-Tree for fast neighbor search
    print(f"\n===== Building KD-Tree (alpha={alpha}) =====")
    kdtree = KDTree(unique_vertices)

    checked_points = set()
    spheres = []

    # Start sphere detection with progress bar
    print(f"\n===== Starting Sphere Detection =====")
    for idx, point in enumerate(tqdm(unique_vertices, desc="Detection Progress")):
        if idx in checked_points:
            continue

        neighbors = kdtree.query_ball_point(point, alpha)
        neighbor_pts = unique_vertices[neighbors]

        if len(neighbor_pts) < 10:
            continue

        # Compute centroid as candidate sphere center
        center = np.mean(neighbor_pts, axis=0)
        dists = np.linalg.norm(neighbor_pts - center, axis=1)
        radius = np.mean(dists)

        # Check if points lie on a spherical surface
        if np.max(dists) - np.min(dists) <= distance_tolerance:
            spheres.append(((float(center[0]), float(center[1]), float(center[2])), float(radius)))
            checked_points.update(neighbors)
            tqdm.write(f"Sphere detected: Center {center} Radius {radius}")
    
    print(f"\n===== Detection Finished: {len(spheres)} spheres found =====")
    return spheres

def vtk_visualization(stl_path:str, spheres:list[tuple[tuple[float, ...], float]]):
    # VTK 3D Visualization
    print("\n===== Starting VTK Visualization =====")
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(1000, 800)
    renderer = vtk.vtkRenderer()
    renderWindow.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)

    # Render STL model
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.3)
    actor.GetProperty().SetColor(0.7, 0.8, 1)
    renderer.AddActor(actor)

    # Render detected spheres
    for center, radius in spheres:
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(center[0], center[1], center[2])
        sphereSource.SetRadius(radius)
        sphereSource.SetThetaResolution(30)
        sphereSource.SetPhiResolution(30)

        sphereMapper = vtk.vtkPolyDataMapper()
        sphereMapper.SetInputConnection(sphereSource.GetOutputPort())
        sphereActor = vtk.vtkActor()
        sphereActor.SetMapper(sphereMapper)
        sphereActor.GetProperty().SetColor(1, 0.2, 0.2)
        sphereActor.GetProperty().SetOpacity(0.6)
        renderer.AddActor(sphereActor)

    # Window configuration
    renderer.SetBackground(0.1, 0.1, 0.1)
    renderWindow.SetWindowName("STL + Sphere Detection")
    renderWindow.Render()
    interactor.Start()

# Main execution
if __name__ == "__main__":
    STL_FILE = "BONE-1.stl"
    ALPHA = 0.01

    detected_spheres = locate_sphere_in_stl(
        stl_path=STL_FILE,
        alpha=ALPHA
    )

    vtk_visualization(
        stl_path=STL_FILE, 
        spheres=detected_spheres
    )
