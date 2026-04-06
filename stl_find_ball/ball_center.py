import numpy as np
from scipy.spatial import KDTree
import trimesh
from tqdm import tqdm
import vtk

# -------------------
# Curvature Radius Calculation
# -------------------
def estimate_curvature_radius(points, center_idx=0):
    pts = np.asarray(points, dtype=np.float64)
    p0 = pts[center_idx:center_idx+1]
    neighbors = np.delete(pts, center_idx, axis=0)
    q = neighbors - p0
    cov = q.T @ q / q.shape[0]
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    n = eig_vecs[:, 0]

    if np.abs(n[0]) < 0.9:
        u = np.cross(n, [1,0,0])
    else:
        u = np.cross(n, [0,1,0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    u_coords = q @ u
    v_coords = q @ v
    z_coords = q @ n

    A = np.column_stack([u_coords**2, u_coords*v_coords, v_coords**2])
    z = z_coords
    abc, *_ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = abc

    S = np.array([[2*a, b], [b, 2*c]])
    k1, k2 = np.linalg.eigvalsh(S)
    eps = 1e-12
    R1 = 1.0 / (np.abs(k1) + eps)
    R2 = 1.0 / (np.abs(k2) + eps)
    R_mean = 2.0 / (np.abs(k1) + np.abs(k2) + eps)
    return k1, k2, R1, R2, R_mean

# -------------------
# Union-Find Data Structure
# -------------------
class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size)
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, x, y):
        fx, fy = self.find(x), self.find(y)
        if fx != fy:
            self.parent[fy] = fx

# -------------------
# Core Function: Remove small spheres contained in larger ones
# -------------------
def locate_sphere_in_stl(
    stl_path: str,
    k: int = 20,
    radius_rtol: float = 0.1,
    dist_tol_ratio: float = 0.1,
    min_cluster_size: int = 15
):
    # Load mesh model
    mesh = trimesh.load(stl_path)
    pts = np.unique(mesh.vertices, axis=0).astype(np.float32)
    n = len(pts)
    print(f"Number of vertices: {n}")

    # Build KDTree and search neighbors
    print("Building KDTree...")
    kdt = KDTree(pts)
    dists, indices = kdt.query(pts, k=k)

    # Compute curvature radius for each point
    print("Calculating curvature radii...")
    radii = np.zeros(n, dtype=np.float32)
    for i in tqdm(range(n)):
        nb_pts = pts[indices[i]]
        *_, R_mean = estimate_curvature_radius(nb_pts, center_idx=0)
        radii[i] = R_mean

    # Clustering with Union-Find
    print("Union-Find clustering...")
    uf = UnionFind(n)
    for i in tqdm(range(n)):
        for j in indices[i]:
            if i == j: continue
            if abs(radii[i] - radii[j]) / max(radii[i], radii[j], 1e-6) < radius_rtol:
                uf.union(i, j)

    # Extract connected components
    print("Extracting connected components...")
    components = {}
    for i in range(n):
        r = uf.find(i)
        if r not in components:
            components[r] = []
        components[r].append(i)

    # Validate and collect candidate spheres
    print("Validating spherical surfaces...")
    spheres = []
    for comp in components.values():
        if len(comp) < min_cluster_size:
            continue
        cpts = pts[comp]
        center = cpts.mean(axis=0)
        d = np.linalg.norm(cpts - center, axis=1)
        r = np.median(d)
        tol = r * dist_tol_ratio
        in_ratio = np.sum(np.abs(d - r) < tol) / len(cpts)
        if in_ratio >= 0.7:
            spheres.append((center, r))

    # ========================
    # Remove small spheres inside larger ones
    # ========================
    if len(spheres) == 0:
        return []

    # Sort spheres from largest to smallest radius
    spheres.sort(key=lambda x: -x[1])
    final_spheres = []

    for c, r in spheres:
        keep = True
        # Check against all larger spheres already kept
        for fc, fr in final_spheres:
            dist = np.linalg.norm(c - fc)
            # If center is inside a larger sphere, discard this one
            if dist < fr:
                keep = False
                break
        if keep:
            final_spheres.append((c, r))

    # ========================
    # ✅ WRAPPER: Convert ALL numpy types to pure Python list / float
    # ========================
    pure_python_spheres = []
    for center, radius in final_spheres:
        # Convert numpy array -> Python list, numpy float -> Python float
        center_list = [float(coord) for coord in center]
        radius_float = float(radius)
        pure_python_spheres.append((center_list, radius_float))

    # Return pure Python types ONLY
    print(f"\nOriginal detected spheres: {len(spheres)}")
    print(f"Final spheres after filtering: {len(pure_python_spheres)}")
    return pure_python_spheres

# -------------------
# VTK 3D Visualization
# -------------------
def vtk_visualization(stl_path:str, spheres:list):
    print("\nStarting VTK visualization...")
    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(1000, 800)
    ren = vtk.vtkRenderer()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.3)
    actor.GetProperty().SetColor(0.7,0.8,1)
    ren.AddActor(actor)

    for c, r in spheres:
        src = vtk.vtkSphereSource()
        src.SetCenter(*c)
        src.SetRadius(r)
        src.SetThetaResolution(30)
        src.SetPhiResolution(30)
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(src.GetOutputPort())
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(1,0.2,0.2)
        a.GetProperty().SetOpacity(0.6)
        ren.AddActor(a)

    ren.SetBackground(0.1,0.1,0.1)
    renWin.Render()
    iren.Start()

# -------------------
# Main Execution
# -------------------
if __name__ == "__main__":
    detected = locate_sphere_in_stl(stl_path="BONE-1.stl")
    vtk_visualization("BONE-1.stl", detected)