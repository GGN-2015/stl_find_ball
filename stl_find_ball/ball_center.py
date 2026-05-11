import numpy as np
from scipy.spatial import KDTree
import trimesh
from tqdm import tqdm
import vtk
import os
from live_chrono import LiveChrono
import math

# -------------------
# Upgrade: Curvature Radius + Normal → Direct Sphere Center Calculation
# -------------------
def estimate_sphere_center(points, center_idx=0):
    pts = np.asarray(points, dtype=np.float64)
    p0 = pts[center_idx]
    neighbors = np.delete(pts, center_idx, 0)
    q = neighbors - p0

    # PCA for normal vector
    cov = q.T @ q / q.shape[0]
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    n = eig_vecs[:, 0]
    n = n / np.linalg.norm(n)

    # Coordinate system
    if np.abs(n[0]) < 0.9:
        u = np.cross(n, [1, 0, 0])
    else:
        u = np.cross(n, [0, 1, 0])
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
    eps = 1e-8
    R_mean = 2.0 / (np.abs(k1) + np.abs(k2) + eps)

    sign = np.sign(z_coords.mean())
    sphere_center = p0 + sign * n * R_mean
    return sphere_center, R_mean

# -------------------
# Union-Find
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
# Subfunctions after Splitting
# -------------------
def load_stl_points(stl_path):
    if not os.path.isfile(stl_path):
        raise FileNotFoundError(f"{stl_path}")
    
    mesh = trimesh.load(stl_path)
    pts = np.unique(mesh.vertices, axis=0).astype(np.float32) # type:ignore
    return mesh, pts

def compute_sphere_centers_and_radii(pts, indices, k):
    n = len(pts)
    sphere_centers = np.zeros((n, 3), dtype=np.float32)
    radii = np.zeros(n, dtype=np.float32)
    for i in tqdm(range(n)):
        nb_pts = pts[indices[i]]
        c, r = estimate_sphere_center(nb_pts, center_idx=0)
        sphere_centers[i] = c
        radii[i] = r
    return sphere_centers, radii

def cluster_by_sphere_center(pts, sphere_centers, radii, indices, dist_tol_ratio):
    n = len(pts)
    uf = UnionFind(n)
    for i in tqdm(range(n)):
        r_i = radii[i]
        c_i = sphere_centers[i]
        max_dist = r_i * dist_tol_ratio
        for j in indices[i]:
            if i == j: continue
            c_j = sphere_centers[j]
            dist = np.linalg.norm(c_i - c_j)
            if dist < max_dist:
                uf.union(i, j)
    return uf

def extract_clusters(uf, min_cluster_size):
    components = {}
    for i in tqdm(range(len(uf.parent))):
        r = uf.find(i)
        if r not in components:
            components[r] = []
        components[r].append(i)
    return [comp for comp in components.values() if len(comp) >= min_cluster_size]

def generate_initial_spheres(clusters, sphere_centers, radii, max_dist, max_ball_siz):
    spheres = []
    for idx in tqdm(range(len(clusters))):
        comp = clusters[idx]
        avg_center = sphere_centers[comp].mean(axis=0)
        avg_radius = np.median(radii[comp])
        if avg_radius <= 0.3 * max_dist and avg_radius <= max_ball_siz:
            spheres.append((avg_center, avg_radius))
    return spheres

# ---------------------------
# SVD Sphere Fitting
# ---------------------------
def fit_sphere_from_points(merge_pts):
    pts = np.asarray(merge_pts, dtype=np.float64)
    n = pts.shape[0]
    if n < 4:
        return None, None

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    A = np.zeros((n, 4))
    b = np.zeros(n)
    for i in range(n):
        xi, yi, zi = x[i], y[i], z[i]
        A[i] = [2*xi, 2*yi, 2*zi, 1.0]
        b[i] = xi**2 + yi**2 + zi**2

    try:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        inv_s = np.diag(1.0 / s)
        coef = Vt.T @ inv_s @ U.T @ b
    except:
        return None, None

    xc, yc, zc, d = coef
    r_sq = xc**2 + yc**2 + zc**2 + d
    if r_sq < 1e-8:
        return None, None
    r = np.sqrt(r_sq)
    return np.array([xc, yc, zc]), r

def get_shell_point_mask(pts, center, r, dist_tol_ratio):
    d = np.linalg.norm(pts - center, axis=1)
    return (d >= (1 - dist_tol_ratio)*r) & (d <= (1 + dist_tol_ratio)*r)

# ========================
# Spherical Shell Point Count Statistics
# ========================
def count_shell_points(pts, center, r, dist_tol_ratio):
    """Count points within the spherical shell range (1±ratio)*r"""
    mask = get_shell_point_mask(pts, center, r, dist_tol_ratio)
    return int(np.sum(mask))

# ---------------------------
# Sphere Merging
# ---------------------------
def merge_overlapping_spheres(pts, spheres_now, dist_tol_ratio, merge_ball_tol_ratio):
    while True:
        spheres_nxt = []
        for idx_now in tqdm(range(len(spheres_now))):
            c, r = spheres_now[idx_now]
            merged = False
            for i in range(len(spheres_nxt)):
                fc, fr = spheres_nxt[i]
                dist = np.linalg.norm(c - fc)
                if dist < max(fr, r):
                    if dist / max(fr, r) <= merge_ball_tol_ratio:
                        # Get point set using abstract function
                        mask1 = get_shell_point_mask(pts, fc, fr, dist_tol_ratio)
                        mask2 = get_shell_point_mask(pts, c, r, dist_tol_ratio)
                        merge_pts = pts[mask1 | mask2]
                        if len(merge_pts) < 10:
                            continue
                        new_c, new_r = fit_sphere_from_points(merge_pts)
                        if new_c is None:
                            continue
                        spheres_nxt[i] = (new_c, new_r)
                        merged = True
                        break
            if not merged:
                spheres_nxt.append((c, r))
        if len(spheres_now) == len(spheres_nxt):
            break
        spheres_now = sorted(spheres_nxt, key=lambda x: -x[1])
    return spheres_now

# ---------------------------
# Sphere Iterative Optimization
# ---------------------------
def refine_spheres(pts, spheres, dist_tol_ratio, max_iter=20):
    optimized = []
    for idx_now in tqdm(range(len(spheres))):
        c0, r0  = spheres[idx_now]
        c, r = np.copy(c0), r0
        for _ in range(max_iter):
            # Count using abstract function
            cnt_now = count_shell_points(pts, c, r, dist_tol_ratio)
            score_now = cnt_now / (r ** 2)
            shell_pts = pts[get_shell_point_mask(pts, c, r, dist_tol_ratio)]

            new_c, new_r = fit_sphere_from_points(shell_pts)
            if new_c is None or (new_r is None or new_r < 1e-6):
                break

            # Count using abstract function
            cnt_nxt = count_shell_points(pts, new_c, new_r, dist_tol_ratio)
            score_nxt = cnt_nxt / (new_r ** 2)

            if score_nxt > score_now:
                c, r = new_c, new_r
            else:
                break
        optimized.append((c, r))
    return optimized

# Calculate the farthest point pair of the point cloud
def compute_max_point_distance(pts):
    # Find extreme points of each axis and calculate the distance between them
    if len(pts) == 0:
        return 0.0
    # Max/min points of each axis
    x_min = pts[np.argmin(pts[:,0])]
    x_max = pts[np.argmax(pts[:,0])]
    y_min = pts[np.argmin(pts[:,1])]
    y_max = pts[np.argmax(pts[:,1])]
    z_min = pts[np.argmin(pts[:,2])]
    z_max = pts[np.argmax(pts[:,2])]
    
    # Only calculate distances between these 6 points to get the approximate farthest point pair (sufficient precision and never out of memory)
    candidates = np.array([x_min, x_max, y_min, y_max, z_min, z_max])
    max_dist = 0.0
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            d = np.linalg.norm(candidates[i] - candidates[j])
            if d > max_dist:
                max_dist = d
    return max_dist

# -------------------
# Main Function
# -------------------
def locate_sphere_in_stl(
    stl_path: str,
    k: int = 20, # Number of neighborhood points
    dist_tol_ratio: float = 0.05, # Distance tolerance ratio
    merge_ball_tol_ratio: float = 0.2, # Sphere merging condition
    min_cluster_size: int = 15, # Minimum cluster size
    max_ball_siz:float = 10.0, # Maximum sphere radius
    max_ball_cnt:int = 8,
) -> list[tuple[tuple[float, float, float] ,float, float]]:
    print(f"Loading model {stl_path} ...")
    mesh, pts = load_stl_points(stl_path)
    print(f"Number of vertices: {len(pts)}")

    # Calculate using the safe function
    print("Calculating the farthest point pair distance ...")
    max_dist = compute_max_point_distance(pts)
    print(f"Farthest point pair distance of point cloud: {max_dist:.4f}")

    with LiveChrono(display_format="Building KDTree, elapsed time: %H:%M:%S"):
        kdt = KDTree(pts)
        _, indices = kdt.query(pts, k=k)

    print("\nAnalyzing curvature radius point by point ...")
    sphere_centers, radii = compute_sphere_centers_and_radii(pts, indices, k)

    print("Building Union-Find structure ...")
    uf = cluster_by_sphere_center(pts, sphere_centers, radii, indices, dist_tol_ratio)

    print("Extracting equivalence classes ...")
    clusters = extract_clusters(uf, min_cluster_size)

    print("Constructing initial sphere set ...")
    spheres = generate_initial_spheres(clusters, sphere_centers, radii, max_dist, max_ball_siz)

    if not spheres:
        return []
    spheres_now = sorted(spheres, key=lambda x: -x[1])

    print(f"\nDetected {len(spheres_now)} initial spheres")
    spheres_now = merge_overlapping_spheres(pts, spheres_now, dist_tol_ratio, merge_ball_tol_ratio)

    round_cnt = 0
    while True:
        round_cnt += 1
        print(f"Round {round_cnt} of sphere merging ...")
        print("    Optimizing sphere size and position ...")
        shpere_nxt = refine_spheres(pts, spheres_now, dist_tol_ratio)
        print("    Merging overlapping spheres ...")
        shpere_nxt = merge_overlapping_spheres(pts, shpere_nxt, dist_tol_ratio, merge_ball_tol_ratio)
        print(f"Remaining spheres: {len(shpere_nxt)} ...")
        if len(spheres_now) == len(shpere_nxt):
            break
        spheres_now = shpere_nxt

    print("Sorting by point density ratio ...")
    spheres_tmp = []
    for _, (c, r) in enumerate(spheres_now):
        pc_rate = count_shell_points(pts, c, r, dist_tol_ratio) / (4 * math.pi * r * r)
        spheres_tmp.append((c, r, pc_rate))
    spheres_tmp.sort(key=lambda x: -x[-1]) # Sort by point density ratio in descending order
    spheres_tmp = spheres_tmp[:min(max_ball_cnt, len(spheres_tmp))]
    spheres = spheres_tmp
    print(f"Final number of spheres: {len(spheres)}")

    return [((float(c[0]), float(c[1]), float(c[2])), float(r), float(pc_rate)) for c, r, pc_rate in spheres]

# -------------------
# VTK Visualization and Segmentation
# -------------------
def color_points_by_shell(polydata, spheres, dist_tol_ratio):
    points = polydata.GetPoints()
    num_pts = points.GetNumberOfPoints()
    is_shell = np.zeros(num_pts, dtype=bool)
    for i in range(num_pts):
        p = np.array(points.GetPoint(i))
        for (c, r, _) in spheres:
            d = np.linalg.norm(p - c)
            if (1 - dist_tol_ratio)*r <= d <= (1 + dist_tol_ratio)*r:
                is_shell[i] = True
                break
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")
    YELLOW = [255,255,0]
    WHITE = [255,255,255]
    for flag in is_shell:
        colors.InsertNextTypedTuple(YELLOW if flag else WHITE)
    polydata.GetPointData().SetScalars(colors)

def create_sphere_actor(c, r, alpha:float):
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
    a.GetProperty().SetOpacity(alpha)
    return a

def vtk_visualization(stl_path: str, spheres: list, dist_tol_ratio: float = 0.1):
    if not os.path.isfile(stl_path):
        raise FileNotFoundError(stl_path)
    print("\nStarting VTK 3D visualization...")

    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(1000, 800)
    ren = vtk.vtkRenderer()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    reader.Update()
    polydata = reader.GetOutput()
    
    print("Spherical shell point count statistics:")
    pc_rate_max = 0
    for i, (c, r, pc_rate) in enumerate(spheres):
        pc_rate_max = max(pc_rate_max, pc_rate)
        print(f"Sphere {i+1:2d} | Radius {r:6.3f} | Point density ratio {pc_rate:6.2f} |")

    color_points_by_shell(polydata, spheres, dist_tol_ratio)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.ScalarVisibilityOn()
    mapper.SetScalarModeToUsePointData()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.5)
    ren.AddActor(actor)

    for c, r, pc_rate in spheres:
        ren.AddActor(create_sphere_actor(c, r, pc_rate / pc_rate_max * 0.8))

    ren.SetBackground(0.1,0.1,0.1)
    renWin.Render()
    iren.Start()

# -------------------
# Main Program
# -------------------
if __name__ == "__main__":
    import os
    DIRNOW = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(DIRNOW)
    TEST_DATA_PATH = os.path.join(ROOT_DIR, "test_data")

    for filename in os.listdir(TEST_DATA_PATH):
        if not filename.endswith(".stl"):
            continue
        print("=" * 60)
        stl_path = os.path.join(TEST_DATA_PATH, filename)
        detected_spheres = locate_sphere_in_stl(
            stl_path=stl_path,
            k=30,
            max_ball_cnt= 8 if filename.find("XH-") != -1 else 4
        )
        vtk_visualization(stl_path, detected_spheres)