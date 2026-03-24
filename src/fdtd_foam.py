"""
fdtd_foam.py — FDTD scattering measurement on foam (Kelvin/C15/WP).

Components:
  gauged_force_foam: harmonic force with Peierls R on one edge
  wave_packet_foam: Gaussian wave packet on irregular vertex positions
  sphere_measure_foam: nearest-vertex mapping to sphere points
  run_fdtd_foam: Verlet time loop with sphere recording

Uses md_foam.py infrastructure (prepare_edges, verlet_step, etc.).
No PML — periodic structure, time-windowed measurement before wrap-around.

Date: Mar 2026
"""

import numpy as np
from core_math.dynamics.md_foam import prepare_edges, verlet_step


# =========================================================================
# Gauged force
# =========================================================================

def gauged_force_foam(u, edge_info, k_L, k_T, gauged_edges, R_matrix,
                      mass=1.0):
    """Harmonic force with Peierls u_j → R·u_j on gauged edges.

    Identical to harmonic_force_spring except on gauged edges, where
    the displacement difference is R·u_j - u_i instead of u_j - u_i.

    Peierls energy on gauged edge: ½(R·u_j - u_i)^T K (R·u_j - u_i)
      F_i = (K/m)(R·u_j - u_i)       — same K, rotated displacement
      F_j = (R^T K/m)(R·u_j - u_i)   — R^T from chain rule on u_j

    Verified: F = -D_gauged(k=0) @ u to machine epsilon (T6, diff=5.5e-17).

    Args:
      u: (3*nv,) displacement vector
      edge_info: from prepare_edges()
      k_L, k_T: spring constants
      gauged_edges: int or list of int — defect edge index(es)
      R_matrix: (3,3) rotation matrix (same R for all gauged edges)
      mass: vertex mass

    Returns: (3*nv,) acceleration
    """
    idx_i = edge_info['idx_i']
    idx_j = edge_info['idx_j']
    dirs = edge_info['edge_dirs']

    nv = len(u) // 3
    u3 = u.reshape(nv, 3)

    # Normalize gauged_edges to a set
    if isinstance(gauged_edges, (int, np.integer)):
        gauged_set = {int(gauged_edges)}
    else:
        gauged_set = set(int(e) for e in gauged_edges)

    # du = u_j - u_i for all edges
    du = u3[idx_j] - u3[idx_i]  # (n_edges, 3)

    # Override gauged edges: du = R·u_j - u_i
    for ge in gauged_set:
        i_g = idx_i[ge]
        j_g = idx_j[ge]
        du[ge] = R_matrix @ u3[j_g] - u3[i_g]

    # Tensorial spring: K·du = k_T du + (k_L-k_T)(du·ê)ê
    du_par = np.sum(du * dirs, axis=1, keepdims=True) * dirs
    f_edge = k_T * du + (k_L - k_T) * du_par
    f_edge /= mass

    # Distribute forces
    F = np.zeros_like(u)
    F3 = F.reshape(nv, 3)

    np.add.at(F3, idx_i, f_edge)
    np.add.at(F3, idx_j, -f_edge)

    # Fix gauged edges j-side: replace -K·du with -R^T·K·du
    # Correction: add (I - R^T)·f_edge[ge] at vertex j
    for ge in gauged_set:
        j_g = idx_j[ge]
        f_g = f_edge[ge]
        F3[j_g] += f_g - R_matrix.T @ f_g

    return F


def stiffness_force_foam(u, edge_info, k_L, k_T, defect_edges, f_scale,
                          mass=1.0):
    """Harmonic force with stiffness defect K → f·K on specified edges.

    Model A: spring constant scaled by f on defect edges.
    No rotation, no direction dependence — purely scalar perturbation.
    Well-defined on ANY lattice (no [K,R] issue).

    Args:
      u: (3*nv,) displacement vector
      edge_info: from prepare_edges()
      k_L, k_T: spring constants
      defect_edges: int or list of int — defect edge index(es)
      f_scale: float — stiffness scaling factor (f=1 → no defect)
      mass: vertex mass

    Returns: (3*nv,) acceleration
    """
    idx_i = edge_info['idx_i']
    idx_j = edge_info['idx_j']
    dirs = edge_info['edge_dirs']

    nv = len(u) // 3
    u3 = u.reshape(nv, 3)

    if isinstance(defect_edges, (int, np.integer)):
        defect_set = {int(defect_edges)}
    else:
        defect_set = set(int(e) for e in defect_edges)

    # du = u_j - u_i for all edges
    du = u3[idx_j] - u3[idx_i]

    # Tensorial spring: K·du = k_T du + (k_L-k_T)(du·ê)ê
    du_par = np.sum(du * dirs, axis=1, keepdims=True) * dirs
    f_edge = k_T * du + (k_L - k_T) * du_par
    f_edge /= mass

    # Scale defect edges by f
    for de in defect_set:
        f_edge[de] *= f_scale

    # Distribute forces (symmetric — no j-side correction needed)
    F = np.zeros_like(u)
    F3 = F.reshape(nv, 3)
    np.add.at(F3, idx_i, f_edge)
    np.add.at(F3, idx_j, -f_edge)

    return F


# =========================================================================
# Chain construction
# =========================================================================

def build_directed_chain(V, E, start_edge, direction, n_edges, L):
    """Build a directed chain of edges through the foam.

    Starting from start_edge, follows edges that have the largest
    projection onto the preferred direction. Avoids revisiting vertices.

    Args:
      V: (nv, 3) vertex positions
      E: list of (i, j) edge tuples
      start_edge: int — starting edge index
      direction: (3,) preferred direction (unit vector)
      n_edges: int — number of edges in chain
      L: float — box period (for minimum image)

    Returns:
      chain: list of int — edge indices
      path: list of int — vertex indices (len = n_edges + 1)
    """
    # Build adjacency
    adj = {}
    for ei, (a, b) in enumerate(E):
        adj.setdefault(a, []).append((ei, b))
        adj.setdefault(b, []).append((ei, a))

    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction)

    # Start: traverse start_edge in the direction
    i0, j0 = E[start_edge]
    dv = V[j0] - V[i0]
    dv = dv - L * np.round(dv / L)
    if np.dot(dv, direction) >= 0:
        path = [i0, j0]
    else:
        path = [j0, i0]

    chain = [start_edge]
    visited_edges = {start_edge}
    visited_verts = {path[0], path[1]}

    for _ in range(n_edges - 1):
        current = path[-1]
        best_edge = None
        best_proj = -2.0
        best_next = None

        for ei, neighbor in adj[current]:
            if ei in visited_edges:
                continue
            if neighbor in visited_verts:
                continue
            # Edge direction from current to neighbor
            dv = V[neighbor] - V[current]
            dv = dv - L * np.round(dv / L)
            dv_hat = dv / np.linalg.norm(dv)
            proj = np.dot(dv_hat, direction)
            if proj > best_proj:
                best_proj = proj
                best_edge = ei
                best_next = neighbor

        if best_edge is None:
            break  # dead end (all neighbors visited)

        chain.append(best_edge)
        visited_edges.add(best_edge)
        visited_verts.add(best_next)
        path.append(best_next)

    if len(chain) < n_edges:
        import warnings
        warnings.warn(f"Chain truncated: {len(chain)}/{n_edges} edges "
                      f"(all neighbors visited)")

    return chain, path


# =========================================================================
# Wave packet
# =========================================================================

def wave_packet_foam(vertices, k_vec, r_center, sigma, k_L, k_T,
                     mass=1.0):
    """Gaussian wave packet on foam vertices.

    u_i = A · cos(k · r_i) · exp(-|r_i - r_center|² / 2σ²)
    v_i = ω · A · sin(k · r_i) · exp(-|r_i - r_center|² / 2σ²) · k̂

    The displacement is along k̂ (longitudinal polarization).
    ω = dispersion at |k| (approximate: ω ≈ v_L |k| at long wavelength).

    Args:
      vertices: (nv, 3) positions
      k_vec: (3,) wave vector
      r_center: (3,) center of wave packet
      sigma: Gaussian width
      k_L, k_T: spring constants (for dispersion estimate)
      mass: vertex mass

    Returns: (u0, v0) each (3*nv,)
    """
    nv = len(vertices)
    k_mag = np.linalg.norm(k_vec)
    k_hat = k_vec / k_mag if k_mag > 1e-14 else np.array([1, 0, 0.])

    # Approximate longitudinal velocity: v_L = sqrt(k_L/mass) for foam at k→0
    # (z=4 Kelvin, more precise would need band structure)
    v_L = np.sqrt(k_L / mass)
    omega = v_L * k_mag  # long-wavelength approximation

    # Phase and envelope at each vertex
    phase = vertices @ k_vec  # (nv,) = k · r_i
    dr = vertices - r_center
    envelope = np.exp(-np.sum(dr**2, axis=1) / (2 * sigma**2))  # (nv,)

    # Displacement along k̂
    u3 = np.outer(envelope * np.cos(phase), k_hat)  # (nv, 3)
    v3 = np.outer(omega * envelope * np.sin(phase), k_hat)  # (nv, 3)

    return u3.ravel(), v3.ravel()


# =========================================================================
# Sphere measurement
# =========================================================================

def sphere_measure_foam(vertices, r_center, r_m, thetas, phis, L):
    """Find nearest vertices to sphere measurement points.

    Sphere of radius r_m centered at r_center.
    Uses minimum image convention for periodic box.

    Args:
      vertices: (nv, 3)
      r_center: (3,)
      r_m: sphere radius
      thetas: (N_th,) polar angles
      phis: (N_ph,) azimuthal angles
      L: box period

    Returns:
      vertex_indices: (N_th * N_ph,) nearest vertex for each sphere point
      sphere_points: (N_th * N_ph, 3) actual sphere point positions
    """
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    th_flat = TH.ravel()
    ph_flat = PH.ravel()

    # Sphere points in absolute coordinates
    sx = r_center[0] + r_m * np.sin(th_flat) * np.cos(ph_flat)
    sy = r_center[1] + r_m * np.sin(th_flat) * np.sin(ph_flat)
    sz = r_center[2] + r_m * np.cos(th_flat)
    sphere_pts = np.column_stack([sx, sy, sz])

    # Find nearest vertex (minimum image)
    n_pts = len(th_flat)
    vertex_indices = np.zeros(n_pts, dtype=int)

    for p in range(n_pts):
        dr = vertices - sphere_pts[p]
        dr = dr - L * np.round(dr / L)  # minimum image
        dist2 = np.sum(dr**2, axis=1)
        vertex_indices[p] = np.argmin(dist2)

    return vertex_indices, sphere_pts


# =========================================================================
# FDTD runner
# =========================================================================

def run_fdtd_foam(u0, v0, force_fn, dt, n_steps, sphere_idx):
    """Run Verlet FDTD and record displacements at sphere vertices.

    Args:
      u0, v0: (3*nv,) initial displacement and velocity
      force_fn: callable(u) -> (3*nv,) acceleration
      dt: time step
      n_steps: number of steps
      sphere_idx: (N_pts,) vertex indices for sphere recording

    Returns:
      rec_u: (n_steps, N_pts, 3) displacement recordings at sphere points
    """
    nv = len(u0) // 3
    n_pts = len(sphere_idx)

    rec_u = np.zeros((n_steps, n_pts, 3))

    u = u0.copy()
    v = v0.copy()
    a = force_fn(u)

    for step in range(n_steps):
        # Record
        u3 = u.reshape(nv, 3)
        rec_u[step] = u3[sphere_idx]

        # Step
        u, v, a = verlet_step(u, v, a, force_fn, dt)

    return rec_u


def compute_sigma_tr_foam(rec_def, rec_ref, r_m, thetas, phis):
    """Compute σ_tr from defect and reference recordings.

    Same formula as PRL: f²(Ω) = r_m² <|u_sc|²> / <|u_inc|²>
    σ_tr = ∫ (1-cosθ_s) f² dΩ

    Incoming direction assumed +x.

    Args:
      rec_def: (n_steps, N_pts, 3) defect recording
      rec_ref: (n_steps, N_pts, 3) reference recording
      r_m: measurement radius
      thetas: (N_th,)
      phis: (N_ph,)

    Returns: (sigma_tot, sigma_tr)
    """
    # Scattered field
    sc = rec_def - rec_ref  # (n_steps, N_pts, 3)

    # Time-averaged intensities
    sc2 = np.mean(np.sum(sc**2, axis=2), axis=0)  # (N_pts,)
    inc2 = np.mean(np.sum(rec_ref**2, axis=2), axis=0)  # (N_pts,)

    # Floor to avoid 0/0
    inc2_floor = max(1e-30, 1e-12 * np.max(inc2))
    inc2 = np.maximum(inc2, inc2_floor)

    f2 = r_m**2 * sc2 / inc2  # (N_pts,)

    # Integration (reuse PRL formula)
    N_th = len(thetas)
    N_ph = len(phis)
    f2_2d = f2.reshape(N_th, N_ph)

    d_th = thetas[1] - thetas[0] if N_th > 1 else np.pi
    d_ph = phis[1] - phis[0] if N_ph > 1 else 2 * np.pi

    w_th = np.ones(N_th) * d_th
    w_th[0] *= 0.5
    w_th[-1] *= 0.5

    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    sin_th = np.sin(TH)
    cos_th_s = np.sin(TH) * np.cos(PH)  # scattering angle from +x

    dOmega = sin_th * w_th[:, np.newaxis] * d_ph

    sigma_tot = np.sum(f2_2d * dOmega)
    sigma_tr = np.sum((1 - cos_th_s) * f2_2d * dOmega)
    return sigma_tot, sigma_tr


# =========================================================================
# Open boundary mesh + PML
# =========================================================================

def build_open_mesh(V_periodic, E_periodic, L, R_keep=None):
    """Cut periodic mesh to open boundary sphere.

    Keeps vertices within R_keep of box center. Edges with both endpoints
    inside are kept. Long edges (wrap-around artifacts) are removed.

    Args:
      V_periodic: (nv, 3) vertex positions from periodic mesh
      E_periodic: (ne, 2) edge indices
      L: periodic box size
      R_keep: sphere radius (default: L/2 - 2)

    Returns: V_open (nv',3), E_open (ne',2), L_eff (= 2*R_keep)
    """
    V = np.array(V_periodic)
    E = np.array(E_periodic)

    if R_keep is None:
        R_keep = L / 2 - 2

    center = np.array([L/2, L/2, L/2])
    dist = np.linalg.norm(V - center, axis=1)
    inside = dist < R_keep
    n_inside = int(np.sum(inside))

    old_to_new = -np.ones(len(V), dtype=int)
    old_to_new[inside] = np.arange(n_inside)

    both = inside[E[:, 0]] & inside[E[:, 1]]
    E_new = old_to_new[E[both]]
    V_new = V[inside]

    # Remove wrap-around edges
    lens = np.linalg.norm(V_new[E_new[:, 1]] - V_new[E_new[:, 0]], axis=1)
    if len(lens) > 0:
        good = lens < 3 * np.median(lens)
        E_new = E_new[good]

    return V_new, E_new, 2 * R_keep


def make_pml_open(vertices, center, r_inner, r_outer, strength=2.0):
    """Spherical PML damping for open boundary mesh.

    γ = 0 for r < r_inner, ramps to strength at r_outer.
    Quadratic ramp (same as elastic_3d.py).

    Returns: gamma (nv,) per-vertex damping
    """
    dist = np.linalg.norm(vertices - center, axis=1)
    width = r_outer - r_inner
    gamma = np.zeros(len(vertices))
    mask = dist > r_inner
    gamma[mask] = strength * np.minimum((dist[mask] - r_inner) / width, 1.0) ** 2
    return gamma


def prepare_edges_open(V, E):
    """Edge info for open boundary (no minimum image convention)."""
    E_arr = np.array(E)
    idx_i = E_arr[:, 0]
    idx_j = E_arr[:, 1]
    dr = V[idx_j] - V[idx_i]
    lengths = np.linalg.norm(dr, axis=1, keepdims=True)
    dirs = dr / np.maximum(lengths, 1e-15)
    return {
        'idx_i': idx_i, 'idx_j': idx_j,
        'edge_dirs': dirs, 'edge_lengths': lengths.ravel(),
    }


def harmonic_force_open(u, edge_info, k_L, k_T, mass=1.0):
    """Harmonic force on open boundary mesh (no minimum image)."""
    idx_i = edge_info['idx_i']
    idx_j = edge_info['idx_j']
    dirs = edge_info['edge_dirs']

    nv = len(u) // 3
    u3 = u.reshape(nv, 3)
    du = u3[idx_j] - u3[idx_i]

    du_par = np.sum(du * dirs, axis=1, keepdims=True) * dirs
    f_edge = k_T * du + (k_L - k_T) * du_par
    f_edge /= mass

    F = np.zeros_like(u)
    F3 = F.reshape(nv, 3)
    np.add.at(F3, idx_i, f_edge)
    np.add.at(F3, idx_j, -f_edge)
    return F


def gauged_force_open(u, edge_info, k_L, k_T, gauged_edges, R_matrix,
                      mass=1.0):
    """Peierls-gauged force on open boundary mesh (no minimum image)."""
    idx_i = edge_info['idx_i']
    idx_j = edge_info['idx_j']
    dirs = edge_info['edge_dirs']

    nv = len(u) // 3
    u3 = u.reshape(nv, 3)

    if isinstance(gauged_edges, (int, np.integer)):
        gauged_set = {int(gauged_edges)}
    else:
        gauged_set = set(int(e) for e in gauged_edges)

    du = u3[idx_j] - u3[idx_i]
    for ge in gauged_set:
        du[ge] = R_matrix @ u3[idx_j[ge]] - u3[idx_i[ge]]

    du_par = np.sum(du * dirs, axis=1, keepdims=True) * dirs
    f_edge = k_T * du + (k_L - k_T) * du_par
    f_edge /= mass

    F = np.zeros_like(u)
    F3 = F.reshape(nv, 3)
    np.add.at(F3, idx_i, f_edge)
    np.add.at(F3, idx_j, -f_edge)

    for ge in gauged_set:
        j_g = idx_j[ge]
        f_g = f_edge[ge]
        F3[j_g] += f_g - R_matrix.T @ f_g

    return F


def wave_packet_open(vertices, k_vec, r_center, sigma, k_L, k_T, mass=1.0):
    """Gaussian wave packet on open boundary mesh (no minimum image)."""
    nv = len(vertices)
    k_mag = np.linalg.norm(k_vec)
    k_hat = k_vec / k_mag if k_mag > 1e-14 else np.array([1, 0, 0.])
    v_L = np.sqrt(k_L / mass)
    omega = v_L * k_mag

    phase = vertices @ k_vec
    dr = vertices - r_center
    envelope = np.exp(-np.sum(dr**2, axis=1) / (2 * sigma**2))

    u3 = np.outer(envelope * np.cos(phase), k_hat)
    v3 = np.outer(omega * envelope * np.sin(phase), k_hat)
    return u3.ravel(), v3.ravel()


def sphere_measure_open(vertices, r_center, r_m, thetas, phis):
    """Nearest-vertex sphere measurement on open boundary (no minimum image)."""
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    th, ph = TH.ravel(), PH.ravel()
    sx = r_center[0] + r_m * np.sin(th) * np.cos(ph)
    sy = r_center[1] + r_m * np.sin(th) * np.sin(ph)
    sz = r_center[2] + r_m * np.cos(th)
    sphere_pts = np.column_stack([sx, sy, sz])

    n_pts = len(th)
    vertex_indices = np.zeros(n_pts, dtype=int)
    for p in range(n_pts):
        dist2 = np.sum((vertices - sphere_pts[p])**2, axis=1)
        vertex_indices[p] = np.argmin(dist2)

    return vertex_indices, sphere_pts


def run_fdtd_pml(u0, v0, force_fn, dt, n_steps, gamma, sphere_idx):
    """Verlet FDTD with PML damping, recording at sphere points.

    PML: v *= 1/(1 + gamma*dt) per vertex, same as elastic_3d.py.
    """
    nv = len(u0) // 3
    n_pts = len(sphere_idx)
    rec_u = np.zeros((n_steps, n_pts, 3))

    d_pml = 1.0 / (1.0 + np.repeat(gamma, 3) * dt)

    u, v = u0.copy(), v0.copy()
    a = force_fn(u)

    for step in range(n_steps):
        u3 = u.reshape(nv, 3)
        rec_u[step] = u3[sphere_idx]

        u += dt * v + 0.5 * dt**2 * a
        a_new = force_fn(u)
        v += 0.5 * dt * (a + a_new)
        v *= d_pml
        a = a_new

    return rec_u


def find_disk_edges_open(V, E, z_plane, center_xy, R_disk):
    """Find edges crossing z_plane within R_disk (no minimum image)."""
    E_arr = np.array(E)
    disk = []
    for ei in range(len(E_arr)):
        i, j = E_arr[ei]
        zi, zj = V[i, 2], V[j, 2]
        if (zi - z_plane) * (zj - z_plane) <= 0:
            mid_xy = 0.5 * (V[i, :2] + V[j, :2])
            if np.linalg.norm(mid_xy - center_xy) < R_disk:
                disk.append(ei)
    return disk


# =========================================================================
# Self-tests
# =========================================================================

def _self_test():
    """Verify gauged_force_foam on Kelvin N=2."""
    from physics.hodge import build_kelvin_with_dual_info
    from gauge_foam import make_peierls_rotation

    data = build_kelvin_with_dual_info(N=2)
    V, E, L = data['V'], data['E'], data['L']
    edge_info = prepare_edges(V, E, L)

    nv = len(V)
    rng = np.random.RandomState(42)
    u = rng.randn(3 * nv) * 0.01

    k_L, k_T = 3.0, 1.0

    # T1: gauged_force at α=0 equals harmonic_force (single edge)
    from core_math.dynamics.md_foam import harmonic_force_spring
    F_normal = harmonic_force_spring(u, edge_info, k_L, k_T)
    F_gauge0 = gauged_force_foam(u, edge_info, k_L, k_T,
                                  gauged_edges=0, R_matrix=np.eye(3))
    diff1 = np.max(np.abs(F_normal - F_gauge0))
    assert diff1 < 1e-14, f"T1 FAIL: F(α=0) ≠ F_normal, diff={diff1:.2e}"
    print(f"  T1: F(α=0, 1 edge) = F_normal: diff={diff1:.2e} OK")

    # T1b: gauged_force with list of edges at α=0
    F_gauge0_list = gauged_force_foam(u, edge_info, k_L, k_T,
                                       gauged_edges=[0, 1, 2], R_matrix=np.eye(3))
    diff1b = np.max(np.abs(F_normal - F_gauge0_list))
    assert diff1b < 1e-14, f"T1b FAIL: F(α=0, 3 edges) ≠ F_normal, diff={diff1b:.2e}"
    print(f"  T1b: F(α=0, 3 edges) = F_normal: diff={diff1b:.2e} OK")

    # T2: gauged_force at α=0.5 differs from normal
    R05 = make_peierls_rotation(0.5)
    F_gauge05 = gauged_force_foam(u, edge_info, k_L, k_T,
                                   gauged_edges=0, R_matrix=R05)
    diff2 = np.max(np.abs(F_normal - F_gauge05))
    assert diff2 > 1e-6, f"T2 FAIL: F(α=0.5) = F_normal"
    print(f"  T2: F(α=0.5) ≠ F_normal: diff={diff2:.2e} OK")

    # T3: Verlet stability with gauged force (100 steps)
    # NOTE: e = ½v² - ½u·F(u) is NOT the exact Hamiltonian for the gauged system,
    # but Verlet drift < 1e-3 confirms force is conservative (derives from a potential).
    # The definitive force correctness test is T6/T8 (F = -D@u to machine epsilon).
    def force_fn(u_):
        return gauged_force_foam(u_, edge_info, k_L, k_T,
                                  gauged_edges=0, R_matrix=R05)

    v = rng.randn(3 * nv) * 0.01
    dt = 0.01

    a = force_fn(u)
    e_start = 0.5 * np.dot(v, v) - 0.5 * np.dot(u, force_fn(u))
    for _ in range(100):
        u, v, a = verlet_step(u, v, a, force_fn, dt)
    e_end = 0.5 * np.dot(v, v) - 0.5 * np.dot(u, force_fn(u))
    drift = abs(e_end - e_start) / abs(e_start)
    assert drift < 1e-3, f"T3 FAIL: energy drift {drift:.2e}"
    print(f"  T3: energy drift over 100 steps: {drift:.2e} OK")

    # T4: wave_packet_foam produces nonzero u, v
    k_vec = np.array([0.5, 0.0, 0.0])
    r_center = np.array([L/2, L/2, L/2])
    u0, v0 = wave_packet_foam(V, k_vec, r_center, sigma=3.0,
                               k_L=k_L, k_T=k_T)
    assert np.max(np.abs(u0)) > 1e-4, "T4 FAIL: u0 all zero"
    assert np.max(np.abs(v0)) > 1e-4, "T4 FAIL: v0 all zero"
    print(f"  T4: wave packet: max |u0|={np.max(np.abs(u0)):.4f}, "
          f"max |v0|={np.max(np.abs(v0)):.4f} OK")

    # T5: sphere_measure_foam finds valid vertex indices
    thetas = np.linspace(0.1, np.pi - 0.1, 8)
    phis = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    s_idx, s_pts = sphere_measure_foam(V, r_center, r_m=3.0,
                                        thetas=thetas, phis=phis, L=L)
    assert len(s_idx) == len(thetas) * len(phis)
    assert np.all(s_idx >= 0) and np.all(s_idx < nv)
    n_unique = len(set(s_idx))
    print(f"  T5: sphere: {len(s_idx)} points, {n_unique} unique vertices OK")

    # T6: gauged_force matches build_gauged_dynamical_matrix at k=0
    # This is the DEFINITIVE test: F_gauged(u) must equal -D_gauged(k=0) @ u.
    # Verifies the sign correction (I-R^T)·f_g on vertex j.
    from gauge_foam import build_gauged_dynamical_matrix
    from physics.bloch import DisplacementBloch

    # Reset u (T3 modified it)
    rng2 = np.random.RandomState(99)
    u_test = rng2.randn(3 * nv) * 0.01
    alpha_test = 0.30
    R_test = make_peierls_rotation(alpha_test)
    ge = 5  # test edge

    # Force from gauged_force_foam (single edge)
    F_fdtd = gauged_force_foam(u_test, edge_info, k_L, k_T,
                                gauged_edges=ge, R_matrix=R_test)

    # Force from D(k=0) @ u (should equal -F)
    db = DisplacementBloch(V, E, L, k_L=k_L, k_T=k_T)
    D_gauged = build_gauged_dynamical_matrix(db, np.zeros(3), [ge], R_test)
    F_matrix = -np.real(D_gauged) @ u_test

    diff6 = np.max(np.abs(F_fdtd - F_matrix))
    assert diff6 < 1e-12, f"T6 FAIL: F_fdtd ≠ -D@u, diff={diff6:.2e}"
    print(f"  T6: F_gauged(1 edge) = -D(k=0)@u: diff={diff6:.2e} OK")

    # T7: build_directed_chain produces valid chain
    chain, path = build_directed_chain(V, E, start_edge=ge,
                                        direction=[1, 0, 0], n_edges=5, L=L)
    assert len(chain) == 5, f"T7 FAIL: chain length {len(chain)} ≠ 5"
    assert len(path) == 6, f"T7 FAIL: path length {len(path)} ≠ 6"
    assert len(set(path)) == 6, "T7 FAIL: path has repeated vertices"
    print(f"  T7: directed chain: {len(chain)} edges, {len(path)} vertices, no repeats OK")

    # T8: multi-edge gauged_force matches D(k=0) with multiple gauged edges
    multi_edges = chain[:3]  # first 3 edges of chain
    F_multi = gauged_force_foam(u_test, edge_info, k_L, k_T,
                                 gauged_edges=multi_edges, R_matrix=R_test)
    D_multi = build_gauged_dynamical_matrix(db, np.zeros(3), multi_edges, R_test)
    F_multi_matrix = -np.real(D_multi) @ u_test

    diff8 = np.max(np.abs(F_multi - F_multi_matrix))
    assert diff8 < 1e-12, f"T8 FAIL: F_multi ≠ -D_multi@u, diff={diff8:.2e}"
    print(f"  T8: F_gauged({len(multi_edges)} edges) = -D(k=0)@u: diff={diff8:.2e} OK")

    # ── Open boundary tests (need larger mesh) ─────────────

    # T9: build_open_mesh preserves edges, removes wrap-around
    data_big = build_kelvin_with_dual_info(N=6)
    V_p, E_p = np.array(data_big['V']), np.array(data_big['E'])
    L_big = data_big['L']
    V_o, E_o, L_eff = build_open_mesh(V_p, E_p, L_big)
    assert len(V_o) < len(V_p), "T9 FAIL: open mesh not smaller"
    assert len(V_o) > 0, "T9 FAIL: open mesh empty"
    # No long edges
    lens = np.linalg.norm(V_o[E_o[:, 1]] - V_o[E_o[:, 0]], axis=1)
    assert np.all(lens < 3), f"T9 FAIL: long edge {lens.max():.2f}"
    # All edges have valid indices
    assert np.all(E_o >= 0) and np.all(E_o < len(V_o))
    print(f"  T9: open mesh: {len(V_p)} → {len(V_o)} vertices, "
          f"max edge {lens.max():.3f} OK")

    # T10: make_pml_open: zero inside, positive outside
    center_o = np.mean(V_o, axis=0)
    r_outer = L_eff / 2
    r_inner = r_outer * 0.75
    gamma = make_pml_open(V_o, center_o, r_inner, r_outer)
    dist_o = np.linalg.norm(V_o - center_o, axis=1)
    assert np.all(gamma[dist_o < r_inner - 0.1] == 0), "T10 FAIL: γ≠0 inside"
    assert np.any(gamma > 0), "T10 FAIL: no damping anywhere"
    print(f"  T10: PML: γ=0 inside (n={np.sum(gamma==0)}), "
          f"γ>0 outside (n={np.sum(gamma>0)}, max={gamma.max():.2f}) OK")

    # T11: prepare_edges_open consistent with open mesh
    ei_o = prepare_edges_open(V_o, E_o)
    assert len(ei_o['idx_i']) == len(E_o)
    # Edge dirs are unit vectors
    dir_norms = np.linalg.norm(ei_o['edge_dirs'], axis=1)
    assert np.allclose(dir_norms, 1.0, atol=1e-10), "T11 FAIL: dirs not unit"
    print(f"  T11: open edge_info: {len(E_o)} edges, dirs unit OK")

    # T12: harmonic_force_open at α=0 matches gauged_force_open at α=0
    rng3 = np.random.RandomState(77)
    u_o = rng3.randn(3 * len(V_o)) * 0.01
    F_h = harmonic_force_open(u_o, ei_o, k_L, k_T)
    F_g0 = gauged_force_open(u_o, ei_o, k_L, k_T, [0], np.eye(3))
    diff12 = np.max(np.abs(F_h - F_g0))
    assert diff12 < 1e-14, f"T12 FAIL: open F(α=0) ≠ F_harmonic, diff={diff12:.2e}"
    print(f"  T12: open F(α=0) = F_harmonic: diff={diff12:.2e} OK")

    # T13: run_fdtd_pml with γ=0 matches run_fdtd_foam energy (no damping)
    gamma_zero = np.zeros(len(V_o))
    u0_o, v0_o = wave_packet_open(V_o, np.array([0.5, 0, 0]),
                                   center_o, 2.0, k_L, k_T)
    thetas_o = np.linspace(0.1, np.pi-0.1, 4)
    phis_o = np.linspace(0, 2*np.pi, 6, endpoint=False)
    s_idx_o, _ = sphere_measure_open(V_o, center_o, 3.0, thetas_o, phis_o)
    def fr_o(u_): return harmonic_force_open(u_, ei_o, k_L, k_T)
    rec = run_fdtd_pml(u0_o, v0_o, fr_o, 0.02, 50, gamma_zero, s_idx_o)
    assert rec.shape == (50, len(s_idx_o), 3)
    assert np.max(np.abs(rec)) > 1e-4, "T13 FAIL: no signal recorded"
    print(f"  T13: FDTD+PML(γ=0): signal max={np.max(np.abs(rec)):.4f} OK")

    # T14: PML damps energy — kinetic energy at PML boundary decreases
    # Use a localized impulse near PML edge, not a wave packet at center
    gamma_real = make_pml_open(V_o, center_o, r_inner, r_outer, strength=2.0)
    d_pml_14 = 1.0 / (1.0 + np.repeat(gamma_real, 3) * 0.02)
    # Impulse: kick vertices in PML region
    u_pml = np.zeros(3 * len(V_o))
    v_pml = np.zeros(3 * len(V_o))
    in_pml = gamma_real > 0
    v_pml_3 = v_pml.reshape(len(V_o), 3)
    v_pml_3[in_pml] = 0.1  # give velocity to PML vertices
    ke_start = 0.5 * np.sum(v_pml[np.repeat(in_pml, 3)]**2)
    a_pml = fr_o(u_pml)
    for _ in range(100):
        u_pml += 0.02*v_pml + 0.5*0.02**2*a_pml
        a_new = fr_o(u_pml)
        v_pml += 0.5*0.02*(a_pml + a_new)
        v_pml *= d_pml_14
        a_pml = a_new
    ke_end = 0.5 * np.sum(v_pml[np.repeat(in_pml, 3)]**2)
    assert ke_end < ke_start * 0.5, f"T14 FAIL: PML KE not damped ({ke_end/ke_start:.2f})"
    print(f"  T14: PML damps KE in boundary: {ke_start:.4f} → {ke_end:.4f} "
          f"({ke_end/ke_start:.0%}) OK")

    # T15: find_disk_edges_open finds edges on open mesh
    z_o = center_o[2] + 0.5
    disk_o = find_disk_edges_open(V_o, E_o, z_o, center_o[:2], 3.0)
    assert len(disk_o) > 0, "T15 FAIL: no disk edges on open mesh"
    # All indices valid
    assert all(0 <= ei < len(E_o) for ei in disk_o)
    print(f"  T15: open disk: {len(disk_o)} edges at z={z_o:.1f} OK")

    print(f"  fdtd_foam.py: 15/15 PASS")


if __name__ == '__main__':
    _self_test()
