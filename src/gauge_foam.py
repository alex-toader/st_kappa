"""
gauge_foam.py — Peierls gauge on arbitrary polyhedral mesh (foam/Voronoi).

Generalization of vortex/gauge_3d.py (SC cubic only) to arbitrary
graph-based meshes. Applies a rotation R(2πα) to elastic bonds
crossing a surface bounded by a loop.

Key difference from SC cubic:
  - Edges are not axis-aligned → rotation plane depends on edge direction
  - "Disk" is a collection of faces, not a flat circle
  - Bond identification uses edge-face topology, not grid indices

Functions:
  find_edges_crossing_plane(V, E, L, z_plane, center_xy, R_disk)
    → edges whose midpoint crosses a horizontal plane within radius R_disk

  make_peierls_rotation(alpha, axis)
    → 3×3 rotation matrix R(2πα) around given axis

  build_gauged_dynamical_matrix(db, k, gauged_edges, alpha, rotation_axis)
    → modified D(k) with Peierls rotation on specified edges

For st_base sync: this module is generic (works on any periodic mesh).
Could be added to st_base/src/physics/ or st_base/src/vortex/.

Date: Mar 2026
"""
import numpy as np


def find_edges_crossing_plane(vertices, edges, L, z_plane,
                               center_xy=None, R_disk=None):
    """Find edges that cross a horizontal plane z = z_plane.

    An edge (i,j) crosses the plane if, using minimum image convention,
    one endpoint is below z_plane and the other above.

    Optional: restrict to edges whose midpoint (xy) is within R_disk
    of center_xy (circular disk selection, analogous to SC gauge_3d).

    Parameters
    ----------
    vertices : (V, 3) array
    edges : list of (i, j) tuples
    L : float — period (cubic box assumed)
    z_plane : float — z coordinate of the plane
    center_xy : (2,) array or None — center of disk in xy
    R_disk : float or None — radius of disk

    Returns
    -------
    crossing_indices : list of int — indices into edges list
    crossing_info : list of dict with keys:
        'edge_idx', 'i', 'j', 'i_below', 'j_above',
        'midpoint', 'edge_vector', 'edge_hat'
    """
    if center_xy is None:
        center_xy = np.array([L / 2, L / 2])

    crossing_indices = []
    crossing_info = []

    for e_idx, (i, j) in enumerate(edges):
        vi, vj = vertices[i], vertices[j]

        # Minimum image edge vector
        dv = vj - vi
        dv = dv - L * np.round(dv / L)

        # Effective positions: vi and vi + dv
        zi = vi[2]
        zj_eff = zi + dv[2]

        # Does the edge cross z_plane?
        # Check if z_plane is strictly between zi and zj_eff
        z_lo, z_hi = min(zi, zj_eff), max(zi, zj_eff)
        if not (z_lo < z_plane <= z_hi):
            continue

        # Midpoint
        t = (z_plane - zi) / dv[2] if abs(dv[2]) > 1e-14 else 0.5
        midpoint = vi + t * dv
        midpoint[:2] = midpoint[:2] % L  # wrap xy

        # Disk filter
        if R_disk is not None:
            dx_xy = midpoint[:2] - center_xy
            dx_xy = dx_xy - L * np.round(dx_xy / L)
            if np.linalg.norm(dx_xy) > R_disk:
                continue

        # Edge unit vector
        edge_len = np.linalg.norm(dv)
        edge_hat = dv / edge_len if edge_len > 1e-14 else dv

        # Which vertex is below, which above
        if zi < z_plane:
            i_below, j_above = i, j
        else:
            i_below, j_above = j, i

        crossing_indices.append(e_idx)
        crossing_info.append({
            'edge_idx': e_idx,
            'i': i, 'j': j,
            'i_below': i_below, 'j_above': j_above,
            'midpoint': midpoint,
            'edge_vector': dv,
            'edge_hat': edge_hat,
        })

    return crossing_indices, crossing_info


def make_peierls_rotation(alpha, axis=None):
    """Build 3×3 rotation matrix R(2πα) around given axis.

    Parameters
    ----------
    alpha : float — vortex strength (0.5 = Z₂)
    axis : (3,) array or None — rotation axis (default: z-axis)

    Returns
    -------
    R : (3, 3) array — rotation matrix
    """
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0])
    axis = axis / np.linalg.norm(axis)

    phi = 2 * np.pi * (float(alpha) % 1.0)
    c = np.cos(phi)
    s = np.sin(phi)

    # Snap near-zero values (numerical precision at half-integer alpha)
    if abs(s) < 1e-12:
        s = 0.0
    if abs(c - 1) < 1e-12:
        c = 1.0
    if abs(c + 1) < 1e-12:
        c = -1.0

    # Rodrigues' rotation formula: R = I cos(φ) + (1-cos(φ)) n⊗n + sin(φ) [n]×
    nx, ny, nz = axis
    K = np.array([[0, -nz, ny],
                   [nz, 0, -nx],
                   [-ny, nx, 0]])
    R = c * np.eye(3) + (1 - c) * np.outer(axis, axis) + s * K
    return R


def build_gauged_dynamical_matrix(db, k, gauged_edges, R_matrix):
    """Build dynamical matrix with Peierls rotation on specified edges.

    Takes a DisplacementBloch object and modifies the coupling for
    edges in gauged_edges by applying rotation R_matrix.

    On a gauged edge (i,j):
      Normal coupling: K · (u_j - u_i) with phase exp(ik·n·L)
      Gauged coupling: K · (R · u_j - u_i) with phase exp(ik·n·L)

    This replaces u_j → R · u_j on the gauged bonds only.
    The stiffness matrix K_e = k_T I + (k_L - k_T) ê⊗ê is unchanged.

    Parameters
    ----------
    db : DisplacementBloch — the elastic system
    k : (3,) wave vector
    gauged_edges : set or list of int — edge indices to gauge
    R_matrix : (3, 3) — rotation matrix to apply

    Returns
    -------
    D : (3V, 3V) complex Hermitian — modified dynamical matrix
    """
    V_count = db.V
    D = np.zeros((3 * V_count, 3 * V_count), dtype=complex)

    gauged_set = set(gauged_edges)

    for e_idx, (i, j) in enumerate(db.edges):
        e_hat = db.edge_vectors[e_idx]
        n = db.crossings[e_idx]

        phase = np.exp(1j * np.dot(k, n * db.L))

        # Spring tensor: K_ab = k_T δ_ab + (k_L - k_T) ê_a ê_b
        K_edge = db.k_T * np.eye(3) + (db.k_L - db.k_T) * np.outer(e_hat, e_hat)
        K_edge = K_edge / db.mass

        # Indices in 3V×3V matrix
        ii = slice(3*i, 3*i+3)
        jj = slice(3*j, 3*j+3)

        if e_idx in gauged_set:
            # Gauged bond: Peierls substitution u_j → R·u_j (Volterra process).
            # Energy: ½(R·u_j - u_i)^T K (R·u_j - u_i)
            # F_i = K(R·u_j - u_i),  F_j = R^T K(R·u_j - u_i)
            #
            # D[i,i] += K           (i-side: lattice unrotated)
            # D[j,j] += R^T K R     (j-side: lattice rotated)
            # D[i,j] -= K R phase   (off-diagonal)
            # D[j,i] -= R^T K phase* (off-diagonal, Hermitian conjugate)
            #
            # Hermitian: D[i,j]† = -(KR·phase)† = -R^T K phase* = D[j,i] ✓
            K_rotated = R_matrix.T @ K_edge @ R_matrix  # R^T K R

            D[ii, ii] += K_edge           # unrotated side
            D[jj, jj] += K_rotated        # rotated side
            D[ii, jj] -= K_edge @ R_matrix * phase       # K·R
            D[jj, ii] -= R_matrix.T @ K_edge * np.conj(phase)  # R^T·K
        else:
            # Normal bond (matching DisplacementBloch convention)
            D[ii, ii] += K_edge
            D[ii, jj] -= K_edge * phase

            D[jj, jj] += K_edge
            D[jj, ii] -= K_edge * np.conj(phase)

    return D


# =========================================================================
# Self-tests
# =========================================================================
def _test_init():
    """Verify gauge_foam on Kelvin N=2."""
    import sys
    sys.path.insert(0, '.')
    from physics.hodge import build_kelvin_with_dual_info
    from physics.bloch import DisplacementBloch

    errors = []
    data = build_kelvin_with_dual_info(N=2)
    V, E, L = data['V'], data['E'], data['L']

    # T1: find edges crossing z=L/2
    idx, info = find_edges_crossing_plane(V, E, L, L/2)
    if len(idx) == 0:
        errors.append("T1: no edges cross z=L/2")
    print(f"  T1: {len(idx)} edges cross z=L/2")

    # T2: with disk R=3, fewer edges
    idx_disk, _ = find_edges_crossing_plane(V, E, L, L/2,
                                             center_xy=np.array([L/2, L/2]),
                                             R_disk=3.0)
    if len(idx_disk) > len(idx):
        errors.append(f"T2: disk filter increased edges ({len(idx_disk)} > {len(idx)})")
    print(f"  T2: {len(idx_disk)} edges in disk R=3 (of {len(idx)} crossing)")

    # T3: rotation matrix at alpha=0 is identity
    R0 = make_peierls_rotation(0.0)
    if np.max(np.abs(R0 - np.eye(3))) > 1e-14:
        errors.append("T3: R(alpha=0) != I")
    print(f"  T3: R(alpha=0) = I: OK")

    # T4: R(alpha=0.5, z-axis) = diag(-1,-1,1) (rotation by π around z)
    R05 = make_peierls_rotation(0.5)
    R05_expected = np.diag([-1.0, -1.0, 1.0])
    if np.max(np.abs(R05 - R05_expected)) > 1e-14:
        errors.append(f"T4: R(alpha=0.5) != diag(-1,-1,1): {R05}")
    print(f"  T4: R(alpha=0.5) = diag(-1,-1,1): OK")

    # T5: gauged D(k) at alpha=0 equals ungauged D(k)
    db = DisplacementBloch(V, E, L, k_L=3.0, k_T=1.0)
    k = np.array([0.3, 0.2, 0.1])
    D_normal = db.build_dynamical_matrix(k)
    D_gauge0 = build_gauged_dynamical_matrix(db, k, idx_disk, np.eye(3))
    diff = np.max(np.abs(D_normal - D_gauge0))
    if diff > 1e-12:
        errors.append(f"T5: D(alpha=0) != D_normal: diff={diff:.2e}")
    print(f"  T5: D(gauge alpha=0) = D(normal): diff={diff:.2e} OK")

    # T6: gauged D(k) at alpha=0.5 differs from normal
    R05 = make_peierls_rotation(0.5)
    D_gauge05 = build_gauged_dynamical_matrix(db, k, idx_disk, R05)
    diff05 = np.max(np.abs(D_normal - D_gauge05))
    if diff05 < 1e-6:
        errors.append(f"T6: D(alpha=0.5) = D(normal) — gauge has no effect!")
    print(f"  T6: D(gauge alpha=0.5) ≠ D(normal): diff={diff05:.2e} OK")

    # T7: gauged D is Hermitian
    herm_err = np.max(np.abs(D_gauge05 - D_gauge05.conj().T))
    if herm_err > 1e-12:
        errors.append(f"T7: D not Hermitian: {herm_err:.2e}")
    print(f"  T7: Hermiticity error: {herm_err:.2e} OK")

    n_tests = 7
    n_pass = n_tests - len(errors)
    for e in errors:
        print(f"  FAIL: {e}")
    print(f"  gauge_foam.py: {n_pass}/{n_tests} PASS")
    if errors:
        raise AssertionError(f"{len(errors)} tests failed")


if __name__ == '__main__':
    _test_init()
