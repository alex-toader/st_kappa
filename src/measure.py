"""
measure.py — Measurement functions for elastic transport on discrete structures.

Provides:
  measure_mr(V, E, L, ...)     — ⟨r⟩ FDTD: energy-weighted mean distance of scattered field
  compute_delta_spectral(mesh)  — Δ_spectral: mean(bandwidth/frequency) across bands
  count_unique_dirs(V, E, L)    — number of distinct edge directions (within 1°)

These are the observables used in the κ paper. Each function is self-contained
and depends only on numpy + the FDTD infrastructure in st_foam_scattering.

Date: 24 Mar 2026
"""
import numpy as np


def measure_mr(V, E, L, R_disk=2.0, alpha=0.5, k_values=(0.3, 0.5, 1.0),
               dt=0.02, min_nv=30, min_disk=2):
    """Measure ⟨r⟩: energy-weighted mean distance of scattered field from defect.

    Runs two Verlet FDTD simulations (reference + defect) for each k in k_values,
    computes scattered field u_sc = u_defect - u_reference, and returns the
    energy-weighted mean distance from defect center.

    Args:
        V: (nv, 3) vertex positions
        E: list of [i, j] edges
        L: periodic box size
        R_disk: radius of Peierls defect disk
        alpha: Peierls rotation strength (0.5 = Z₂)
        k_values: wave vector magnitudes to average over
        dt: Verlet time step
        min_nv: minimum vertices for reliable measurement
        min_disk: minimum disk edges required

    Returns: (mr, n_disk) where mr is ⟨r⟩ or None if measurement impossible.
    """
    # Lazy imports to avoid hard dependency at module level
    from core_math.dynamics.md_foam import prepare_edges, harmonic_force_spring
    from gauge_foam import find_edges_crossing_plane, make_peierls_rotation
    from fdtd_foam import gauged_force_foam, wave_packet_foam

    V = np.asarray(V, dtype=float)
    nv = len(V)

    if nv < min_nv:
        return None, 0

    R_MAT = make_peierls_rotation(alpha)
    edge_info = prepare_edges(V, E, L)

    # Find best z-plane for disk
    cxy = np.array([L / 2, L / 2])
    best_z, best_n = L / 2, 0
    for zo in np.arange(0, 2, 0.1):
        z = L / 2 + zo
        idx, _ = find_edges_crossing_plane(V, E, L, z, cxy, L / 3)
        if len(idx) > best_n:
            best_n = len(idx)
            best_z = z

    if best_n == 0:
        return None, 0  # no edges cross any z-plane in scan range

    disk_idx, _ = find_edges_crossing_plane(V, E, L, best_z, cxy, R_disk)
    if len(disk_idx) < min_disk:
        return None, len(disk_idx)

    # Distances from defect center (minimum image)
    defect_center = np.array([L / 2, L / 2, best_z])
    r_start = np.array([L / 4, L / 2, L / 2])
    dr_all = V - defect_center
    dr_all = dr_all - L * np.round(dr_all / L)
    dist = np.linalg.norm(dr_all, axis=1)

    # Force functions
    def fr(u_):
        return harmonic_force_spring(u_, edge_info, 1.0, 0.0)

    def fd(u_):
        return gauged_force_foam(u_, edge_info, 1.0, 0.0, disk_idx, R_MAT)

    # FDTD at each k
    mrs = []
    for k0 in k_values:
        u0, v0 = wave_packet_foam(V, np.array([k0, 0, 0]),
                                   r_start, 2.0, 1.0, 0.0)
        # n_steps: wave must traverse domain. Assumes v_group ≈ 1.0 (lattice units).
        # Factor 0.9: stop before wrap-around on PBC.
        n_steps = int(L / 1.0 * 0.9 / dt)

        # Reference (no defect)
        u_r, v_r = u0.copy(), v0.copy()
        a_r = fr(u_r)
        # Defect
        u_d, v_d = u0.copy(), v0.copy()
        a_d = fd(u_d)

        for _ in range(n_steps):
            u_r += dt * v_r + 0.5 * dt**2 * a_r
            a_new = fr(u_r)
            v_r += 0.5 * dt * (a_r + a_new)
            a_r = a_new

            u_d += dt * v_d + 0.5 * dt**2 * a_d
            a_new = fd(u_d)
            v_d += 0.5 * dt * (a_d + a_new)
            a_d = a_new

        u_sc = (u_d - u_r).reshape(nv, 3)
        e_pv = np.sum(u_sc**2, axis=1)
        E_tot = np.sum(e_pv)
        if E_tot > 1e-30:
            mrs.append(float(np.sum(dist * e_pv) / E_tot))

    mr = float(np.mean(mrs)) if mrs else None
    return mr, len(disk_idx)


def compute_delta_spectral(mesh, n_k=15):
    """Compute Δ_spectral = mean(bandwidth / mean_frequency) across optical bands.

    Measures how much eigenvalues vary with k — a proxy for dispersion.
    Δ >> 0: dispersive bands (coherent structure).
    Δ ≈ 0: flat bands (incoherent structure).

    NOTE: This is NOT the same as Σv² (which is in compute_sv2_from_mesh in
    builders/random_graphs.py). Δ_spectral and Σv² are correlated (ρ≈0.98)
    but not identical. Σv² uses group velocity; Δ_spectral uses bandwidth ratio.
    Paper cites Σv² as the primary observable. Δ_spectral is used only in §6
    (continuum analysis).

    Args:
        mesh: dict with V, E, L (and optionally edge_dirs)
        n_k: number of k-points along [1,0,0]

    Returns: float Δ_spectral
    """
    from builders.random_graphs import bloch_H_from_mesh

    V = np.asarray(mesh['V'])
    L = mesh['L']
    nv = len(V)
    ndof = 3 * nv
    k_max = np.pi / L
    k_vals = np.linspace(0, k_max, n_k)

    all_omega = []
    for k in k_vals:
        H = bloch_H_from_mesh(mesh, np.array([k, 0, 0]))
        evals = np.linalg.eigvalsh(H)
        all_omega.append(np.sqrt(np.maximum(np.sort(evals.real), 0)))
    all_omega = np.array(all_omega)

    bw_ratios = []
    for band in range(ndof):
        mean_w = np.mean(all_omega[:, band])
        bw = np.max(all_omega[:, band]) - np.min(all_omega[:, band])
        if mean_w > 0.01:
            bw_ratios.append(bw / mean_w)

    return float(np.mean(bw_ratios)) if bw_ratios else 0


def count_unique_dirs(V, E, L, tol=0.9998):
    """Count distinct edge directions (up to sign, within angular tolerance).

    Two directions are considered the same if |ê₁·ê₂| > tol or |ê₁·(-ê₂)| > tol.

    Args:
        V: vertex positions
        E: edge list
        L: box size (for minimum image)
        tol: dot product threshold (0.9998 ≈ 1°)

    Returns: int number of unique directions
    """
    V = np.asarray(V, dtype=float)
    dirs = []
    for i, j in E:
        dr = V[j] - V[i]
        dr -= L * np.round(dr / L)
        ell = np.linalg.norm(dr)
        if ell > 1e-10:
            dirs.append(dr / ell)

    unique = []
    for d in dirs:
        is_new = True
        for u in unique:
            if abs(np.dot(d, u)) > tol:  # covers both d≈u and d≈-u
                is_new = False
                break
        if is_new:
            unique.append(d)

    return len(unique)


def mean_edge_length(V, E, L):
    """Mean edge length (minimum image convention)."""
    V = np.asarray(V, dtype=float)
    lens = []
    for i, j in E:
        dr = V[j] - V[i]
        dr -= L * np.round(dr / L)
        lens.append(np.linalg.norm(dr))
    return float(np.mean(lens)) if lens else 0


# =============================================================================
# SELF-TESTS
# =============================================================================

def _self_test():
    """Quick validation of all functions."""
    import sys, os
    _math = os.path.dirname(os.path.abspath(__file__))
    if _math not in sys.path:
        sys.path.insert(0, _math)

    from builders.structure_catalog import build_structure
    from builders.random_graphs import compute_sv2_from_mesh

    print("measure.py — self-tests")
    print("=" * 50)
    n_pass = 0

    # T1: count_unique_dirs on Kelvin (6 dirs) vs random (~190)
    m_k = build_structure('bcc')
    m_r = build_structure('random_z4', seed=42)
    nd_k = count_unique_dirs(m_k['V'], m_k['E'], m_k['L'])
    nd_r = count_unique_dirs(m_r['V'], m_r['E'], m_r['L'])
    assert nd_k == 6, f"Kelvin should have 6 dirs: {nd_k}"
    assert nd_r > 100, f"Random should have >100 dirs: {nd_r}"
    print(f"  T1: count_unique_dirs — Kelvin={nd_k}, random={nd_r}. PASS.")
    n_pass += 1

    # T2: compute_delta_spectral — crystal > random
    d_k = compute_delta_spectral(m_k)
    d_r = compute_delta_spectral(m_r)
    assert d_k > d_r, f"Crystal Δ should exceed random: {d_k:.4f} vs {d_r:.4f}"
    print(f"  T2: Δ_spectral — Kelvin={d_k:.4f} > random={d_r:.4f}. PASS.")
    n_pass += 1

    # T3: mean_edge_length > 0
    ell_k = mean_edge_length(m_k['V'], m_k['E'], m_k['L'])
    assert ell_k > 0, f"Edge length should be positive: {ell_k}"
    print(f"  T3: mean_edge_length — Kelvin={ell_k:.4f}. PASS.")
    n_pass += 1

    # T4: measure_mr on random (should be O(1))
    _foam = os.path.normpath(os.path.join(_math, '..', '..', '..',
                                           'st_foam_scattering', 'src'))
    if _foam not in sys.path:
        sys.path.insert(0, _foam)
    mr, nd = measure_mr(np.array(m_r['V']), m_r['E'], m_r['L'])
    assert mr is not None, "measure_mr should return value on random"
    assert 0.5 < mr < 3.0, f"⟨r⟩ should be O(1): {mr:.3f}"
    print(f"  T4: measure_mr — random ⟨r⟩={mr:.3f}, n_disk={nd}. PASS.")
    n_pass += 1

    print(f"\n{n_pass}/4 PASS")
    return n_pass


if __name__ == '__main__':
    _self_test()
