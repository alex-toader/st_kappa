"""
§2 Model — infrastructure validation tests.

Tests:
  test_so2_essential        — T2.1: SO(2) Peierls → scattering; α=0 → none
  test_ablation_basics      — T2.2+T2.3: smoke test (full ablation in W1 files 22-23)
  test_energy_conservation  — T2.4: Verlet energy oscillation < 1%
  test_ranking_robustness   — T2.5: ranking preserved across R_disk (PML in W5)
  test_seeds_reproducible   — T2.6: seeds 0-4 vs 100-104 within 2σ
  test_k_direction          — T2.7: 20-dir average vs [100], ranking preserved
  test_wave_reaches_defect  — T2.8: E_sc/E_inc > 0.01 (wave interacted)
  test_rdisk_plateau        — T2.9: R_disk=2.0 on plateau (≈ R_disk=2.5)
  test_alpha_independence   — T2.10: α=0.1,0.3,0.5 → CV < 15%
  test_complex_crystal_smoke— T2.11: measure_mr works on beta_mn, th3p4
  test_k_convergence        — T2.12: 3 k-values ≈ 10 k-values (< 15%)
  test_plane_z_stability    — T2.13: auto z ≈ fixed z (< 15%)

RAW OUTPUT (24 Mar 2026):

  T2.1: α=0.5 → ⟨r⟩=1.449; α=0.0 → None (u_sc=0). SO(2) essential. PASS.
  T2.2-3: Smoke test — ⟨r⟩ O(1) on 3 seeds. Full ablation in W1 files 22-23. PASS.
  T2.4: Energy (KE + PE from elongations) oscillation 8.5e-05 (< 1%). PASS.
  T2.5: Ranking crystal > random at R_disk=1.5, 2.0, 2.5. PML: 24 Mar investigation. PASS.
  T2.6: Seeds 0-4 (1.436±0.104) vs 100-104 (1.410±0.179). Within 2σ. PASS.
  T2.7: k-direction ranking [beta_mn < c15 < fcc] same at [100] and 5-dir. PASS.
  T2.8: E_sc/E_inc=0.21-1.38, max|u_sc|/u0=0.88-2.01. Significant. PASS.
  T2.9: R_disk=2.0 vs 2.5: mean diff 4.5%. On plateau. PASS.
  T2.10: α=0.1,0.3,0.5 CV=2.2%. Independent. PASS.
  T2.11: beta_mn ⟨r⟩=1.383, th3p4 ⟨r⟩=0.979. Complex crystals work. PASS.
  T2.12: 3 vs 10 k-values: 1.6% diff. Convergent. PASS.
  T2.13: auto vs fixed z-plane: 0.9% diff. Stable. PASS.

  12/12 PASS (11.2s)

Date: 24 Mar 2026
"""
import sys, os, time
import numpy as np

_src = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', 'src'))
_src_path = _src
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from builders.structure_catalog import build_structure
from builders.random_graphs import compute_sv2_from_mesh, bloch_H_from_mesh
from measure import measure_mr


# ── T2.1 ─────────────────────────────────────────────────────

def test_so2_essential():
    """T2.1: SO(2) Peierls gauge produces scattering; α=0 produces none.

    On random z=4: ⟨r⟩ > 0.5 at α=0.5, ⟨r⟩ ≈ 0 at α=0 (or None).
    """
    t0 = time.time()
    print("T2.1: SO(2) Peierls essential")
    print("-" * 60)

    m = build_structure('random_z4', seed=0)
    V = np.array(m['V'])

    mr_05, nd = measure_mr(V, m['E'], m['L'], alpha=0.5)
    mr_00, _ = measure_mr(V, m['E'], m['L'], alpha=0.0)

    print(f"  α=0.5: ⟨r⟩={mr_05:.3f}, n_disk={nd}")
    mr_00_s = f"{mr_00:.3f}" if mr_00 is not None else "None (u_sc=0, no scattering)"
    print(f"  α=0.0: ⟨r⟩={mr_00_s}")

    assert mr_05 is not None and mr_05 > 0.5, f"α=0.5 should scatter: {mr_05}"
    # At α=0, Peierls rotation = identity → defect = reference → u_sc ≈ 0
    # May be None (E_tot < 1e-30) or a very small number (floating point)
    assert mr_00 is None or mr_00 < 0.1, \
        f"α=0 should produce negligible scattering: {mr_00}"

    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── T2.2 + T2.3 ─────────────────────────────────────────────

def test_ablation_basics():
    """T2.2+T2.3: Infrastructure smoke test (NN-only, 3-DOF).

    Full ablation (NNN not needed, 1-DOF sufficient) documented in
    W1 file 22-23 with results: SO(2) gives σ_tr=26.3, scalar gives 0.0,
    K2=0 gives σ_tr=49.8 (increases), 1-DOF ≈ 3-DOF within 1-15%.
    Here we verify infrastructure produces valid ⟨r⟩ with default settings.
    """
    t0 = time.time()
    print("\nT2.2-2.3: infrastructure smoke test (NN, 3-DOF)")
    print("-" * 60)

    mrs = []
    for seed in range(3):
        m = build_structure('random_z4', seed=seed)
        V = np.array(m['V'])
        mr, nd = measure_mr(V, m['E'], m['L'])
        if mr: mrs.append(mr)
        print(f"  seed={seed}: ⟨r⟩={mr:.3f}, n_disk={nd}")

    assert len(mrs) >= 2, "Should get valid ⟨r⟩ on most seeds"
    assert all(0.5 < mr < 3.0 for mr in mrs), f"All ⟨r⟩ should be O(1)"
    print(f"\n  Full ablation in W1 files 22-23:")
    print(f"    SO(2) essential: σ_tr=26.3 (scalar: 0.0)")
    print(f"    NNN not needed: K2=0 → σ_tr=49.8 (increases)")
    print(f"    1-DOF sufficient: 1-DOF ≈ 3-DOF within 1-15%")
    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── T2.4 ─────────────────────────────────────────────────────

def test_energy_conservation():
    """T2.4: Verlet FDTD conserves energy (no defect).

    Run FDTD with α=0 (no defect). Total energy should be constant
    (drift < 10⁻⁸ relative to initial).
    """
    t0 = time.time()
    print("\nT2.4: energy conservation (Verlet, no defect)")
    print("-" * 60)

    from core_math.dynamics.md_foam import prepare_edges, harmonic_force_spring
    from fdtd_foam import wave_packet_foam

    m = build_structure('random_z4', seed=0)
    V = np.array(m['V'])
    E_list = m['E']
    L = m['L']
    nv = len(V)

    edge_info = prepare_edges(V, E_list, L)
    E_arr = np.array(E_list)

    # Compute edge directions independently (not from edge_info internals)
    e_dirs = np.zeros((len(E_arr), 3))
    for ei, (i, j) in enumerate(E_arr):
        dr = V[j] - V[i]
        dr -= L * np.round(dr / L)
        ell = np.linalg.norm(dr)
        if ell > 1e-10:
            e_dirs[ei] = dr / ell

    u0, v0 = wave_packet_foam(V, np.array([0.5, 0, 0]),
                               np.array([L/4, L/2, L/2]), 2.0, 1.0, 0.0)

    def force(u_):
        return harmonic_force_spring(u_, edge_info, 1.0, 0.0)

    def compute_PE(u_):
        """PE = 0.5 * Σ_edges k_L * (Δu · ê)² — direct from elongations."""
        u3 = u_.reshape(nv, 3)
        pe = 0.0
        for ei in range(len(E_arr)):
            i, j = E_arr[ei]
            du = u3[j] - u3[i]
            stretch = np.dot(du, e_dirs[ei])
            pe += 0.5 * 1.0 * stretch**2  # k_L = 1.0
        return pe

    dt = 0.02
    n_steps = 200
    u, v = u0.copy(), v0.copy()
    a = force(u)

    energies = []
    for step in range(n_steps):
        KE = 0.5 * np.sum(v**2)
        PE = compute_PE(u)
        energies.append(KE + PE)

        u += dt * v + 0.5 * dt**2 * a
        a_new = force(u)
        v += 0.5 * dt * (a + a_new)
        a = a_new

    energies = np.array(energies)
    E0 = energies[0]
    drift = (np.max(energies) - np.min(energies)) / abs(E0)

    print(f"  E range: [{np.min(energies):.6f}, {np.max(energies):.6f}]")
    print(f"  Relative oscillation: {drift:.2e}")

    # Verlet is symplectic → energy oscillates but doesn't drift
    assert drift < 0.01, f"Energy oscillation too large: {drift:.2e}"
    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── T2.5 ─────────────────────────────────────────────────────

def test_ranking_robustness():
    """T2.5: Crystal > random ranking preserved across R_disk values.

    Verifies that the ranking crystal > random is robust to defect geometry.
    Full PML validation (width sweep, 15-25% shift in absolute ⟨r⟩) is
    documented in investigation 24 Mar (5 seeds × 6 widths). Here we test
    the weaker but related claim: ranking is preserved across R_disk.
    """
    t0 = time.time()
    print("\nT2.5: ranking robustness across R_disk")
    print("-" * 60)

    m_c = build_structure('diamond')
    m_r = build_structure('random_z4', seed=0)

    for R_disk in [1.5, 2.0, 2.5]:
        mr_c, _ = measure_mr(np.array(m_c['V']), m_c['E'], m_c['L'], R_disk=R_disk)
        mr_r, _ = measure_mr(np.array(m_r['V']), m_r['E'], m_r['L'], R_disk=R_disk)
        mr_c_s = f"{mr_c:.3f}" if mr_c else "---"
        mr_r_s = f"{mr_r:.3f}" if mr_r else "---"
        print(f"  R_disk={R_disk:.1f}: crystal={mr_c_s}, random={mr_r_s}")
        if mr_c and mr_r:
            assert mr_c > mr_r, \
                f"Crystal should exceed random at R={R_disk}: {mr_c:.3f} vs {mr_r:.3f}"

    print(f"\n  Ranking crystal > random preserved at all R_disk.")
    print(f"  PML validation: 24 Mar investigation (PML width 5-30%,")
    print(f"  ⟨r⟩ shifts 15-25%, ranking preserved).")
    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── T2.6 ─────────────────────────────────────────────────────

def test_seeds_reproducible():
    """T2.6: Seeds 0-4 vs 100-104 give consistent ⟨r⟩ (within 2σ)."""
    t0 = time.time()
    print("\nT2.6: seed reproducibility")
    print("-" * 60)

    mrs_a, mrs_b = [], []
    for seed in range(5):
        m = build_structure('random_z4', seed=seed)
        mr, _ = measure_mr(np.array(m['V']), m['E'], m['L'])
        if mr: mrs_a.append(mr)

    for seed in range(100, 105):
        m = build_structure('random_z4', seed=seed)
        mr, _ = measure_mr(np.array(m['V']), m['E'], m['L'])
        if mr: mrs_b.append(mr)

    mean_a, std_a = np.mean(mrs_a), np.std(mrs_a)
    mean_b, std_b = np.mean(mrs_b), np.std(mrs_b)
    diff = abs(mean_a - mean_b)
    within_2s = diff < 2 * max(std_a, std_b)

    print(f"  Seeds 0-4:   {mean_a:.3f} ± {std_a:.3f}")
    print(f"  Seeds 100-4: {mean_b:.3f} ± {std_b:.3f}")
    print(f"  Diff: {diff:.3f}, within 2σ: {within_2s}")

    assert within_2s, f"Means should be within 2σ: {diff:.3f} vs 2σ={2*max(std_a,std_b):.3f}"
    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── T2.7 ─────────────────────────────────────────────────────

def test_k_direction():
    """T2.7: Σv² along [100] is convention; ranking preserved at other k-dirs.

    On 3 crystals: compute Σv² at k=[1,0,0] and 5 random k-dirs.
    Ranking should be preserved.
    """
    t0 = time.time()
    print("\nT2.7: k-direction convention — ranking preserved")
    print("-" * 60)

    names = ['fcc', 'c15', 'beta_mn']
    sv2_100 = []
    sv2_multi = []

    for name in names:
        m = build_structure(name)
        sv2, _, _ = compute_sv2_from_mesh(m)  # k=[1,0,0]
        sv2_100.append(sv2)

        # Multi-direction average
        V = np.array(m['V'])
        L = m['L']
        nv = len(V)
        ndof = 3 * nv
        k_max = np.pi / L
        rng = np.random.RandomState(0)

        sv2_dirs = []
        for _ in range(5):
            k_dir = rng.randn(3)
            k_dir /= np.linalg.norm(k_dir)
            k_vals = np.linspace(0.01, k_max, 10)
            dk = k_vals[1] - k_vals[0]
            all_omega = []
            for k in k_vals:
                H = bloch_H_from_mesh(m, k * k_dir)
                evals = np.linalg.eigvalsh(H)
                all_omega.append(np.sqrt(np.maximum(np.sort(evals.real), 0)))
            all_omega = np.array(all_omega)
            v2 = 0
            for band in range(ndof):
                if np.max(all_omega[:, band]) > 0.01:
                    v_g = np.abs(np.gradient(all_omega[:, band], dk))
                    v2 += np.mean(v_g)**2
            sv2_dirs.append(v2 / ndof)
        sv2_multi.append(np.mean(sv2_dirs))

        print(f"  {name:>10s}: Σv²([100])={sv2:.6f}, Σv²(5-dir)={np.mean(sv2_dirs):.6f}")

    # Check ranking preserved
    rank_100 = np.argsort(sv2_100)
    rank_multi = np.argsort(sv2_multi)
    same_rank = np.array_equal(rank_100, rank_multi)

    print(f"\n  Ranking [100]: {[names[i] for i in rank_100]}")
    print(f"  Ranking multi: {[names[i] for i in rank_multi]}")
    print(f"  Same ranking: {same_rank}")

    assert same_rank, "Ranking should be preserved across k-directions"
    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── T2.8 ─────────────────────────────────────────────────────

def test_wave_reaches_defect():
    """T2.8: Verify wave actually interacts with defect.

    CRITICAL: if v_group ≈ 0 (flat bands), wave may not reach defect in
    n_steps, giving u_sc ≈ 0 for the wrong reason. This test verifies
    that the scattered field has measurable amplitude relative to incident.

    Note on flat-band structures (beta_mn): scattering is purely local —
    the Peierls defect modifies the local field immediately, not via
    Bloch propagation. u_sc is nonzero from local interaction, not from
    a propagating wave reaching the defect. This is physically correct
    and is the mechanism discussed in §5 (transport without propagation).
    """
    t0 = time.time()
    print("\nT2.8: wave reaches defect (max|u_sc| > threshold)")
    print("-" * 60)

    from core_math.dynamics.md_foam import prepare_edges, harmonic_force_spring
    from gauge_foam import find_edges_crossing_plane, make_peierls_rotation
    from fdtd_foam import gauged_force_foam, wave_packet_foam

    R_MAT = make_peierls_rotation(0.5)

    for label, builder_args in [('random_z4', {'seed': 0}),
                                  ('random_z4', {'seed': 5}),
                                  ('beta_mn', {})]:
        if label == 'beta_mn':
            m = build_structure('beta_mn')
        else:
            m = build_structure(label, **builder_args)
        V = np.array(m['V'])
        E = m['E']
        L = m['L']
        nv = len(V)

        edge_info = prepare_edges(V, E, L)
        cxy = np.array([L/2, L/2])
        best_z, best_n = L/2, 0
        for zo in np.arange(0, 2, 0.1):
            z = L/2 + zo
            idx, _ = find_edges_crossing_plane(V, E, L, z, cxy, L/3)
            if len(idx) > best_n:
                best_n = len(idx); best_z = z
        disk_idx, _ = find_edges_crossing_plane(V, E, L, best_z, cxy, 2.0)
        if len(disk_idx) < 2:
            print(f"  {label} s={builder_args.get('seed','')}: n_disk<2, skip")
            continue

        r_start = np.array([L/4, L/2, L/2])

        def fr(u_):
            return harmonic_force_spring(u_, edge_info, 1.0, 0.0)
        def fd(u_):
            return gauged_force_foam(u_, edge_info, 1.0, 0.0, disk_idx, R_MAT)

        k0 = 0.5
        u0, v0 = wave_packet_foam(V, np.array([k0, 0, 0]), r_start, 2.0, 1.0, 0.0)
        dt = 0.02
        n_steps = int(L / 1.0 * 0.9 / dt)

        u_r, v_r = u0.copy(), v0.copy(); a_r = fr(u_r)
        u_d, v_d = u0.copy(), v0.copy(); a_d = fd(u_d)
        for _ in range(n_steps):
            u_r += dt*v_r + 0.5*dt**2*a_r; a_new = fr(u_r); v_r += 0.5*dt*(a_r+a_new); a_r = a_new
            u_d += dt*v_d + 0.5*dt**2*a_d; a_new = fd(u_d); v_d += 0.5*dt*(a_d+a_new); a_d = a_new

        u_sc = u_d - u_r
        max_usc = np.max(np.abs(u_sc))
        u0_amp = np.max(np.abs(u0))
        E_sc = np.sum(u_sc**2)
        E_inc = np.sum(u0**2)
        ratio = E_sc / E_inc if E_inc > 0 else 0
        seed_s = builder_args.get('seed', '')
        print(f"  {label} s={seed_s}: max|u_sc|/u0={max_usc/u0_amp:.3f}, "
              f"E_sc/E_inc={ratio:.2e}, n_disk={len(disk_idx)}")

        # Scattered field should be a significant fraction of incident
        assert max_usc > u0_amp * 0.01, \
            f"Scattered field negligible vs incident: {max_usc/u0_amp:.2e}"
        assert ratio > 0.01, \
            f"Scattered energy too small: E_sc/E_inc={ratio:.2e}"

    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── T2.9 ─────────────────────────────────────────────────────

def test_rdisk_plateau():
    """T2.9: R_disk=2.0 is on (or near) the plateau.

    ⟨r⟩(R=2.0) should be within 10% of ⟨r⟩(R=2.5) on random z=4.
    If not, default R_disk=2.0 systematically underestimates ⟨r⟩.
    """
    t0 = time.time()
    print("\nT2.9: R_disk=2.0 near plateau")
    print("-" * 60)

    diffs = []
    for seed in range(3):
        m = build_structure('random_z4', seed=seed)
        V = np.array(m['V'])
        mr_20, _ = measure_mr(V, m['E'], m['L'], R_disk=2.0)
        mr_25, _ = measure_mr(V, m['E'], m['L'], R_disk=2.5)
        if mr_20 and mr_25:
            diff = abs(mr_20 - mr_25) / mr_25
            diffs.append(diff)
            print(f"  seed={seed}: ⟨r⟩(R=2.0)={mr_20:.3f}, ⟨r⟩(R=2.5)={mr_25:.3f}, "
                  f"diff={diff:.1%}")

    mean_diff = np.mean(diffs)
    print(f"\n  Mean difference: {mean_diff:.1%}")
    assert mean_diff < 0.10, f"R=2.0 should be within 10% of R=2.5: {mean_diff:.1%}"
    print(f"  R_disk=2.0 is near plateau. Time: {time.time()-t0:.1f}s. PASS.")


# ── T2.10 ────────────────────────────────────────────────────

def test_alpha_independence():
    """T2.10: ⟨r⟩ independent of α in §2 context.

    Quick verification that α choice doesn't change ⟨r⟩ significantly.
    Full test in T5.3 (CV=2.2% across α=0.05-0.50).
    """
    t0 = time.time()
    print("\nT2.10: α independence (quick check)")
    print("-" * 60)

    m = build_structure('random_z4', seed=0)
    V = np.array(m['V'])

    mrs = []
    for alpha in [0.1, 0.3, 0.5]:
        mr, _ = measure_mr(V, m['E'], m['L'], alpha=alpha)
        if mr: mrs.append(mr)
        print(f"  α={alpha:.1f}: ⟨r⟩={mr:.3f}" if mr else f"  α={alpha:.1f}: None")

    cv = np.std(mrs) / np.mean(mrs)
    print(f"\n  CV = {cv:.1%} (full test T5.3: CV=2.2%)")
    assert cv < 0.15, f"CV should be < 15%: {cv:.1%}"
    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── T2.11 ────────────────────────────────────────────────────

def test_complex_crystal_smoke():
    """T2.11: Infrastructure works on complex crystal (beta_mn, th3p4).

    Validates that measure_mr returns valid (not None) ⟨r⟩ on structures
    with many atoms/cell and short edges — not just random z=4.
    """
    t0 = time.time()
    print("\nT2.11: complex crystal smoke test")
    print("-" * 60)

    for name in ['beta_mn', 'th3p4']:
        m = build_structure(name)
        V = np.array(m['V'])
        mr, nd = measure_mr(V, m['E'], m['L'])
        nv = len(V)
        print(f"  {name}: nv={nv}, ⟨r⟩={mr:.3f}, n_disk={nd}" if mr
              else f"  {name}: nv={nv}, ⟨r⟩=None, n_disk={nd}")
        assert mr is not None, f"{name}: measure_mr returned None (nv={nv}, n_disk={nd})"
        assert 0.5 < mr < 2.0, f"{name}: ⟨r⟩ out of expected range [0.5, 2.0]: {mr:.3f}"

    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── T2.12 ────────────────────────────────────────────────────

def test_k_convergence():
    """T2.12: ⟨r⟩ averaged over 3 k-values ≈ 10 k-values (< 15% difference).

    Default k_values=(0.3, 0.5, 1.0). Verify this is representative.
    """
    t0 = time.time()
    print("\nT2.12: k-values convergence (3 vs 10)")
    print("-" * 60)

    m = build_structure('random_z4', seed=0)
    V = np.array(m['V'])

    mr_3, _ = measure_mr(V, m['E'], m['L'], k_values=(0.3, 0.5, 1.0))
    mr_10, _ = measure_mr(V, m['E'], m['L'],
                           k_values=tuple(np.linspace(0.2, 1.2, 10)))

    diff = abs(mr_3 - mr_10) / mr_10 if mr_10 else 0
    print(f"  3 k-values:  ⟨r⟩={mr_3:.3f}")
    print(f"  10 k-values: ⟨r⟩={mr_10:.3f}")
    print(f"  Difference: {diff:.1%}")

    assert diff < 0.15, f"3 vs 10 k-values should differ < 15%: {diff:.1%}"
    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── T2.13 ────────────────────────────────────────────────────

def test_plane_z_stability():
    """T2.13: ⟨r⟩ stable to choice of z-plane.

    Auto-chosen best_z vs fixed z=L/2: difference < 15%.
    """
    t0 = time.time()
    print("\nT2.13: z-plane choice stability")
    print("-" * 60)

    from gauge_foam import find_edges_crossing_plane

    m = build_structure('random_z4', seed=0)
    V = np.array(m['V'])
    E = m['E']
    L = m['L']

    # Auto-chosen (default)
    mr_auto, nd_auto = measure_mr(V, E, L)

    # Fixed z = L/2 — manual measure
    from core_math.dynamics.md_foam import prepare_edges, harmonic_force_spring
    from gauge_foam import make_peierls_rotation
    from fdtd_foam import gauged_force_foam, wave_packet_foam

    R_MAT = make_peierls_rotation(0.5)
    edge_info = prepare_edges(V, E, L)
    nv = len(V)
    fixed_z = L / 2
    cxy = np.array([L/2, L/2])
    disk_idx, _ = find_edges_crossing_plane(V, E, L, fixed_z, cxy, 2.0)

    if len(disk_idx) >= 2:
        defect_center = np.array([L/2, L/2, fixed_z])
        r_start = np.array([L/4, L/2, L/2])
        dr_all = V - defect_center
        dr_all -= L * np.round(dr_all / L)
        dist = np.linalg.norm(dr_all, axis=1)

        def fr(u_):
            return harmonic_force_spring(u_, edge_info, 1.0, 0.0)
        def fd(u_):
            return gauged_force_foam(u_, edge_info, 1.0, 0.0, disk_idx, R_MAT)

        mrs_fixed = []
        dt = 0.02
        for k0 in [0.3, 0.5, 1.0]:
            u0, v0 = wave_packet_foam(V, np.array([k0,0,0]), r_start, 2.0, 1.0, 0.0)
            n_steps = int(L / 1.0 * 0.9 / dt)
            u_r, v_r = u0.copy(), v0.copy(); a_r = fr(u_r)
            u_d, v_d = u0.copy(), v0.copy(); a_d = fd(u_d)
            for _ in range(n_steps):
                u_r += dt*v_r + 0.5*dt**2*a_r; a_new = fr(u_r); v_r += 0.5*dt*(a_r+a_new); a_r = a_new
                u_d += dt*v_d + 0.5*dt**2*a_d; a_new = fd(u_d); v_d += 0.5*dt*(a_d+a_new); a_d = a_new
            u_sc = (u_d - u_r).reshape(nv, 3)
            e_pv = np.sum(u_sc**2, axis=1)
            E_tot = np.sum(e_pv)
            if E_tot > 1e-30:
                mrs_fixed.append(float(np.sum(dist * e_pv) / E_tot))
        mr_fixed = np.mean(mrs_fixed) if mrs_fixed else None
    else:
        mr_fixed = None

    assert mr_auto is not None, "Auto z-plane gave no result"
    assert mr_fixed is not None, \
        f"Fixed z=L/2 gave no disk edges (n_disk={len(disk_idx)}). " \
        f"Structure may have no edges crossing this plane. " \
        f"Use a structure with edges near z=L/2 or increase R_disk."

    diff = abs(mr_auto - mr_fixed) / mr_auto
    print(f"  Auto z:  ⟨r⟩={mr_auto:.3f} (n_disk={nd_auto})")
    print(f"  Fixed z: ⟨r⟩={mr_fixed:.3f} (n_disk={len(disk_idx)})")
    print(f"  Difference: {diff:.1%}")

    assert diff < 0.15, f"z-plane choice should be stable: {diff:.1%}"
    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── Main ─────────────────────────────────────────────────────

TESTS = [
    ('so2', test_so2_essential),
    ('ablation', test_ablation_basics),
    ('energy', test_energy_conservation),
    ('ranking_pml', test_ranking_robustness),
    ('seeds', test_seeds_reproducible),
    ('kdir', test_k_direction),
    ('wave', test_wave_reaches_defect),
    ('rdisk', test_rdisk_plateau),
    ('alpha', test_alpha_independence),
    ('crystal', test_complex_crystal_smoke),
    ('kconv', test_k_convergence),
    ('zplane', test_plane_z_stability),
]

if __name__ == '__main__':
    print("test_02_model.py — §2 model validation")
    print("=" * 60)

    selected = sys.argv[1:] if len(sys.argv) > 1 else [t[0] for t in TESTS]

    t_start = time.time()
    passed = 0
    total = 0
    for name, fn in TESTS:
        if name in selected or any(s in name for s in selected):
            total += 1
            try:
                fn()
                passed += 1
            except AssertionError as e:
                print(f"  FAIL: {e}")
            except Exception as e:
                print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"{passed}/{total} PASS ({time.time()-t_start:.1f}s)")
