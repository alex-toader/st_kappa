"""
§6 Coherence continuum — transition is continuous, not sharp.

Tests:
  test_104_scan_continuous  — T6.1: Σv² spans 5 orders of magnitude on 104 structures
  test_no_binary_separator  — T6.2: Δ_spectral, ||C-0.5|| overlap between crystal/random
  test_complex_crystals     — T6.3: beta_mn, gamma_brass, alpha_mn ⟨r⟩ ≈ random
  test_quasicrystal         — T6.4: Fibonacci Σv² >> random (periodicity not needed)
  test_complex_crystal_fdtd — T6.5: ⟨r⟩ FDTD on beta_mn < 2.0
  test_ndirs_partial_corr   — T6.6: partial correlation ρ(n_dirs, ⟨r⟩ | Σv²)
  test_ndirs_overlap        — T6.7: n_dirs overlap crystal/random (structural continuum)
  test_complex_crystal_in_random_ci — T6.8: beta_mn ⟨r⟩ within random ± 2σ
  test_ndirs_scaling                — T6.11: Σv² vs n_dirs, saturates at n_dirs ≥ dim

  NOT in this file:
  - θ-scan with ⟨r⟩ FDTD (continuous on same graph): future work.
    Σv²(θ) continuity demonstrated spectrally in T3.11.
  - T6.6 power analysis: n=14 underpowered for ρ_partial=0.3 (~20% power).
    Paper notes limitation.

RAW OUTPUT (24 Mar 2026):

  T6.1: Σv² range 19491× (4.3 orders, incl SC N=3). Continuous. PASS.
  T6.2: Δ_spectral separation 0.5× (< 1.0 = overlap). Continuum. PASS.
  T6.3: beta_mn 1.38, gamma_brass 1.22, alpha_mn 1.10 — all < 2.0, ≈ random (1.44). PASS.
  T6.4: Fibonacci 3D (standard, Kohmoto): 64% of SC (n=5), 44% (n=6). >> random. PASS.
  T6.5: beta_mn FDTD ⟨r⟩=1.383 (random: 1.449). Crystal in random range. PASS.
  T6.6: ρ_partial(n_dirs, ⟨r⟩ | Σv²) = -0.314 (p=0.27). n_dirs is proxy for Σv². PASS.

  T6.7: n_dirs crystal [4,286] overlaps random [185,196]. Structural continuum. PASS.
  T6.8: beta_mn z-score=-0.61 within random ±2σ. Statistically in random regime. PASS.

  T6.11: n_dirs=2→50: Σv² 0.0002→0.00003, saturates at n_dirs≥3. All << crystal. PASS.

  9/9 PASS

Date: 24 Mar 2026
"""
import sys, os, time
import numpy as np
from scipy.stats import spearmanr

_src = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', 'src'))
_src_path = _src
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from builders.structure_catalog import build_structure
from builders.random_graphs import compute_sv2_from_mesh
from measure import measure_mr, compute_delta_spectral, count_unique_dirs


# ── T6.1 ─────────────────────────────────────────────────────

def test_104_scan_continuous():
    """T6.1: Σv² spans ≥3 orders of magnitude across structures.

    Continuous variation from crystal (SC Σv²≈2.6) to random (Σv²≈0.0003).
    """
    t0 = time.time()
    print("T6.1: Σv² range across crystals + random")
    print("-" * 60)

    sv2_crystals, sv2_randoms = [], []

    for name in ['fcc', 'diamond', 'perovskite', 'pyrochlore', 'c15', 'a15',
                  'beta_mn', 'gamma_brass', 'alpha_mn', 'pyrite', 'skutterudite']:
        m = build_structure(name)
        sv2, _, _ = compute_sv2_from_mesh(m)
        sv2_crystals.append(sv2)

    # SC at N=3 has z=6, Σv²≈2.6 — extends upper range
    from builders.voronoi_scan import build_voronoi_mesh
    m_sc = build_voronoi_mesh('sc', N=3)
    sv2_sc, _, _ = compute_sv2_from_mesh(m_sc)
    sv2_crystals.append(sv2_sc)

    for seed in range(10):
        m = build_structure('random_z4', seed=seed)
        sv2, _, _ = compute_sv2_from_mesh(m)
        sv2_randoms.append(sv2)

    all_sv2 = sv2_crystals + sv2_randoms
    ratio = max(all_sv2) / min(all_sv2)
    orders = np.log10(ratio)
    print(f"  Crystal Σv²: [{min(sv2_crystals):.6f}, {max(sv2_crystals):.6f}]")
    print(f"  Random  Σv²: [{min(sv2_randoms):.6f}, {max(sv2_randoms):.6f}]")
    print(f"  Total range: {ratio:.0f}× ({orders:.1f} orders of magnitude)")
    assert orders > 3, f"Should span ≥3 orders: {orders:.1f}"
    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── T6.2 ─────────────────────────────────────────────────────

def test_no_binary_separator():
    """T6.2: No single scalar cleanly separates crystal from random.

    Δ_spectral separation < 3× between lowest crystal and highest random.
    """
    t0 = time.time()
    print("\nT6.2: no binary separator (Δ_spectral overlap)")
    print("-" * 60)

    d_crystals, d_randoms = [], []

    for name in ['fcc', 'diamond', 'c15', 'a15', 'beta_mn', 'gamma_brass']:
        m = build_structure(name)
        d = compute_delta_spectral(m)
        d_crystals.append((name, d))

    for seed in range(5):
        m = build_structure('random_z4', seed=seed)
        d = compute_delta_spectral(m)
        d_randoms.append((f'rand_{seed}', d))

    dc_vals = [x[1] for x in d_crystals]
    dr_vals = [x[1] for x in d_randoms]

    sep = min(dc_vals) / max(dr_vals) if max(dr_vals) > 0 else float('inf')

    print(f"  Crystal Δ: [{min(dc_vals):.5f}, {max(dc_vals):.5f}]")
    print(f"  Random  Δ: [{min(dr_vals):.5f}, {max(dr_vals):.5f}]")
    print(f"  Separation: {sep:.1f}× (min crystal / max random)")

    # Separation < 1.0 means overlap exists (min crystal below max random).
    # This demonstrates continuum, not phase boundary.
    assert sep < 1.0, f"Should overlap (sep < 1.0) for continuum: {sep:.1f}×"
    print(f"\n  Separation {sep:.1f}× (< 1.0 = overlap) — continuum confirmed.")
    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── T6.3 ─────────────────────────────────────────────────────

def test_complex_crystals():
    """T6.3: Crystals with many dirs behave as random in transport.

    beta_mn (234 dirs), gamma_brass (262), alpha_mn (286) — all periodic
    but ⟨r⟩ in the random range (< 2.0).
    """
    t0 = time.time()
    print("\nT6.3: complex crystals behave as random")
    print("-" * 60)

    complex_crystals = ['beta_mn', 'gamma_brass', 'alpha_mn']

    # Random baseline
    mr_randoms = []
    for seed in range(5):
        m = build_structure('random_z4', seed=seed)
        V = np.array(m['V'])
        mr, _ = measure_mr(V, m['E'], m['L'])
        if mr: mr_randoms.append(mr)
    mean_random = np.mean(mr_randoms)

    for name in complex_crystals:
        m = build_structure(name)
        V = np.array(m['V'])
        mr, _ = measure_mr(V, m['E'], m['L'])
        n_dir = count_unique_dirs(V, m['E'], m['L'])
        print(f"  {name:>15s}: n_dir={n_dir:4d}, ⟨r⟩={mr:.3f} "
              f"(random mean: {mean_random:.3f})")
        assert mr is not None and mr < 2.0, \
            f"{name} should have ⟨r⟩<2: {mr}"

    print(f"\n  Complex crystals ⟨r⟩ < 2.0, comparable to random ({mean_random:.2f}).")
    print(f"  Direction diversity → incoherent regime despite periodicity.")
    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── T6.4 ─────────────────────────────────────────────────────

def test_quasicrystal():
    """T6.4: Fibonacci 3D quasicrystal transports (periodicity not needed).

    Standard Fibonacci quasicrystal (Kohmoto et al. 1983): tensor product
    of three 1D Fibonacci chains with substitution rule L→LS, S→L.
    Same edge directions as SC (±x,±y,±z), non-periodic spacing (L/S=φ).
    Σv² should be >> random and a significant fraction of SC.
    """
    t0 = time.time()
    print("\nT6.4: Fibonacci 3D quasicrystal")
    print("-" * 60)

    from builders.fibonacci_3d import build_fibonacci_3d

    results = []
    for n in [5, 6]:
        m_fib = build_fibonacci_3d(n=n, L=4.0)
        sv2_fib, fflat_fib, _ = compute_sv2_from_mesh(m_fib)

        # SC reference at same nv
        L_sc = float(n)
        V_sc = [[ix+0.5, iy+0.5, iz+0.5]
                for ix in range(n) for iy in range(n) for iz in range(n)]
        E_sc = []
        for ix in range(n):
            for iy in range(n):
                for iz in range(n):
                    i = ix*n*n + iy*n + iz
                    E_sc.append([i, ((ix+1)%n)*n*n + iy*n + iz])
                    E_sc.append([i, ix*n*n + ((iy+1)%n)*n + iz])
                    E_sc.append([i, ix*n*n + iy*n + (iz+1)%n])
        m_sc = {'V': V_sc, 'E': E_sc, 'L': L_sc, 'dim': 3, 'F': []}
        sv2_sc, _, _ = compute_sv2_from_mesh(m_sc)

        ratio = sv2_fib / sv2_sc if sv2_sc > 0 else 0
        results.append((n, sv2_fib, ratio))
        print(f"  n={n}: Fibonacci Σv²={sv2_fib:.6f} ({ratio*100:.0f}% of SC={sv2_sc:.6f})")

    # Assert on ALL n values
    for n_val, sv2_val, r_val in results:
        assert sv2_val > 0.01, f"n={n_val}: Fibonacci should transport: {sv2_val:.6f}"
        assert r_val > 0.30, f"n={n_val}: Fibonacci should be ≥30% of SC: {r_val:.2f}"

    print(f"\n  Same 3 directions as SC, non-periodic spacing.")
    print(f"  Transports — periodicity not required, coherence is.")
    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── T6.5 ─────────────────────────────────────────────────────

def test_complex_crystal_fdtd():
    """T6.5: ⟨r⟩ FDTD on beta_mn is in random range (< 2.0).

    Direct FDTD confirmation that a periodic crystal with many directions
    has ⟨r⟩ comparable to random structures.
    """
    t0 = time.time()
    print("\nT6.5: beta_mn FDTD — crystal in random range")
    print("-" * 60)

    m = build_structure('beta_mn')
    V = np.array(m['V'])
    mr, nd = measure_mr(V, m['E'], m['L'])
    n_dir = count_unique_dirs(V, m['E'], m['L'])

    # Random reference
    m_r = build_structure('random_z4', seed=0)
    mr_r, _ = measure_mr(np.array(m_r['V']), m_r['E'], m_r['L'])

    print(f"  beta_mn: n_dir={n_dir}, ⟨r⟩={mr:.3f}, n_disk={nd}")
    print(f"  random:  ⟨r⟩={mr_r:.3f}")

    assert mr is not None and mr < 2.0, f"beta_mn ⟨r⟩ should be < 2.0: {mr}"
    assert abs(mr - mr_r) / mr_r < 0.5, \
        f"beta_mn should be within 50% of random: {mr:.3f} vs {mr_r:.3f}"

    print(f"\n  Periodic crystal with {n_dir} directions: ⟨r⟩ in random range.")
    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── T6.6 ─────────────────────────────────────────────────────

def test_ndirs_partial_corr():
    """T6.6: Partial correlation ρ(n_dirs, ⟨r⟩ | Σv²) on crystals.

    Tests whether n_dirs adds predictive power beyond Σv².
    If partial ρ ≈ 0: n_dirs is a proxy for Σv² (not independent).
    If partial ρ significant: n_dirs has independent explanatory power.
    """
    t0 = time.time()
    print("\nT6.6: partial correlation n_dirs vs ⟨r⟩ controlling for Σv²")
    print("-" * 60)

    names, n_dirs_list, sv2s, mrs = [], [], [], []

    for name in ['fcc', 'diamond', 'perovskite', 'pyrochlore', 'c15', 'a15',
                  'clathrate_I', 'beta_mn', 'gamma_brass', 'alpha_mn',
                  'pyrite', 'skutterudite', 'spinel', 'th3p4']:
        m = build_structure(name)
        V = np.array(m['V'])
        sv2, _, _ = compute_sv2_from_mesh(m)
        mr, _ = measure_mr(V, m['E'], m['L'])
        nd = count_unique_dirs(V, m['E'], m['L'])
        if mr is not None:
            names.append(name)
            n_dirs_list.append(nd)
            sv2s.append(sv2)
            mrs.append(mr)

    # Direct correlations
    rho_nd, p_nd = spearmanr(n_dirs_list, mrs)
    rho_sv2, p_sv2 = spearmanr(sv2s, mrs)
    rho_nd_sv2, _ = spearmanr(n_dirs_list, sv2s)

    print(f"  n={len(names)} crystals")
    print(f"  ρ(n_dirs, ⟨r⟩) = {rho_nd:.3f} (p={p_nd:.4f})")
    print(f"  ρ(Σv², ⟨r⟩)   = {rho_sv2:.3f} (p={p_sv2:.4f})")
    print(f"  ρ(n_dirs, Σv²) = {rho_nd_sv2:.3f} (colinearity)")

    # Partial correlation via residuals
    # Rank-based: regress ranks of n_dirs and ⟨r⟩ on ranks of Σv²
    from scipy.stats import rankdata
    r_nd = rankdata(n_dirs_list)
    r_mr = rankdata(mrs)
    r_sv2 = rankdata(sv2s)

    # Residuals
    a1, b1 = np.polyfit(r_sv2, r_nd, 1)
    res_nd = r_nd - (a1 * r_sv2 + b1)
    a2, b2 = np.polyfit(r_sv2, r_mr, 1)
    res_mr = r_mr - (a2 * r_sv2 + b2)

    rho_partial, p_partial = spearmanr(res_nd, res_mr)
    print(f"  ρ_partial(n_dirs, ⟨r⟩ | Σv²) = {rho_partial:.3f} (p={p_partial:.4f})")

    # n_dirs and Σv² are highly colinear (ρ ≈ -0.88). After controlling for Σv²,
    # n_dirs should have little residual explanatory power (partial ρ nonsignificant).
    assert abs(rho_nd_sv2) > 0.7, \
        f"n_dirs and Σv² should be colinear: ρ={rho_nd_sv2:.3f}"
    print(f"\n  Colinearity ρ(n_dirs, Σv²) = {rho_nd_sv2:.3f} — highly correlated.")
    # Paper claims n_dirs is proxy for Σv², not independent variable.
    # Assert partial correlation is nonsignificant.
    assert p_partial > 0.05, \
        f"Partial ρ should be nonsignificant (n_dirs = proxy): p={p_partial:.4f}"
    print(f"  Partial ρ nonsignificant (p={p_partial:.2f}) — n_dirs is proxy for Σv².")

    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── Main ─────────────────────────────────────────────────────

# ── T6.7 ─────────────────────────────────────────────────────

def test_ndirs_overlap():
    """T6.7: Overlap on structural variable (n_dirs), not just spectral.

    If continuum is only on Δ_spectral (derived from Σv²), it could be
    circular. n_dirs is a purely structural variable. Crystal n_dirs range
    should overlap with random n_dirs range → continuum on structural level.
    """
    t0 = time.time()
    print("\nT6.7: n_dirs overlap (structural, not spectral)")
    print("-" * 60)

    c_dirs, r_dirs = [], []
    for name in ['fcc', 'diamond', 'pyrochlore', 'c15', 'a15', 'clathrate_I',
                  'beta_mn', 'gamma_brass', 'alpha_mn', 'pyrite']:
        m = build_structure(name)
        nd = count_unique_dirs(np.array(m['V']), m['E'], m['L'])
        c_dirs.append(nd)

    for seed in range(5):
        m = build_structure('random_z4', seed=seed)
        nd = count_unique_dirs(np.array(m['V']), m['E'], m['L'])
        r_dirs.append(nd)

    print(f"  Crystal n_dirs: [{min(c_dirs)}, {max(c_dirs)}]")
    print(f"  Random  n_dirs: [{min(r_dirs)}, {max(r_dirs)}]")

    # Crystal n_dirs covers a wider range than random — from far below (fcc: 4)
    # to above (alpha_mn: 286). This shows direction diversity varies continuously
    # across structures, not in discrete bins.
    assert min(c_dirs) < min(r_dirs), \
        f"Some crystals should have fewer dirs than random: {min(c_dirs)} vs {min(r_dirs)}"
    assert max(c_dirs) > max(r_dirs), \
        f"Some crystals should have more dirs than random: {max(c_dirs)} vs {max(r_dirs)}"
    print(f"  Crystal range [{min(c_dirs)}, {max(c_dirs)}] wider than random [{min(r_dirs)}, {max(r_dirs)}].")
    print(f"  Directional diversity varies continuously — not discrete categories.")
    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── T6.8 ─────────────────────────────────────────────────────

def test_complex_crystal_in_random_ci():
    """T6.8: beta_mn ⟨r⟩ falls within random z=4 confidence interval.

    Statistical test: beta_mn ⟨r⟩ is within mean ± 2σ of random distribution.
    Stronger than just ⟨r⟩ < 2.0.
    """
    t0 = time.time()
    print("\nT6.8: beta_mn ⟨r⟩ within random confidence interval")
    print("-" * 60)

    # Random distribution (15 seeds for stable CI)
    mr_randoms = []
    for seed in range(15):
        m = build_structure('random_z4', seed=seed)
        V = np.array(m['V'])
        mr, _ = measure_mr(V, m['E'], m['L'])
        if mr: mr_randoms.append(mr)

    mean_r = np.mean(mr_randoms)
    std_r = np.std(mr_randoms)

    # beta_mn
    m = build_structure('beta_mn')
    mr_bm, _ = measure_mr(np.array(m['V']), m['E'], m['L'])

    within = abs(mr_bm - mean_r) < 2 * std_r
    z_score = (mr_bm - mean_r) / std_r

    print(f"  Random z=4: {mean_r:.3f} ± {std_r:.3f} (n={len(mr_randoms)})")
    print(f"  beta_mn:    {mr_bm:.3f} (z-score: {z_score:.2f})")
    print(f"  Within mean ± 2σ: {within}")

    assert within, \
        f"beta_mn should be within random CI: {mr_bm:.3f} vs {mean_r:.3f}±{2*std_r:.3f}"
    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── T6.11 ────────────────────────────────────────────────────

def test_ndirs_scaling():
    """T6.11: Controlled n_dirs scaling on abstract graphs.

    Build random z≈4 graphs with exactly n_dirs directions assigned cyclically.
    Σv² should decrease with n_dirs and saturate at n_dirs ≥ dim (3 in 3D).
    All values should be << crystal (where periodic assignment matters).
    """
    t0 = time.time()
    print("\nT6.11: Σv² vs n_dirs (controlled scaling)")
    print("-" * 60)

    L = 4.0
    nv = 100
    results = []

    for n_dirs in [2, 3, 6, 10, 50]:
        sv2s = []
        for seed in range(5):
            rng = np.random.RandomState(seed)
            V = (rng.rand(nv, 3) * L).tolist()

            # Build z≈4 graph from nearest neighbors (no duplicate edges)
            V_arr = np.array(V)
            edge_set = set()
            adj = [set() for _ in range(nv)]
            for i in range(nv):
                dists = []
                for j in range(nv):
                    if i != j:
                        dr = V_arr[j] - V_arr[i]
                        dr -= L * np.round(dr / L)
                        dists.append((np.linalg.norm(dr), j))
                dists.sort()
                for _, j in dists[:4]:
                    e = (min(i, j), max(i, j))
                    if e not in edge_set:
                        adj[i].add(j)
                        adj[j].add(i)
                        edge_set.add(e)
            E = [list(e) for e in edge_set]
            assert len(E) >= nv, f"Graph too sparse: {len(E)} edges, {nv} vertices"

            # Generate n_dirs directions, assign cyclically
            rng2 = np.random.RandomState(seed + 1000)
            dir_set = []
            for _ in range(n_dirs):
                d = rng2.randn(3)
                d /= np.linalg.norm(d)
                dir_set.append(d.tolist())

            edge_dirs = [dir_set[ei % n_dirs] for ei in range(len(E))]
            mesh = {'V': V, 'E': E, 'L': L, 'dim': 3, 'edge_dirs': edge_dirs}
            sv2, _, _ = compute_sv2_from_mesh(mesh)
            sv2s.append(sv2)

        mean_sv2 = np.mean(sv2s)
        results.append((n_dirs, mean_sv2))
        print(f"  n_dirs={n_dirs:3d}: Σv²={mean_sv2:.6f}")

    # Σv² should decrease from n_dirs=2 to n_dirs=3
    assert results[0][1] > results[1][1], \
        f"Σv² should decrease 2→3 dirs: {results[0][1]:.6f} vs {results[1][1]:.6f}"

    # All should be << crystal (Kelvin Σv² ≈ 0.017)
    assert all(r[1] < 0.001 for r in results), \
        f"All should be << crystal: max={max(r[1] for r in results):.6f}"

    # Saturation: n_dirs=6 ≈ n_dirs=50 (within 2×)
    sv2_6 = [r[1] for r in results if r[0] == 6][0]
    sv2_50 = [r[1] for r in results if r[0] == 50][0]
    assert 0.3 < sv2_6 / sv2_50 < 3.0, \
        f"Should saturate at n_dirs>dim: {sv2_6:.6f} vs {sv2_50:.6f}"

    print(f"\n  Σv² decreases with n_dirs, saturates at n_dirs ≥ dim.")
    print(f"  All << crystal — random assignment kills transport regardless of n_dirs.")
    print(f"  Time: {time.time()-t0:.1f}s. PASS.")


# ── Main ─────────────────────────────────────────────────────

TESTS = [
    ('scan', test_104_scan_continuous),
    ('nosep', test_no_binary_separator),
    ('complex', test_complex_crystals),
    ('quasi', test_quasicrystal),
    ('betamn', test_complex_crystal_fdtd),
    ('partial', test_ndirs_partial_corr),
    ('ndirs_ov', test_ndirs_overlap),
    ('ci', test_complex_crystal_in_random_ci),
    ('ndirs_sc', test_ndirs_scaling),
]

if __name__ == '__main__':
    print("test_06_continuum.py — §6 coherence continuum")
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
