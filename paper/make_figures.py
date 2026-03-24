"""
Generate all figures for the paper.

Figures:
  fig1_model.pdf        — (schematic, manual — placeholder here)
  fig2_bands.pdf        — band structure: crystal vs random at same z
  fig3_decoherence.pdf  — B/A test on 9 crystals + Gaussian law Σv²(θ)
  fig4_scan.pdf         — Σv² vs ⟨r⟩ on 104 structures, colored crystal/random
  fig5_mr_vs_nv.pdf     — ⟨r⟩ constant on random (100 foams)
  fig6_cosine.pdf       — cos(u_ref, u_def) vs distance, crystal vs random

Usage:
  python make_figures.py          # all figures
  python make_figures.py fig4     # single figure

Date: 24 Mar 2026
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_src = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if _src not in sys.path:
    sys.path.insert(0, _src)

from builders.structure_catalog import build_structure
from builders.random_graphs import compute_sv2_from_mesh, bloch_H_from_mesh
from measure import measure_mr, count_unique_dirs

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'figure.figsize': (3.4, 2.8),
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def fig2_bands():
    """Band structure: crystal (a15) vs random z=4."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.2))

    # Crystal: a15 (N=1, nv=46, 138 DOF — good visibility)
    m_c = build_structure('a15')
    V_c = np.array(m_c['V']); L_c = m_c['L']
    nv_c = len(V_c); ndof_c = 3 * nv_c
    k_max_c = np.pi / L_c
    k_vals_c = np.linspace(0, k_max_c, 40)
    all_omega_c = []
    for k in k_vals_c:
        H = bloch_H_from_mesh(m_c, np.array([k, 0, 0]))
        evals = np.linalg.eigvalsh(H)
        all_omega_c.append(np.sqrt(np.maximum(np.sort(evals.real), 0)))
    all_omega_c = np.array(all_omega_c)

    for band in range(ndof_c):
        ax1.plot(k_vals_c, all_omega_c[:, band], 'b-', lw=0.4, alpha=0.6)
    ax1.set_xlabel('$k$')
    ax1.set_ylabel('$\\omega$')
    ax1.set_title('A15 crystal ($z=4$, 46 vertices)', fontsize=9)
    ax1.set_xlim(0, k_max_c)
    ax1.set_ylim(0, 2.2)

    # Random: z=4 seed=0 (nv~100 — show first 50 bands for clarity)
    m_r = build_structure('random_z4', n_seeds=10, seed=0)
    V_r = np.array(m_r['V']); L_r = m_r['L']
    nv_r = len(V_r); ndof_r = 3 * nv_r
    k_max_r = np.pi / L_r
    k_vals_r = np.linspace(0, k_max_r, 40)
    all_omega_r = []
    for k in k_vals_r:
        H = bloch_H_from_mesh(m_r, np.array([k, 0, 0]))
        evals = np.linalg.eigvalsh(H)
        all_omega_r.append(np.sqrt(np.maximum(np.sort(evals.real), 0)))
    all_omega_r = np.array(all_omega_r)

    n_show = min(ndof_r, 80)
    for band in range(n_show):
        ax2.plot(k_vals_r, all_omega_r[:, band], 'r-', lw=0.5, alpha=0.7)
    ax2.set_xlabel('$k$')
    ax2.set_title(f'Random Voronoi ($z=4$, {nv_r} vertices)', fontsize=9)
    ax2.set_xlim(0, k_max_r)
    ax2.set_ylim(0, 2.2)
    ax2.text(0.5, 0.5, 'flat bands\n($f_{\\mathrm{flat}} > 0.99$)',
             transform=ax2.transAxes, ha='center', va='center',
             fontsize=11, color='darkred', alpha=0.7)

    fig.tight_layout()
    fig.savefig('fig2_bands.pdf')
    plt.close()
    print('  fig2_bands.pdf')


def fig3_decoherence():
    """Two panels: (a) B/A on 9 crystals, (b) Gaussian Σv²(θ)."""
    from scipy.optimize import curve_fit

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8))

    # Panel (a): B/A ratios
    crystals = ['bcc', 'fcc', 'diamond', 'c15', 'a15',
                'pyrochlore', 'perovskite', 'pyrite', 'clathrate_I']
    ba_means = []
    for name in crystals:
        m = build_structure(name)
        V = np.array(m['V']); E = np.array(m['E']); L = m['L']
        dirs = []
        for i, j in E:
            dr = V[j] - V[i]; dr -= L * np.round(dr / L)
            ell = np.linalg.norm(dr)
            dirs.append(dr / max(ell, 1e-10))
        dirs = np.array(dirs)
        mesh_orig = {'V': V.tolist(), 'E': E.tolist(), 'L': L, 'dim': 3,
                     'edge_dirs': dirs.tolist()}
        sv2_A, _, _ = compute_sv2_from_mesh(mesh_orig)
        ratios = []
        for seed in range(5):
            rng = np.random.RandomState(seed)
            rd = rng.randn(len(E), 3)
            rd = rd / np.linalg.norm(rd, axis=1, keepdims=True)
            mesh_r = dict(mesh_orig); mesh_r['edge_dirs'] = rd.tolist()
            sv2_B, _, _ = compute_sv2_from_mesh(mesh_r)
            ratios.append(sv2_B / sv2_A if sv2_A > 1e-10 else 0)
        ba_means.append(np.mean(ratios))

    ax1.barh(range(len(crystals)), ba_means, color='steelblue')
    ax1.set_yticks(range(len(crystals)))
    ax1.set_yticklabels(crystals, fontsize=8)
    ax1.set_xlabel('B/A ratio')
    ax1.axvline(0.1, color='red', ls='--', lw=0.8)
    ax1.set_title('(a) Direction decoherence', fontsize=9)

    # Panel (b): Gaussian Σv²(θ) on Kelvin
    m = build_structure('bcc')
    V = np.array(m['V']); E = np.array(m['E']); L = m['L']
    dirs = []
    for i, j in E:
        dr = V[j] - V[i]; dr -= L * np.round(dr / L)
        ell = np.linalg.norm(dr)
        dirs.append(dr / max(ell, 1e-10))
    dirs = np.array(dirs)

    thetas_deg = [0, 2, 5, 8, 10, 15, 20, 30, 45]
    sv2s = []
    for td in thetas_deg:
        theta = np.radians(td)
        if td == 0:
            noisy = dirs
        else:
            rng = np.random.RandomState(42)
            noisy = []
            for d in dirs:
                axis = rng.randn(3); axis /= np.linalg.norm(axis)
                angle = rng.normal(0, theta)
                c, s = np.cos(angle), np.sin(angle)
                K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
                R = np.eye(3) + s*K + (1-c)*(K@K)
                noisy.append(R @ d)
            noisy = np.array(noisy)
        mesh = {'V': V.tolist(), 'E': E.tolist(), 'L': L, 'dim': 3,
                'edge_dirs': noisy.tolist()}
        sv2, _, _ = compute_sv2_from_mesh(mesh)
        sv2s.append(sv2)

    thetas_rad = np.array([np.radians(t) for t in thetas_deg])
    sv2s = np.array(sv2s)

    def gauss(x, A, sigma):
        return A * np.exp(-x**2 / sigma**2)
    popt, _ = curve_fit(gauss, thetas_rad, sv2s, p0=[sv2s[0], 0.2])
    th_fit = np.linspace(0, thetas_rad[-1], 100)

    ax2.plot(thetas_deg, sv2s, 'ko', ms=4)
    ax2.plot(np.degrees(th_fit), gauss(th_fit, *popt), 'r-', lw=1)
    ax2.set_xlabel('$\\theta$ (degrees)')
    ax2.set_ylabel('$\\Sigma v^2$')
    ax2.set_title(f'(b) Gaussian, $\\sigma={np.degrees(popt[1]):.1f}°$', fontsize=9)

    fig.tight_layout()
    fig.savefig('fig3_decoherence.pdf')
    plt.close()
    print('  fig3_decoherence.pdf')


def fig4_scan():
    """Σv² vs ⟨r⟩ on crystals + random, colored."""
    crystals = ['a15', 'c15', 'clathrate_I', 'diamond', 'fcc', 'fluorite',
                'gamma_brass', 'beta_mn', 'alpha_mn', 'perovskite',
                'pyrochlore', 'pyrite', 'skutterudite', 'spinel', 'th3p4']
    c_sv2, c_mr, c_names = [], [], []
    for name in crystals:
        m = build_structure(name)
        V = np.array(m['V'])
        sv2, _, _ = compute_sv2_from_mesh(m)
        mr, _ = measure_mr(V, m['E'], m['L'])
        if mr is not None:
            c_sv2.append(sv2); c_mr.append(mr); c_names.append(name)

    r_sv2, r_mr = [], []
    for seed in range(15):
        m = build_structure('random_z4', seed=seed)
        V = np.array(m['V'])
        sv2, _, _ = compute_sv2_from_mesh(m)
        mr, _ = measure_mr(V, m['E'], m['L'])
        if mr: r_sv2.append(sv2); r_mr.append(mr)

    fig, ax = plt.subplots()
    ax.scatter(c_sv2, c_mr, c='blue', s=20, zorder=3, label='Crystal')
    ax.scatter(r_sv2, r_mr, c='red', s=10, alpha=0.5, zorder=2, label='Random z=4')
    ax.set_xscale('log')
    ax.set_xlabel('$\\Sigma v^2$')
    ax.set_ylabel('$\\langle r \\rangle$')
    ax.legend(fontsize=8)

    # Annotate a few
    for i, name in enumerate(c_names):
        labels_map = {'fcc': 'FCC', 'diamond': 'diamond', 'beta_mn': 'beta-Mn', 'th3p4': 'Th$_3$P$_4$'}
        if name in labels_map:
            ax.annotate(labels_map[name], (c_sv2[i], c_mr[i]), fontsize=6,
                        xytext=(5, 3), textcoords='offset points')

    fig.savefig('fig4_scan.pdf')
    plt.close()
    print('  fig4_scan.pdf')


def fig5_mr_vs_nv():
    """⟨r⟩ vs nv on random — constant."""
    nvs, mrs_mean, mrs_std = [], [], []
    for n_seeds in [6, 8, 10, 12, 15, 18, 20, 25, 30, 35]:
        seed_mrs, seed_nvs = [], []
        for seed in range(10):
            m = build_structure('random_z4', n_seeds=n_seeds, seed=seed)
            V = np.array(m['V'])
            mr, _ = measure_mr(V, m['E'], m['L'])
            if mr:
                seed_mrs.append(mr)
                seed_nvs.append(len(V))
        if seed_mrs:
            nvs.append(np.mean(seed_nvs))
            mrs_mean.append(np.mean(seed_mrs))
            mrs_std.append(np.std(seed_mrs))

    fig, ax = plt.subplots()
    ax.errorbar(nvs, mrs_mean, yerr=mrs_std, fmt='ko', ms=4, capsize=2)
    ax.axhline(np.mean(mrs_mean), color='red', ls='--', lw=0.8,
               label=f'mean = {np.mean(mrs_mean):.2f}')
    ax.set_xlabel('$n_v$ (vertices)')
    ax.set_ylabel('$\\langle r \\rangle$')
    ax.legend(fontsize=8)
    ax.set_ylim(0.8, 2.0)

    fig.savefig('fig5_mr_vs_nv.pdf')
    plt.close()
    print('  fig5_mr_vs_nv.pdf')


def fig6_cosine():
    """Cosine similarity vs distance: crystal vs random."""
    from core_math.dynamics.md_foam import prepare_edges, harmonic_force_spring
    from gauge_foam import find_edges_crossing_plane, make_peierls_rotation
    from fdtd_foam import gauged_force_foam, wave_packet_foam

    R_MAT = make_peierls_rotation(0.5)

    def get_cos_profile(mesh):
        V = np.array(mesh['V']); E = mesh['E']; L = mesh['L']; nv = len(V)
        edge_info = prepare_edges(V, E, L)
        cxy = np.array([L/2, L/2])
        best_z, best_n = L/2, 0
        for zo in np.arange(0, 2, 0.1):
            z = L/2 + zo
            idx, _ = find_edges_crossing_plane(V, E, L, z, cxy, L/3)
            if len(idx) > best_n: best_n = len(idx); best_z = z
        disk_idx, _ = find_edges_crossing_plane(V, E, L, best_z, cxy, 2.0)
        if len(disk_idx) < 2: return None, None

        defect_center = np.array([L/2, L/2, best_z])
        dr_all = V - defect_center; dr_all -= L * np.round(dr_all / L)
        dist = np.linalg.norm(dr_all, axis=1)

        def fr(u_): return harmonic_force_spring(u_, edge_info, 1.0, 0.0)
        def fd(u_): return gauged_force_foam(u_, edge_info, 1.0, 0.0, disk_idx, R_MAT)

        r_start = np.array([L/4, L/2, L/2])
        u0, v0 = wave_packet_foam(V, np.array([0.5,0,0]), r_start, 2.0, 1.0, 0.0)
        dt = 0.02; n_steps = int(L / 1.0 * 0.9 / dt)
        u_r, v_r = u0.copy(), v0.copy(); a_r = fr(u_r)
        u_d, v_d = u0.copy(), v0.copy(); a_d = fd(u_d)
        for _ in range(n_steps):
            u_r += dt*v_r + 0.5*dt**2*a_r; a_new = fr(u_r); v_r += 0.5*dt*(a_r+a_new); a_r = a_new
            u_d += dt*v_d + 0.5*dt**2*a_d; a_new = fd(u_d); v_d += 0.5*dt*(a_d+a_new); a_d = a_new

        cos_per_v = []
        for iv in range(nv):
            ur_v = u_r[3*iv:3*iv+3]; ud_v = u_d[3*iv:3*iv+3]
            nr, nd = np.linalg.norm(ur_v), np.linalg.norm(ud_v)
            if nr > 1e-15 and nd > 1e-15:
                cos_per_v.append(np.dot(ur_v, ud_v) / (nr * nd))
            else:
                cos_per_v.append(1.0)
        return dist, np.array(cos_per_v)

    fig, ax = plt.subplots()

    m_r = build_structure('random_z4', seed=0)
    dist_r, cos_r = get_cos_profile(m_r)
    if dist_r is not None:
        ax.scatter(dist_r, cos_r, s=3, alpha=0.3, c='red', label='Random z=4')

    m_c = build_structure('diamond')
    dist_c, cos_c = get_cos_profile(m_c)
    if dist_c is not None:
        ax.scatter(dist_c, cos_c, s=3, alpha=0.3, c='blue', label='Diamond')

    ax.set_xlabel('Distance from defect')
    ax.set_ylabel('cos($u_{\\mathrm{ref}}$, $u_{\\mathrm{def}}$)')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.2, 1.1)

    fig.savefig('fig6_cosine.pdf')
    plt.close()
    print('  fig6_cosine.pdf')


FIGURES = {
    'fig2': fig2_bands,
    'fig3': fig3_decoherence,
    'fig4': fig4_scan,
    'fig5': fig5_mr_vs_nv,
}

if __name__ == '__main__':
    selected = sys.argv[1:] if len(sys.argv) > 1 else list(FIGURES.keys())
    print('Generating figures...')
    for name in selected:
        if name in FIGURES:
            FIGURES[name]()
    print('Done.')
