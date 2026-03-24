# Tests Map

52 tests across 5 files (~3 min total). Each file corresponds to one paper section.
Every claim has at least one test with a real assert.

---

## test_02_model.py (12 tests) — §2 Infrastructure validation

| ID | Test | What it verifies |
|----|------|-----------------|
| T2.1 | test_so2_essential | SO(2) Peierls → scattering; α=0 → none (E_sc=0) |
| T2.2-3 | test_ablation_basics | Infrastructure smoke test (full ablation in W1 files 22-23) |
| T2.4 | test_energy_conservation | Verlet KE+PE oscillation < 0.01% over 200 steps |
| T2.5 | test_ranking_robustness | Crystal > random ranking preserved at R_disk=1.5, 2.0, 2.5 |
| T2.6 | test_seeds_reproducible | Seeds 0-4 vs 100-104: means within 2σ |
| T2.7 | test_k_direction | Σv² ranking preserved across 5 random k-directions |
| T2.8 | test_wave_reaches_defect | E_sc/E_inc > 0.01 on random + beta_mn (wave interacts) |
| T2.9 | test_rdisk_plateau | R_disk=2.0 within 4.5% of R=2.5 (on plateau) |
| T2.10 | test_alpha_independence | ⟨r⟩ CV=2.2% across α=0.1, 0.3, 0.5 |
| T2.11 | test_complex_crystal_smoke | measure_mr valid on beta_mn (nv=130) and th3p4 (nv=188) |
| T2.12 | test_k_convergence | 3 vs 10 k-values: 1.6% difference |
| T2.13 | test_plane_z_stability | Auto vs fixed z=L/2: 0.9% difference |

## test_03_flat_bands.py (14 tests) — §3 Central mechanism

| ID | Test | What it verifies |
|----|------|-----------------|
| T3.1 | test_tensor_isotropy | ‖⟨ê⊗ê⟩ − I/3‖ < 0.10 on 15 random structures (5 z × 3 seeds) |
| T3.2 | test_fflat_universal | fflat > 0.99 on 42 random (5z × 7 seeds + 7 edge-swap) |
| T3.3 | test_crystal_random_ratio | Crystal/random Σv²: 70× at z=4, 1097× at z≈6 |
| T3.4 | test_direction_kills | B/A < 0.2 on 9 crystals (8/9 < 0.1, clathrate_I=0.15) |
| T3.5 | test_positions_minor | C/A ∈ [0.1, 4.0] on 9 crystals (positions < directions) |
| T3.6 | test_reverse | Crystal dirs on random graph → Σv²=0.000027 (dead) |
| T3.7 | test_gaussian_decoherence | Σv²(θ) Gaussian R² > 0.93 on bcc, diamond, fcc |
| T3.8 | test_all_or_nothing | 5% swap → 9× drop; 90% restored → 31× below crystal |
| T3.9 | test_modes_extended | IPR×nv ∈ [0.5, 6.0] — modes extended, not Anderson |
| T3.10 | test_geometric_vs_random | Both geometric and random dirs give Σv² << crystal |
| T3.11 | test_theta_scan_causal | Σv² monotonically decreases 0°→45° (129×) |
| T3.12 | test_tensor_vs_transport | Kelvin ⟨ê⊗ê⟩=I/3 exact but Σv²=62× random |
| T3.13 | test_fflat_multi_k | fflat=0.990 on all 5 k-directions (worst case z=4) |
| T3.14 | test_ipr_multi_k | IPR×nv ∈ [2.7, 3.0] at 5 k-values — extended at all k |

## test_04_coherent.py (9 tests) — §4 Crystal transport

| ID | Test | What it verifies |
|----|------|-----------------|
| T4.1 | test_sv2_predicts_mr | Spearman(Σv², ⟨r⟩) = 0.800 (p=0.0003) on 15 crystals |
| T4.2 | test_green_derivation | ⟨v²⟩/⟨v⟩ vs ⟨r⟩: ρ=0.943 (p=0.005) on 6 crystals |
| T4.3 | test_high_transport | fcc, diamond, pyrochlore, perovskite: all ⟨r⟩ > 3 |
| T4.4 | test_low_transport | beta_mn, gamma_brass, alpha_mn, skutterudite, th3p4, spinel: all ⟨r⟩ < 2 |
| T4.5 | test_n_convergence | N=1→N=2: a15 +18%, c15 +15%, pyrite +52% (lower bound) |
| T4.6 | test_unit_effect | th3p4 ⟨r⟩/ℓ=1.79 ≈ random ⟨r⟩/ℓ=1.95 (unit effect, not physics) |
| T4.7 | test_intra_group_correlation | Intra-LOW ρ(Σv², ⟨r⟩) = 0.455 (positive, p=0.19) |
| T4.8 | test_borderline_placement | Borderline Σv² between HIGH and LOW groups |
| T4.10 | test_gap_in_distribution | Largest gap = 1.205 between fluorite and perovskite |

## test_05_incoherent.py (9 tests) — §5 Transport without propagation

| ID | Test | What it verifies |
|----|------|-----------------|
| T5.1 | test_mr_on_scan | ⟨r⟩ ∈ [0.5, 2.5] on 18 random (z=3,4,6 + edge-swap) |
| T5.2 | test_mr_independent_nv | ⟨r⟩ ~ nv^(-0.09) on 100 foams (10 sizes × 10 seeds) |
| T5.3 | test_mr_independent_alpha | CV=2.2% across α=0.05-0.50; α=0.05 within 15% of α=0.50 |
| T5.4 | test_sv2_not_predictive | Intra-z ρ all p > 0.05 on 5 z-values × 15 seeds |
| T5.5-7 | test_beyond_z4 | z=6 ⟨r⟩=1.50, edge-swap 1.79, z6/z4 ratio=1.04 |
| T5.8 | test_cosine_similarity | cos(u_ref, u_def) > 0.90 at r > 3 on random |
| T5.9 | test_rdisk_scaling | ⟨r⟩ grows 1.16→1.44, saturates (last step 0%) |
| T5.10 | test_mr_constant_nv_pbc | ⟨r⟩ ~ nv^0.09 (PBC). Open boundary: W5 file 09 |
| T5.11 | test_no_power_tail | Gaussian R² ≥ power law R² (no long-range tail) |

## test_06_continuum.py (8 tests) — §6 Coherence continuum

| ID | Test | What it verifies |
|----|------|-----------------|
| T6.1 | test_104_scan_continuous | Σv² spans 4.3 orders (0.0002 to 2.6, incl SC N=3) |
| T6.2 | test_no_binary_separator | Δ_spectral separation 0.5× (< 1.0 = overlap → continuum) |
| T6.3 | test_complex_crystals | beta_mn, gamma_brass, alpha_mn: all ⟨r⟩ < 2.0 ≈ random |
| T6.4 | test_quasicrystal | Fibonacci 3D (Kohmoto 1983): 44-64% of SC, >> random |
| T6.5 | test_complex_crystal_fdtd | beta_mn FDTD ⟨r⟩=1.38 within 50% of random (1.45) |
| T6.6 | test_ndirs_partial_corr | ρ_partial(n_dirs, ⟨r⟩ | Σv²) NS — n_dirs is proxy |
| T6.7 | test_ndirs_overlap | Crystal n_dirs [4, 286] spans random [185, 196] |
| T6.8 | test_complex_crystal_in_random_ci | beta_mn z-score=-0.46, within random ± 2σ |
