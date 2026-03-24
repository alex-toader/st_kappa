"""
random_graphs.py — Random graph builders for z≠4 investigations.

Builds random z-regular graphs embedded in 3D periodic box.
Purpose: test whether κ≈1 is z=4 specific or extends to other coordinations.

Strategy for each z:
  z=4: use build_random_voronoi (already exact, Euler relation)
  z=3: start from z=4 Voronoi, remove edges (keeping connected)
  z=5: start from z=4 Voronoi, add short-range edges
  z=6: start from z=4 Voronoi, add 2 edges per vertex (nearest non-neighbors)
  z=8: start from z=4 Voronoi, add 4 edges per vertex

All builders preserve: 3D periodic embedding, vertex positions, spring model compatibility.
All return same dict format as build_random_voronoi: {V, E, L, name, ...}

Maxwell prediction (k_T=0, 1 constraint per edge, 3 DOF per vertex):
  floppy = 3nV - nE = nV*(3 - z/2)
  z=3: 1.5nV floppy    (very under-constrained, many flat bands)
  z=4: 1.0nV floppy    (under-constrained, flat bands → κ≈1 on random)
  z=5: 0.5nV floppy    (mildly under-constrained)
  z=6: 0 floppy        (isostatic — transition point?)
  z=8: -nV overconstrained (SSS > 0)

Results (23 Mar 2026):
  z=3: Σv² ≈ 0.000013 (3 seeds, SLOW)
  z=4: Σv² ≈ 0.000268 (from W3, confirmed)
  z=5: Σv² ≈ 0.000075 (3 seeds, SLOW)
  z=6: Σv² ≈ 0.000076 (3 seeds, SLOW — Maxwell predicts isostatic transition!)
  z=8: Σv² ≈ 0.000145 (3 seeds, SLOW)
  ALL z give Σv²≈0 on random → κ≈1 is NOT z-specific, it's from DISORDER.
  Crystal z=6.67 (diamond): Σv²=0.086 (1000× larger at same z).

Related file: open_random.py
  FDTD measurement of ⟨r⟩ on open-boundary random Voronoi (time-domain).
  This file does SPECTRAL analysis (Σv² from eigenvalues of H(k)).
  open_random.py does TIME-DOMAIN simulation (⟨r⟩ from Verlet FDTD + PML).
  Both measure transport on random foam but via different methods.
  On crystals: Σv² predicts ⟨r⟩ (ρ=0.85). On random: Σv² ≈ 0 always,
  ⟨r⟩ ≈ 1.4 from mode redistribution — the two observables DECOUPLE.

Known limitations:
  - z=3 on odd nV: one vertex stays at z=4 (graph theory constraint)
  - z=3 builder removes longest edges first → slightly more clustered than true random z=3
  - z>4 builders add edges to nearest non-neighbors → hybrid Voronoi+geometric graph
  - These are confounds but Σv²≈0 on ALL z (3-8) makes confounds irrelevant
  - Practical limit: n_seeds≤25 for z=3 (BFS connectivity check O(nV²×nE))

API:
  build_random_z_graph(n_seeds, z_target, L, seed) → mesh dict (3D, z=3..12)
  build_2d_voronoi_primal(n_seeds, L, seed) → mesh dict (2D, z≈3)
  build_abstract_z_graph(nv, z_target, L, seed) → mesh dict (no 3D embedding)

Self-tests: 34/34 (T1-T8: z accuracy, connectivity, no defects, edge count,
  reproducibility, sizes, 2D, abstract)
"""

import numpy as np
from .voronoi_scan import build_random_voronoi


def _periodic_distance_matrix(V, L):
    """Compute pairwise periodic distance matrix."""
    nv = len(V)
    V = np.array(V)
    dist = np.zeros((nv, nv))
    for i in range(nv):
        dr = V - V[i]
        dr = dr - L * np.round(dr / L)
        dist[i] = np.linalg.norm(dr, axis=1)
    np.fill_diagonal(dist, 1e10)
    return dist


def _adjacency_from_edges(nv, E):
    """Build adjacency sets from edge list."""
    adj = [set() for _ in range(nv)]
    for i, j in np.array(E):
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _edges_from_adjacency(adj):
    """Build sorted edge list from adjacency sets."""
    nv = len(adj)
    seen = set()
    E = []
    for i in range(nv):
        for j in adj[i]:
            edge = (min(i, j), max(i, j))
            if edge not in seen:
                seen.add(edge)
                E.append(list(edge))
    return E


def _degrees(adj):
    """Return degree array."""
    return np.array([len(adj[i]) for i in range(len(adj))])


def _is_connected_without(adj, i, j, nv):
    """Check if graph remains connected after removing edge (i,j). BFS."""
    visited = [False] * nv
    visited[i] = True
    queue = [i]
    head = 0
    while head < len(queue):
        v = queue[head]
        head += 1
        for u in adj[v]:
            if (v == i and u == j) or (v == j and u == i):
                continue  # skip the removed edge
            if not visited[u]:
                visited[u] = True
                queue.append(u)
    return visited[j]


def build_random_z3(n_seeds, L=4.0, seed=42):
    """Build random z≈3 graph by removing edges from z=4 Voronoi.

    Strategy: start from z=4 Voronoi, iteratively remove edges
    from vertices with z>3, preferring long edges, checking connectivity via BFS.
    May not reach exact z=3 on all vertices (some stuck at z=4 to keep connected).
    """
    m = build_random_voronoi(n_seeds, L=L, seed=seed)
    V = np.array(m['V'])
    nv = len(V)
    adj = _adjacency_from_edges(nv, m['E'])
    dist = _periodic_distance_matrix(V, L)
    rng = np.random.RandomState(seed + 1000)

    target_ne = int(round(3 * nv / 2))

    for _ in range(nv * 20):
        degs = _degrees(adj)
        current_ne = sum(degs) // 2
        if current_ne <= target_ne:
            break

        # Find vertices with z > 3
        candidates = np.where(degs > 3)[0]
        if len(candidates) == 0:
            break

        i = rng.choice(candidates)
        neighbors = list(adj[i])

        # Score: prefer removing edge where both endpoints have z>3, longest first
        scored = []
        for j in neighbors:
            if len(adj[j]) <= 3:
                continue  # would make j z=2
            scored.append((j, dist[i][j], len(adj[j])))

        if not scored:
            # All neighbors at z=3 — can't remove without creating z=2
            continue

        scored.sort(key=lambda x: (-x[2], -x[1]))

        removed = False
        for j, _, _ in scored:
            if _is_connected_without(adj, i, j, nv):
                adj[i].discard(j)
                adj[j].discard(i)
                removed = True
                break

        # If nothing removable, skip this vertex

    E = _edges_from_adjacency(adj)
    degs = _degrees(adj)

    return {
        'V': V.tolist(), 'E': E, 'F': [], 'L': L,
        'name': f'random_z3_{n_seeds}_s{seed}',
        'n_sites': n_seeds,
        'is_plateau': False,
        'z_stats': {
            'z_mean': float(np.mean(degs)),
            'z_min': int(np.min(degs)),
            'z_max': int(np.max(degs)),
            'z_target': 3,
        },
    }


def _add_edges_to_z(adj, dist, z_target, rng, nv):
    """Add edges to increase mean z to z_target. Prefer short edges.

    Recalculates edge count each iteration to avoid drift.
    Stops when target nE = z_target * nV / 2 reached or no candidates.
    """
    target_ne = int(round(z_target * nv / 2))
    max_iter = target_ne * 5

    for _ in range(max_iter):
        current_ne = sum(len(adj[i]) for i in range(nv)) // 2
        if current_ne >= target_ne:
            break

        degs = _degrees(adj)
        under = np.where(degs < z_target)[0]
        if len(under) == 0:
            break

        i = rng.choice(under)

        # Find nearest non-neighbor that also has z < z_target
        candidates = []
        for j in range(nv):
            if j == i or j in adj[i]:
                continue
            if degs[j] < z_target:
                candidates.append((j, dist[i][j]))

        if not candidates:
            # Accept any non-neighbor with z < z_target + 1 (allow slight over)
            for j in range(nv):
                if j == i or j in adj[i]:
                    continue
                if degs[j] <= z_target:
                    candidates.append((j, dist[i][j]))

        if not candidates:
            continue

        # Pick nearest
        candidates.sort(key=lambda x: x[1])
        j = candidates[0][0]

        adj[i].add(j)
        adj[j].add(i)


def _build_random_z_above4(n_seeds, z_target, L=4.0, seed=42):
    """Build random z>4 graph by adding edges to z=4 Voronoi.

    Adds shortest available edges to under-connected vertices until
    mean z reaches z_target. Same vertex positions as base Voronoi.
    """
    m = build_random_voronoi(n_seeds, L=L, seed=seed)
    V = np.array(m['V'])
    nv = len(V)
    adj = _adjacency_from_edges(nv, m['E'])
    dist = _periodic_distance_matrix(V, L)
    rng = np.random.RandomState(seed + z_target * 1000)

    _add_edges_to_z(adj, dist, z_target, rng, nv)

    E = _edges_from_adjacency(adj)
    degs = _degrees(adj)

    return {
        'V': V.tolist(), 'E': E, 'F': [], 'L': L,
        'name': f'random_z{z_target}_{n_seeds}_s{seed}',
        'n_sites': n_seeds,
        'is_plateau': False,
        'z_stats': {
            'z_mean': float(np.mean(degs)),
            'z_min': int(np.min(degs)),
            'z_max': int(np.max(degs)),
            'z_target': z_target,
        },
    }


# --- 2D Voronoi primal graph (z≈3) ---

def build_2d_voronoi_primal(n_seeds, L=4.0, seed=42):
    """Build 2D Voronoi primal graph (vertices at cell wall intersections, z≈3).

    The primal graph of a 2D Voronoi has vertices where three cell walls meet
    and edges along cell walls. Generic 2D Voronoi has z=3 (trivalent).

    Args:
        n_seeds: number of random seed points
        L: periodic box size
        seed: random seed

    Returns: dict with V (nv,2), E, L, z_stats

    Result (23 Mar 2026):
      Σv² ≈ 0.0003-0.003 on 2D random Voronoi (same order as 3D).
      Mechanism universal across dimensions.
    """
    from scipy.spatial import Voronoi as Voronoi2D

    rng = np.random.RandomState(seed)
    pts = rng.rand(n_seeds, 2) * L

    # 9 periodic images (2D)
    images = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            images.append(pts + np.array([dx, dy]) * L)
    all_pts = np.vstack(images)
    vor = Voronoi2D(all_pts)

    # Map Voronoi vertex index → our vertex index (dedup by periodic wrapping)
    vertex_map = {}
    V = []

    for vi, v in enumerate(vor.vertices):
        vw = v % L
        found = False
        for existing_idx, existing_v in enumerate(V):
            dr = np.abs(np.array(existing_v) - vw)
            dr = np.minimum(dr, L - dr)
            if np.linalg.norm(dr) < 1e-6:
                vertex_map[vi] = existing_idx
                found = True
                break
        if not found:
            vertex_map[vi] = len(V)
            V.append(vw.tolist())

    # Edges from Voronoi ridges
    edge_set = set()
    for ridge_verts in vor.ridge_vertices:
        if -1 in ridge_verts or len(ridge_verts) < 2:
            continue
        v1 = vertex_map.get(ridge_verts[0])
        v2 = vertex_map.get(ridge_verts[1])
        if v1 is not None and v2 is not None and v1 != v2:
            edge_set.add((min(v1, v2), max(v1, v2)))

    V = np.array(V)
    E = [list(e) for e in edge_set]
    nv = len(V)

    degs = np.zeros(nv, dtype=int)
    for i, j in E:
        degs[i] += 1
        degs[j] += 1

    return {
        'V': V.tolist(), 'E': E, 'F': [], 'L': L,
        'name': f'voronoi2d_{n_seeds}_s{seed}',
        'n_sites': n_seeds,
        'is_plateau': False,
        'dim': 2,
        'z_stats': {
            'z_mean': float(np.mean(degs)),
            'z_min': int(np.min(degs)),
            'z_max': int(np.max(degs)),
            'z_target': 3,
        },
    }


# --- Abstract random graph (no 3D embedding) ---

def build_abstract_z_graph(nv, z_target=4, L=4.0, seed=42):
    """Build random z-regular graph with random positions and random edge directions.

    No geometric relationship between vertex positions and edge connectivity.
    Tests whether 3D embedding matters for κ≈1 or just graph randomness.

    Uses configuration model with repair: pair random half-edges, then fix
    under-connected vertices by adding edges to other under-connected vertices.

    Args:
        nv: number of vertices
        z_target: target coordination
        L: box size (for Bloch phases)
        seed: random seed

    Returns: dict with V (nv,3), E, L, edge_dirs (ne,3), z_stats

    Result (23 Mar 2026):
      Σv² ≈ 0.0003 — same as geometric Voronoi.
      3D embedding is irrelevant for κ≈1.
    """
    rng = np.random.RandomState(seed)

    # Configuration model: pair half-edges randomly
    stubs = []
    for v in range(nv):
        for _ in range(z_target):
            stubs.append(v)
    rng.shuffle(stubs)

    edges = set()
    adj = [set() for _ in range(nv)]

    for k in range(0, len(stubs) - 1, 2):
        i, j = stubs[k], stubs[k + 1]
        if i == j:
            continue
        edge = (min(i, j), max(i, j))
        if edge in edges:
            continue
        edges.add(edge)
        adj[i].add(j)
        adj[j].add(i)

    # Repair: fix under-connected vertices
    for _ in range(nv * 10):
        degs = _degrees(adj)
        under = np.where(degs < z_target)[0]
        if len(under) < 2:
            break

        # Pick two under-connected vertices that aren't already connected
        rng.shuffle(under)
        added = False
        for idx_a in range(len(under)):
            if added:
                break
            i = under[idx_a]
            if len(adj[i]) >= z_target:
                continue
            for idx_b in range(idx_a + 1, len(under)):
                j = under[idx_b]
                if len(adj[j]) >= z_target:
                    continue
                if j in adj[i]:
                    continue
                edge = (min(i, j), max(i, j))
                if edge not in edges:
                    edges.add(edge)
                    adj[i].add(j)
                    adj[j].add(i)
                    added = True
                    break

        if not added:
            # No under-under pair available — try under-any
            for i in under:
                if len(adj[i]) >= z_target:
                    continue
                for j in range(nv):
                    if j == i or j in adj[i]:
                        continue
                    if len(adj[j]) >= z_target + 1:
                        continue  # don't make j too over-connected
                    edge = (min(i, j), max(i, j))
                    if edge not in edges:
                        edges.add(edge)
                        adj[i].add(j)
                        adj[j].add(i)
                        added = True
                        break
                if added:
                    break
            if not added:
                break

    E = [list(e) for e in edges]

    # Random positions (for Bloch phases only — no geometric meaning)
    V = rng.rand(nv, 3) * L

    # Random edge directions (unit vectors, no relation to vertex positions)
    edge_dirs = []
    for _ in range(len(E)):
        d = rng.randn(3)
        d = d / np.linalg.norm(d)
        edge_dirs.append(d.tolist())

    degs = np.array([len(adj[i]) for i in range(nv)])

    return {
        'V': V.tolist(), 'E': E, 'F': [], 'L': L,
        'name': f'abstract_z{z_target}_n{nv}_s{seed}',
        'n_sites': nv,
        'is_plateau': False,
        'edge_dirs': edge_dirs,
        'z_stats': {
            'z_mean': float(np.mean(degs)),
            'z_min': int(np.min(degs)),
            'z_max': int(np.max(degs)),
            'z_target': z_target,
        },
    }


# --- Bloch Hamiltonian helpers ---

def bloch_H_from_mesh(mesh, k_vec):
    """Build Bloch Hamiltonian from mesh dict. Uses edge_dirs if present.

    For geometric graphs (Voronoi, z>4): computes e_hat from V[j]-V[i].
    For abstract graphs: uses mesh['edge_dirs'] (prescribed random directions).

    Bloch phases always from vertex positions (minimum image convention).

    Args:
        mesh: dict with V, E, L, optionally edge_dirs
        k_vec: wave vector (3,)

    Returns: H (3nv × 3nv complex)
    """
    V = np.array(mesh['V'])
    E = np.array(mesh['E'])
    L = mesh['L']
    nv = len(V)
    has_dirs = 'edge_dirs' in mesh and len(mesh['edge_dirs']) == len(E)

    if has_dirs:
        edge_dirs = np.array(mesh['edge_dirs'])

    H = np.zeros((3 * nv, 3 * nv), dtype=complex)
    for ei in range(len(E)):
        i, j = E[ei]
        dr = V[j] - V[i]
        delta = -L * np.round(dr / L)
        dr_real = dr + delta

        if has_dirs:
            e_hat = edge_dirs[ei]
        else:
            ell = np.linalg.norm(dr_real)
            if ell < 1e-10:
                continue
            e_hat = dr_real / ell

        K = np.outer(e_hat, e_hat)
        phase = np.exp(1j * np.dot(k_vec, delta))
        for a in range(3):
            for b in range(3):
                H[3*i+a, 3*j+b] -= K[a, b] * phase
                H[3*j+a, 3*i+b] -= K[a, b] * np.conj(phase)
                H[3*i+a, 3*i+b] += K[a, b]
                H[3*j+a, 3*j+b] += K[a, b]
    return H


def bloch_H_2d_from_mesh(mesh, k_vec):
    """Build 2D Bloch Hamiltonian from mesh dict. Uses edge_dirs if present.

    For 2D meshes (dim=2). Uses 2-DOF per vertex, k_T=0 springs.

    Args:
        mesh: dict with V (nv,2), E, L, optionally edge_dirs
        k_vec: wave vector (2,)

    Returns: H (2nv × 2nv complex)
    """
    V = np.array(mesh['V'])
    E = np.array(mesh['E'])
    L = mesh['L']
    nv = len(V)
    has_dirs = 'edge_dirs' in mesh and len(mesh['edge_dirs']) == len(E)

    if has_dirs:
        edge_dirs = np.array(mesh['edge_dirs'])

    H = np.zeros((2 * nv, 2 * nv), dtype=complex)
    for ei in range(len(E)):
        i, j = E[ei]
        dr = V[j] - V[i]
        delta = -L * np.round(dr / L)
        dr_real = dr + delta

        if has_dirs:
            e_hat = edge_dirs[ei]
        else:
            ell = np.linalg.norm(dr_real)
            if ell < 1e-10:
                continue
            e_hat = dr_real / ell

        K = np.outer(e_hat, e_hat)
        phase = np.exp(1j * np.dot(k_vec, delta))
        for a in range(2):
            for b in range(2):
                H[2*i+a, 2*j+b] -= K[a, b] * phase
                H[2*j+a, 2*i+b] -= K[a, b] * np.conj(phase)
                H[2*i+a, 2*i+b] += K[a, b]
                H[2*j+a, 2*j+b] += K[a, b]
    return H


def compute_sv2_from_mesh(mesh, n_k=15):
    """Compute Σv²/mode from mesh dict. Handles 2D and 3D, geometric and abstract.

    Returns: (sv2, fflat, n_zero)
    """
    dim = mesh.get('dim', 3)
    V = np.array(mesh['V'])
    L = mesh['L']
    nv = len(V)
    ndof = dim * nv
    k_max = np.pi / L
    k_vals = np.linspace(0, k_max, n_k)
    dk = k_vals[1] - k_vals[0]

    all_omega = []
    for k in k_vals:
        if dim == 2:
            H = bloch_H_2d_from_mesh(mesh, np.array([k, 0]))
        else:
            H = bloch_H_from_mesh(mesh, np.array([k, 0, 0]))
        evals = np.linalg.eigvalsh(H)
        all_omega.append(np.sqrt(np.maximum(np.sort(evals.real), 0)))
    all_omega = np.array(all_omega)

    omega_gamma = all_omega[0]
    n_zero = int(np.sum(omega_gamma < 0.01))
    n_opt = ndof - n_zero

    v2_sum = 0
    flat_count = 0
    for band in range(ndof):
        bw = np.max(all_omega[:, band]) - np.min(all_omega[:, band])
        if omega_gamma[band] >= 0.01 and bw < 0.05:
            flat_count += 1
        if np.max(all_omega[:, band]) > 0.01:
            v_g = np.abs(np.gradient(all_omega[:, band], dk))
            v2_sum += np.mean(v_g)**2

    sv2 = v2_sum / ndof
    fflat = flat_count / n_opt if n_opt > 0 else 0
    return sv2, fflat, n_zero


# --- Dispatcher ---

def build_random_z_graph(n_seeds, z_target, L=4.0, seed=42):
    """Build random z-graph. Dispatches to appropriate builder.

    Args:
        n_seeds: number of seed points (determines nv)
        z_target: target coordination (3, 4, 5, 6, 8)
        L: box size
        seed: random seed

    Returns: mesh dict with V, E, L, z_stats
    """
    if z_target == 3:
        return build_random_z3(n_seeds, L, seed)
    elif z_target == 4:
        m = build_random_voronoi(n_seeds, L, seed)
        degs = np.array([0] * len(m['V']))
        for i, j in np.array(m['E']):
            degs[i] += 1
            degs[j] += 1
        m['z_stats'] = {
            'z_mean': float(np.mean(degs)),
            'z_min': int(np.min(degs)),
            'z_max': int(np.max(degs)),
            'z_target': 4,
        }
        return m
    elif z_target in (5, 6, 8, 10, 12):
        return _build_random_z_above4(n_seeds, z_target, L, seed)
    else:
        raise ValueError(f"z_target={z_target} not supported. Use 3, 4, 5, 6, 8, 10, 12.")


# =============================================================================
# SELF-TESTS
# =============================================================================

def _is_connected(adj, nv):
    """Check if graph is connected via BFS."""
    if nv == 0:
        return True
    visited = [False] * nv
    visited[0] = True
    queue = [0]
    head = 0
    while head < len(queue):
        v = queue[head]
        head += 1
        for u in adj[v]:
            if not visited[u]:
                visited[u] = True
                queue.append(u)
    return all(visited)


def _self_test():
    """Test all z-graph builders: z accuracy, connectivity, no defects, sizes."""
    print("random_graphs.py self-tests")
    print("-" * 60)

    total_pass = 0
    total_tests = 0

    # T1: z accuracy and basic validity on multiple seeds
    for z in [3, 4, 5, 6, 8]:
        for seed in [42, 123, 7]:
            total_tests += 1
            try:
                m = build_random_z_graph(15, z, seed=seed)
                V = np.array(m['V'])
                E = np.array(m['E'])
                nv, ne = len(V), len(E)
                s = m['z_stats']

                assert abs(s['z_mean'] - z) < 0.5, \
                    f"z_mean={s['z_mean']:.2f} too far from {z}"
                assert s['z_min'] >= max(z - 1, 2), \
                    f"z_min={s['z_min']} too low"

                # No self-loops
                for i, j in E:
                    assert i != j, f"Self-loop ({i},{j})"

                # No duplicates
                edge_set = set((min(i, j), max(i, j)) for i, j in E)
                assert len(edge_set) == ne, \
                    f"Duplicates: {ne} edges but {len(edge_set)} unique"

                # All vertex indices valid
                for i, j in E:
                    assert 0 <= i < nv and 0 <= j < nv, \
                        f"Invalid index: ({i},{j}), nv={nv}"

                total_pass += 1
                print(f"  T1 z={z} s={seed}: nv={nv} ne={ne} "
                      f"z={s['z_mean']:.2f} [{s['z_min']}-{s['z_max']}] PASS")
            except Exception as e:
                print(f"  T1 z={z} s={seed}: FAIL ({str(e)[:60]})")

    # T2: connectivity (graph must be connected)
    for z in [3, 4, 6]:
        total_tests += 1
        m = build_random_z_graph(15, z, seed=42)
        nv = len(m['V'])
        adj = _adjacency_from_edges(nv, m['E'])
        conn = _is_connected(adj, nv)
        if conn:
            total_pass += 1
            print(f"  T2 z={z}: connected. PASS.")
        else:
            print(f"  T2 z={z}: DISCONNECTED. FAIL.")

    # T3: edge count accuracy (nE should be close to z*nV/2)
    for z in [3, 5, 6, 8]:
        total_tests += 1
        m = build_random_z_graph(20, z, seed=42)
        nv = len(m['V'])
        ne = len(m['E'])
        expected_ne = z * nv / 2
        ratio = ne / expected_ne
        if 0.90 < ratio < 1.10:
            total_pass += 1
            print(f"  T3 z={z}: ne={ne}, expected={expected_ne:.0f}, "
                  f"ratio={ratio:.3f}. PASS.")
        else:
            print(f"  T3 z={z}: ne={ne}, expected={expected_ne:.0f}, "
                  f"ratio={ratio:.3f}. FAIL.")

    # T4: reproducibility
    total_tests += 1
    m1 = build_random_z_graph(15, 6, seed=42)
    m2 = build_random_z_graph(15, 6, seed=42)
    e1 = set((min(i, j), max(i, j)) for i, j in np.array(m1['E']))
    e2 = set((min(i, j), max(i, j)) for i, j in np.array(m2['E']))
    if e1 == e2:
        total_pass += 1
        print(f"  T4 reproducibility: PASS.")
    else:
        print(f"  T4 reproducibility: FAIL.")

    # T5: different sizes
    for n in [10, 20, 25]:
        total_tests += 1
        try:
            m = build_random_z_graph(n, 5, seed=42)
            s = m['z_stats']
            assert abs(s['z_mean'] - 5) < 0.5
            total_pass += 1
            print(f"  T5 n={n} z=5: nv={len(m['V'])} "
                  f"z={s['z_mean']:.2f}. PASS.")
        except Exception as e:
            print(f"  T5 n={n} z=5: FAIL ({str(e)[:50]})")

    # T6: 2D Voronoi primal
    for n in [15, 20, 25]:
        total_tests += 1
        try:
            m = build_2d_voronoi_primal(n, seed=42)
            V = np.array(m['V'])
            E = np.array(m['E'])
            nv, ne = len(V), len(E)
            s = m['z_stats']

            assert V.shape[1] == 2 or len(V[0]) == 2, "Should be 2D"
            assert s['z_mean'] > 2.5, f"z too low: {s['z_mean']}"
            assert s['z_mean'] < 4.5, f"z too high: {s['z_mean']}"

            # No self-loops, no duplicates
            edge_set = set((min(i, j), max(i, j)) for i, j in E)
            assert len(edge_set) == ne, "Duplicate edges"

            # Connected
            adj_2d = _adjacency_from_edges(nv, E)
            assert _is_connected(adj_2d, nv), "Disconnected"

            total_pass += 1
            print(f"  T6 2D n={n}: nv={nv} ne={ne} "
                  f"z={s['z_mean']:.2f} connected. PASS.")
        except Exception as e:
            print(f"  T6 2D n={n}: FAIL ({str(e)[:60]})")

    # T7: abstract graph (no 3D embedding)
    for nv_target in [30, 50, 80]:
        total_tests += 1
        try:
            m = build_abstract_z_graph(nv_target, z_target=4, seed=42)
            V = np.array(m['V'])
            E = np.array(m['E'])
            nv, ne = len(V), len(E)
            s = m['z_stats']
            dirs = np.array(m['edge_dirs'])

            assert abs(s['z_mean'] - 4) < 0.5, \
                f"z_mean={s['z_mean']:.2f} too far from 4"
            assert dirs.shape == (ne, 3), \
                f"edge_dirs shape {dirs.shape} vs ({ne},3)"

            # Directions are unit vectors
            norms = np.linalg.norm(dirs, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-10), "Dirs not unit"

            # No self-loops
            for i, j in E:
                assert i != j, f"Self-loop ({i},{j})"

            total_pass += 1
            print(f"  T7 abstract nv={nv_target}: nv={nv} ne={ne} "
                  f"z={s['z_mean']:.2f} dirs OK. PASS.")
        except Exception as e:
            print(f"  T7 abstract nv={nv_target}: FAIL ({str(e)[:60]})")

    # T8: abstract graph different z
    for z in [3, 6]:
        total_tests += 1
        try:
            m = build_abstract_z_graph(50, z_target=z, seed=42)
            s = m['z_stats']
            assert abs(s['z_mean'] - z) < 0.5, \
                f"z_mean={s['z_mean']:.2f} too far from {z}"
            total_pass += 1
            print(f"  T8 abstract z={z}: z={s['z_mean']:.2f}. PASS.")
        except Exception as e:
            print(f"  T8 abstract z={z}: FAIL ({str(e)[:60]})")

    # T9: abstract graph with edge_dirs produces Σv²≈0 via bloch_H_from_mesh
    total_tests += 1
    try:
        m = build_abstract_z_graph(50, z_target=4, seed=42)
        sv2, fflat, n_zero = compute_sv2_from_mesh(m)
        assert sv2 < 0.01, f"Abstract Σv²={sv2:.6f} too large"
        total_pass += 1
        print(f"  T9 abstract Σv²: {sv2:.6f} (< 0.01). edge_dirs used. PASS.")
    except Exception as e:
        print(f"  T9 abstract Σv²: FAIL ({str(e)[:60]})")

    # T10: 2D Voronoi Σv² via compute_sv2_from_mesh
    total_tests += 1
    try:
        m2d = build_2d_voronoi_primal(20, seed=42)
        sv2_2d, fflat_2d, nz_2d = compute_sv2_from_mesh(m2d)
        assert sv2_2d < 0.01, f"2D Σv²={sv2_2d:.6f} too large"
        total_pass += 1
        print(f"  T10 2D Σv²: {sv2_2d:.6f} (< 0.01). PASS.")
    except Exception as e:
        print(f"  T10 2D Σv²: FAIL ({str(e)[:60]})")

    # T11: geometric 3D z=6 Σv² via compute_sv2_from_mesh
    total_tests += 1
    try:
        m3d = build_random_z_graph(15, 6, seed=42)
        sv2_3d, _, _ = compute_sv2_from_mesh(m3d)
        assert sv2_3d < 0.01, f"3D z=6 Σv²={sv2_3d:.6f} too large"
        total_pass += 1
        print(f"  T11 3D z=6 Σv²: {sv2_3d:.6f} (< 0.01). PASS.")
    except Exception as e:
        print(f"  T11 3D z=6 Σv²: FAIL ({str(e)[:60]})")

    print("-" * 60)
    print(f"{total_pass}/{total_tests} PASS")
    assert total_pass == total_tests, \
        f"Not all tests passed: {total_pass}/{total_tests}"


if __name__ == '__main__':
    _self_test()
