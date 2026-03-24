"""
fibonacci_3d.py — 3D Fibonacci quasicrystal builder.

Tensor product of three 1D Fibonacci chains. Standard construction
(Kohmoto, Kadanoff, Tang 1983). Two tile types S and L with L/S = φ.

Properties:
  - Edge directions: only ±x, ±y, ±z (3 unique, same as SC)
  - Spacing: aperiodic (two values S and L, ratio φ)
  - Topology: nearest-neighbor on grid (z=6, same as SC)
  - Periodicity: NONE (quasiperiodic)

Key for paper: same directional coherence as SC, non-periodic spacing.
Tests whether periodicity is required for transport.

Date: 24 Mar 2026
"""
import numpy as np


def _fibonacci_chain(n_points, L):
    """Generate 1D Fibonacci sequence of positions.

    Uses substitution rule: L → LS, S → L (Fibonacci word).
    Two tile types with spacing ratio L/S = φ = (1+√5)/2.
    Positions normalized to fit in [0, L).

    Args:
        n_points: number of points on chain
        L: total chain length

    Returns: array of positions, length n_points
    """
    phi = (1 + np.sqrt(5)) / 2

    # Generate Fibonacci word by substitution until long enough
    word = ['L']
    while len(word) < n_points:
        new_word = []
        for tile in word:
            if tile == 'L':
                new_word.extend(['L', 'S'])
            else:  # S
                new_word.append('L')
        word = new_word
    word = word[:n_points]

    # Assign spacings: n_L * a_L + n_S * a_S = L, a_L / a_S = phi
    n_L = sum(1 for t in word if t == 'L')
    n_S = sum(1 for t in word if t == 'S')
    a_S = L / (n_L * phi + n_S)
    a_L = phi * a_S

    # Cumulative positions
    positions = [0.0]
    for tile in word[:-1]:
        step = a_L if tile == 'L' else a_S
        positions.append(positions[-1] + step)

    return np.array(positions)


def build_fibonacci_3d(n=8, L=4.0):
    """Build 3D Fibonacci quasicrystal.

    Tensor product of three 1D Fibonacci chains with periodic NN edges.
    nv = n³, ne = 3n³ (each vertex has 6 neighbors: ±x, ±y, ±z).

    Args:
        n: points per axis (nv = n³)
        L: cubic box size

    Returns: mesh dict with V, E, L, dim, name
    """
    xs = _fibonacci_chain(n, L)
    ys = _fibonacci_chain(n, L)
    zs = _fibonacci_chain(n, L)

    # Build vertices
    V = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                V.append([float(xs[ix]), float(ys[iy]), float(zs[iz])])

    # Index map: (ix, iy, iz) → vertex index
    def idx(ix, iy, iz):
        return ix * n * n + iy * n + iz

    # Build edges: NN along each axis (periodic)
    E = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                v = idx(ix, iy, iz)
                # +x neighbor
                E.append([v, idx((ix + 1) % n, iy, iz)])
                # +y neighbor
                E.append([v, idx(ix, (iy + 1) % n, iz)])
                # +z neighbor
                E.append([v, idx(ix, iy, (iz + 1) % n)])

    nv = n ** 3
    ne = 3 * nv
    assert len(V) == nv
    assert len(E) == ne

    return {
        'V': V, 'E': E, 'F': [], 'L': L, 'dim': 3,
        'name': f'fibonacci_3d(n={n})',
    }


# =============================================================================
# SELF-TESTS
# =============================================================================

def _self_test():
    import sys, os
    _math = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(_math)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    print("fibonacci_3d.py — self-tests")
    print("=" * 50)

    # T1: chain has correct length, spacing ratio, and aperiodicity
    phi = (1 + np.sqrt(5)) / 2
    L = 4.0
    n = 8
    chain = _fibonacci_chain(n, L)
    assert len(chain) == n, f"Chain length: {len(chain)}"
    spacings = np.diff(chain)
    unique_s = sorted(set(np.round(spacings, 6)))
    assert len(unique_s) == 2, f"Should have 2 tile types: {unique_s}"
    ratio = unique_s[1] / unique_s[0]
    assert abs(ratio - phi) < 0.01, f"Ratio should be φ: {ratio:.4f}"
    n_L = sum(1 for s in spacings if abs(s - unique_s[1]) < 1e-4)
    n_S = len(spacings) - n_L
    assert n_L > n_S, f"Should have more L than S tiles: n_L={n_L}, n_S={n_S}"
    # Verify aperiodicity: no period p < n divides the spacing sequence
    for p in range(1, len(spacings) // 2 + 1):
        if len(spacings) % p == 0:
            is_periodic = all(abs(spacings[i] - spacings[i % p]) < 1e-10
                              for i in range(len(spacings)))
            assert not is_periodic, f"Chain is periodic with period {p}"
    print(f"  T1: chain OK. Tiles: S={unique_s[0]:.4f}, L={unique_s[1]:.4f}, "
          f"ratio={ratio:.4f}, n_L={n_L}, n_S={n_S}, aperiodic.")

    # T2: 3D mesh has correct counts
    m = build_fibonacci_3d(n=5, L=4.0)
    assert len(m['V']) == 125, f"nv: {len(m['V'])}"
    assert len(m['E']) == 375, f"ne: {len(m['E'])}"
    print(f"  T2: n=5 → nv=125, ne=375. OK.")

    # T3: all edges are axis-aligned (3 unique directions)
    from measure import count_unique_dirs
    n_dir = count_unique_dirs(m['V'], m['E'], m['L'])
    assert n_dir == 3, f"Should have 3 dirs (±x,±y,±z): {n_dir}"
    print(f"  T3: n_dir=3. Same as SC. OK.")

    # T4: z=6 on all vertices
    V = np.array(m['V'])
    E = np.array(m['E'])
    nv = len(V)
    deg = np.zeros(nv, dtype=int)
    for i, j in E:
        deg[i] += 1
        deg[j] += 1
    assert np.all(deg == 6), f"z should be 6: min={deg.min()}, max={deg.max()}"
    print(f"  T4: z=6 on all vertices. OK.")

    # T5: Σv² > 0 and significant fraction of SC
    from builders.random_graphs import compute_sv2_from_mesh
    sv2_fib, fflat, _ = compute_sv2_from_mesh(m)
    # SC reference at same n
    n_sc = 5
    V_sc = [[ix+0.5, iy+0.5, iz+0.5]
            for ix in range(n_sc) for iy in range(n_sc) for iz in range(n_sc)]
    E_sc = []
    for ix in range(n_sc):
        for iy in range(n_sc):
            for iz in range(n_sc):
                i = ix*n_sc*n_sc + iy*n_sc + iz
                E_sc.append([i, ((ix+1)%n_sc)*n_sc*n_sc + iy*n_sc + iz])
                E_sc.append([i, ix*n_sc*n_sc + ((iy+1)%n_sc)*n_sc + iz])
                E_sc.append([i, ix*n_sc*n_sc + iy*n_sc + (iz+1)%n_sc])
    m_sc = {'V': V_sc, 'E': E_sc, 'L': float(n_sc), 'dim': 3, 'F': []}
    sv2_sc, _, _ = compute_sv2_from_mesh(m_sc)
    ratio = sv2_fib / sv2_sc
    assert sv2_fib > 0.01, f"Fibonacci should transport: Σv²={sv2_fib:.6f}"
    assert ratio > 0.30, f"Should be ≥30% of SC: {ratio:.2f}"
    print(f"  T5: Σv²_fib={sv2_fib:.6f}, Σv²_SC={sv2_sc:.6f}, "
          f"ratio={ratio:.2f}. Transports. OK.")

    print(f"\n5/5 PASS")


if __name__ == '__main__':
    _self_test()
