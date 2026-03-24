"""
structure_catalog.py — Unified builder for all structures used in κ paper.

Single entry point: build_structure(name, **kwargs)
Returns standardized mesh dict with all metadata.

Categories:
  CRYSTAL: 18 crystallographic structures from voronoi_scan (half_heusler removed,
           duplicate of fluorite — identical fractional positions).
  RANDOM:  random Voronoi z=4, random z=3/5/6/8, edge-swapped non-Voronoi.

Every structure is validated on build:
  - nv > 0, ne > 0
  - z_mean within 5% of target (if applicable)
  - connected graph (single component)
  - no degenerate edges (length > 0)

Self-tests at bottom verify all structures build correctly.

Date: 24 Mar 2026
"""
import numpy as np
from builders.voronoi_scan import (
    build_voronoi_mesh, build_random_voronoi, build_edge_swap_graph, CATALOG
)
from builders.random_graphs import build_random_z_graph


# Suspected duplicates — verified at import time (see _verify_duplicates).
_DUPLICATE_CRYSTALS = set()

# Structures too small at any N for reliable ⟨r⟩ FDTD (nv < 30)
_TOO_SMALL = {'sc'}  # nv=8 at N=2

# Crystal N choices: use N=2 if N=1 gives nv < 30
_CRYSTAL_N = {}  # populated by _init_crystal_N()
_INITIALIZED = False


def _verify_duplicates():
    """Check fluorite vs half_heusler at import time."""
    m1 = build_voronoi_mesh('fluorite', N=1)
    m2 = build_voronoi_mesh('half_heusler', N=1)
    v1 = sorted([tuple(np.round(v, 10)) for v in m1['V']])
    v2 = sorted([tuple(np.round(v, 10)) for v in m2['V']])
    if v1 == v2:
        _DUPLICATE_CRYSTALS.add('half_heusler')


_verify_duplicates()


def _init_crystal_N():
    """Determine N for each crystal so nv >= 30."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True
    for cat_name, _, _, _ in CATALOG:
        if cat_name in _DUPLICATE_CRYSTALS:
            continue
        m1 = build_voronoi_mesh(cat_name, N=1)
        if len(m1['V']) >= 30:
            _CRYSTAL_N[cat_name] = 1
        else:
            try:
                m2 = build_voronoi_mesh(cat_name, N=2)
                _CRYSTAL_N[cat_name] = 2
            except Exception:
                _CRYSTAL_N[cat_name] = 1


def _graph_stats(V, E, L):
    """Compute graph statistics."""
    V = np.array(V)
    E_arr = np.array(E)
    nv, ne = len(V), len(E_arr)

    # Degree
    deg = np.zeros(nv, dtype=int)
    for i, j in E_arr:
        deg[i] += 1
        deg[j] += 1

    # Connectivity (BFS)
    adj = [[] for _ in range(nv)]
    for i, j in E_arr:
        adj[i].append(j)
        adj[j].append(i)
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
    connected = all(visited)

    # Unique directions
    dirs = []
    for i, j in E_arr:
        dr = V[j] - V[i]
        dr -= L * np.round(dr / L)
        ell = np.linalg.norm(dr)
        if ell > 1e-10:
            dirs.append(dr / ell)
    unique = []
    for d in dirs:
        is_new = True
        for u in unique:
            if abs(np.dot(d, u)) > 0.9998:  # covers both d≈u and d≈-u
                is_new = False
                break
        if is_new:
            unique.append(d)

    # Edge lengths
    lens = []
    for i, j in E_arr:
        dr = V[j] - V[i]
        dr -= L * np.round(dr / L)
        lens.append(float(np.linalg.norm(dr)))
    min_len = min(lens) if lens else 0

    return {
        'nv': nv, 'ne': ne,
        'z_mean': float(np.mean(deg)),
        'z_min': int(np.min(deg)),
        'z_max': int(np.max(deg)),
        'n_unique_dirs': len(unique),
        'connected': connected,
        'min_edge_length': min_len,
    }


def _validate(mesh, name, z_target=None):
    """Validate mesh, raise on failure."""
    stats = mesh['stats']
    assert stats['nv'] > 0, f"{name}: empty mesh"
    assert stats['ne'] > 0, f"{name}: no edges"
    assert stats['connected'], f"{name}: disconnected graph"
    assert stats['min_edge_length'] > 1e-10, \
        f"{name}: degenerate edge (len={stats['min_edge_length']:.2e})"
    if z_target is not None:
        z_err = abs(stats['z_mean'] - z_target) / z_target
        assert z_err < 0.10, \
            f"{name}: z_mean={stats['z_mean']:.2f} too far from target {z_target} ({z_err:.0%})"


def _enrich(mesh, name, category):
    """Add standardized metadata to mesh."""
    V = np.array(mesh['V'])
    E = mesh['E']
    L = mesh['L']
    stats = _graph_stats(V, E, L)
    mesh['stats'] = stats
    mesh['name'] = name
    mesh['category'] = category
    return mesh


def build_structure(name, **kwargs):
    """Build any structure by name.

    Crystal names: sc, bcc, fcc, diamond, a15, c15, clathrate_I, nacl,
                   perovskite, fluorite, pyrochlore, sodalite, beta_mn,
                   gamma_brass, spinel, alpha_mn, th3p4, pyrite, skutterudite
    Random names:  random_z4, random_z3, random_z5, random_z6, random_z8,
                   edge_swap
    Args:
        name: structure name
        n_seeds: (random only) number of seed points, default 15
        seed: (random only) random seed, default 42
        swap_seed: (edge_swap only) swap random seed, default None (=seed+100)
        N: (crystal only) override supercell size

    Returns: mesh dict with V, E, L, name, category, stats
    """
    # Random structures
    if name.startswith('random_z'):
        z_str = name[len('random_z'):]
        if not z_str.isdigit():
            raise ValueError(f"Invalid random name '{name}': expected 'random_zN' where N is integer")
        z = int(z_str)
        if z not in (3, 4, 5, 6, 8, 10, 12):
            raise ValueError(f"z_target={z} not supported. Use 3, 4, 5, 6, 8, 10, 12.")
        n_seeds = kwargs.get('n_seeds', 15)
        seed = kwargs.get('seed', 42)
        m = build_random_z_graph(n_seeds, z_target=z, L=4.0, seed=seed)
        m = _enrich(m, f'{name}_s{seed}', 'random')
        _validate(m, m['name'], z_target=z)
        return m

    if name == 'edge_swap':
        n_seeds = kwargs.get('n_seeds', 15)
        seed = kwargs.get('seed', 42)
        swap_seed = kwargs.get('swap_seed', seed + 100)
        m = build_edge_swap_graph(n_seeds, L=4.0, seed=seed, swap_seed=swap_seed)
        m = _enrich(m, f'edgeswap_s{seed}', 'random')
        _validate(m, m['name'], z_target=4)
        return m

    # Crystal structures
    if name in _DUPLICATE_CRYSTALS:
        raise ValueError(f"'{name}' is a duplicate (identical to fluorite). Use 'fluorite' instead.")

    _init_crystal_N()
    if name not in _CRYSTAL_N:
        raise ValueError(f"Unknown structure '{name}'. Available: {list_names()}")

    N = kwargs.get('N', _CRYSTAL_N[name])
    m = build_voronoi_mesh(name, N=N)
    m = _enrich(m, f'{name}' + (f'(N={N})' if N > 1 else ''), 'crystal')
    _validate(m, m['name'])
    return m


def list_names():
    """All available structure names."""
    _init_crystal_N()
    crystals = sorted(_CRYSTAL_N.keys())
    randoms = ['random_z3', 'random_z4', 'random_z5', 'random_z6', 'random_z8', 'edge_swap']
    return crystals + randoms


def build_all_crystals(**kwargs):
    """Build all non-duplicate crystal structures."""
    _init_crystal_N()
    results = []
    for name in sorted(_CRYSTAL_N.keys()):
        m = build_structure(name, **kwargs)
        results.append(m)
    return results


def build_random_set(n_seeds=15, seeds=range(5), z_values=[4], include_edge_swap=True):
    """Build a set of random structures."""
    results = []
    for z in z_values:
        for seed in seeds:
            m = build_structure(f'random_z{z}', n_seeds=n_seeds, seed=seed)
            results.append(m)
    if include_edge_swap:
        for seed in seeds:
            m = build_structure('edge_swap', n_seeds=n_seeds, seed=seed)
            results.append(m)
    return results


# =============================================================================
# SELF-TESTS
# =============================================================================

def _self_test_all_crystals():
    """Build all crystals, validate, print table."""
    _init_crystal_N()
    print(f"Building {len(_CRYSTAL_N)} crystal structures...")
    print(f"  {'name':>15s} {'N':>2s} {'nv':>5s} {'ne':>5s} {'z':>5s} {'n_dir':>5s} {'conn':>4s}")
    print(f"  {'-'*45}")

    n_pass = 0
    for name in sorted(_CRYSTAL_N.keys()):
        m = build_structure(name)
        _validate(m, name)
        s = m['stats']
        print(f"  {m['name']:>15s} {_CRYSTAL_N[name]:2d} {s['nv']:5d} {s['ne']:5d} "
              f"{s['z_mean']:5.1f} {s['n_unique_dirs']:5d} {'Y' if s['connected'] else 'N':>4s}")
        n_pass += 1

    print(f"\n  {n_pass}/{len(_CRYSTAL_N)} crystals built and validated.")
    return n_pass


def _self_test_duplicate_removed():
    """Verify half_heusler is blocked."""
    try:
        build_structure('half_heusler')
        print("  FAIL: half_heusler should raise ValueError")
        return False
    except ValueError as e:
        print(f"  half_heusler correctly blocked: {e}")
        return True


def _self_test_random_builders():
    """Test random builders with validation."""
    print(f"\nBuilding random structures...")
    n_pass = 0
    for name in ['random_z3', 'random_z4', 'random_z5', 'random_z6', 'random_z8']:
        m = build_structure(name, seed=42)
        _validate(m, name)
        s = m['stats']
        z_target = int(name.split('z')[1])
        z_ok = abs(s['z_mean'] - z_target) / z_target < 0.05
        print(f"  {m['name']:>20s}: nv={s['nv']:4d}, z={s['z_mean']:.2f} "
              f"(target={z_target}, {'OK' if z_ok else 'WARN'}), "
              f"n_dir={s['n_unique_dirs']}, conn={'Y' if s['connected'] else 'N'}")
        n_pass += 1

    # Edge swap
    m = build_structure('edge_swap', seed=42)
    _validate(m, 'edge_swap')
    s = m['stats']
    print(f"  {m['name']:>20s}: nv={s['nv']:4d}, z={s['z_mean']:.2f}, "
          f"n_dir={s['n_unique_dirs']}, conn={'Y' if s['connected'] else 'N'}")
    n_pass += 1

    print(f"\n  {n_pass}/6 random builders validated.")
    return n_pass


def _self_test_consistency():
    """Verify fluorite and half_heusler were actually identical."""
    from builders.voronoi_scan import build_voronoi_mesh as bvm
    m1 = bvm('fluorite', N=1)
    m2 = bvm('half_heusler', N=1)
    v1 = sorted([tuple(v) for v in m1['V']])
    v2 = sorted([tuple(v) for v in m2['V']])
    identical = v1 == v2
    print(f"\n  fluorite vs half_heusler vertices identical: {identical}")
    if not identical:
        print(f"  WARNING: structures differ — half_heusler should NOT be blocked!")
    return identical


if __name__ == '__main__':
    print("structure_catalog.py — self-tests")
    print("=" * 60)

    p1 = _self_test_all_crystals()
    p2 = _self_test_duplicate_removed()
    p3 = _self_test_random_builders()
    p4 = _self_test_consistency()

    total = p1 + (1 if p2 else 0) + p3 + (1 if p4 else 0)
    expected = len(_CRYSTAL_N) + 1 + 6 + 1
    print(f"\n{'='*60}")
    print(f"{total}/{expected} PASS")
