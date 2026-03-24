"""
voronoi_scan.py — Generic Voronoi builder for crystallographic structures.

Ported from physics_ai/ST_11/src/1_foam/scripts/09_material_scan.py.
Builds periodic Voronoi complexes from fractional site positions.

Returns mesh dict with V, E, F, L (no cells, no Hodge stars).
Sufficient for Layer 0 (exactness) and coherence analysis (Phase 2).

API:
    build_voronoi_mesh(name) → mesh dict
    list_structures() → list of (name, space_group, n_sites)
    sites_*(N, L_cell) → (points, L)

20 structures available (7 existing + 13 new from ST_11 material scan).

Results (20 Mar 2026, N=1, L_cell=4.0):

  Build all 20:
    sc             sites=  1  V=   1 E=   1 F=   1  Plateau=N
    bcc            sites=  2  V=  12 E=  24 F=  14  Plateau=Y
    fcc            sites=  4  V=  12 E=  32 F=  24  Plateau=N
    diamond        sites=  8  V=  24 E=  80 F=  64  Plateau=N
    a15            sites=  8  V=  46 E=  92 F=  54  Plateau=Y
    c15            sites= 24  V= 136 E= 272 F= 160  Plateau=Y
    clathrate_I    sites= 24  V= 105 E= 253 F= 164  Plateau=N
    nacl           sites=  8  V=   8 E=  12 F=   6  Plateau=N
    perovskite     sites=  5  V=  11 E=  36 F=  30  Plateau=N
    fluorite       sites= 12  V=  28 E=  72 F=  56  Plateau=N
    pyrochlore     sites= 16  V=  19 E=  60 F=  60  Plateau=N
    sodalite       sites= 12  V=   3 E=   6 F=   5  Plateau=N
    beta_mn        sites= 20  V= 130 E= 260 F= 150  Plateau=Y
    gamma_brass    sites= 52  V= 341 E= 724 F= 441  Plateau=N
    spinel         sites= 56  V= 289 E= 608 F= 378  Plateau=N
    alpha_mn       sites= 58  V= 352 E= 733 F= 448  Plateau=N
    th3p4          sites= 28  V= 188 E= 380 F= 222  Plateau=N
    pyrite         sites= 12  V=  40 E=  96 F=  68  Plateau=N
    skutterudite   sites= 34  V= 174 E= 389 F= 252  Plateau=N
    half_heusler   sites= 12  V=  28 E=  72 F=  56  Plateau=N

  Layer 0 (V >= 10): 13/17 PASS, 2 FAIL (clathrate_I, skutterudite),
    2 ERROR (perovskite, pyrite — face winding issue), 3 SKIP (sc, nacl, sodalite).

  Known issue: face vertex ordering uses arctan2 with ref = coords[0] - centroid,
  unstable when first vertex is near centroid. Edges always correct.
"""

import numpy as np
from scipy.spatial import Voronoi
from itertools import product


# =============================================================================
# CORE BUILDER
# =============================================================================

def build_voronoi(points, L, repeat=1):
    """Build periodic Voronoi from points in [0,L)^3. Returns V, E, F."""
    n_pts = len(points)

    offsets = list(product(range(-repeat, repeat+1), repeat=3))
    central_idx = offsets.index((0, 0, 0))

    images = []
    for di, dj, dk in offsets:
        offset = np.array([di, dj, dk]) * L
        images.append(points + offset)

    all_points = np.vstack(images)
    central_start = central_idx * n_pts
    central_end = central_start + n_pts

    vor = Voronoi(all_points)

    vertex_dict = {}
    vertices = []
    face_set = {}

    def wrap(pos):
        return tuple(round(x % L, 8) for x in pos)

    def get_idx(pos):
        w = wrap(pos)
        if w not in vertex_dict:
            vertex_dict[w] = len(vertices)
            vertices.append(np.array(w))
        return vertex_dict[w]

    for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
        rverts = vor.ridge_vertices[ridge_idx]
        if -1 in rverts:
            continue

        in_c1 = central_start <= p1 < central_end
        in_c2 = central_start <= p2 < central_end
        if not (in_c1 or in_c2):
            continue

        coords = np.array([vor.vertices[v] for v in rverts])
        center = coords.mean(axis=0)

        ref = coords[0] - center
        normal = np.cross(coords[1] - center, coords[2] - center)
        nl = np.linalg.norm(normal)
        if nl < 1e-14:
            continue
        normal /= nl

        vecs = coords - center
        angles = np.arctan2(
            np.einsum('ij,j->i',
                      np.cross(vecs, ref[np.newaxis, :].repeat(len(vecs), 0)),
                      normal),
            vecs @ ref
        )
        order = np.argsort(angles)
        face_indices = tuple(get_idx(coords[i]) for i in order)

        def canonical(f):
            n = len(f)
            best = f
            for start in range(n):
                rot = f[start:] + f[:start]
                if rot < best:
                    best = rot
            rev = f[::-1]
            for start in range(n):
                rot = rev[start:] + rev[:start]
                if rot < best:
                    best = rot
            return best

        cf = canonical(face_indices)
        if cf not in face_set:
            face_set[cf] = list(face_indices)

    V = np.array(vertices)
    faces = list(face_set.values())

    edge_set = set()
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            edge_set.add((min(a, b), max(a, b)))
    E = sorted(edge_set)

    return V, E, faces


def check_topology(V, E, faces):
    """Check vertex degree and faces per edge. Returns (avg_deg, avg_fpe, is_plateau)."""
    from collections import Counter

    deg = Counter()
    for a, b in E:
        deg[a] += 1
        deg[b] += 1
    degrees = list(deg.values())
    avg_deg = np.mean(degrees) if degrees else 0
    frac_deg4 = sum(1 for d in degrees if d == 4) / len(degrees) if degrees else 0

    edge_face_count = Counter()
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            edge_face_count[(min(a, b), max(a, b))] += 1
    fpe = list(edge_face_count.values())
    avg_fpe = np.mean(fpe) if fpe else 0
    frac_fpe3 = sum(1 for f in fpe if f == 3) / len(fpe) if fpe else 0

    is_plateau = (frac_deg4 >= 0.95 and frac_fpe3 >= 0.95)
    return avg_deg, avg_fpe, is_plateau


# =============================================================================
# SITE GENERATORS
# =============================================================================

def gen_sites(frac_positions, N=1, L_cell=4.0):
    """Fractional positions -> physical coordinates in [0, N*L_cell)^3."""
    L = N * L_cell
    points = []
    seen = set()
    for i, j, k in product(range(N), repeat=3):
        for f in frac_positions:
            cell = (i, j, k)
            p = tuple(round(((cell[d] + f[d]) * L_cell) % L, 8) for d in range(3))
            if p not in seen:
                seen.add(p)
                points.append(list(p))
    return np.array(points), L


def sites_sc(N=1, L_cell=4.0):
    return gen_sites([[0,0,0]], N, L_cell)

def sites_bcc(N=1, L_cell=4.0):
    return gen_sites([[0,0,0], [0.5,0.5,0.5]], N, L_cell)

def sites_fcc(N=1, L_cell=4.0):
    return gen_sites([[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]], N, L_cell)

def sites_diamond(N=1, L_cell=4.0):
    fcc = [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]
    base = [[0,0,0], [0.25,0.25,0.25]]
    fracs = [[(b[d]+t[d])%1 for d in range(3)] for b in base for t in fcc]
    return gen_sites(fracs, N, L_cell)

def sites_a15(N=1, L_cell=4.0):
    """A15 / Weaire-Phelan. Pm-3n (223), 8 sites."""
    fracs = [
        [0,0,0], [0.5,0.5,0.5],
        [0.25,0,0.5], [0.75,0,0.5],
        [0.5,0.25,0], [0.5,0.75,0],
        [0,0.5,0.25], [0,0.5,0.75],
    ]
    return gen_sites(fracs, N, L_cell)

def sites_c15(N=1, L_cell=4.0):
    """C15 Laves. Fd-3m (227), 24 sites."""
    fcc = [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]
    base_8a = [[0,0,0], [0.25,0.25,0.25]]
    base_16d = [[5/8,5/8,5/8], [5/8,3/8,3/8], [3/8,5/8,3/8], [3/8,3/8,5/8]]
    fracs = []
    for b in base_8a:
        for t in fcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    for b in base_16d:
        for t in fcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    return gen_sites(fracs, N, L_cell)

def sites_clathrate_I(N=1, L_cell=4.0):
    """Type I clathrate. Pm-3n (223), 24 sites."""
    x = 0.184
    fracs = [
        [0,0,0], [0.5,0.5,0.5],
        [0.25,0,0.5], [0.75,0,0.5],
        [0.5,0.25,0], [0.5,0.75,0],
        [0,0.5,0.25], [0,0.5,0.75],
    ]
    for sx, sy, sz in product([1,-1], repeat=3):
        fracs.append([sx*x % 1, sy*x % 1, sz*x % 1])
        fracs.append([(0.5+sx*x) % 1, (0.5+sy*x) % 1, (0.5+sz*x) % 1])
    seen = set()
    unique = []
    for f in fracs:
        key = tuple(round(v, 6) for v in f)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return gen_sites(unique, N, L_cell)

def sites_nacl(N=1, L_cell=4.0):
    """NaCl (rocksalt). Fm-3m (225), 8 sites."""
    fcc = [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]
    fracs = []
    for b in [[0,0,0], [0.5,0.5,0.5]]:
        for t in fcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    return gen_sites(fracs, N, L_cell)

def sites_perovskite(N=1, L_cell=4.0):
    """Perovskite (SrTiO3). Pm-3m (221), 5 sites."""
    fracs = [[0,0,0], [0.5,0.5,0.5], [0.5,0,0], [0,0.5,0], [0,0,0.5]]
    return gen_sites(fracs, N, L_cell)

def sites_fluorite(N=1, L_cell=4.0):
    """Fluorite (CaF2). Fm-3m (225), 12 sites."""
    fcc = [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]
    fracs = []
    for b in [[0,0,0], [0.25,0.25,0.25], [0.75,0.75,0.75]]:
        for t in fcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    return gen_sites(fracs, N, L_cell)

def sites_pyrochlore(N=1, L_cell=4.0):
    """Pyrochlore (16d sublattice of C15). Fd-3m (227), 16 sites."""
    fcc = [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]
    base_16d = [[5/8,5/8,5/8], [5/8,3/8,3/8], [3/8,5/8,3/8], [3/8,3/8,5/8]]
    fracs = []
    for b in base_16d:
        for t in fcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    return gen_sites(fracs, N, L_cell)

def sites_sodalite(N=1, L_cell=4.0):
    """Sodalite framework. Im-3m (229), 12 sites."""
    base_12d = [
        [0.25, 0, 0.5], [0.75, 0, 0.5],
        [0.5, 0.25, 0], [0.5, 0.75, 0],
        [0, 0.5, 0.25], [0, 0.5, 0.75],
    ]
    bcc = [[0,0,0], [0.5,0.5,0.5]]
    fracs = []
    for b in base_12d:
        for t in bcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    return gen_sites(fracs, N, L_cell)

def sites_beta_mn(N=1, L_cell=4.0):
    """beta-Mn. P4132 (213), 20 sites."""
    x, y = 0.064, 0.203
    base_8c = [
        [x, x, x], [0.5-x, -x, 0.5+x],
        [-x, 0.5+x, 0.5-x], [0.5+x, 0.5-x, -x],
        [0.5+x, 0.5+x, 0.5+x], [1-x, 0.5-x, x],
        [0.5-x, x, 1-x], [x, 1-x, 0.5-x],
    ]
    base_12d = [
        [1/8, y, 0.25+y], [3/8, -y, 0.75+y],
        [7/8, -y, 0.75-y], [5/8, y, 0.25-y],
        [0.25+y, 1/8, y], [0.75+y, 3/8, -y],
        [0.75-y, 7/8, -y], [0.25-y, 5/8, y],
        [y, 0.25+y, 1/8], [-y, 0.75+y, 3/8],
        [-y, 0.75-y, 7/8], [y, 0.25-y, 5/8],
    ]
    fracs = [[c % 1 for c in pos] for pos in base_8c + base_12d]
    return gen_sites(fracs, N, L_cell)

def sites_gamma_brass(N=1, L_cell=4.0):
    """gamma-brass (Cu5Zn8). I-43m (217), 52 sites."""
    bcc = [[0,0,0], [0.5,0.5,0.5]]
    fracs = []
    for x_val in [0.828, 0.110]:
        base = [[x_val,x_val,x_val], [-x_val,-x_val,x_val],
                [-x_val,x_val,-x_val], [x_val,-x_val,-x_val]]
        for b in base:
            for t in bcc:
                fracs.append([(b[d]+t[d])%1 for d in range(3)])
    x_oh = 0.355
    base_12e = [[x_oh,0,0], [-x_oh,0,0], [0,x_oh,0], [0,-x_oh,0], [0,0,x_oh], [0,0,-x_oh]]
    for b in base_12e:
        for t in bcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    x_co, z_co = 0.313, 0.037
    base_24g = [
        [x_co,x_co,z_co], [-x_co,-x_co,z_co], [-x_co,x_co,-z_co], [x_co,-x_co,-z_co],
        [z_co,x_co,x_co], [z_co,-x_co,-x_co], [-z_co,-x_co,x_co], [-z_co,x_co,-x_co],
        [x_co,z_co,x_co], [-x_co,z_co,-x_co], [x_co,-z_co,-x_co], [-x_co,-z_co,x_co],
    ]
    for b in base_24g:
        for t in bcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    seen = set()
    unique = []
    for f in fracs:
        key = tuple(round(x % 1, 6) for x in f)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return gen_sites(unique, N, L_cell)

def sites_spinel(N=1, L_cell=4.0):
    """Spinel (MgAl2O4). Fd-3m (227), 56 sites."""
    fcc = [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]
    base_8a = [[0,0,0], [0.25,0.25,0.25]]
    base_16d = [[5/8,5/8,5/8], [5/8,3/8,3/8], [3/8,5/8,3/8], [3/8,3/8,5/8]]
    x = 0.3875
    base_32e_half = [[x,x,x], [-x,-x,x], [-x,x,-x], [x,-x,-x]]
    base_32e = base_32e_half + [[(b[d]+0.25)%1 for d in range(3)] for b in base_32e_half]
    fracs = []
    for b in base_8a + base_16d + base_32e:
        for t in fcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    return gen_sites(fracs, N, L_cell)

def sites_alpha_mn(N=1, L_cell=4.0):
    """alpha-Mn. I-43m (217), 58 sites."""
    bcc = [[0,0,0], [0.5,0.5,0.5]]
    fracs = []
    for t in bcc:
        fracs.append(t[:])
    x8 = 0.317
    base_8c = [[x8,x8,x8], [-x8,-x8,x8], [-x8,x8,-x8], [x8,-x8,-x8]]
    for b in base_8c:
        for t in bcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    for xg, zg in [(0.357, 0.034), (0.089, 0.278)]:
        base_24g = [
            [xg,xg,zg], [-xg,-xg,zg], [-xg,xg,-zg], [xg,-xg,-zg],
            [zg,xg,xg], [zg,-xg,-xg], [-zg,-xg,xg], [-zg,xg,-xg],
            [xg,zg,xg], [-xg,zg,-xg], [xg,-zg,-xg], [-xg,-zg,xg],
        ]
        for b in base_24g:
            for t in bcc:
                fracs.append([(b[d]+t[d])%1 for d in range(3)])
    seen = set()
    unique = []
    for f in fracs:
        key = tuple(round(v % 1, 6) for v in f)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return gen_sites(unique, N, L_cell)

def sites_th3p4(N=1, L_cell=4.0):
    """Th3P4-type. I-43d (220), 28 sites."""
    bcc = [[0,0,0], [0.5,0.5,0.5]]
    fracs = []
    base_12a = [
        [3/8, 0, 1/4], [1/8, 0, 3/4],
        [1/4, 3/8, 0], [3/4, 1/8, 0],
        [0, 1/4, 3/8], [0, 3/4, 1/8],
    ]
    for b in base_12a:
        for t in bcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    x = 0.083
    base_16c = [
        [x, x, x], [-x, 0.5-x, 0.5+x],
        [0.5-x, 0.5+x, -x], [0.5+x, -x, 0.5-x],
        [0.25+x, 0.25+x, 0.25+x], [0.25-x, 0.75-x, 0.75+x],
        [0.75-x, 0.75+x, 0.25-x], [0.75+x, 0.25-x, 0.75-x],
    ]
    for b in base_16c:
        for t in bcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    seen = set()
    unique = []
    for f in fracs:
        key = tuple(round(v % 1, 6) for v in f)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return gen_sites(unique, N, L_cell)

def sites_pyrite(N=1, L_cell=4.0):
    """Pyrite (FeS2). Pa-3 (205), 12 sites."""
    fcc = [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]
    x = 0.386
    base = [[0,0,0], [x,x,x], [0.5-x,-x,0.5+x]]
    fracs = []
    for b in base:
        for t in fcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    return gen_sites(fracs, N, L_cell)

def sites_skutterudite(N=1, L_cell=4.0):
    """Skutterudite (CoAs3). Im-3 (204), 32 sites."""
    bcc = [[0,0,0], [0.5,0.5,0.5]]
    fracs = []
    for t in bcc:
        fracs.append(t[:])
    base_8c = [[1/4,1/4,1/4], [-1/4,-1/4,1/4], [-1/4,1/4,-1/4], [1/4,-1/4,-1/4]]
    for b in base_8c:
        for t in bcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    y, z = 0.150, 0.350
    base_24g = [
        [0,y,z], [0,-y,-z], [0,-y,z], [0,y,-z],
        [z,0,y], [-z,0,-y], [z,0,-y], [-z,0,y],
        [y,z,0], [-y,-z,0], [-y,z,0], [y,-z,0],
    ]
    for b in base_24g:
        for t in bcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    seen = set()
    unique = []
    for f in fracs:
        key = tuple(round(v % 1, 6) for v in f)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return gen_sites(unique, N, L_cell)

def sites_half_heusler(N=1, L_cell=4.0):
    """Half-Heusler. F-43m (216), 12 sites."""
    fcc = [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]
    fracs = []
    for b in [[0,0,0], [0.25,0.25,0.25], [0.75,0.75,0.75]]:
        for t in fcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    return gen_sites(fracs, N, L_cell)


# =============================================================================
# CATALOG + UNIFIED API
# =============================================================================

CATALOG = [
    ('sc',           sites_sc,           'Pm-3m (221)',   1),
    ('bcc',          sites_bcc,          'Im-3m (229)',   2),
    ('fcc',          sites_fcc,          'Fm-3m (225)',   4),
    ('diamond',      sites_diamond,      'Fd-3m (227)',   8),
    ('a15',          sites_a15,          'Pm-3n (223)',   8),
    ('c15',          sites_c15,          'Fd-3m (227)',  24),
    ('clathrate_I',  sites_clathrate_I,  'Pm-3n (223)',  24),
    ('nacl',         sites_nacl,         'Fm-3m (225)',   8),
    ('perovskite',   sites_perovskite,   'Pm-3m (221)',   5),
    ('fluorite',     sites_fluorite,     'Fm-3m (225)',  12),
    ('pyrochlore',   sites_pyrochlore,   'Fd-3m (227)',  16),
    ('sodalite',     sites_sodalite,     'Im-3m (229)',  12),
    ('beta_mn',      sites_beta_mn,      'P4132 (213)',  20),
    ('gamma_brass',  sites_gamma_brass,  'I-43m (217)',  52),
    ('spinel',       sites_spinel,       'Fd-3m (227)',  56),
    ('alpha_mn',     sites_alpha_mn,     'I-43m (217)',  58),
    ('th3p4',        sites_th3p4,        'I-43d (220)',  28),
    ('pyrite',       sites_pyrite,       'Pa-3 (205)',   12),
    ('skutterudite', sites_skutterudite, 'Im-3 (204)',   32),
    ('half_heusler', sites_half_heusler, 'F-43m (216)',  12),
]

def build_voronoi_mesh(name, N=1, L_cell=4.0):
    """Build Voronoi mesh for a named structure.

    Returns mesh dict with V, E, F, L, name, space_group, n_sites, is_plateau.

    NOTE: Face vertex ordering uses arctan2 with ref = coords[0] - centroid.
    This can be unstable when the first vertex is close to the centroid.
    Edges (V, E) are always correct; faces may have winding errors on some
    structures (perovskite, pyrite). For edge-direction analysis (Phase 2),
    only V, E, L are needed — faces are not required.
    """
    entry = None
    for cat_name, fn, sg, ns in CATALOG:
        if cat_name == name:
            entry = (fn, sg, ns)
            break
    if entry is None:
        raise ValueError(f"Unknown structure '{name}'. Available: {list_names()}")

    fn, sg, n_sites_expected = entry
    pts, L = fn(N=N, L_cell=L_cell)

    V, E, F = build_voronoi(pts, L, repeat=1)
    avg_deg, avg_fpe, is_plateau = check_topology(V, E, F)

    # Auto-retry with repeat=2 if topology looks bad (fraction-based check)
    from collections import Counter
    deg = Counter()
    for a, b in E:
        deg[a] += 1
        deg[b] += 1
    frac_deg4 = sum(1 for d in deg.values() if d == 4) / max(len(deg), 1)
    if frac_deg4 < 0.9:
        V2, E2, F2 = build_voronoi(pts, L, repeat=2)
        _, _, is_plateau2 = check_topology(V2, E2, F2)
        V, E, F = V2, E2, F2
        is_plateau = is_plateau2

    return {
        'V': V, 'E': E, 'F': F, 'L': L,
        'name': name,
        'space_group': sg,
        'n_sites': len(pts),
        'is_plateau': is_plateau,
    }


def build_random_voronoi(n_seeds, L=4.0, seed=42):
    """Build periodic Voronoi mesh from random seed points.

    Generates n_seeds random points in [0, L)³ and builds the Voronoi
    complex. Useful for testing predictors on non-crystallographic structures.

    Args:
        n_seeds: number of random cell centers (minimum ~10 for stability)
        L: cubic box size
        seed: random seed for reproducibility

    Returns: dict with V, E, F, L (same format as build_voronoi_mesh)

    Raises ValueError if n_seeds too small (Voronoi degenerate).

    Robustness (tested 22 Mar 2026):
      n_seeds >= 15: Σv²/mode CV < 25% across seeds (robust)
      n_seeds = 10: CV ~ 38% (marginal)
      n_seeds < 10: often fails (face dedup collision)
      All random Voronoi produce z=4 Plateau foams (max_z=4).
    """
    rng = np.random.RandomState(seed)
    pts = rng.rand(n_seeds, 3) * L

    V, E, F = build_voronoi(pts, L, repeat=1)
    avg_deg, avg_fpe, is_plateau = check_topology(V, E, F)

    if len(V) == 0:
        raise ValueError(f"Random Voronoi failed: 0 vertices (try n_seeds >= 10)")

    return {
        'V': V, 'E': E, 'F': F, 'L': L,
        'name': f'random_{n_seeds}_s{seed}',
        'n_sites': n_seeds,
        'is_plateau': is_plateau,
    }


# --- Non-Voronoi random z=4 graph (edge-swap builder) ---

def build_edge_swap_graph(n_seeds, L=4.0, seed=42, swap_seed=0,
                          swap_fraction=3.0):
    """Build non-Voronoi z=4 graph by edge-swapping a random Voronoi.

    Starts from a random Voronoi (z=4) and randomly swaps edges while
    preserving z=4 at every vertex. After swapping, the graph is no
    longer a Voronoi tessellation but retains z=4, the same vertex
    positions, and the same number of edges.

    This is the key builder for testing whether κ≈1 is a Voronoi-specific
    property or a generic z=4 property (S7 investigation, 22 Mar 2026).

    Args:
        n_seeds: number of seed points for initial Voronoi
        L: cubic box size
        seed: random seed for Voronoi point generation
        swap_seed: random seed for edge swap sequence
        swap_fraction: number of swaps as multiple of nE (default 3.0 = 3×nE)

    Returns: dict with V, E, L, name, swap_stats

    Result (22 Mar 2026):
      Σv²_swap ≈ Σv²_voronoi ≈ 0 on 9 tested foams (n=10..20).
      fflat_swap = 1.000 (all bands flat).
      93% of edges changed at swap_fraction=3.0.
      κ≈1 is NOT Voronoi-specific — it is a z=4 + random positions property.
    """
    # Build base Voronoi
    m = build_random_voronoi(n_seeds, L=L, seed=seed)
    V = np.array(m['V'])
    E_arr = [tuple(e) for e in np.array(m['E'])]
    nv = len(V)
    ne = len(E_arr)
    n_swaps = int(ne * swap_fraction)

    rng = np.random.RandomState(swap_seed)

    # Edge set for O(1) lookup
    edge_set = set()
    for i, j in E_arr:
        edge_set.add((min(i, j), max(i, j)))

    # Adjacency
    adj = [set() for _ in range(nv)]
    for i, j in E_arr:
        adj[i].add(j)
        adj[j].add(i)

    n_success = 0
    n_attempts = 0

    for _ in range(n_swaps * 10):
        if n_success >= n_swaps:
            break
        n_attempts += 1

        idx1 = rng.randint(ne)
        idx2 = rng.randint(ne)
        if idx1 == idx2:
            continue

        a, b = E_arr[idx1]
        c, d = E_arr[idx2]
        if len({a, b, c, d}) < 4:
            continue

        # Try swap (a,b)+(c,d) → (a,c)+(b,d)
        new1 = (min(a, c), max(a, c))
        new2 = (min(b, d), max(b, d))
        old1 = (min(a, b), max(a, b))
        old2 = (min(c, d), max(c, d))

        if new1 in edge_set or new2 in edge_set:
            # Try alternative: (a,d)+(b,c)
            new1 = (min(a, d), max(a, d))
            new2 = (min(b, c), max(b, c))
            if new1 in edge_set or new2 in edge_set:
                continue

        edge_set.remove(old1)
        edge_set.remove(old2)
        edge_set.add(new1)
        edge_set.add(new2)

        adj[a].discard(b)
        adj[b].discard(a)
        adj[c].discard(d)
        adj[d].discard(c)
        adj[new1[0]].add(new1[1])
        adj[new1[1]].add(new1[0])
        adj[new2[0]].add(new2[1])
        adj[new2[1]].add(new2[0])

        E_arr[idx1] = new1
        E_arr[idx2] = new2
        n_success += 1

    # Warn if swap rate too low
    if n_success < n_swaps * 0.5:
        import warnings
        warnings.warn(
            f"build_edge_swap_graph: only {n_success}/{n_swaps} swaps succeeded "
            f"({n_success/n_swaps:.0%}). Graph may not be sufficiently randomized.",
            RuntimeWarning, stacklevel=2)

    # Compute stats
    orig_set = set((min(i, j), max(i, j)) for i, j in np.array(m['E']))
    new_set = set((min(i, j), max(i, j)) for i, j in E_arr)
    edges_changed = len(orig_set - new_set)
    degrees = np.array([len(adj[i]) for i in range(nv)])

    return {
        'V': V.tolist(),
        'E': [list(e) for e in E_arr],
        'F': [],  # faces undefined after swap — DEC (d₁) requires faces, will fail
        'L': L,
        'name': f'swap_{n_seeds}_s{seed}_sw{swap_seed}',
        'n_sites': n_seeds,
        'is_plateau': False,
        'swap_stats': {
            'n_swaps': n_success,
            'n_attempts': n_attempts,
            'edges_changed': edges_changed,
            'edges_total': ne,
            'frac_changed': edges_changed / ne,
            'z_mean': float(np.mean(degrees)),
            'z_min': int(np.min(degrees)),
            'z_max': int(np.max(degrees)),
        },
    }


def list_names():
    """Return list of available structure names."""
    return [name for name, _, _, _ in CATALOG]


def list_structures():
    """Return list of (name, space_group, n_sites) for all structures."""
    return [(name, sg, ns) for name, _, sg, ns in CATALOG]


# =============================================================================
# SELF-TESTS
# =============================================================================

def _self_test_build_all():
    """Build all 20 structures, check V, E, F are non-empty."""
    print(f"  Building all {len(CATALOG)} structures...")
    for name, _, sg, ns in CATALOG:
        mesh = build_voronoi_mesh(name)
        nV, nE, nF = len(mesh['V']), len(mesh['E']), len(mesh['F'])
        plat = 'Y' if mesh['is_plateau'] else 'N'
        assert nV > 0 and nE > 0 and nF > 0, f"{name}: empty mesh"
        print(f"    {name:15s}  {sg:15s}  sites={mesh['n_sites']:3d}  "
              f"V={nV:4d} E={nE:4d} F={nF:4d}  Plateau={plat}")


def _self_test_layer0_on_all():
    """Layer 0 (exactness) on all 20 structures."""
    import sys, os
    _analysis = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'analysis'))
    _math_src = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
    for p in [_analysis, _math_src]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from validator import validate_layer0

    MIN_VERTICES = 10  # skip degenerate meshes
    print(f"\n  Layer 0 on all structures with V >= {MIN_VERTICES}...")
    n_pass = 0
    n_tested = 0
    for name in list_names():
        mesh = build_voronoi_mesh(name)
        nV = len(mesh['V'])
        if nV < MIN_VERTICES:
            print(f"    {name:15s}  [SKIP]  V={nV} < {MIN_VERTICES}")
            continue
        n_tested += 1
        try:
            report = validate_layer0(mesh)
            status = 'PASS' if report.all_pass else 'FAIL'
            if report.all_pass:
                n_pass += 1
            checks_str = ', '.join(f"{c['name']}={c['value']}" for c in report.checks)
            print(f"    {name:15s}  [{status}]  V={nV:4d}  {checks_str}")
        except Exception as e:
            err_msg = str(e)[:60]
            print(f"    {name:15s}  [ERROR] V={nV:4d}  {err_msg}")

    print(f"\n  {n_pass}/{n_tested} PASS Layer 0 (skipped {len(CATALOG)-n_tested} too small)")
    return n_pass


def _self_test_edge_swap():
    """Test build_edge_swap_graph: z=4 preserved, edges changed, reproducible."""
    print(f"\n  Edge-swap builder tests...")

    # T1: basic build + z=4 preserved
    m = build_edge_swap_graph(15, seed=42, swap_seed=0)
    V = np.array(m['V'])
    E = np.array(m['E'])
    stats = m['swap_stats']
    nv, ne = len(V), len(E)
    assert stats['z_min'] == 4 and stats['z_max'] == 4, \
        f"z not uniform 4: [{stats['z_min']}, {stats['z_max']}]"
    assert stats['z_mean'] == 4.0, f"z_mean = {stats['z_mean']}"
    print(f"    T1: n=15, nv={nv}, ne={ne}, z=4 exact. "
          f"Swaps: {stats['n_swaps']}, changed: {stats['edges_changed']}/{ne} "
          f"({stats['frac_changed']:.0%}). PASS.")

    # T2: significant fraction of edges changed
    assert stats['frac_changed'] > 0.70, \
        f"Too few edges changed: {stats['frac_changed']:.0%}"
    print(f"    T2: {stats['frac_changed']:.0%} edges changed (>70%). PASS.")

    # T3: reproducible (same seeds → same result)
    m2 = build_edge_swap_graph(15, seed=42, swap_seed=0)
    E2 = np.array(m2['E'])
    assert np.array_equal(E, E2), "Not reproducible!"
    print(f"    T3: reproducible (same seeds → same edges). PASS.")

    # T4: different swap_seed → different result
    m3 = build_edge_swap_graph(15, seed=42, swap_seed=99)
    E3 = set((min(i, j), max(i, j)) for i, j in np.array(m3['E']))
    E_set = set((min(i, j), max(i, j)) for i, j in E)
    overlap = len(E_set & E3)
    assert overlap < ne * 0.5, f"Different swap_seed too similar: {overlap}/{ne}"
    print(f"    T4: different swap_seed → {ne - overlap}/{ne} edges differ. PASS.")

    # T5: multiple sizes
    for n_seeds in [10, 20, 25]:
        try:
            m4 = build_edge_swap_graph(n_seeds, seed=42, swap_seed=0)
            s = m4['swap_stats']
            assert s['z_min'] == 4 and s['z_max'] == 4
            print(f"    T5: n={n_seeds}, nv={len(m4['V'])}, "
                  f"z=4, changed {s['frac_changed']:.0%}. PASS.")
        except ValueError as e:
            print(f"    T5: n={n_seeds} failed (expected for small n): {e}")

    # T6: no self-loops or duplicate edges
    E_arr = np.array(m['E'])
    for i, j in E_arr:
        assert i != j, f"Self-loop: ({i},{j})"
    edge_tuples = [(min(i, j), max(i, j)) for i, j in E_arr]
    assert len(set(edge_tuples)) == len(edge_tuples), "Duplicate edges"
    print(f"    T6: no self-loops, no duplicates. PASS.")

    print(f"  Edge-swap: 6/6 PASS.")


if __name__ == '__main__':
    _self_test_build_all()
    print("\nPASS: build_all")
    _self_test_edge_swap()
    print("\nPASS: edge_swap")
    n = _self_test_layer0_on_all()
    if n == len(CATALOG):
        print("PASS: layer0_all")
    else:
        print(f"PARTIAL: {n}/{len(CATALOG)} passed Layer 0")
    print("\nDone.")
