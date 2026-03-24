# st_kappa

Directional coherence controls elastic wave transport on discrete structures.

**Author:** Alexandru Toader (toader_alexandru@yahoo.com)

## Result

On discrete elastic structures, transport of scattered waves is controlled by
directional coherence of edges. When edge directions are spatially coherent
(few distinct directions, periodically assigned), bands are dispersive and
perturbations propagate. When directions are incoherent — whether from
structural disorder or crystallographic complexity — bands flatten and
propagation is suppressed, yet local redistribution persists (⟨r⟩ ≈ 1-2
lattice spacings, independent of system size).

The mechanism: diverse edge directions → ⟨ê⊗ê⟩ → I/d (LLN on projectors)
→ H(k) weakly k-dependent → flat bands → Σv² ≈ 0. The transition is
continuous, not sharp. Demonstrated on 104 structures (19 crystallographic
+ 85 random, 5 coordination numbers).

## Paper

Target: J. Phys.: Condensed Matter.

- `paper/` — manuscript (to be written)

## Tests

52 tests across 5 files (~3 min total). Each file corresponds to one paper
section. Every claim has at least one test with a real assert.

See `tests/tests_map.md` for the complete inventory with per-test descriptions.

```
tests/
├── test_02_model.py          (12 tests)  Infrastructure validation
├── test_03_flat_bands.py     (14 tests)  Central mechanism: flat bands from incoherence
├── test_04_coherent.py       ( 9 tests)  Crystal transport: Σv² as predictor
├── test_05_incoherent.py     ( 9 tests)  Transport without propagation
├── test_06_continuum.py      ( 8 tests)  Coherence continuum, quasicrystals
└── tests_map.md                          Complete test inventory
```

## Source code

```
src/
├── builders/
│   ├── structure_catalog.py   Unified builder for 19 crystals + random
│   ├── fibonacci_3d.py        3D Fibonacci quasicrystal (Kohmoto 1983)
│   ├── voronoi_scan.py        20 crystallographic Voronoi meshes
│   └── random_graphs.py       Random z=3-8, edge-swap, abstract builders
├── measure.py                 measure_mr, compute_delta_spectral, count_unique_dirs
```

## Running tests

```bash
# All tests (~3 min)
for f in tests/test_0*.py; do python3 "$f"; done

# Single section
python3 tests/test_03_flat_bands.py

# Single test
python3 tests/test_03_flat_bands.py theta
```

## Requirements

- Python 3.9+
- NumPy, SciPy

## License

MIT
