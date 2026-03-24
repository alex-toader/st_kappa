# Directional coherence controls elastic wave transport on discrete structures

Alexandru Toader
Independent researcher, Buzău, Romania
toader_alexandru@yahoo.com

## Abstract

We identify directional coherence — the spatial arrangement of edge directions
in the bonding network — as the mechanism controlling elastic wave transport
on discrete structures. On structures with few, periodically assigned directions,
the dynamical matrix varies strongly with wavevector, producing dispersive
bands and long-range propagation. On structures with many, incoherently
assigned directions, phase-dependent anisotropic contributions cancel
statistically, bands flatten, and propagation is suppressed. Yet a local
defect still redistributes energy over a finite, system-size-independent
distance. This establishes a distinction between propagation (ballistic,
$v_g \neq 0$) and transport (finite spatial support of the scattered
response). The transition between regimes is continuous, not sharp. Crystals
with large unit cells approach the behavior of random structures despite
perfect periodicity. A Fibonacci quasicrystal with coherent directions
retains 44–64% of optimal crystal transport, demonstrating that periodicity
is not required. The mechanism is validated across 104 structures (19 crystallographic and 85
random) spanning five coordination numbers.

## §1. Introduction

Phonon transport on periodic lattices is well understood through Bloch theory
and Boltzmann transport [3,4]. On non-periodic discrete structures — amorphous
solids, foams, biological networks — periodicity is absent and the standard
framework does not apply. What controls wave propagation and transport on discrete structures without
translational symmetry?

We show that the controlling variable is *directional coherence*: the number
of distinct edge directions in the bonding network and whether they are
spatially arranged in a coherent (periodic) or incoherent (random) pattern.
Directional coherence is not a single scalar quantity but reflects the joint
effect of directional diversity and spatial correlations in their assignment.
As an operational proxy, we define the *coherence ratio* as
$\Sigma v^2_{\text{shuffled}} / \Sigma v^2_{\text{original}}$: the fraction
of transport retained when edge directions are randomly reassigned across the
same graph (keeping the direction set fixed). Coherent structures have coherence ratio $\ll 1$ (transport depends on
assignment pattern); incoherent structures have coherence ratio $\approx 1$
(no pattern to destroy). For example, a BCC crystal with 6 directions
assigned periodically has $\Sigma v^2 = 0.017$; shuffling the same directions
randomly drops it to $0.00015$ — a coherence ratio of $0.0084$.

Our central claim, stated in falsifiable form: *elastic wave transport on
discrete structures is controlled by the spatial coherence of edge-direction
assignments, which governs the k-dependence of the dynamical matrix.* If a
structure's coherence ratio is $\ll 1$, its transport is assignment-dependent
and suppressible by shuffling.
Structures with few, periodically assigned directions retain dispersive bands
and long-range propagation. Structures with many, incoherently assigned
directions — whether structurally disordered or crystallographically complex
— flatten their bands and suppress propagation entirely.

Yet a restricted notion of transport persists without propagation. A local
defect on an incoherent structure redistributes energy over a finite distance
$\langle r \rangle \approx 1$–$2$ lattice spacings, independent of system
size. This establishes a distinction between *propagation* ($v_g \neq 0$,
energy moves ballistically) and *transport* (finite spatial support of the
defect-induced response). We use the latter term strictly to denote the spatial
extent of the scattered response, not a transport coefficient in the
Boltzmann or diffusive sense.

This mechanism differs from Anderson localization [1], where disorder in the
potential on a regular lattice localizes eigenmodes spatially. Here,
eigenmodes remain extended (IPR $\sim 1/N$); the suppression is spectral
(group velocities vanish), not spatial. Periodicity is not required: a
Fibonacci quasicrystal with coherent directions retains 44–64% of optimal
crystal transport despite non-periodic spacing.

Our results predict that crystalline materials with large unit cells suppress
elastic wave propagation in the same way as amorphous materials. Complexity
of the unit cell, not structural disorder, is the relevant parameter for
phonon engineering.

We demonstrate these results on 104 structures (19 crystallographic + 85
random) spanning 5 coordination numbers, with 56 validated tests covering
mechanism, prediction, and robustness [see tests_map.md in supplementary
code].

## §2. Model

We study elastic wave transport on discrete structures modeled as spring
networks. Each vertex carries 3 degrees of freedom (displacement). Edges
connect nearest neighbors with tensorial springs $K_e = k_L (\hat{e} \otimes
\hat{e})$, where $\hat{e}$ is the unit vector along the edge and $k_L$ is
the longitudinal spring constant ($k_T = 0$). The dynamical matrix is

$$H(k) = \sum_{\text{edges}} (\hat{e} \otimes \hat{e}) \, e^{ik \cdot \delta}$$

where $\delta$ is the displacement vector between connected vertices under
periodic boundary conditions. We work in dimensionless units where vertex
mass, spring constant $k_L$, and lattice spacing are set to 1. The box size
$L$ is measured in these units.

**Defect.** Scattering is induced by a Peierls defect: an SO(2) rotation of
strength $\alpha = 0.5$ applied to edges crossing a disk of radius
$R_{\text{disk}} = 2.0$. Ablation tests confirm that the SO(2) gauge
structure is essential ($\alpha = 0$ produces zero scattering; T2.1); NNN
springs and vectorial DOF beyond 1 component are not needed [W1 files 22–23].

**FDTD.** We use a Verlet integrator ($dt = 0.02$) with periodic boundary
conditions and time-windowed measurement. A Gaussian wave packet is launched,
and the scattered field is computed as $u_{\text{sc}} = u_{\text{defect}} -
u_{\text{ref}}$, the difference between displacement fields with and without
the Peierls defect, both evolved from the same initial wavepacket.
The observable is

$$\langle r \rangle = \frac{\sum_i d_i |u_{\text{sc},i}|^2}{\sum_i |u_{\text{sc},i}|^2}$$

the energy-weighted mean distance of the scattered field from the defect
center. This quantity measures the spatial support of the defect-induced
response.

**Structures.** We use 19 crystallographic structures from a validated catalog
(structure_catalog.py, 27 self-tests), random Voronoi foams at $z = 3$–$8$,
edge-swapped non-Voronoi graphs, and a Fibonacci 3D quasicrystal [Kohmoto
et al., 1983]. Band structure quantities ($\Sigma v^2$, $f_{\text{flat}}$)
are measured along $k = [1,0,0]$; ranking is preserved across 5 random
$k$-directions (T2.7) and 3 vs 10 $k$-values differ by 1.6% (T2.12).

**Validation.** Energy oscillation in the Verlet integrator is below $10^{-4}$
(T2.4). Results are insensitive to defect geometry ($R_{\text{disk}} = 2.0$ is
within 4.5% of the plateau at $R = 2.5$; T2.9), random seeds (means within
$2\sigma$ across independent seed sets; T2.6), and initial wave phase
(CV = 5.6%; T5.12). The infrastructure is validated on complex crystals
(beta-Mn $n_v = 130$, Th$_3$P$_4$ $n_v = 188$; T2.11) and verified to
produce significant scattering ($E_{\text{sc}}/E_{\text{inc}} = 0.21$–$1.38$;
T2.8).

All simulation code is publicly available at
https://github.com/alex-toader/st_kappa.

## §3. Flat bands from directional incoherence

This section establishes the mechanism underlying all subsequent results.

Directional incoherence suppresses transport through a statistical
cancellation of phase-dependent anisotropic contributions in the dynamical
matrix $H(k)$ defined in §2. The eigenvalues of $H(k)$ determine the phonon
dispersion; their variation with $k$ determines group velocities and
transport.

On structures with few distinct edge directions (3–10), periodically assigned,
the sum retains strong $k$-dependent anisotropy: specific Bloch phases
reinforce constructively along symmetry directions, producing dispersive bands
with $\Sigma v^2 \gg 0$. On structures with many diverse directions (50+),
incoherently assigned, the phase-dependent anisotropic contributions cancel
statistically. The $k$-dependent anisotropic components of $H(k)$ become small
relative to its trace, eigenvalues become nearly degenerate, bands flatten
($f_{\text{flat}} \approx 1$), and group velocities vanish
($\Sigma v^2 \approx 0$).

We verify this on 42 random structures spanning five coordination numbers
($z = 3, 4, 5, 6, 8$) and edge-swapped non-Voronoi graphs. All give
$f_{\text{flat}} > 0.99$ (T3.2). At the same coordination, crystals have
$63\times$ (at $z=4$) to $1097\times$ (at $z \approx 6$) larger $\Sigma v^2$
than random structures (T3.3).

**Directions, not positions, control transport.** We decouple the two
contributions using a controlled test on 9 crystallographic structures.
Randomizing edge directions on a crystal graph reduces $\Sigma v^2$ by
$19\times$ to $338\times$ (B/A test, T3.4). Randomizing vertex positions
while keeping crystal directions changes $\Sigma v^2$ by less than a factor
of 2 (C/A test, T3.5). The reverse test confirms that both directions and
graph topology are necessary: crystal directions assigned to a random graph
produce $\Sigma v^2 = 0.16\%$ of crystal (T3.6).

The most direct demonstration is the shuffle test (T3.16): we take a crystal
(Kelvin/BCC), keep its exact set of 6 directions, but randomly reassign which
edge receives which direction. The result: $\Sigma v^2$ drops to $0.84\%$ of crystal — a coherence ratio of
$0.0084$, indistinguishable from fully random directions. The direction set is
not the variable; the spatial pattern of assignment (periodic vs random) is.

The decoherence follows a Gaussian law: $\Sigma v^2(\theta) = \Sigma v^2_0
\exp(-\theta^2/\sigma^2)$ where $\theta$ is the standard deviation of
directional noise applied to crystal edges. The fit gives $R^2 > 0.93$ on
three crystals (BCC $\sigma = 6.5°$, diamond $8.8°$, FCC $8.9°$; T3.7).
A few degrees of directional noise collapse the band structure. This is
phenomenologically equivalent to dephasing: directional noise progressively
washes out the coherent anisotropic contributions that sustain dispersion.

The suppression is sharp: $5\%$ edge replacement reduces
$\Sigma v^2$ by $8.9\times$, and even restoring $90\%$ of crystal edges
leaves $\Sigma v^2$ at $31\times$ below the unperturbed crystal (T3.8).
This is sharper than Anderson localization, where a percolation-like threshold
exists near $50\%$ disorder.

Despite suppressed transport, eigenmodes remain spatially extended:
$\text{IPR} \times n_v \approx 3$ (consistent with 3 DOF per site) at all
$k$-values tested (T3.9, T3.14). This is not Anderson localization [1]:
eigenmodes remain extended, and the suppression arises from spectral
flattening (group velocities vanish) rather than spatial confinement of
modes.

Finally, the tensor average $\langle \hat{e} \otimes \hat{e} \rangle$ does
not by itself distinguish the two regimes: Kelvin (BCC) has $\langle \hat{e}
\otimes \hat{e} \rangle = I/3$ exactly, yet its $\Sigma v^2$ is $62\times$
larger than random (T3.12). The controlling variable is not the tensor average
but the number of distinct directions and their spatial coherence — how they
are distributed across the structure. The flat-band result does not depend on
how directions are assigned algorithmically: on random vertex positions,
geometric directions (from $V_j - V_i$) and prescribed random directions both
give $\Sigma v^2 \ll$ crystal (T3.10).

## §4. Transport on coherent structures

Having established that directional incoherence flattens bands, we now examine
the opposite limit: crystals with few coherent directions, where bands are
dispersive and $\Sigma v^2$ predicts $\langle r \rangle$.

**Predictive relation.** Across 15 crystallographic structures, the Spearman
correlation between $\Sigma v^2$ and $\langle r \rangle$ is $\rho = 0.800$
($p = 0.0003$; T4.1). This relation holds across structure classes but
weakens within the low-transport group (§4 Limitations). The connection is derived from the eigenmode expansion
of the Green's function: $\langle r \rangle \propto \langle v^2 \rangle /
\langle v \rangle$, verified at $\rho = 0.943$ ($p = 0.005$) on 6 crystals
(T4.2).

**Classification.** The 15 measurable crystals split into two groups separated
by a natural gap of 1.205 in $\langle r \rangle$ (between fluorite at 1.89
and perovskite at 3.09; T4.10):

- HIGH transport ($\langle r \rangle > 3$): FCC, diamond, pyrochlore,
  perovskite — few unique directions (4–10), $z > 5$ (T4.3).
- LOW transport ($\langle r \rangle < 2$): C15, beta-Mn, gamma-brass,
  alpha-Mn, skutterudite, Th$_3$P$_4$, spinel — many directions (34–286)
  and/or large unit cells (T4.4).

Pyrite ($\langle r \rangle = 1.51$ at $N = 1$, 2.30 at $N = 2$; 20 dirs)
and fluorite ($\langle r \rangle = 1.89$; 9 dirs) are borderline. Fluorite
has few directions like HIGH crystals but lower $\langle r \rangle$; its
$\Sigma v^2 = 0.033$ overlaps with pyrochlore ($0.021$). $\Sigma v^2$ alone
does not fully predict borderline behavior (T4.8).

**Limitations.** Within the LOW group, the intra-group correlation
$\rho(\Sigma v^2, \langle r \rangle) = 0.455$ ($p = 0.19$) is positive but
not significant at $n = 10$ (T4.7). The predictor works across groups but is
weak within the LOW group.

Crystal $\langle r \rangle$ at $N = 1$ is a lower bound: $N = 2$ gives
$+17\%$ (A15), $+15\%$ (C15), $+52\%$ (pyrite; T4.5). Structures with
shorter edges (e.g., Th$_3$P$_4$, $\ell = 0.55$) have smaller absolute
$\langle r \rangle$ but similar $\langle r \rangle / \ell \approx 1.8$,
comparable to random ($\approx 2.0$) — a unit effect, not different physics
(T4.6).

Edge lengths are irrelevant when directions are fixed: with original crystal
directions prescribed explicitly and vertex positions perturbed (changing
edge lengths), $\Sigma v^2$ remains identical to the unperturbed crystal
(ratio 1.00; T4.11).
Metric geometry does not control transport — only the spatial pattern of
directions matters.

SC cubic and NaCl are excluded from $\langle r \rangle$ measurements (disk
geometry incompatible with their edge structure). Both are extreme TYPE I
with $\Sigma v^2 = 0.66$–$2.6$.

## §5. Transport without propagation

In incoherent structures, wave propagation is suppressed ($v_g \approx 0$).
Yet a local defect still redistributes energy over a finite distance — in the
restricted sense of §1.

**⟨r⟩ is independent of system size.** We measure $\langle r \rangle$ on 100
random Voronoi foams ($z=4$, 10 sizes $\times$ 10 seeds, $n_v = 40$–238). A
power law fit gives $\langle r \rangle \sim n_v^{-0.09}$ — effectively
constant (T5.2). This rules out a mean-free-path interpretation, where
$\langle r \rangle$ would scale with domain size. No thermodynamic limit is
needed.

**Absolute values.** On open boundary (30 foams, $n_v$ up to 7408),
$\langle r \rangle_{\text{open}} = 1.52 \pm 0.05$, constant across system
sizes [W5 file 09]. On PBC, $\langle r \rangle \approx 1.4$. FDTD
measurements on random $z=6$ ($\langle r \rangle = 1.50$) and edge-swapped
non-Voronoi ($\langle r \rangle = 1.74$) confirm that the result is not
$z=4$-specific; the $z=6/z=4$ ratio is 1.04 (T5.5–5.7).

**Independence tests.** $\langle r \rangle$ is independent of:
- Peierls strength $\alpha$: CV = 2.2% across $\alpha = 0.05$–$0.50$ (T5.3).
  $Z_2$ topology plays no special role on random structures.
- Band structure ($\Sigma v^2$): at fixed $z$, intra-$z$ Spearman
  correlations are all nonsignificant ($p > 0.05$, tested on 5 $z$-values
  $\times$ 15 seeds each; T5.4). The global $\rho = 0.37$ is a $z$-confound
  artefact.
- Initial wave phase: CV = 5.6% across $k = 0.2$–$1.2$ (T5.12).

**⟨r⟩ scales with defect size, then saturates.** $\langle r \rangle$ grows
sublinearly with $R_{\text{disk}}$ from 1.16 ($R = 1.0$) to 1.44 ($R = 2.5$),
then plateaus — the last step gives 0% change (T5.9). The observable is
geometric (set by defect coupling range), not spectral.

**Mechanism: local perturbation stays local.** On random structures, the
cosine similarity between reference and defect fields exceeds 0.90 at
$r > 3$ lattice spacings (T5.8) — the fields are nearly identical far from
the defect. On crystals at $L = 16$, the same quantity drops to 0.45 at
$r = 3$, indicating propagation [W5 file 11]. The scattered energy profile
fits a Gaussian better than a power law (T5.11), ruling out long-range tails.

$\langle r \rangle$ should not be interpreted as a mean free path or diffusion
length. It is a property of the Green's function response to a local
perturbation — the spatial support of the scattered field — and does not
arise from asymptotic dynamics. This behavior resembles evanescent response in continuous media but arises
here without an underlying band gap, purely from directional incoherence. It
is distinct from the diffuson/locon classification of Allen and Feldman [5,6]
in that our modes are fully extended (IPR $\sim 1/N$) yet non-propagating.

## §6. The coherence continuum

The transition from coherent to incoherent transport is continuous, not a
phase transition.

$\Sigma v^2$ spans 4.3 orders of magnitude across our structures — from
$\sim 0.0002$ (random) to $\sim 2.6$ (SC cubic at $N = 3$; T6.1). No binary
separator cleanly divides crystal from random: the spectral measure
$\Delta_{\text{spectral}}$ shows overlap at $0.5\times$ between the lowest
crystal and highest random (T6.2). Crystals with many atoms per unit cell
— beta-Mn (20 sites), gamma-brass (52), alpha-Mn (58) — approach the
transport regime of random structures ($\langle r \rangle \approx 1$–$1.5$)
despite perfect periodicity (T6.3). Beta-Mn at $\langle r \rangle = 1.38$ is
within $\pm 2\sigma$ of the random distribution ($z$-score $= -0.46$; T6.8).

This is not an artefact of coordination or topology — it is directional
diversity. Crystal $n_{\text{dirs}}$ ranges from 4 (FCC) to 286 (alpha-Mn);
random $z=4$ structures have $n_{\text{dirs}} \approx 185$–$196$. Direction
diversity varies continuously across structures, not in discrete categories
(T6.7).

On controlled abstract graphs with exactly $n_{\text{dirs}}$ directions
assigned cyclically, $\Sigma v^2$ decreases from $n_{\text{dirs}} = 2$ to 3,
then saturates once $n_{\text{dirs}}$ exceeds the embedding dimension (T6.11).
All values are $\ll$ crystal — random assignment suppresses transport regardless of
how many directions are available. The spatial pattern of assignment, not the
direction count, determines transport.

Directional coherence does not reduce to a single scalar order parameter but
emerges from the interplay of directional diversity and spatial correlations
in their assignment. The coherence ratio (defined in §1) is $0.0084$ on
Kelvin, confirming that the periodic assignment pattern — not the direction
set — controls transport (T3.16). Partial correlation analysis confirms
that $n_{\text{dirs}}$ is a proxy for $\Sigma v^2$, not an independent
predictor ($\rho_{\text{partial}} = -0.31$, $p = 0.27$; T6.6).

**Test of the theory: quasicrystals.** A standard Fibonacci 3D quasicrystal
(Kohmoto et al., 1983) has the same 3 edge directions as SC cubic but
non-periodic spacing ($L/S = \varphi$). It retains 64% of SC transport at
$n = 5$ and 44% at $n = 6$ — far above random ($\sim 0.2\%$ of SC; T6.4).
Periodicity is not required for transport; directional coherence is.
Extension to other quasicrystal types (icosahedral, Ammann-Beenker) is left
for future work.

## §7. Discussion

**Relation to Anderson localization.** Our mechanism differs fundamentally
from Anderson localization. In Anderson's picture, disorder in the potential
on a regular lattice produces spatially localized eigenmodes. Here, disorder
is in the graph topology and edge directions, not the potential. Eigenmodes
remain spatially extended (IPR $\sim 1/n_v$ at all $k$; T3.9, T3.14). The
suppression is spectral — group velocities vanish — not spatial.

**Complex crystals as incoherent structures.** Any crystalline material with
a sufficiently large unit cell (many atoms, many distinct edge directions)
will suppress elastic wave propagation in the same way as an amorphous
material. This is not disorder-driven — it is complexity-driven. The relevant
parameter for phonon engineering is directional diversity of the bonding
network, not structural order or periodicity.

**Design principle.** Our results suggest that phonon transport can be
controlled by tuning directional coherence: few, periodically arranged
directions enable propagation; many, incoherently arranged directions suppress
it. This provides a structural design principle independent of chemical
composition.

**Scope.** All results are for longitudinal springs ($k_T = 0$) in three
dimensions. Extension to $k_T > 0$ and 2D is expected on physical grounds
(the mechanism is spectral, not geometric) but is left for future work.

**Limitations.** We do not derive the Gaussian decoherence law analytically
($\Sigma v^2(\theta) \propto \exp(-\theta^2/\sigma^2)$ is empirical with
$R^2 > 0.93$). We do not identify a single scalar order parameter for
coherence. The cosine similarity demonstration of local perturbation (T5.8)
shows weak separation on small domains ($L = 8$); clearer results require
$L \geq 16$ [W5 file 11]. Crystal $\langle r \rangle$ at $N = 1$ is a lower
bound; larger supercells would refine the classification.

**Conclusion.** Directional coherence, not periodicity, determines elastic
wave transport on discrete structures.

## References

[1] P. W. Anderson, Phys. Rev. 109, 1492 (1958).
[2] M. Kohmoto, L. P. Kadanoff, C. Tang, Phys. Rev. Lett. 50, 1870 (1983).
[3] N. W. Ashcroft and N. D. Mermin, *Solid State Physics* (Holt, 1976).
[4] J. M. Ziman, *Electrons and Phonons* (Oxford, 1960).
[5] P. B. Allen and J. L. Feldman, Phys. Rev. B 48, 12581 (1993).
[6] P. B. Allen, J. L. Feldman, J. Fabian, F. Wooten, Phil. Mag. B 79, 1715 (1999).
[7] M. Wyart, Ann. Phys. Fr. 30, 1 (2005).
[8] A. J. Liu and S. R. Nagel, Annu. Rev. Condens. Matter Phys. 1, 347 (2010).
[9] C. S. O'Hern et al., Phys. Rev. E 68, 011306 (2003).
[10] D. Shechtman et al., Phys. Rev. Lett. 53, 1951 (1984).
[11] T. C. Lubensky et al., Rep. Prog. Phys. 78, 073901 (2015).
[12] C. R. Calladine, Int. J. Solids Struct. 14, 161 (1978).
[13] D. M. Sussman, C. P. Goodrich, A. J. Liu, Soft Matter 12, 3982 (2016).
[14] K. Sun et al., Proc. Natl. Acad. Sci. 109, 12369 (2012).
[15] L. Yan, E. DeGiuli, M. Wyart, Europhys. Lett. 114, 26003 (2016).
[16] S. Gelin, H. Tanaka, A. Lemaître, Nature Mater. 15, 1177 (2016).
[17] M. Baggioli and A. Zaccone, Phys. Rev. Lett. 122, 145501 (2019).
