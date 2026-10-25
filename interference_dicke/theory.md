# Fixed-excitation interference: analytical foundations

Evidence labels: **Proven** denotes the derivations below; numerical results
are explicitly separate in RESULTS.md. Frequencies are angular frequencies;
angles are radians; simulation sets hbar=1. Qubit i is mask bit i. The input
occupies sites 0,...,l-1, i.e. |1^l 0^(n-l)> in site order.

## Conservation and currents — Proven

With Z|0>=|0>, Z|1>=-|1>, the question's sigma^+=(X-iY)/2
is |1><0| (excitation creation); sigma^- is its adjoint. Thus
[n_i,sigma_i^+]=sigma_i^+, [n_i,sigma_i^-]=-sigma_i^-.
For each exchange edge, the two changes in [N,sigma_i^+ sigma_j^-]
cancel. The diagonal Z_i and n_i n_j commute with N individually. Hence
[H(t),N]=0 even for time-dependent coefficients, and the sector dimension
is binomial(n,l).

For weight-l bitmasks x, the diagonal energy is

    E_x = sum_i h_i (1-2x_i) + sum_(i,j) chi_ij x_i x_j.

Physical graph vertices are qubits. The configuration graph vertices are
l-element subsets of those qubits. A configuration edge moves exactly one
excitation along a physical edge to an empty site. Its matrix element is
J_ij, not an independent control for each many-body edge. Therefore

    i da_x/dt = E_x a_x + sum_(y~x) J_xy a_y.
    d |a_x|²/dt = 2 sum_(y~x) J_xy Im(a_x* a_y)
                = 2 sum_(y~x) J_xy r_x r_y sin(phi_y-phi_x).

Positive J gives maximal instantaneous influx along one edge for
phi_y-phi_x=pi/2; negative J reverses the preferred phase. All edge currents
cannot generally be maximized simultaneously: phases satisfy cycle
consistency, and one physical control moves many configuration pairs.
Diagonal gates have no instantaneous probability current but change later
currents. Equal populations alone do not imply high coherent fidelity.

## Path histories — Proven; acceleration remains unproven

For constant H, (H^m)_(x,x0) is a sum of products along length-m walks,
including diagonal self-steps. Each term carries (-it)^m/m!.
For time dependence use the Dyson expansion with ordered integrals, not
powers of an average Hamiltonian. In the interaction picture of E_x(t),
each hop y->x contributes J_xy(t) exp[i integral_0^t(E_x-E_y) ds],
in addition to the -i factor. Between hops, histories accrue the diagonal
phase of their occupied configuration. Local Z gives additive site phases;
CP adds a phase only while both endpoints are occupied. Different histories
with the same endpoints can therefore interfere differently. This is an
identity, not a claim of faster mixing; the experiments measure actual
overlap and compiled cost. Quantum walks do not converge monotonically.

## Free-chain reachability — Proven

For an OPEN nearest-neighbor chain, Jordan-Wigner maps XY exchange to
quadratic number-conserving hopping and Z to chemical potentials. The
Jordan-Wigner strings cancel only for adjacent sites in that ordering.
An initial basis state is a Slater determinant of l occupied orbitals.
Time-dependent quadratic evolution rotates its orbitals by U(n) and
preserves decomposability of its exterior product. With independent local
Z controls and independently controlled adjacent exchanges on a connected
path, commutators generate u(n): adjacent real hopping plus local diagonal
differences generate imaginary hopping, then more distant matrix units.
Thus all Slaters, and only Slaters, are reachable in this ideal control
family with unrestricted time/depth. This is not a shallow-circuit claim.

For a pure fixed-l state, Slater decomposability is equivalent to the full
set of Grassmann-Plucker relations, or equivalently to an idempotent
one-particle reduced density matrix (with trace l). These give necessary
and sufficient unrestricted-chain criteria; satisfying just one Plucker
equation is sufficient only in the four-mode two-particle case.

For n=4,l=2, write p_ij in increasing orbital order. Every Slater obeys

    p12 p34 - p13 p24 + p14 p23 = 0.

Dicke has all p_ij=1/sqrt(6), so the left side is 1/6. Exact preparation is
impossible by open-chain XY+Z. More generally for 2<=l<=n-2, take the
Plucker relation with common occupied prefix S of size l-2 and four later
sites. All three products of Dicke coefficients are 1/binomial(n,l), so
the relation fails. The sectors l=0,n are trivial; l=1,n-1 are unrestricted
one-particle/one-hole Slater spaces and present no such obstruction.

### Exact D_2^4 fidelity bound

Let A be the real skew-symmetric 4x4 matrix with upper-triangle entries
1/sqrt(6). For orthonormal complex orbitals u,v, the overlap with Dicke is
the bilinear form u^T A v, up to conjugation and sign convention. Its modulus
is at most ||A||_2. The canonical real skew eigenplane achieves this bound
with real orthonormal u,v. The two repeated singular-value squares are

    (3+2 sqrt(2))/6, (3-2 sqrt(2))/6.

Therefore max_Slater F = (3+2 sqrt(2))/6 = 0.9714045207910317.
Indeed tr(A†A)=2 and pf(A)=1/6, giving the same roots independently.
tests/test_research.py checks both SVD and an explicitly constructed Slater
maximizer, and independently verifies circuit Plucker invariance. Depth
experiments provide an additional constructive numerical check. No optimizer
failure is used as an impossibility argument.

For non-chain spin couplings, Jordan-Wigner introduces occupation strings;
the same spin XY operation is not generically quadratic in that ordering.
The chain ceiling must NOT be imposed on arbitrary Willow patches. This
does not itself prove controllability or a performance advantage there.
CP also introduces a quartic fermionic term and can leave the Slater family.

## Symmetry, eigenstates, and locality — Proven

Let S^±=sum sigma_i^± and S_z=sum Z_i/2 (opposite sign to excitation
magnetization). The anticommutator S^+S^-+S^-S^+ equals
2[S²-S_z²]. In fixed weight l the fully symmetric representation has
S=n/2 and |S_z|=|n/2-l|. Also

    S^+S^-+S^-S^+ = n I + 2 sum_(i<j) XY_ij.

Uniform complete-graph exchange is the adjacency of the Johnson graph
J(n,l), regular of degree l(n-l). Dicke is its Perron eigenvector with
eigenvalue l(n-l); with negative exchange it is the unique sector ground
state when 0<l<n. BUT time evolution under an H for which Dicke is always
an eigenstate conserves its overlap. Uniform exchange plus uniform Z cannot
prepare it from a basis input: F remains 1/binomial(n,l).

On a general connected physical graph the positive configuration Laplacian

    sum_(i,j) g_ij [n_i+n_j-2n_i n_j - XY_ij], g_ij>0,

has Dicke as its unique zero-energy sector ground state: its quadratic form
is sum_edges g |a_x-a_y|² and the configuration graph is connected. Merely
evolving under this Laplacian also conserves target overlap. An adiabatic
construction would require a specified initial Hamiltonian, gap estimate,
and runtime; none is claimed here.

Reflection symmetry of a chain does not fix the contiguous initial state
unless combined with additional transformations. Restricting controls must
be checked for compatibility with both input and target; permutation
symmetry alone can freeze the target overlap, as above.

In a depth-d local two-qubit circuit from a product input, observables on
sites at graph distance >2d have disjoint backward cones and zero connected
covariance. For Dicke, Cov(n_i,n_j)=-l(n-l)/[n²(n-1)] for i!=j, which is
nonzero for 0<l<n. Hence exact preparation needs 2d at least the graph
diameter. This bound concerns two-qubit layer depth without measurements
or pre-entangled ancillas. For approximate preparation it does not directly
impose the same bound: if trace distance is delta, the covariance error is
at most 6delta, giving only 1-F >= (|Cov|/6)² for pure circuit outputs with
disjoint cones. This weak bound vanishes with n; do not claim a strong
approximate-depth lower bound from exact correlations alone.

## Hybrid constraint

A local block substituted into recursion must work coherently for every
populated child weight j, not just the optimized root weight l. Relative
phases between j branches must agree. The required split coefficients are
sqrt[binomial(a,j) binomial(n-a,l-j) / binomial(n,l)]. Independent
high-fidelity single-sector optimizations are insufficient to certify a
hybrid. Integration is deferred pending multi-sector block verification.

## Verified references (accessed 2026-10-25)

- Andreas Bärtschi and Stephan Eidenbenz (2019), [Deterministic Preparation
  of Dicke States](https://arxiv.org/abs/1904.07358), FCT 2019, pp.126–139,
  DOI 10.1007/978-3-030-25027-0_9. SCS baseline source; O(nl) logical gates,
  O(n) depth on a line. Those are asymptotic results, not our compiled counts.
- Andreas Bärtschi and Stephan Eidenbenz (2022), [Short-Depth Circuits for
  Dicke State Preparation](https://arxiv.org/abs/2207.09998), QCE 2022,
  pp.87–96, DOI 10.1109/QCE53715.2022.00027. Grid constructions are literature
  comparisons, not separately implemented here.
- Google Quantum AI, [Google devices](https://quantumai.google/cirq/google/devices).
  The page describes a fixed Willow exchange gate and hardware-dependent
  support. This experiment's installed static Willow105 record accepts CZ,
  not arbitrary FSim(theta,phi). The actual installed validator determines
  the gate set used here; no adjustable-pulse access is assumed.
