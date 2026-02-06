# Central Calculation / Derivation

## Recent Formulation (Pasted text)

### 1. Setup and definition

We consider a composite Hilbert space $H = H_S \otimes H_B$ with total Hamiltonian

$$
H_{\text{tot}} = H_Q + H_B + H_I, \quad H_I = f \otimes B,
$$

where $f$ acts on $H_S$ and $B$ on $H_B$. The reduced equilibrium state is defined by

$$
\rho_S = \text{Tr}_B \, e^{-\beta H_{\text{tot}}}.
$$

The Hamiltonian of mean force $H_{\text{MF}}$ is defined implicitly by

$$
\rho_S = \frac{e^{-\beta H_{\text{MF}}}}{Z_{\text{MF}}}, \quad Z_{\text{MF}} = \text{Tr}_S e^{-\beta H_{\text{MF}}}.
$$

No assumption of weak coupling, Markovianity, or locality is made.

### 2. Imaginary-time interaction picture and Gaussian representation

Introduce the imaginary-time interaction picture with respect to $H_0 = H_Q + H_B$. For any operator $O$,

$$
\tilde{O}(\tau) = e^{\tau H_0} O e^{-\tau H_0}, \quad \tau \in [0, \beta].
$$

The partition operator admits the exact Dyson expansion

$$
e^{-\beta H_{\text{tot}}} = e^{-\beta H_0} \, T_\tau \exp \left( -\int_0^\beta d\tau \, \tilde{H}_I(\tau) \right),
$$

where $T_\tau$ denotes imaginary-time ordering.

Assume the bath is Gaussian with respect to $B$, i.e. all cumulants beyond second order vanish. Tracing out the bath yields an exact influence functional,

$$
\rho_S = e^{-\beta H_Q} \, T_\tau \exp \left( -\frac{1}{2} \int_0^\beta d\tau \int_0^\beta d\tau' \, \tilde{f}(\tau) \, K(\tau - \tau') \, \tilde{f}(\tau') \right),
$$

where

$$
K(\tau - \tau') = \langle T_\tau \tilde{B}(\tau) \tilde{B}(\tau') \rangle_B
$$

is the exact thermal bath correlation function.

Equivalently, this object may be written as a quenched Gaussian average,

$$
\rho_S = e^{-\beta H_Q} \, \langle T_\tau \exp \left( -\int_0^\beta d\tau \, \xi(\tau) \tilde{f}(\tau) \right) \rangle_\xi,
$$

with $\langle \xi(\tau) \xi(\tau') \rangle = K(\tau - \tau')$. This reformulation is exact.

### 3. Origin of imaginary-time dynamics of $f$

Although $f$ is time-independent in the Schrödinger picture, time ordering forces the appearance of

$$
\tilde{f}(\tau) = e^{\tau H_Q} f e^{-\tau H_Q}.
$$

This is not optional: the bilocal object $T_\tau \, \tilde{f}(\tau) \tilde{f}(\tau')$ cannot be reduced to $f^2$ unless $[H_Q, f] = 0$. The noncommutativity of $H_Q$ and $f$ is the sole source of imaginary-time nonlocality on the system side.

### 4. Adjoint-action expansion

Introduce the adjoint action

$$
\text{ad}_{H_Q}(X) = [H_Q, X], \quad \text{ad}_{H_Q}^n(X) = [H_Q, \text{ad}_{H_Q}^{n-1}(X)].
$$

The imaginary-time evolved operator admits the exact series

$$
\tilde{f}(\tau) = \sum_{n=0}^\infty \frac{\tau^n}{n!} \text{ad}_{H_Q}^n(f).
$$

Substituting into the bilocal exponent yields

$$
\int_0^\beta d\tau \int_0^\beta d\tau' \tilde{f}(\tau) K(\tau - \tau') \tilde{f}(\tau') = \sum_{n,m} \mu_{nm} \, \text{ad}_{H_Q}^n(f) \text{ad}_{H_Q}^m(f),
$$

with kernel moments

$$
\mu_{nm} = \frac{1}{n! m!} \int_0^\beta d\tau \int_0^\beta d\tau' \, \tau^n (\tau')^m K(\tau - \tau').
$$

This representation is exact and purely algebraic.

### 5. Closure criterion

Define the operator set

$$
A_f = \text{span}\{ \text{ad}_{H_Q}^n(f) \}_{n=0}^\infty.
$$

**Criterion (exact):**
A strictly local Hamiltonian of mean force $H_{\text{MF}}$ exists as a finite operator polynomial if and only if $A_f$ is finite-dimensional and closed under multiplication (or, equivalently, forms a finite Lie algebra together with the identity).

If $A_f$ does not close, the influence functional generates an infinite operator series and no exact local representation exists.

No approximation is involved in this statement.

### 6. Construction via BCH and point of failure

When closure holds, one may write

$$
\rho_S = \exp \left( -\beta H_Q - \sum_{i,j} c_{ij} O_i O_j \right), \quad O_i \in A_f,
$$

and use the Baker–Campbell–Hausdorff formula to combine terms into

$$
\rho_S = e^{-\beta H_{\text{MF}}}.
$$

When closure fails, the BCH series does not truncate. Any local Hamiltonian obtained by truncation is approximate, with the approximation entering only through the discarded adjoint sector.

---

## Manuscript Logic Plan

### Plan for `sections/02_derivation.tex`

1.  **Subsection: Formalism**
    *   Definitions of $H_{\text{tot}}$, $\rho_S$.
    *   Definition of HMF.
2.  **Subsection: Path Integral Representation**
    *   Imaginary-time picture.
    *   Gaussian bath integration (Exact Influence Functional).
    *   Auxiliary field $\xi(\tau)$ interpretation (Quenched Average).
3.  **Subsection: The Non-Commutativity Problem**
    *   Why $\tilde{f}(\tau)$ matters.
    *   Adjoint action expansion (The key algebraic step).
4.  **Subsection: Exact Local Existence Condition**
    *   The Lie Algebra closure condition ($A_f$).
    *   This is the pivotal theoretical result: exactness $\iff$ closure.
5.  **Subsection: Approximation Strategy**
    *   What happens when closure fails (Truncation).
    *   Link to BCH.

This structure moves from standard definitions -> novel algebraic insight -> classification of exact vs approximate regimes.
