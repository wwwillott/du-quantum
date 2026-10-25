"""Batched quantum-trajectory noise simulator + fast official scorer for the Dicke challenge.

Noise model (per moment):
  * PhasedXZ: depolarising Pauli error with prob sq_pauli_error_rate[q] (X/Y/Z each p/3).
  * CZ: two-qubit depolarising with prob cz_pauli_error_rate[pair] (15 Paulis each p/15).
  * Every qubit, every moment: amplitude damping (exact jump unravelling, T1), white dephasing (Tphi),
    quasi-static detuning (per shot, Gaussian, gives T2*), optional residual ZZ between coupled idle qubits.
    Moment duration: 42 ns if it contains a CZ, 160 ns if a reset, else 25 ns.
    Qubits stay in |0> before their first gate (no decoherence there anyway).
  * Readout: classical bit flips 0->1 with zero_errors[q], 1->0 with one_errors[q].
  * Optional coherent errors: CZ conditional-phase error (rad) and 1q over-rotation (fractional).
Outputs ideal-shot distributions for Z and X basis on the state qubits.
"""
import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import cirq


@dataclass
class HW:
    T1: float = 68e-6
    Tphi_white: float = 200e-6
    T2star: float = 30e-6          # quasi-static Gaussian dephasing (echoable)
    zz_hz: float = 0.0             # residual ZZ on idle coupled pairs (Hz)
    t_1q: float = 25e-9
    t_cz: float = 42e-9
    t_reset: float = 160e-9
    cz_phase_err_std: float = 0.0  # static coherent CZ phase error per pair (rad), drawn once per seed
    sq_overrot_std: float = 0.0    # static fractional over-rotation per qubit, drawn once per seed
    gate_noise_scale: float = 1.0  # multiply Pauli error rates
    readout_scale: float = 1.0
    seed: int = 1234


def _key(a, b):
    return (a, b) if a < b else (b, a)


def _ops_from_circuit(circuit, qorder):
    idx = {q: i for i, q in enumerate(qorder)}
    moments = []
    for m in circuit:
        ops = []
        kind = '1q'
        for op in m.operations:
            g = op.gate
            qs = [idx[q] for q in op.qubits]
            if isinstance(g, cirq.CZPowGate) and len(qs) == 2:
                ops.append(('cz', qs, g.exponent, op.qubits))
                kind = 'cz'
            elif isinstance(g, cirq.ResetChannel):
                ops.append(('reset', qs, None, op.qubits))
                if kind != 'cz':
                    kind = 'reset'
            elif cirq.has_unitary(op) and len(qs) == 1:
                ops.append(('u1', qs, cirq.unitary(op), op.qubits))
            elif isinstance(g, cirq.MeasurementGate):
                raise ValueError('measurement not supported in sim')
            elif len(qs) == 1 and isinstance(g, (cirq.IdentityGate, cirq.WaitGate)):
                continue
            else:
                raise ValueError(f'unsupported op {op}')
        moments.append((kind, ops))
    return moments


PAULI = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


class Batch:
    def __init__(self, nq, B):
        self.nq, self.B = nq, B
        self.psi = np.zeros((B,) + (2,) * nq, dtype=np.complex128)
        self.psi[(slice(None),) + (0,) * nq] = 1.0

    def apply1(self, q, U, mask=None):
        ax = q + 1
        if mask is None:
            self.psi = np.moveaxis(np.tensordot(U, self.psi, axes=([1], [ax])), 0, ax)
        else:
            if not mask.any():
                return
            sub = self.psi[mask]
            self.psi[mask] = np.moveaxis(np.tensordot(U, sub, axes=([1], [ax])), 0, ax)

    def apply1_per_traj(self, q, Us):
        """Us: (B,2,2)"""
        ax = q + 1
        p = np.moveaxis(self.psi, ax, 1)  # (B,2,...)
        sh = p.shape
        p = p.reshape(sh[0], 2, -1)
        p = np.einsum('bij,bjk->bik', Us, p)
        self.psi = np.moveaxis(p.reshape(sh), 1, ax)

    def cz(self, a, b, phase=np.pi):
        sl = [slice(None)] * (self.nq + 1)
        sl[a + 1] = 1
        sl[b + 1] = 1
        self.psi[tuple(sl)] *= np.exp(1j * phase)

    def pop1(self, q):
        ax = q + 1
        sl = [slice(None)] * (self.nq + 1)
        sl[ax] = 1
        x = self.psi[tuple(sl)]
        return np.sum(np.abs(x) ** 2, axis=tuple(range(1, self.nq)))

    def normalize(self):
        n = np.sqrt(np.sum(np.abs(self.psi) ** 2, axis=tuple(range(1, self.nq + 1))))
        self.psi /= n.reshape((-1,) + (1,) * self.nq)


def simulate(circuit, state_qubits, cz_err, sq_err, zero_err, one_err, hw=HW(), n_traj=2000,
             rng=None, noiseless=False):
    """Return (PZ, PX, info): distributions over 2^N bitstrings of the sorted state qubits (qubit 0 = MSB),
    INCLUDING readout error. Ancillas traced out."""
    rng = np.random.default_rng(hw.seed) if rng is None else rng
    state_q = sorted(state_qubits)
    anc = sorted(set(circuit.all_qubits()) - set(state_q))
    qorder = state_q + anc
    nq, N = len(qorder), len(state_q)
    moments = _ops_from_circuit(circuit, qorder)
    czmap = {}
    for (a, b), (p, _) in cz_err.items():
        czmap[_key(a, b)] = p
    # static coherent errors (same for all shots)
    crng = np.random.default_rng(hw.seed + 7)
    cz_phase = {k: crng.normal(0, hw.cz_phase_err_std) for k in czmap}
    overrot = {q: crng.normal(0, hw.sq_overrot_std) for q in qorder}
    # coupled pairs among used qubits (for residual ZZ)
    used_pairs = [(i, j) for i in range(nq) for j in range(i + 1, nq)
                  if _key(qorder[i], qorder[j]) in czmap]
    first_gate = [None] * nq
    for t, (kind, ops) in enumerate(moments):
        for op in ops:
            for qi in op[1]:
                if first_gate[qi] is None:
                    first_gate[qi] = t

    B = max(1, min(n_traj, int(2 ** 19 // 2 ** nq)))
    PZ = np.zeros(2 ** N)
    PX = np.zeros(2 ** N)
    done = 0
    Hm = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    while done < n_traj:
        b = min(B, n_traj - done)
        st = Batch(nq, b)
        detune = rng.normal(0, np.sqrt(2) / hw.T2star, size=(b, nq)) if (hw.T2star and not noiseless) else None
        for t, (kind, ops) in enumerate(moments):
            dt = hw.t_cz if kind == 'cz' else (hw.t_reset if kind == 'reset' else hw.t_1q)
            active = set()
            for (typ, qs, g, cqs) in ops:
                active.update(qs)
                if typ == 'u1':
                    U = g
                    if hw.sq_overrot_std and not noiseless:
                        # over-rotation: apply U^(1+eps) via eigen-decomposition
                        w, v = np.linalg.eig(U)
                        eps = overrot[qorder[qs[0]]]
                        ang = np.angle(w)
                        U = v @ np.diag(np.exp(1j * ang * (1 + eps))) @ np.linalg.inv(v)
                    st.apply1(qs[0], U)
                    if not noiseless and not (abs(U[0, 1]) < 1e-9 and abs(U[1, 0]) < 1e-9):
                        p = sq_err.get(cqs[0], (0.0, 0))[0] * hw.gate_noise_scale
                        r = rng.random(b)
                        for k, P in enumerate('XYZ'):
                            m = (r >= p * k / 3) & (r < p * (k + 1) / 3)
                            st.apply1(qs[0], PAULI[P], m)
                elif typ == 'cz':
                    a, c = qs
                    k = _key(cqs[0], cqs[1])
                    if k not in czmap:
                        raise ValueError(f'CZ on non-allowed pair {k}')
                    ph = np.pi * g + (cz_phase[k] if not noiseless else 0.0)
                    st.cz(a, c, ph)
                    if not noiseless:
                        p = czmap[k] * hw.gate_noise_scale
                        r = rng.random(b)
                        hit = r < p
                        if hit.any():
                            which = rng.integers(1, 16, size=b)
                            for w in range(1, 16):
                                m = hit & (which == w)
                                if not m.any():
                                    continue
                                pa, pb = 'IXYZ'[w // 4], 'IXYZ'[w % 4]
                                if pa != 'I':
                                    st.apply1(a, PAULI[pa], m)
                                if pb != 'I':
                                    st.apply1(c, PAULI[pb], m)
                elif typ == 'reset':
                    q = qs[0]
                    # reset: project-and-flip: |1> -> |0>, sample outcome per traj
                    p1 = st.pop1(q)
                    out = rng.random(b) < p1
                    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
                    P1to0 = np.array([[0, 1], [0, 0]], dtype=complex)
                    Us = np.where(out[:, None, None], P1to0[None], P0[None])
                    st.apply1_per_traj(q, Us)
                    st.normalize()
            if noiseless:
                continue
            # decoherence on every qubit that has started
            gam = 1 - np.exp(-dt / hw.T1)
            for q in range(nq):
                if first_gate[q] is None or t < first_gate[q]:
                    continue
                p1 = st.pop1(q)
                jump = rng.random(b) < gam * p1
                Kj = np.array([[0, 1], [0, 0]], dtype=complex)
                Kn = np.array([[1, 0], [0, np.sqrt(1 - gam)]], dtype=complex)
                Us = np.where(jump[:, None, None], Kj[None], Kn[None])
                st.apply1_per_traj(q, Us)
                # white dephasing
                if hw.Tphi_white:
                    pz = 0.5 * (1 - np.exp(-dt / hw.Tphi_white))
                    st.apply1(q, PAULI['Z'], rng.random(b) < pz)
                if detune is not None:
                    ph = detune[:, q] * dt
                    Us = np.zeros((b, 2, 2), dtype=complex)
                    Us[:, 0, 0] = np.exp(-0.5j * ph)
                    Us[:, 1, 1] = np.exp(0.5j * ph)
                    st.apply1_per_traj(q, Us)
            st.normalize()
            if hw.zz_hz:
                for (i, j) in used_pairs:
                    if i in active or j in active:
                        continue
                    if first_gate[i] is None or first_gate[j] is None or t < max(first_gate[i], first_gate[j]):
                        continue
                    st.cz(i, j, 2 * np.pi * hw.zz_hz * dt)
        psi = st.psi.reshape(b, 2 ** N, -1)
        PZ += np.sum(np.abs(psi) ** 2, axis=(0, 2))
        px = st.psi
        for q in range(N):
            px = np.moveaxis(np.tensordot(Hm, px, axes=([1], [q + 1])), 0, q + 1)
        PX += np.sum(np.abs(px.reshape(b, 2 ** N, -1)) ** 2, axis=(0, 2))
        done += b
    PZ /= n_traj
    PX /= n_traj
    if not noiseless:
        e0 = np.array([zero_err.get(q, (0, 0))[0] for q in state_q]) * hw.readout_scale
        e1 = np.array([one_err.get(q, (0, 0))[0] for q in state_q]) * hw.readout_scale
        PZ = apply_readout(PZ, e0, e1)
        PX = apply_readout(PX, e0, e1)
    return PZ, PX, dict(depth=len(moments), nq=nq)


def apply_readout(P, e0, e1):
    N = len(e0)
    P = P.reshape((2,) * N)
    for q in range(N):
        M = np.array([[1 - e0[q], e1[q]], [e0[q], 1 - e1[q]]])
        P = np.moveaxis(np.tensordot(M, P, axes=([1], [q])), 0, q)
    return P.reshape(-1)


# ---------------------------------------------------------------- scoring
def _bits_table(N):
    idx = np.arange(2 ** N)
    return ((idx[:, None] >> (N - 1 - np.arange(N))) & 1).astype(np.int8)


def krawtchouk(N, k, w):
    return sum((-1) ** j * math.comb(w, j) * math.comb(N - w, k - j) for j in range(0, k + 1))


def score_from_counts(cz_counts, cx_counts, N, n, K):
    """cz_counts, cx_counts: arrays over 2^N bitstrings (qubit 0 = MSB). Exactly the official formulas."""
    S = cz_counts.sum()
    wt = _bits_table(N).sum(1)
    M = math.comb(N, n)
    shell = wt == n
    f = cz_counts[shell].sum() / S
    if f == 0:
        return 0.0, np.inf, 0.0, 0.0
    E = S * f / M
    chi2 = np.sum((cz_counts[shell] - E) ** 2 / E)
    Z = chi2 / (M * f)
    comb = lambda a, b: math.comb(a, b) if (b >= 0 and a >= b) else 0
    X = 0.0
    Sx = cx_counts.sum()
    hw_hist = np.bincount(wt, weights=cx_counts, minlength=N + 1)
    for k in range(1, K + 1):
        exp = comb(k, k // 2) * comb(N - k, n - k // 2) / comb(N, n) if k % 2 == 0 else 0.0
        meas = sum(hw_hist[w] * krawtchouk(N, k, w) for w in range(N + 1)) / Sx / math.comb(N, k)
        X += (exp - meas) ** 2 / exp ** 2 if exp != 0 else (exp - meas) ** 2
    return 1.0 / (Z + X), Z, X, f


def score_stats(PZ, PX, N, n, K=None, shots=10000, reps=400, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    K = N if K is None else K
    PZ = np.clip(PZ, 0, None); PZ /= PZ.sum()
    PX = np.clip(PX, 0, None); PX /= PX.sum()
    out = []
    for _ in range(reps):
        cz = rng.multinomial(shots, PZ)
        cx = rng.multinomial(shots, PX)
        out.append(score_from_counts(cz, cx, N, n, K))
    out = np.array(out)
    return dict(median=float(np.median(out[:, 0])), mean=float(np.mean(out[:, 0])),
                Zmed=float(np.median(out[:, 1])), Xmed=float(np.median(out[:, 2])),
                f=float(np.mean(out[:, 3])))


def dicke_vec(N, n):
    psi = np.zeros(2 ** N)
    for c in itertools.combinations(range(N), n):
        psi[sum(1 << (N - 1 - q) for q in c)] = 1
    return psi / np.linalg.norm(psi)


def check_exact(circuit, state_qubits, n, atol=1e-6):
    """Fidelity of noiseless circuit (ancillas traced) with D(N,n). Uses cirq density matrix if resets present."""
    state_q = sorted(state_qubits)
    anc = sorted(set(circuit.all_qubits()) - set(state_q))
    order = state_q + anc
    N = len(state_q)
    has_reset = any(isinstance(op.gate, cirq.ResetChannel) for op in circuit.all_operations())
    target = dicke_vec(N, n)
    if has_reset:
        rho = cirq.DensityMatrixSimulator().simulate(circuit, qubit_order=order).final_density_matrix
        rho = cirq.partial_trace(rho.reshape((2,) * (2 * len(order))), list(range(N))).reshape(2 ** N, 2 ** N)
        F = float(np.real(target @ rho @ target))
    else:
        psi = cirq.final_state_vector(circuit, qubit_order=order, dtype=np.complex128)
        psi = psi.reshape(2 ** N, -1)
        amp = target @ psi
        F = float(np.sum(np.abs(amp) ** 2))
    return F
