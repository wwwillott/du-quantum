"""Score circuits the way the organizer most likely does: Google's Willow QVM noise model
(cirq_google NoiseModelFromGoogleNoiseProperties(load_device_noise_properties('willow_pink'))), error maps derived
from the same calibration, exact density-matrix simulation (N<=11), Z and X basis, readout confusion,
harmonic aggregation 1/mean(Z+X) over 10k-shot trials.

usage: python tools/qvm_eval.py N n [variants=dd,nodd] [Ks=2,4,N]
"""
import sys, os, time, math, itertools
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np
import cirq, cirq_google as cg
from cirq_google.engine import virtual_engine_factory as vef
from sim import score_from_counts

PROPS = vef.load_device_noise_properties('willow_pink')
NOISE = cg.NoiseModelFromGoogleNoiseProperties(PROPS)


def maps_from_props(p=PROPS):
    cz, sq, z0, z1 = {}, {}, {}, {}
    for k, val in p.gate_pauli_errors.items():
        if k.gate_type == cirq.CZPowGate:
            a, b = k.qubits
            key = (a, b) if (b, a) not in cz else None
            if key:
                cz[key] = (float(val), 5e-4)
        elif k.gate_type == cirq.PhasedXZGate:
            sq[k.qubits[0]] = (float(val), 2e-4)
    for q, (e0, e1) in p.readout_errors.items():
        z0[q] = (float(e0), 2e-4); z1[q] = (float(e1), 3e-4)
    return cz, sq, z0, z1


def noisy_probs(circuit, outs, xbasis):
    c = circuit.copy()
    if xbasis:
        c.append(cirq.Moment([cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.H)).on(q) for q in outs]))
    order = sorted(outs) + sorted(set(c.all_qubits()) - set(outs))
    sim = cirq.DensityMatrixSimulator(noise=NOISE, dtype=np.complex128)
    rho = sim.simulate(c, qubit_order=order).final_density_matrix
    K = len(order); N = len(outs)
    d = np.real(np.diag(rho)).reshape(2 ** N, -1).sum(1)
    # readout confusion (independent, per qubit)
    P = d.reshape((2,) * N)
    for i, q in enumerate(sorted(outs)):
        e0, e1 = PROPS.readout_errors[q]
        M = np.array([[1 - e0, e1], [e0, 1 - e1]])
        P = np.moveaxis(np.tensordot(M, P, axes=([1], [i])), 0, i)
    P = np.clip(P.reshape(-1), 0, None)
    return P / P.sum()


def harmonic_score(PZ, PX, N, n, K, trials=300, rng=None):
    rng = np.random.default_rng(1) if rng is None else rng
    tot = []
    for _ in range(trials):
        s, Z, X, f = score_from_counts(rng.multinomial(10000, PZ), rng.multinomial(10000, PX), N, n, K)
        tot.append(Z + X)
    tot = np.array(tot)
    return 1 / tot.mean(), tot.mean()


if __name__ == '__main__':
    import importlib.util
    N, n = int(sys.argv[1]), int(sys.argv[2])
    variants = sys.argv[3].split(',') if len(sys.argv) > 3 else ['dd', 'nodd']
    Ks = [int(x) if x != 'N' else N for x in (sys.argv[4].split(',') if len(sys.argv) > 4 else ['2', '4', 'N'])]
    spec = importlib.util.spec_from_file_location('subq', os.path.join(ROOT, 'submission.py'))
    S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
    cz, sq, z0, z1 = maps_from_props()
    orig = S.build_circuit
    for var in variants:
        S.build_circuit = (lambda t, m, _dd=(var == 'dd'), **kw: orig(t, m, dd=_dd))
        t0 = time.time()
        c, qs = S.create_dicke_state(N, n, cz, sq, z0, z1)
        PZ = noisy_probs(c, qs, False); PX = noisy_probs(c, qs, True)
        shell = np.array([bin(i).count('1') == n for i in range(2 ** N)])
        res = [f'K={K}: score={harmonic_score(PZ, PX, N, n, K)[0]:.4f}' for K in Ks]
        print(f'D({N},{n}) {var:5s} cz={sum(len(o.qubits)==2 for o in c.all_operations())} moments={len(c)} '
              f'1q={sum(len(o.qubits)==1 for o in c.all_operations())} f={PZ[shell].sum():.4f}  ' + '  '.join(res) +
              f'  ({time.time()-t0:.0f}s)', flush=True)
    S.build_circuit = orig
