"""GPU global CZ compression of exact Dicke circuits: greedy CZ deletion + full-circuit refit.

Circuit form (k nodes, qubit 0 = MSB, state on nodes 0..N-1, ancillas after):
    U3 on every node, then for each CZ j=(a_j,b_j): CZ, U3(a_j), U3(b_j).   U3 = Rz(b) Ry(t) Rz(a)
Every exact circuit with that CZ sequence is representable, so a deletion + refit explores the
whole family of circuits with one CZ fewer.

Round: try deleting every CZ (batched on the GPU as masks; warm start + perturbed copies), Adam
screen in complex64, Levenberg-Marquardt polish (complex128, forward-mode Jacobian) of the best
few; accept the first that becomes exact (1 - F < 1e-11, final polish ~1e-20).

usage (dqgpu env): python tools/gpu_compress.py data/peep/10_5_dc2_cz79.json [--minutes 120] [--copies 2]
    writes data/peep/{N}_{n}_gpu_cz{m}.json for every accepted m (peep format, picked up by build_submission.py)
"""
import argparse, itertools, json, math, os, sys, time
import torch
import numpy as np

import torch

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------------------------------------------------------ circuit <-> slots
def zyz(U):
    U = np.asarray(U, dtype=complex)
    V = U / np.sqrt(np.linalg.det(U))
    t = 2 * math.atan2(abs(V[1, 0]), abs(V[0, 0]))
    p00 = np.angle(V[0, 0]) if abs(V[0, 0]) > 1e-14 else 0.0
    p10 = np.angle(V[1, 0]) if abs(V[1, 0]) > 1e-14 else 0.0
    a, b = -p00 - p10, -p00 + p10
    R = u3_np(a, t, b)
    ph = np.vdot(R.ravel(), U.ravel())
    assert abs(abs(ph) - 2) < 1e-9, (U, R)
    return a, t, b


def u3_np(a, t, b):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c * np.exp(-0.5j * (a + b)), -s * np.exp(0.5j * (a - b))],
                     [s * np.exp(-0.5j * (a - b)), c * np.exp(0.5j * (a + b))]])


def ops_to_slots(k, ops):
    """ops: [('u', q, 2x2) | ('cz', a, b)] -> (czs, init[k], post[m][2]) as 2x2 unitaries."""
    init = [np.eye(2, dtype=complex) for _ in range(k)]
    czs, post, last = [], [], [None] * k
    for op in ops:
        if op[0] == 'u':
            q, U = op[1], op[2]
            if last[q] is None:
                init[q] = U @ init[q]
            else:
                j, s = last[q]
                post[j][s] = U @ post[j][s]
        else:
            a, b = op[1], op[2]
            czs.append((a, b))
            post.append([np.eye(2, dtype=complex), np.eye(2, dtype=complex)])
            last[a], last[b] = (len(czs) - 1, 0), (len(czs) - 1, 1)
    return czs, init, post


def slots_to_ops(k, czs, init, post):
    ops = [('u', q, init[q]) for q in range(k)]
    for j, (a, b) in enumerate(czs):
        ops += [('cz', a, b), ('u', a, post[j][0]), ('u', b, post[j][1])]
    return ops


def params_to_slots(k, czs, theta):
    th = np.asarray(theta).reshape(-1, 3)
    init = [u3_np(*th[q]) for q in range(k)]
    post = [[u3_np(*th[k + 2 * j]), u3_np(*th[k + 2 * j + 1])] for j in range(len(czs))]
    return init, post


def slots_to_params(k, czs, init, post):
    th = [zyz(init[q]) for q in range(k)]
    for j in range(len(czs)):
        th += [zyz(post[j][0]), zyz(post[j][1])]
    return np.array(th, dtype=float).ravel()


def delete_cz(k, czs, init, post, j):
    """Remove CZ j and merge its trailing U3s into the previous slot of each wire."""
    init = [u.copy() for u in init]
    post = [[p[0].copy(), p[1].copy()] for p in post]
    for s, q in enumerate(czs[j]):
        prev = None
        for i in range(j - 1, -1, -1):
            if q in czs[i]:
                prev = (i, czs[i].index(q))
                break
        if prev is None:
            init[q] = post[j][s] @ init[q]
        else:
            post[prev[0]][prev[1]] = post[j][s] @ post[prev[0]][prev[1]]
    czs = czs[:j] + czs[j + 1:]
    post = post[:j] + post[j + 1:]
    return czs, init, post


# ------------------------------------------------------------------ numpy reference simulation
def state_np(k, czs, init, post):
    psi = np.zeros((2,) * k, dtype=complex)
    psi[(0,) * k] = 1

    def ap(psi, q, U):
        return np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q)

    for q in range(k):
        psi = ap(psi, q, init[q])
    for j, (a, b) in enumerate(czs):
        sl = [slice(None)] * k
        sl[a] = 1
        sl[b] = 1
        psi[tuple(sl)] *= -1
        psi = ap(psi, a, post[j][0])
        psi = ap(psi, b, post[j][1])
    return psi.reshape(-1)


def dicke_vec(N, n):
    psi = np.zeros(2 ** N)
    for c in itertools.combinations(range(N), n):
        psi[sum(1 << (N - 1 - q) for q in c)] = 1
    return psi / np.linalg.norm(psi)


def infid_np(N, n, k, czs, init, post):
    amp = dicke_vec(N, n) @ state_np(k, czs, init, post).reshape(2 ** N, -1)
    return 1 - float(np.sum(np.abs(amp) ** 2))


# ------------------------------------------------------------------ torch simulation
