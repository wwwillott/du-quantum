"""Monte-Carlo analysis of the DuQuantum / Google Dicke-state scoring function.

Noise model (deliberately simple, Pauli/classical):
  Z basis: each qubit's measured bit flips independently; 1->0 with prob a1, 0->1 with prob a0.
  X basis: each qubit's measured bit flips with prob q (dephasing during the circuit + readout).
Also: a coherent amplitude bias (non-uniformity) within the Hamming-weight shell.
"""
import itertools, math
import numpy as np

rng = np.random.default_rng(0)
SHOTS = 10_000

# ---------------- scoring functions copied from the notebook ----------------
def check_z(bitstrings, n_exc):
    n_shots, n_q = bitstrings.shape
    meas = bitstrings.sum(axis=1)
    f = np.mean(meas == n_exc)
    chi2 = 0.0
    p = 1 / math.comb(n_q, n_exc)
    # vectorised version of the notebook loop: count each weight-n string
    w = bitstrings[meas == n_exc]
    keys = (w * (1 << np.arange(n_q))).sum(axis=1)
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    E = n_shots * f * p
    seen = 0.0
    for c in counts.values():
        seen += (c - E) ** 2 / E
    unseen = math.comb(n_q, n_exc) - len(counts)
    chi2 = seen + unseen * E
    return p * chi2 / f, f

def corr_avg(bitstrings, k):
    n_shots, n_q = bitstrings.shape
    s = 0.0
    for idx in itertools.combinations(range(n_q), k):
        s += 1 - 2 * np.mean(bitstrings[:, idx].sum(axis=1) % 2)
    return s / math.comb(n_q, k)

def check_x(bitstrings, n_exc, K):
    comb = lambda n, k: math.comb(n, k) if (k >= 0 and n >= k) else 0
    n_shots, n_q = bitstrings.shape
    score, terms = 0.0, []
    for k in range(1, K + 1):
        exp = comb(k, k // 2) * comb(n_q - k, n_exc - k // 2) / comb(n_q, n_exc) if k % 2 == 0 else 0.0
        m = corr_avg(bitstrings, k)
        t = (exp - m) ** 2 / exp ** 2 if exp != 0 else (exp - m) ** 2
        terms.append(t)
        score += t
    return score, terms

# ---------------- ideal distributions ----------------
def dicke_vec(N, n):
    psi = np.zeros(2 ** N)
    for c in itertools.combinations(range(N), n):
        psi[sum(1 << q for q in c)] = 1
    return psi / np.linalg.norm(psi)

def fwht(a):
    a = a.copy(); h = 1
    while h < len(a):
        a = a.reshape(-1, 2 * h)
        x, y = a[:, :h].copy(), a[:, h:].copy()
        a[:, :h], a[:, h:] = x + y, x - y
        a = a.reshape(-1); h *= 2
    return a / math.sqrt(len(a))

def to_bits(idx, N):
    return ((idx[:, None] >> np.arange(N)) & 1).astype(np.int8)

def sample_z(N, n, a0, a1, bias=0.0):
    combos = list(itertools.combinations(range(N), n))
    w = np.ones(len(combos))
    if bias:
        w = w * (1 + bias * rng.standard_normal(len(combos)))
        w = np.clip(w, 0, None)
    w /= w.sum()
    pick = rng.choice(len(combos), size=SHOTS, p=w)
    bits = np.zeros((SHOTS, N), dtype=np.int8)
    for i, c in enumerate(combos):
        rows = pick == i
        bits[np.ix_(rows, list(c))] = 1
    r = rng.random((SHOTS, N))
    flip = np.where(bits == 1, r < a1, r < a0)
    return bits ^ flip

def sample_x(N, n, q):
    probs = fwht(dicke_vec(N, n)) ** 2
    idx = rng.choice(len(probs), size=SHOTS, p=probs / probs.sum())
    bits = to_bits(idx, N)
    flip = rng.random(bits.shape) < q
    return bits ^ flip

def run(N, n, K, a0, a1, q, bias=0.0):
    zs, f = check_z(sample_z(N, n, a0, a1, bias), n)
    xs, terms = check_x(sample_x(N, n, q), n, K)
    return zs, xs, f, 1 / (zs + xs)

if __name__ == "__main__":
    print("== 1. Noise-free floor (shot noise only), K = N ==")
    for N, n in [(2,1),(3,1),(4,1),(4,2),(6,3),(8,1),(8,4),(10,5),(12,1),(12,6)]:
        zs, xs, f, s = run(N, n, N, 0, 0, 0)
        M = math.comb(N, n)
        print(f"N={N:2d} n={n} M={M:4d}  Zscore={zs:.3f} (pred (M-1)/M={(M-1)/M:.3f})  Xscore={xs:.4f}  overall={s:.3f}")

    print("\n== 2. Realistic readout only (a0=0.5%, a1=1.0%, q=0.75%), K=4 ==")
    for N, n in [(4,1),(4,2),(8,1),(8,4),(12,1),(12,6)]:
        zs, xs, f, s = run(N, n, 4, 0.005, 0.01, 0.0075)
        print(f"N={N:2d} n={n}  f={f:.3f}  Z={zs:.3f}  X={xs:.4f}  overall={s:.3f}")

    print("\n== 3. Sensitivity at N=8,n=4,K=8: vary Z-flip (Hamming leakage) vs X-flip (dephasing) ==")
    for a in [0.0, 0.01, 0.02, 0.04]:
        zs, xs, f, s = run(8, 4, 8, a, a, 0.0)
        print(f"  Z-flip {a:.2f}: f={f:.3f} Z={zs:.3f} X={xs:.4f} overall={s:.3f}")
    for q in [0.0, 0.01, 0.02, 0.04]:
        zs, xs, f, s = run(8, 4, 8, 0, 0, q)
        _, terms = check_x(sample_x(8, 4, q), 4, 8)
        print(f"  X-flip {q:.2f}: Z={zs:.3f} X={xs:.4f} overall={s:.3f}  per-k X terms={[round(t,4) for t in terms]}")

    print("\n== 4. Non-uniformity within the shell (relative amplitude spread), no other noise ==")
    for N, n in [(2,1),(4,1),(4,2),(8,4)]:
        for b in [0.0, 0.02, 0.05]:
            zs, xs, f, s = run(N, n, 2, 0, 0, 0, bias=b)
            print(f"  N={N} n={n} spread={b:.2f}: Z={zs:.3f} overall={s:.3f}")
