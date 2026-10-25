# ------------------------------------------------------------------ torch simulation (fused kernels)
class Model:
    """Each CZ and its two trailing U3s are fused into one 4x4 gate; all U3s are built in one call
    (kernel launches dominate on this GPU)."""

    def __init__(self, N, n, k, czs, cdtype=torch.complex128):
        self.N, self.n, self.k, self.czs = N, n, k, list(czs)
        self.m = len(czs)
        self.cdtype = cdtype
        self.rdtype = torch.float32 if cdtype == torch.complex64 else torch.float64
        self.tgt = torch.tensor(dicke_vec(N, n), dtype=cdtype, device=dev)

    def u3_all(self, th):  # (B,S,3) -> (B,S,2,2)
        a, t, b = th[..., 0], th[..., 1], th[..., 2]
        c = torch.cos(t / 2).to(self.cdtype)
        s = torch.sin(t / 2).to(self.cdtype)
        ep = torch.polar(torch.ones_like(a), -0.5 * (a + b))
        em = torch.polar(torch.ones_like(a), 0.5 * (a - b))
        r0 = torch.stack([c * ep, -s * em], -1)
        r1 = torch.stack([s * em.conj(), c * ep.conj()], -1)
        return torch.stack([r0, r1], -2)

    def state(self, theta, mask=None):
        """theta (B, 3*(k+2m)); mask (B, m) of {0,1} (1 = CZ present)."""
        B, k, m = theta.shape[0], self.k, self.m
        U = self.u3_all(theta.reshape(B, -1, 3))
        cols = U[:, :k, :, 0]
        psi = cols[:, 0]
        for q in range(1, k):
            psi = (psi[:, :, None] * cols[:, q][:, None, :]).reshape(B, -1)
        if m == 0:
            return psi
        Ua, Ub = U[:, k::2], U[:, k + 1::2]                     # (B,m,2,2)
        G = torch.einsum('bmij,bmkl->bmikjl', Ua, Ub).reshape(B, m, 4, 4)
        if mask is None:
            d = -torch.ones((B, m), dtype=self.cdtype, device=dev)
        else:
            d = (1 - 2 * mask).to(self.cdtype)
        G = torch.cat([G[..., :3], G[..., 3:] * d[:, :, None, None]], -1)  # G @ diag(1,1,1,d)
        G = G.reshape(B, m, 2, 2, 2, 2)
        for j, (a, b) in enumerate(self.czs):
            g = G[:, j]
            if a > b:
                a, b = b, a
                g = g.permute(0, 2, 1, 4, 3)
            v = psi.reshape(B, 2 ** a, 2, 2 ** (b - a - 1), 2, 2 ** (k - b - 1))
            psi = torch.einsum('bijkl,bxkylz->bxiyjz', g, v).reshape(B, -1)
        return psi

    def loss(self, theta, mask=None):
        psi = self.state(theta, mask)
        B = psi.shape[0]
        amp = torch.einsum('s,bsa->ba', self.tgt.conj(), psi.reshape(B, 2 ** self.N, -1))
        return 1 - (amp.abs() ** 2).sum(1)

    def residual(self, theta1):
        psi = self.state(theta1[None])[0].reshape(2 ** self.N, -1)
        amp = self.tgt.conj() @ psi
        r = psi - self.tgt[:, None] * amp[None, :]
        return torch.view_as_real(r).reshape(-1)


def lm_polish(model, theta, iters=100, tol=1e-24, verbose=False, time_limit=60, giveup=None):
    """Levenberg-Marquardt on the residual; returns (theta, loss).
    giveup: list of (iteration, loss) thresholds - abort if loss is still above them."""
    from torch.func import jacfwd
    th = torch.tensor(np.asarray(theta), dtype=torch.float64, device=dev)
    f = model.residual
    r = f(th)
    L = float(r @ r)
    lam = 1e-4
    t0 = time.time()
    P = th.numel()
    eye = torch.eye(P, dtype=torch.float64, device=dev)
    for it in range(iters):
        if L < tol or time.time() - t0 > time_limit:
            break
        if giveup and any(it >= gi and L > gl for gi, gl in giveup):
            break
        J = jacfwd(f)(th)
        g = J.T @ r
        H = J.T @ J
        dg = torch.diagonal(H).clone() + 1e-9 * float(torch.diagonal(H).max()) + 1e-14
        improved = False
        for _ in range(10):
            try:
                step = torch.linalg.solve(H + lam * torch.diag(dg), -g)
            except Exception:
                lam *= 10
                continue
            th2 = th + step
            r2 = f(th2)
            L2 = float(r2 @ r2)
            if L2 < L:
                th, r, L = th2, r2, L2
                lam = max(lam / 4, 1e-15)
                improved = True
                break
            lam *= 5
        if verbose:
            print(f'      LM it {it} loss {L:.3e} lam {lam:.1e}', flush=True)
        if not improved:
            break
    return th.cpu().numpy(), L


# ------------------------------------------------------------------ IO
def load_ops(path):
    d = json.load(open(path))
    ops = []
    for op in d['ops']:
        if op[0] == 'cz':
            ops.append(('cz', int(op[1]), int(op[2])))
        else:
            U = np.array([complex(r, i) for r, i in op[2]]).reshape(2, 2)
            ops.append(('u', int(op[1]), U))
    return d['N'], d['n'], d['k'], ops


def save_ops(path, N, n, k, czs, init, post, tag, loss):
    ops = []
    for op in slots_to_ops(k, czs, init, post):
        if op[0] == 'cz':
            ops.append(['cz', op[1], op[2]])
        else:
            ops.append(['u', op[1], [[float(z.real), float(z.imag)] for z in np.asarray(op[2]).ravel()]])
    json.dump(dict(N=N, n=n, k=k, ops=ops, tag=tag, ncz=len(czs), loss=loss), open(path, 'w'))


# ------------------------------------------------------------------ main loop
def try_delete(N, n, k, czs, init, post, j, rng, restarts=1, sigma=0.03, log=None):
    czs2, init2, post2 = delete_cz(k, czs, init, post, j)
    model = Model(N, n, k, czs2)
    th0 = slots_to_params(k, czs2, init2, post2)
    best = (None, 1.0)
    for r in range(restarts):
        th = th0 if r == 0 else th0 + rng.normal(0, sigma * r, th0.shape)
        th, L = lm_polish(model, th, iters=120, tol=1e-24, giveup=[(15, 0.3), (30, 1e-2), (60, 1e-5)])
        if L < best[1]:
            best = (th, L)
        if L < 1e-12:
            break
    th, L = best
    if L < 1e-11:
        i2, p2 = params_to_slots(k, czs2, th)
        F = infid_np(N, n, k, czs2, i2, p2)
        if abs(F) < 1e-11:
            return czs2, i2, p2, F
    return None, None, None, L


def compress(N, n, k, czs, init, post, minutes=120, outdir='data/peep', tag='gpu', seed=0, restarts=2,
             log=print):
    rng = np.random.default_rng(seed)
    t_end = time.time() + 60 * minutes
    F0 = infid_np(N, n, k, czs, init, post)
    log(f'D({N},{n}) k={k} start CZ={len(czs)} infid={F0:.2e}')
    while time.time() < t_end and czs:
        m = len(czs)
        t0 = time.time()
        # rank deletions by the immediate damage (one batched forward)
        model = Model(N, n, k, czs)
        th = torch.tensor(slots_to_params(k, czs, init, post)[None].repeat(m, 0), device=dev)
        with torch.no_grad():
            dmg = model.loss(th, torch.tensor(1 - np.eye(m), device=dev)).cpu().numpy()
        order = list(np.argsort(dmg))
        accepted = False
        for cnt, j in enumerate(order):
            if time.time() > t_end:
                break
            c2, i2, p2, F = try_delete(N, n, k, czs, init, post, j, rng, restarts=restarts)
            if c2 is not None:
                log(f'  m={m}: delete #{cnt} (cz {j} {czs[j]}, damage {dmg[j]:.2e}) -> CZ={len(c2)} '
                    f'infid={F:.1e} ({time.time() - t0:.0f}s)')
                czs, init, post = c2, i2, p2
                save_ops(os.path.join(outdir, f'{N}_{n}_{tag}_cz{len(czs)}.json'), N, n, k, czs, init, post, tag, F)
                accepted = True
                break
        if not accepted:
            log(f'  m={m}: no deletion succeeded ({time.time() - t0:.0f}s)')
            break
    log(f'FINAL D({N},{n}) CZ={len(czs)}')
    return czs, init, post


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--minutes', type=float, default=120)
    ap.add_argument('--restarts', type=int, default=2)
    ap.add_argument('--tag', default='gpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--outdir', default='data/peep')
    a = ap.parse_args()
    N, n, k, ops = load_ops(a.src)
    czs, init, post = ops_to_slots(k, ops)
    os.makedirs('logs/gpu_compress', exist_ok=True)
    lf = open(f'logs/gpu_compress/{N}_{n}_{a.tag}.log', 'a')

    def log(s):
        print(s, flush=True)
        lf.write(s + '\n')
        lf.flush()

    compress(N, n, k, czs, init, post, minutes=a.minutes, outdir=a.outdir, tag=a.tag, seed=a.seed,
             restarts=a.restarts, log=log)
