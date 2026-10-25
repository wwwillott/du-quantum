def lm_polish(model, theta, iters=300, tol=1e-22, verbose=False, time_limit=120):
    """Levenberg-Marquardt on the residual; returns (theta, loss)."""
    from torch.func import jacfwd
    th = torch.tensor(theta, dtype=torch.float64, device=dev)
    f = model.residual
    r = f(th)
    L = float(r @ r)
    lam = 1e-3
    t0 = time.time()
    P = th.numel()
    chunk = max(16, min(P, int(2.0e8 / max(1, r.numel() * 16 * 2))))
    for it in range(iters):
        if L < tol or time.time() - t0 > time_limit:
            break
        J = jacfwd(f)(th)
        g = J.T @ r
        H = J.T @ J
        dg = torch.diagonal(H).clone()
        improved = False
        for _ in range(12):
            A = H + lam * (dg + 1e-12 * dg.max() + 1e-15) * torch.eye(P, dtype=H.dtype, device=dev)
            try:
                step = torch.linalg.solve(A, -g)
            except Exception:
                lam *= 10
                continue
            th2 = th + step
            r2 = f(th2)
            L2 = float(r2 @ r2)
            if L2 < L:
                th, r, L = th2, r2, L2
                lam = max(lam / 5, 1e-12)
                improved = True
                break
            lam *= 6
        if verbose:
            print(f'      LM it {it} loss {L:.3e} lam {lam:.1e}', flush=True)
        if not improved:
            break
    return th.cpu().numpy(), L


