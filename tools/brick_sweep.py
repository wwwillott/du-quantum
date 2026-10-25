"""Sweep brickwork depth-minimal synthesis over bonus instances.
usage: python tools/brick_sweep.py "N:n,N:n,..." [restarts] [steps]
For each instance: try a guessed depth; if success walk down until 2 consecutive failures, else walk up.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from brick import search

SHAPES = {2: [(1, 2)], 3: [(1, 3), (2, 2)], 4: [(2, 2)], 5: [(2, 3)], 6: [(2, 3)], 7: [(2, 4), (3, 3)], 8: [(2, 4), (3, 3)],
          9: [(3, 3)], 10: [(3, 4), (2, 5)], 11: [(3, 4)], 12: [(3, 4)], 13: [(4, 4), (3, 5)], 14: [(4, 4), (3, 5)],
          15: [(4, 4), (3, 5)]}


def try_d(N, n, R, C, d, restarts, steps, seed):
    vals, best, cells, classes, seq = search(N, n, R, C, d, 'real', restarts, steps, seed=seed)
    ok = best is not None and best[0] < 1e-12
    if ok:
        os.makedirs('data/brick', exist_ok=True)
        json.dump(dict(N=N, n=n, R=R, C=C, d=d, mode='real', cells=cells, classes=classes, seq=seq,
                       params=best[1].tolist(), loss=float(best[0])),
                  open(f'data/brick/{N}_{n}_{R}x{C}_d{d}.json', 'w'))
    return ok, (vals[0] if len(vals) else None), (best[0] if best else None)


if __name__ == '__main__':
    inst = [tuple(map(int, x.split(':'))) for x in sys.argv[1].split(',')]
    restarts = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 2500
    cur = json.load(open('logs/bonus_current_moments.json'))
    for N, n in inst:
        target = cur[f'{N},{n}']
        dcap = (target - 2) // 2          # need 2d+1 < target
        for (R, C) in SHAPES[N]:
            guess = dcap
            best_d = None
            d = guess
            tried = set()
            # walk up until success (bounded by dcap)
            while d <= dcap and best_d is None:
                t0 = time.time()
                ok, a, b = try_d(N, n, R, C, d, restarts, steps, seed=d)
                tried.add(d)
                print(f'D({N},{n}) {R}x{C} d={d} mom={2*d+1} target<{target}: {"OK" if ok else "fail"} adam={a} pol={b} ({time.time()-t0:.0f}s)', flush=True)
                if ok:
                    best_d = d
                else:
                    break
            if best_d is None:
                continue
            # walk down
            fails = 0
            d = best_d - 1
            while d >= 1 and fails < 2:
                t0 = time.time()
                ok, a, b = try_d(N, n, R, C, d, restarts, steps, seed=100 + d)
                print(f'D({N},{n}) {R}x{C} d={d} mom={2*d+1}: {"OK" if ok else "fail"} adam={a} pol={b} ({time.time()-t0:.0f}s)', flush=True)
                if ok:
                    best_d, fails = d, 0
                else:
                    fails += 1
                d -= 1
            print(f'BEST D({N},{n}) {R}x{C}: d={best_d} moments={2*best_d+1} (was {target})', flush=True)
