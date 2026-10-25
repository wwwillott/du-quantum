"""Independent 4000-trial shot statistics with fixed per-case seeds.

Identical distributions get identical trials regardless of earlier cases.
The saved actual QVM measurements and their literal notebook scores stay intact.
"""
import argparse
import json
import math
import numpy as np
from .compare import OUT, fast_score


def score_batch(z, x, n, k, max_k):
    weights = np.array([i.bit_count() for i in range(1 << n)])
    m = math.comb(n,k)
    shell = z[:,weights==k]
    shots = z.sum(axis=1)
    f = shell.sum(axis=1)/shots
    expected = shell.sum(axis=1)/m
    penalty = np.sum((shell-expected[:,None])**2/expected[:,None],axis=1)/(m*f)
    def comb(a,b):
        return math.comb(a,b) if 0<=b<=a else 0
    for ell in range(1,max_k+1):
        by_weight = np.array([sum((-1)**j*comb(w,j)*comb(n-w,ell-j) for j in range(ell+1))/comb(n,ell) for w in range(n+1)])
        measured = x @ by_weight[weights]/x.sum(axis=1)
        ideal = comb(ell,ell//2)*comb(n-ell,k-ell//2)/comb(n,k) if ell%2==0 else 0.
        penalty += (ideal-measured)**2/(ideal**2 if ideal else 1)
    return 1/penalty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('labels',nargs='+')
    args = parser.parse_args()
    for label in args.labels:
        path = OUT/(label+'.json')
        data = json.loads(path.read_text())
        assert data['complete']
        for row in data['rows']:
            if not row['accepted']:
                continue
            n,k = row['n'],row['k']
            saved = np.load(OUT/f'{label}-{n}-{k}.npz')
            for max_k in sorted({min(4,n),n}):
                rng = np.random.default_rng(43210+n*1000+k*31+max_k)
                z = rng.multinomial(10000,saved['PZ'],size=4000)
                x = rng.multinomial(10000,saved['PX'],size=4000)
                scores = score_batch(z,x,n,k,max_k)
                assert np.isclose(scores[0],fast_score(z[0],x[0],n,k,max_k),atol=1e-10,rtol=1e-10)
                row['shot_statistics'][str(max_k)] = dict(median=float(np.median(scores)),mean=float(np.mean(scores)),
                    q025=float(np.quantile(scores,.025)),q975=float(np.quantile(scores,.975)))
        data['bootstrap_repetitions'] = 4000
        data['bootstrap_seed_rule'] = '43210+1000*N+31*k+K; same per-case seed for every candidate'
        path.write_text(json.dumps(data,indent=2),encoding='utf8')
        print(label,'rescored',len(data['rows']),'cases',flush=True)


if __name__ == '__main__':
    main()
