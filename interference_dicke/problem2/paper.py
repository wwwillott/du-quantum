"""Yu et al., arXiv:2411.03428: collective rotations and weight projections.

This reduced symmetric-space calculation is a research model, NOT a submitted
Willow circuit or a replacement of compiled moment counts by expected rounds.
"""
import math
import numpy as np
from scipy.linalg import expm


def collective_rotation(n, theta):
    # -i*Jy in weight order 0..n. A real, skew-symmetric generator.
    generator = np.zeros((n+1, n+1))
    for w in range(n):
        value = math.sqrt((n-w)*(w+1))/2
        generator[w+1, w] = value
        generator[w, w+1] = -value
    return expm(theta*generator)


def feedback_statistics(n, k, rounds=8):
    """Geometric-angle policy, Eq. (2), with k made absorbing.

    For general k this uses the paper's geometric policy without its optional
    reset heuristic. The first rotation is the binomial-mode angle.
    """
    j, mt = n/2, n/2-k
    r0sq = j*(j+1)
    rt = math.sqrt(r0sq-mt*mt)
    transition = np.zeros((n+1, n+1))
    for w in range(n+1):
        if w == k:
            transition[w, w] = 1
            continue
        m = j-w
        rm = math.sqrt(r0sq-m*m)
        theta = math.asin(np.clip((m*rt-mt*rm)/r0sq, -1, 1))
        if w == 0:
            theta = 2*math.asin(math.sqrt(k/n))
        transition[w] = abs(collective_rotation(n, theta)[:, w])**2
    distribution = np.zeros(n+1); distribution[0] = 1
    failure = []
    for _ in range(rounds):
        distribution = distribution @ transition
        failure.append(float(1-distribution[k]))
    remaining = [w for w in range(n+1) if w != k]
    expected = np.linalg.solve(np.eye(n)-transition[np.ix_(remaining, remaining)], np.ones(n))
    return dict(n=n, k=k, failure_after_rounds=failure,
                expected_collective_rounds=float(expected[remaining.index(0)]),
                compiled_depth=None, eligible_for_score=False)
