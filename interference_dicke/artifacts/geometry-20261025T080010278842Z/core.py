"""Frequency units, little-endian masks (qubit i is bit i), complex128.

XY(theta) = exp[-i theta (XX+YY)/2], Z(alpha) = exp[-i alpha Z],
CP(beta) = exp[-i beta n_i n_j]. All caches are built before optimization.
"""
from itertools import combinations
from math import comb
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, diags


class Sector:
    def __init__(self, n, l, edges):
        if not 0 <= l <= n or n < 1:
            raise ValueError("require n >= 1 and 0 <= l <= n")
        if comb(n, l) > 100_000 or n > 62:
            raise ValueError("sector exceeds this experiment's explicit memory budget")
        self.n, self.l = n, l
        self.edges = tuple(sorted(set(tuple(sorted(e)) for e in edges)))
        if any(i < 0 or j >= n or i == j for i, j in self.edges):
            raise ValueError("invalid physical edge")
        self.masks = np.array(sorted(sum(1 << i for i in c)
                                     for c in combinations(range(n), l)), dtype=np.int64)
        self.index = {int(x): k for k, x in enumerate(self.masks)}
        self.occupied = ((self.masks[:, None] >> np.arange(n)) & 1).astype(float)
        self.z = 1 - 2 * self.occupied
        self.target = np.ones(len(self.masks), complex) / np.sqrt(len(self.masks))
        self.initial = np.zeros(len(self.masks), complex)
        self.initial[self.index[(1 << l) - 1]] = 1
        self.pairs = {}
        for i, j in self.edges:
            a = np.flatnonzero((self.occupied[:, i] == 1) & (self.occupied[:, j] == 0))
            b = np.array([self.index[int(self.masks[k]) ^ (1 << i) ^ (1 << j)]
                          for k in a], dtype=int)
            self.pairs[i, j] = a, b

    def diagonal(self, kind, sites):
        if kind == "z":
            return self.z[:, sites[0]]
        if kind == "cp":
            return np.prod(self.occupied[:, sites], axis=1)
        raise ValueError(kind)

    def apply(self, state, gate, angle, derivative=False):
        kind, sites = gate
        if kind != "xy":
            diagonal = self.diagonal(kind, sites)
            out = np.exp(-1j * angle * diagonal) * state
            return -1j * diagonal * out if derivative else out
        a, b = self.pairs[sites]
        out = np.zeros_like(state) if derivative else state.copy()
        c, s = np.cos(angle), np.sin(angle)
        if derivative:
            out[a] = -s * state[a] - 1j * c * state[b]
            out[b] = -s * state[b] - 1j * c * state[a]
        else:
            out[a] = c * state[a] - 1j * s * state[b]
            out[b] = c * state[b] - 1j * s * state[a]
        return out

    def evolve(self, schedule, parameters):
        if len(schedule) != len(parameters):
            raise ValueError("one parameter required per gate")
        state = self.initial.copy()
        for gate, angle in zip(schedule, parameters):
            state = self.apply(state, gate, angle)
        return state

    def objective(self, schedule, parameters):
        states = [self.initial]
        for gate, angle in zip(schedule, parameters):
            states.append(self.apply(states[-1], gate, angle))
        overlap = np.vdot(self.target, states[-1])
        adjoint = self.target * overlap
        gradient = np.empty(len(schedule))
        for k in reversed(range(len(schedule))):
            derivative = self.apply(states[k], schedule[k], parameters[k], derivative=True)
            gradient[k] = -2 * np.vdot(adjoint, derivative).real
            adjoint = self.apply(adjoint, schedule[k], -parameters[k])
        return 1 - float(abs(overlap) ** 2), gradient

    def hamiltonian(self, exchange, fields, interactions):
        """Sparse exact restriction; dictionaries keyed by physical edges."""
        diagonal = self.z @ np.asarray(fields)
        rows, cols, values = [], [], []
        for edge, value in exchange.items():
            a, b = self.pairs[tuple(sorted(edge))]
            rows.extend(a); rows.extend(b)
            cols.extend(b); cols.extend(a)
            values.extend([value] * (2 * len(a)))
        for edge, value in interactions.items():
            edge = tuple(sorted(edge))
            if edge not in self.pairs:
                raise ValueError("interaction must use an allowed edge")
            diagonal = diagonal + value * self.diagonal("cp", edge)
        return (coo_matrix((values, (rows, cols)), shape=(len(self.masks),) * 2)
                + diags(diagonal)).tocsr()


def schedule_for(sector, depth, conditional_phases=0):
    """Greedy disjoint-edge matchings; local Z gauge omits the last site.

    depth counts matching layers, NOT native depth. CP gates are spread
    through the schedule, on the central physical edge.
    """
    if depth < 1 or not 0 <= conditional_phases <= depth:
        raise ValueError("invalid depth or CP count")
    matchings = []
    for edge in sector.edges:
        for matching in matchings:
            if not set(edge) & {q for pair in matching for q in pair}:
                matching.append(edge)
                break
        else:
            matchings.append([edge])
    if not matchings:
        raise ValueError("ansatz needs an edge")
    positions = set(np.linspace(0, depth - 1, conditional_phases + 2, dtype=int)[1:-1])
    schedule = []
    for layer in range(depth):
        schedule.extend(("xy", e) for e in matchings[layer % len(matchings)])
        schedule.extend(("z", (i,)) for i in range(sector.n - 1))
        if layer in positions:
            schedule.append(("cp", sector.edges[len(sector.edges) // 2]))
    assert sum(kind == "cp" for kind, _ in schedule) == conditional_phases
    return schedule


def optimize(sector, schedule, seed=42, starts=4, maxiter=350, seconds=15):
    """Adjoint L-BFGS-B; shared elapsed budget checked on each evaluation."""
    rng = np.random.default_rng(seed)
    start = perf_counter()
    best = {"loss": 1., "parameters": [0.] * len(schedule)}
    evaluations, trials = 0, []

    class BudgetExpired(Exception):
        pass

    def objective(x):
        nonlocal evaluations
        if perf_counter() - start >= seconds:
            raise BudgetExpired
        loss, gradient = sector.objective(schedule, x)
        evaluations += 1
        if loss < best["loss"]:
            best.update(loss=loss, parameters=x.tolist())
        return loss, gradient

    for _ in range(starts):
        try:
            result = minimize(objective, rng.uniform(-np.pi, np.pi, len(schedule)),
                              jac=True, method="L-BFGS-B",
                              options={"maxiter": maxiter, "maxfun": 2 * maxiter,
                                       "ftol": 1e-13, "gtol": 1e-8, "maxls": 25})
            trials.append({"iterations": int(result.nit), "evaluations": int(result.nfev),
                           "message": str(result.message), "fidelity": 1 - float(result.fun)})
        except BudgetExpired:
            trials.append({"message": "elapsed budget exhausted"})
            break
        if best["loss"] < 1e-11:
            break
    return {"fidelity": 1 - best["loss"], "parameters": best["parameters"],
            "seconds": perf_counter() - start, "evaluations": evaluations,
            "trials": trials, "optimizer": "L-BFGS-B, analytic adjoint",
            "budget": {"starts": starts, "maxiter": maxiter, "maxfun_per_start": 2 * maxiter,
                       "seconds": seconds}, "seed": seed}
