import heapq
import itertools
import math
import time

import cirq
import cirq_google
import networkx as nx
import numpy as np


# ==============================================================================
# Step 1. Make all gates needed out of Willow-native gates
# ==============================================================================

# These are all helper functions that return the desired gate as a Willow-native gate.

def native_x(qubit: cirq.Qid):
    """Return a Pauli-X gate expressed as a Willow-native PhasedXZGate."""
    return cirq.PhasedXZGate(
        x_exponent=1.0,
        z_exponent=0.0,
        axis_phase_exponent=0.0,
    )(qubit)


def native_h(qubit: cirq.Qid):
    """Return a Hadamard gate expressed as a Willow-native PhasedXZGate."""
    return cirq.PhasedXZGate(
        x_exponent=0.5,
        z_exponent=1.0,
        axis_phase_exponent=1.5,
    )(qubit)


def native_ry(qubit: cirq.Qid, theta: float):
    """Return an Ry(theta) rotation expressed as a Willow-native PhasedXZGate."""
    return cirq.PhasedXZGate(
        x_exponent=theta / np.pi,
        z_exponent=0.0,
        axis_phase_exponent=0.5,
    )(qubit)


def native_cnot(control: cirq.Qid, target: cirq.Qid):
    """Yield a CNOT decomposed into Willow-native CZ and PhasedXZ gates.

    CNOT(c, t) = (I x H_t) . CZ(c, t) . (I x H_t)
    """
    yield native_h(target)
    yield cirq.CZ(control, target)
    yield native_h(target)


# ==============================================================================
# LEVEL 2: SCS Building Blocks (Gate i and Gate ii_l CCRy Rotations)
# ==============================================================================


def build_cry_core(q_ctrl: cirq.Qid, q_target: cirq.Qid, theta: float):
    """Decompose a controlled Ry(2 * theta) using two CNOTs and two rotations.

    When q_ctrl is 1, applies Ry(2 * theta) to q_target.
    """
    yield native_ry(q_target, theta)
    yield native_cnot(q_ctrl, q_target)
    yield native_ry(q_target, -theta)
    yield native_cnot(q_ctrl, q_target)


def build_gate_i(q_ctrl: cirq.Qid, q_target: cirq.Qid, theta: float):
    """Build gate (i) from Bärtschi and Eidenbenz.

    Applies controlled Ry(2 * theta) with q_target as the control and q_ctrl
    as the target, conjugated by CNOT(q_ctrl, q_target).
    """
    yield native_cnot(q_ctrl, q_target)
    yield build_cry_core(q_target, q_ctrl, theta)
    yield native_cnot(q_ctrl, q_target)


def build_ccry_core(
    q_ctrl1: cirq.Qid,
    q_ctrl2: cirq.Qid,
    q_target: cirq.Qid,
    theta: float,
):
    """Decompose CCRy(2*theta) using four Ry rotations and four CNOTs."""
    yield native_ry(q_target, theta / 2)
    yield native_cnot(q_ctrl2, q_target)
    yield native_ry(q_target, -theta / 2)
    yield native_cnot(q_ctrl1, q_target)
    yield native_ry(q_target, theta / 2)
    yield native_cnot(q_ctrl2, q_target)
    yield native_ry(q_target, -theta / 2)
    yield native_cnot(q_ctrl1, q_target)


def build_gate_ii(
    q_top: cirq.Qid,
    q_mid: cirq.Qid,
    q_bot: cirq.Qid,
    theta: float,
):
    """Build gate (ii)_l from Figure 3 of Bärtschi and Eidenbenz.

    Acts on the triple (m-l-1, m-l, m-1) = (q_top, q_mid, q_bot).
    """
    yield native_cnot(q_top, q_bot)
    yield build_ccry_core(
        q_ctrl1=q_mid,
        q_ctrl2=q_bot,
        q_target=q_top,
        theta=theta,
    )
    yield native_cnot(q_top, q_bot)


# ==============================================================================
# Step 3. Create SCS blocks out of the SCS-subgates.
# ==============================================================================


def build_scs_block(
    path_qubits: list[cirq.Qid],
    m: int,
    k: int,
):
    """Construct the SCS_{m,k} block acting on path_qubits[:m]."""
    sub_q = path_qubits[:m]

    # Gate (i) on (m-2, m-1).
    theta_i = np.arccos(1.0 / np.sqrt(m))
    yield build_gate_i(sub_q[m - 2], sub_q[m - 1], theta_i)

    # Gate (ii)_l for l in 2..min(k, m-1).
    for l in range(2, min(k, m - 1) + 1):
        q_top = sub_q[m - l - 1]
        q_mid = sub_q[m - l]
        q_bot = sub_q[m - 1]
        theta_l = np.arccos(np.sqrt(l / m))
        yield build_gate_ii(q_top, q_mid, q_bot, theta_l)


# ==============================================================================
# Step 4. Find the optimal path present on the Google Willow grid..
# ==============================================================================

class MultiModePathfinder:
    """Find a low-error connected qubit chain on the Google Willow grid."""

    def __init__(self, mode: str = "branch_and_bound"):
        self.mode = mode

    def calculate_cz_counts(self, N: int, n: int) -> dict:
        """Count the emitted CZ gates on each relative path coupler."""
        cz_counts = {i: 0 for i in range(N - 1)}
        logical_qubits = list(cirq.LineQubit.range(N))
        index_by_qubit = {
            qubit: index for index, qubit in enumerate(logical_qubits)
        }
        circuit = build_full_u_nk(logical_qubits, N, n)

        for operation in circuit.all_operations():
            if operation.gate != cirq.CZ:
                continue
            first, second = (
                index_by_qubit[qubit] for qubit in operation.qubits
            )
            if abs(first - second) != 1:
                raise ValueError(
                    f"non-nearest CZ remained between {first} and {second}"
                )
            cz_counts[min(first, second)] += 1

        return cz_counts

    def compute_path_cost(
        self,
        path: list,
        sq_errors: dict,
        cz_errors: dict,
        readout_errors: dict,
        cz_counts: dict,
    ) -> float:
        """Compute the total additive logarithmic noise cost of a path."""
        cost = 0.0

        # Single-qubit and readout errors.
        for q in path:
            sq_e = sq_errors.get(q, 0.001)
            ro_e = readout_errors.get(q, 0.01)
            cost += -np.log(max(1.0 - sq_e, 1e-6))
            cost += -np.log(max(1.0 - ro_e, 1e-6))

        # CZ errors, weighted by each coupler's gate activity.
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            cz_e = cz_errors.get((u, v), cz_errors.get((v, u), 0.01))
            activity = cz_counts.get(i, 1)
            cost += activity * (-np.log(max(1.0 - cz_e, 1e-6)))

        return cost

    def find_path(
        self,
        qubits: list,
        couplers: list,
        N: int,
        n: int,
        sq_errors: dict,
        cz_errors: dict,
        readout_errors: dict,
        override_mode: str | None = None,
    ):
        """Dispatch to the selected pathfinding strategy."""
        search_mode = override_mode or self.mode
        cz_counts = self.calculate_cz_counts(N, n)

        adjacency = {q: [] for q in qubits}
        for u, v in couplers:
            if u in adjacency and v in adjacency:
                adjacency[u].append(v)
                adjacency[v].append(u)

        search_args = (
            qubits,
            adjacency,
            N,
            sq_errors,
            cz_errors,
            readout_errors,
            cz_counts,
        )
        if search_mode == "branch_and_bound":
            return self._branch_and_bound(*search_args)
        if search_mode == "dijkstra":
            return self._dijkstra(*search_args)
        if search_mode == "geometric_snake":
            return self._geometric_snake(*search_args)
        raise ValueError(f"Unknown pathfinding mode: {search_mode}")

    # --------------------------------------------------------------------------
    # Mode 1: Branch and bound (DFS with pruning)
    # --------------------------------------------------------------------------

    def _branch_and_bound(
        self,
        qubits,
        adjacency,
        N,
        sq_errors,
        cz_errors,
        readout_errors,
        cz_counts,
    ):
        best_path = None
        best_cost = float("inf")

        def dfs(current_path, current_cost, visited):
            nonlocal best_path, best_cost

            if current_cost >= best_cost:
                return

            if len(current_path) == N:
                best_cost = current_cost
                best_path = list(current_path)
                return

            last_q = current_path[-1]
            step_idx = len(current_path) - 1
            neighbors = [
                neighbor
                for neighbor in adjacency[last_q]
                if neighbor not in visited
            ]
            neighbors.sort(
                key=lambda neighbor: cz_errors.get(
                    (last_q, neighbor),
                    cz_errors.get((neighbor, last_q), 0.01),
                )
            )

            for neighbor in neighbors:
                sq_e = sq_errors.get(neighbor, 0.001)
                ro_e = readout_errors.get(neighbor, 0.01)
                cz_e = cz_errors.get(
                    (last_q, neighbor),
                    cz_errors.get((neighbor, last_q), 0.01),
                )
                step_cost = (
                    -np.log(max(1.0 - sq_e, 1e-6))
                    - np.log(max(1.0 - ro_e, 1e-6))
                    + cz_counts.get(step_idx, 1)
                    * (-np.log(max(1.0 - cz_e, 1e-6)))
                )

                visited.add(neighbor)
                current_path.append(neighbor)
                dfs(current_path, current_cost + step_cost, visited)
                current_path.pop()
                visited.remove(neighbor)

        for start_q in qubits:
            sq_e = sq_errors.get(start_q, 0.001)
            ro_e = readout_errors.get(start_q, 0.01)
            initial_cost = -np.log(max(1.0 - sq_e, 1e-6)) - np.log(
                max(1.0 - ro_e, 1e-6)
            )
            dfs([start_q], initial_cost, {start_q})

        return best_path, best_cost

    # --------------------------------------------------------------------------
    # Mode 2: Dijkstra / uniform-cost search
    # --------------------------------------------------------------------------

    def _dijkstra(
        self,
        qubits,
        adjacency,
        N,
        sq_errors,
        cz_errors,
        readout_errors,
        cz_counts,
    ):
        priority_queue = []
        path_counter = 0

        for start_q in qubits:
            sq_e = sq_errors.get(start_q, 0.001)
            ro_e = readout_errors.get(start_q, 0.01)
            initial_cost = -np.log(max(1.0 - sq_e, 1e-6)) - np.log(
                max(1.0 - ro_e, 1e-6)
            )
            heapq.heappush(
                priority_queue,
                (initial_cost, path_counter, (start_q,), {start_q}),
            )
            path_counter += 1

        while priority_queue:
            cost, _, path, visited = heapq.heappop(priority_queue)
            if len(path) == N:
                return list(path), cost

            last_q = path[-1]
            step_idx = len(path) - 1
            for neighbor in adjacency[last_q]:
                if neighbor in visited:
                    continue

                sq_e = sq_errors.get(neighbor, 0.001)
                ro_e = readout_errors.get(neighbor, 0.01)
                cz_e = cz_errors.get(
                    (last_q, neighbor),
                    cz_errors.get((neighbor, last_q), 0.01),
                )
                step_cost = (
                    -np.log(max(1.0 - sq_e, 1e-6))
                    - np.log(max(1.0 - ro_e, 1e-6))
                    + cz_counts.get(step_idx, 1)
                    * (-np.log(max(1.0 - cz_e, 1e-6)))
                )
                new_visited = set(visited)
                new_visited.add(neighbor)
                heapq.heappush(
                    priority_queue,
                    (
                        cost + step_cost,
                        path_counter,
                        path + (neighbor,),
                        new_visited,
                    ),
                )
                path_counter += 1

        return None, float("inf")

    # --------------------------------------------------------------------------
    # Mode 3: Geometric snake fallback
    # --------------------------------------------------------------------------

    def _geometric_snake(
        self,
        qubits,
        adjacency,
        N,
        sq_errors,
        cz_errors,
        readout_errors,
        cz_counts,
    ):
        best_path = None
        best_cost = float("inf")

        def search_snake(path, visited):
            nonlocal best_path, best_cost

            if len(path) == N:
                cost = self.compute_path_cost(
                    path,
                    sq_errors,
                    cz_errors,
                    readout_errors,
                    cz_counts,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_path = list(path)
                return

            last_q = path[-1]
            for neighbor in adjacency[last_q]:
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                path.append(neighbor)
                search_snake(path, visited)
                path.pop()
                visited.remove(neighbor)

        for qubit in qubits:
            search_snake([qubit], {qubit})

        return best_path, best_cost


# ==============================================================================
# Step 5. Build the full U_{N,n} Dicke-state circuit.
# ==============================================================================

def build_unrouted_u_nk(
    path_qubits: list[cirq.Qid],
    N: int,
    n: int,
) -> cirq.Circuit:
    """Construct U_{N,n} before enforcing hardware connectivity."""
    circuit = cirq.Circuit()

    for qubit in path_qubits[N - n :]:
        circuit.append(native_x(qubit))

    for m in range(N, n, -1):
        circuit.append(build_scs_block(path_qubits, m=m, k=n))

    for m in range(n, 1, -1):
        circuit.append(build_scs_block(path_qubits, m=m, k=m - 1))

    return circuit


def build_full_u_nk(
    path_qubits: list[cirq.Qid],
    N: int,
    n: int,
) -> cirq.Circuit:
    """Construct and route U_{N,n} over the selected active path.

    RouteCQC tracks the evolving logical-to-physical permutation, avoiding
    the depth cost of undoing every remote interaction. Dicke states are
    permutation-invariant, so its final logical permutation needs no repair.
    """
    unrouted = build_unrouted_u_nk(path_qubits, N, n)
    path_graph = nx.Graph()
    path_graph.add_nodes_from(path_qubits)
    path_graph.add_edges_from(zip(path_qubits, path_qubits[1:]))
    initial_mapper = cirq.HardCodedInitialMapper(
        {qubit: qubit for qubit in path_qubits}
    )
    routed = cirq.RouteCQC(path_graph)(
        unrouted,
        initial_mapper=initial_mapper,
    )
    return cirq.optimize_for_target_gateset(
        routed,
        gateset=cirq.CZTargetGateset(allow_partial_czs=False),
    )


# ==============================================================================
# Challenge entry point
# ==============================================================================


def create_dicke_state(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict[
        tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]
    ],
    sq_pauli_error_rate: dict[cirq.GridQubit, tuple[float, float]],
    zero_errors: dict[cirq.GridQubit, tuple[float, float]],
    one_errors: dict[cirq.GridQubit, tuple[float, float]],
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Create a hardware-native circuit for an (N, n) Dicke state on Willow105.

    Returns:
        The compiled circuit and the physical GridQubits it uses.
    """
    # Parse error-rate dictionaries containing (error_rate, uncertainty) tuples.
    cz_errors = {
        pair: error_rate
        for pair, (error_rate, _) in cz_pauli_error_rate.items()
    }
    sq_errors = {
        qubit: error_rate
        for qubit, (error_rate, _) in sq_pauli_error_rate.items()
    }
    readout_errors = {
        qubit: (
            zero_errors.get(qubit, (0.01, 0))[0]
            + one_errors.get(qubit, (0.01, 0))[0]
        )
        / 2.0
        for qubit in sq_pauli_error_rate
    }

    active_qubits = list(sq_pauli_error_rate)
    active_couplers = list(cz_pauli_error_rate)

    # Find the best connected one-dimensional path of the requested length.
    pathfinder = MultiModePathfinder(mode="branch_and_bound")
    best_path, _ = pathfinder.find_path(
        qubits=active_qubits,
        couplers=active_couplers,
        N=num_qubits,
        n=num_excitations,
        sq_errors=sq_errors,
        cz_errors=cz_errors,
        readout_errors=readout_errors,
    )
    if best_path is None:
        raise NotImplementedError(
            f"No valid connected 1D path found for N={num_qubits}"
        )

    # Build U_{N,n} along the selected physical path.
    circuit = build_full_u_nk(
        path_qubits=best_path,
        N=num_qubits,
        n=num_excitations,
    )

    circuit = cirq.drop_negligible_operations(circuit, atol=1e-8)
    return circuit, set(best_path)


# ==============================================================================
# State-vector validation helpers
# ==============================================================================


def create_dicke_statevector(
    num_qubits: int,
    num_excitations: int,
) -> np.ndarray:
    psi = np.zeros(2**num_qubits)
    normalization = 1 / np.sqrt(math.comb(num_qubits, num_excitations))
    qubits = range(num_qubits)
    for excited_qubits in itertools.combinations(qubits, num_excitations):
        idx = sum(2**qubit_number for qubit_number in excited_qubits)
        psi[idx] = normalization
    return psi


def test_circuit(
    circuit: cirq.Circuit,
    num_excitations: int,
    qubits: set[cirq.GridQubit],
    sampler=cirq.Simulator(),
):
    all_qubits = circuit.all_qubits()
    ancilla_qubits = all_qubits.difference(qubits)
    ordered_qubits = sorted(qubits) + sorted(ancilla_qubits)
    keep_indices = range(len(qubits))
    num_qubits = len(qubits)
    psi_circuit_full = sampler.simulate(
        circuit,
        qubit_order=ordered_qubits,
    ).final_state_vector
    psi_circuit = (
        cirq.sub_state_vector(
            state_vector=psi_circuit_full,
            keep_indices=keep_indices,
        )
        if len(all_qubits) > len(qubits)
        else psi_circuit_full
    )
    expected = create_dicke_statevector(num_qubits, num_excitations)
    assert np.isclose(np.abs(psi_circuit @ expected) ** 2, 1.0)


def calculate_state_fidelity(
    circuit: cirq.Circuit,
    num_excitations: int,
    qubits: set[cirq.GridQubit],
    *,
    dtype: type[np.complexfloating] = np.complex64,
) -> float:
    """Calculate Dicke-state fidelity using the judge's qubit ordering."""
    all_qubits = circuit.all_qubits()
    ancilla_qubits = all_qubits.difference(qubits)
    ordered_qubits = sorted(qubits) + sorted(ancilla_qubits)
    simulator = cirq.Simulator(dtype=dtype)
    full_state = simulator.simulate(
        circuit,
        qubit_order=ordered_qubits,
    ).final_state_vector
    state = (
        cirq.sub_state_vector(
            state_vector=full_state,
            keep_indices=range(len(qubits)),
        )
        if ancilla_qubits
        else full_state
    )
    ideal_state = create_dicke_statevector(
        len(qubits),
        num_excitations,
    )
    return float(np.abs(state @ ideal_state) ** 2)


def validate_submission_contract(
    circuit: cirq.Circuit,
    qubits: set[cirq.GridQubit],
    num_qubits: int,
    construction_seconds: float,
    cz_pauli_error_rate: dict[
        tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]
    ],
    sq_pauli_error_rate: dict[cirq.GridQubit, tuple[float, float]],
) -> list[str]:
    """Return violations of the standard challenge circuit contract."""
    violations = []
    active_qubits = set(sq_pauli_error_rate)
    active_couplers = {
        frozenset(pair) for pair in cz_pauli_error_rate
    }

    if len(qubits) != num_qubits:
        violations.append(
            f"returned {len(qubits)} result qubits, expected {num_qubits}"
        )
    if not qubits.issubset(active_qubits):
        violations.append("result includes unavailable single-qubit sites")
    if len(circuit) > 1750:
        violations.append(f"circuit has {len(circuit)} moments (limit 1750)")
    if construction_seconds >= 60:
        violations.append(
            f"construction took {construction_seconds:.3f}s (limit <60s)"
        )

    for operation in circuit.all_operations():
        gate = operation.gate
        if isinstance(gate, cirq.PhasedXZGate):
            if operation.qubits[0] not in active_qubits:
                violations.append(
                    f"single-qubit gate uses unavailable {operation.qubits[0]}"
                )
        elif gate == cirq.CZ:
            pair = frozenset(operation.qubits)
            if pair not in active_couplers:
                violations.append(
                    f"CZ uses unavailable coupler {operation.qubits}"
                )
        elif isinstance(gate, cirq.ResetChannel):
            if any(qubit not in active_qubits for qubit in operation.qubits):
                violations.append(
                    f"reset uses unavailable qubits {operation.qubits}"
                )
        else:
            violations.append(f"unsupported operation {operation!r}")

    try:
        cirq_google.Willow105.validate_circuit(circuit)
    except ValueError as error:
        violations.append(f"Willow105 validation failed: {error}")

    return list(dict.fromkeys(violations))


# ==============================================================================
# Competition scoring
# ==============================================================================


def check_z_bitstring_distribution(
    bitstrings: np.ndarray,
    num_excitations: int,
) -> float:
    """Measure how closely Z-basis samples match the ideal distribution."""
    n_shots, n_qubits = bitstrings.shape
    measured_excitations = np.sum(bitstrings, axis=1)
    fraction_in_subspace = np.mean(
        measured_excitations == num_excitations
    )

    chi2 = 0.0
    ideal_probability = 1 / math.comb(n_qubits, num_excitations)
    for indices in itertools.combinations(
        range(n_qubits),
        num_excitations,
    ):
        ideal_state = np.zeros(n_qubits)
        ideal_state[np.array(indices)] = 1
        match_count = np.sum(np.all(bitstrings == ideal_state, axis=1))
        expected_count = (
            n_shots * fraction_in_subspace * ideal_probability
        )
        chi2 += (match_count - expected_count) ** 2 / expected_count

    return ideal_probability * chi2 / fraction_in_subspace


def get_correlator_average(
    bitstrings: np.ndarray,
    correlator_size: int,
) -> float:
    """Return the average correlator of the requested weight."""
    correlator_sum = 0.0
    _, n_qubits = bitstrings.shape
    for indices in itertools.combinations(
        range(n_qubits),
        correlator_size,
    ):
        parity = np.sum(bitstrings[:, indices], axis=1) % 2
        correlator_sum += 1 - 2 * np.mean(parity)
    return correlator_sum / math.comb(n_qubits, correlator_size)


def check_x_bitstring_distribution(
    bitstrings: np.ndarray,
    num_excitations: int,
    max_correlator_size: int,
) -> float:
    """Measure how closely X-basis samples match ideal correlations."""

    def safe_combination(n: int, k: int) -> int:
        return math.comb(n, k) if 0 <= k <= n else 0

    _, n_qubits = bitstrings.shape
    score = 0.0
    for correlator_size in range(1, max_correlator_size + 1):
        expected_correlator = (
            safe_combination(
                correlator_size,
                correlator_size // 2,
            )
            * safe_combination(
                n_qubits - correlator_size,
                num_excitations - correlator_size // 2,
            )
            / safe_combination(n_qubits, num_excitations)
            if correlator_size % 2 == 0
            else 0.0
        )
        measured_correlator = get_correlator_average(
            bitstrings,
            correlator_size,
        )
        difference = expected_correlator - measured_correlator
        score += (
            difference**2 / expected_correlator**2
            if expected_correlator != 0
            else difference**2
        )

    return score


def overall_score(
    z_bitstrings: np.ndarray,
    x_bitstrings: np.ndarray,
    num_excitations: int,
    max_correlator_size: int,
) -> float:
    """Calculate the competition score from Z- and X-basis samples."""
    return 1 / (
        check_z_bitstring_distribution(
            z_bitstrings,
            num_excitations,
        )
        + check_x_bitstring_distribution(
            x_bitstrings,
            num_excitations,
            max_correlator_size,
        )
    )


def estimate_score_locally(
    circuit: cirq.Circuit,
    qubits: set[cirq.GridQubit],
    num_excitations: int,
    repetitions: int = 10_000,
) -> float:
    """Estimate the competition score with noiseless local sampling."""
    ordered_qubits = sorted(qubits)
    simulator = cirq.Simulator()

    z_circuit = circuit.copy()
    z_circuit.append(cirq.measure(*ordered_qubits, key="result"))
    z_bitstrings = simulator.run(
        z_circuit,
        repetitions=repetitions,
    ).measurements["result"]

    x_circuit = circuit.copy()
    x_circuit.append(cirq.H.on_each(*ordered_qubits))
    x_circuit.append(cirq.measure(*ordered_qubits, key="result"))
    x_bitstrings = simulator.run(
        x_circuit,
        repetitions=repetitions,
    ).measurements["result"]

    return overall_score(
        z_bitstrings=z_bitstrings,
        x_bitstrings=x_bitstrings,
        num_excitations=num_excitations,
        max_correlator_size=len(ordered_qubits),
    )


# ==============================================================================
# Synthetic hardware-error generation
# ==============================================================================


def pick_single_qubit_dropouts(
    dropout_rate: float,
    rng: np.random.Generator = np.random.default_rng(),
) -> set[cirq.GridQubit]:
    """Select simulated broken qubits at the requested dropout rate."""
    all_qubits = np.array(list(cirq_google.Willow105.metadata.qubit_set))
    is_dropout = rng.random(len(all_qubits)) < dropout_rate
    return set(all_qubits[is_dropout])


def generate_qubit_error_rates(
    median: float,
    spread: float,
    dropouts: set[cirq.GridQubit],
    uncertainty: float,
    rng: np.random.Generator = np.random.default_rng(),
) -> dict[cirq.GridQubit, tuple[float, float]]:
    """Generate synthetic single-qubit error rates and uncertainties."""
    all_qubits = np.array(list(cirq_google.Willow105.metadata.qubit_set))
    is_working = np.array([qubit not in dropouts for qubit in all_qubits])
    working_qubits = all_qubits[is_working]
    error_rates = rng.lognormal(np.log(median), spread, len(working_qubits))
    uncertainties = uncertainty * np.ones(len(working_qubits))
    return dict(zip(working_qubits, zip(error_rates, uncertainties)))


def generate_two_qubit_pauli_error_rates(
    median: float = 2.5e-3,
    spread: float = 0.5,
    dropout_rate: float = 0.01,
    uncertainty: float = 5e-4,
    rng: np.random.Generator = np.random.default_rng(),
    single_qubit_dropouts: set[cirq.GridQubit] | None = None,
) -> dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]]:
    """Generate synthetic two-qubit Pauli error rates and uncertainties."""
    if single_qubit_dropouts is None:
        single_qubit_dropouts = set()

    all_pairs = np.array(list(cirq_google.Willow105.metadata.qubit_pairs))
    no_qubit_dropouts = np.array(
        [
            all(qubit not in single_qubit_dropouts for qubit in pair)
            for pair in all_pairs
        ]
    )
    is_working = (
        rng.random(len(all_pairs)) > dropout_rate
    ) * no_qubit_dropouts
    working_pairs = [tuple(pair) for pair in all_pairs[is_working]]
    error_rates = rng.lognormal(np.log(median), spread, len(working_pairs))
    uncertainties = uncertainty * np.ones(len(working_pairs))
    return dict(zip(working_pairs, zip(error_rates, uncertainties)))


single_qubit_dropouts = pick_single_qubit_dropouts(0.01)
sq_pauli_error_rate = generate_qubit_error_rates(
    4e-4,
    0.5,
    single_qubit_dropouts,
    2e-4,
)
cz_pauli_error_rate = generate_two_qubit_pauli_error_rates(
    single_qubit_dropouts=single_qubit_dropouts,
)
zero_errors = generate_qubit_error_rates(
    5e-3,
    0.5,
    single_qubit_dropouts,
    2e-4,
)
one_errors = generate_qubit_error_rates(
    1e-2,
    0.5,
    single_qubit_dropouts,
    3e-4,
)


def evaluate_submission_case(
    num_qubits: int,
    num_excitations: int,
) -> dict:
    """Build and evaluate one case against the standard challenge contract."""
    start_time = time.perf_counter()
    circuit, qubits = create_dicke_state(
        num_qubits=num_qubits,
        num_excitations=num_excitations,
        cz_pauli_error_rate=cz_pauli_error_rate,
        sq_pauli_error_rate=sq_pauli_error_rate,
        zero_errors=zero_errors,
        one_errors=one_errors,
    )
    construction_seconds = time.perf_counter() - start_time

    judge_fidelity = calculate_state_fidelity(
        circuit,
        num_excitations,
        qubits,
    )
    strict_fidelity = calculate_state_fidelity(
        circuit,
        num_excitations,
        qubits,
        dtype=np.complex128,
    )
    violations = validate_submission_contract(
        circuit=circuit,
        qubits=qubits,
        num_qubits=num_qubits,
        construction_seconds=construction_seconds,
        cz_pauli_error_rate=cz_pauli_error_rate,
        sq_pauli_error_rate=sq_pauli_error_rate,
    )
    judge_passed = bool(np.isclose(judge_fidelity, 1.0))
    strict_passed = bool(
        np.isclose(
            strict_fidelity,
            1.0,
            rtol=0.0,
            atol=1e-8,
        )
    )

    return {
        "num_qubits": num_qubits,
        "num_excitations": num_excitations,
        "judge_fidelity": judge_fidelity,
        "strict_fidelity": strict_fidelity,
        "judge_passed": judge_passed,
        "strict_passed": strict_passed,
        "violations": violations,
        "moments": len(circuit),
        "operations": sum(1 for _ in circuit.all_operations()),
        "construction_seconds": construction_seconds,
        "passed": judge_passed and strict_passed and not violations,
    }


def print_fidelity_report(
    min_num_qubits: int = 2,
    max_num_qubits: int = 15,
) -> None:
    """Print full judge-contract results for every valid (N, n) pair."""
    results = []

    print("DuQuantum judge-compatibility report")
    print("=" * 110)
    for num_qubits in range(min_num_qubits, max_num_qubits + 1):
        for num_excitations in range(1, num_qubits // 2 + 1):
            result = evaluate_submission_case(
                num_qubits,
                num_excitations,
            )
            results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(
                f"N={num_qubits:2d} n={num_excitations:2d} "
                f"judge={result['judge_fidelity']:.9f} "
                f"strict={result['strict_fidelity']:.12f} "
                f"moments={result['moments']:4d} "
                f"ops={result['operations']:5d} "
                f"build={result['construction_seconds']:7.3f}s "
                f"{status}"
            )
            for violation in result["violations"]:
                print(f"    - {violation}")

    passed_count = sum(result["passed"] for result in results)
    max_moments = max(result["moments"] for result in results)
    max_runtime = max(
        result["construction_seconds"] for result in results
    )
    print("=" * 110)
    print(f"Cases passed: {passed_count}/{len(results)}")
    print(f"Maximum circuit depth: {max_moments}/1750 moments")
    print(f"Maximum construction time: {max_runtime:.3f}/60 seconds")


if __name__ == "__main__":
    print_fidelity_report()
