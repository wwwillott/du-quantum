import heapq

import cirq
import numpy as np
import heapq


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
    """Decompose CCRy(2 * theta) on q_target using four CNOTs.

    Applies Ry(2 * theta) to q_target only when both controls are 1.
    """
    yield native_ry(q_target, theta / 2)
    yield native_cnot(q_ctrl2, q_target)
    yield native_ry(q_target, -theta / 2)
    yield native_cnot(q_ctrl1, q_target)
    yield native_ry(q_target, theta / 2)
    yield native_cnot(q_ctrl2, q_target)
    yield native_ry(q_target, -theta / 2)
    yield native_cnot(q_ctrl1, q_target)


def build_gate_ii(
    q_ctrl1: cirq.Qid,
    q_ctrl2: cirq.Qid,
    q_target: cirq.Qid,
    theta: float,
):
    """Build gate (ii)_l from Bärtschi and Eidenbenz Figure 3.

    Triggers for q_ctrl1=0, q_ctrl2=1, and q_target=1 on qubits
    (n-l, n-l+1, n). The surrounding CNOTs map 01 to 11 and rotate the
    target excitation.
    """
    yield native_cnot(q_ctrl2, q_ctrl1)
    yield native_cnot(q_ctrl1, q_target)
    yield build_ccry_core(q_ctrl1, q_ctrl2, q_target, theta)
    yield native_cnot(q_ctrl1, q_target)
    yield native_cnot(q_ctrl2, q_ctrl1)


# ==============================================================================
# Step 3. Create SCS blocks out of the SCS-subgates.
# ==============================================================================


def build_scs_block(path_qubits, m, k):
    """Construct SCS_{m, k} acting on path_qubits[:m].

    Args:
        path_qubits: Physical qubits arranged along a one-dimensional path.
        m: Current system-size layer, where m <= N.
        k: Excitation count, where k <= n.
    """
    sub_q = path_qubits[:m]

    # Gate (i): two-qubit Givens rotation on qubits m - 2 and m - 1.
    theta_i = np.arccos(1.0 / np.sqrt(m))
    yield build_gate_i(sub_q[m - 2], sub_q[m - 1], theta_i)

    # Gates (ii)_l: three-qubit CCRy Givens rotations.
    for l in range(2, min(k, m - 1) + 1):
        q_ctrl1 = sub_q[m - l - 1]
        q_ctrl2 = sub_q[m - l]
        q_target = sub_q[m - 1]
        theta_l = np.arccos(np.sqrt(l / m))
        yield build_gate_ii(q_ctrl1, q_ctrl2, q_target, theta_l)


# ==============================================================================
# Step 4. Find the optimal path present on the Google Willow grid..
# ==============================================================================

class MultiModePathfinder:
    """Find a low-error connected qubit chain on the Google Willow grid."""

    def __init__(self, mode: str = "branch_and_bound"):
        self.mode = mode

    def calculate_cz_counts(self, N: int, n: int) -> dict:
        """Count CZ gates on each relative coupler for a U_{N,n} circuit."""
        cz_counts = {i: 0 for i in range(N - 1)}

        # Traverse SCS blocks across sub-register sizes from N down to 2.
        for m in range(N, 1, -1):
            k_eff = min(n, m - 1)

            # Gate (i) contains two CNOTs, and therefore two CZs.
            cz_counts[m - 2] += 2

            # Account for gate (ii)_l blocks for l in 2..k_eff.
            for l in range(2, k_eff + 1):
                cz_counts[m - l - 1] += 2
                cz_counts[m - 2] += 2

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

def build_full_u_nk(
    path_qubits: list[cirq.GridQubit],
    N: int,
    n: int,
) -> cirq.Circuit:
    """Construct the complete U_{N,n} Dicke-state circuit along a 1D path.

    Args:
        path_qubits: Ordered physical GridQubits selected by the pathfinder.
        N: Total number of qubits.
        n: Number of excitations in the Dicke state.

    Returns:
        A circuit composed strictly of native CZ and PhasedXZGate operations.
    """
    circuit = cirq.Circuit()

    # Prepare the initial computational state |0...01...1>.
    for qubit in path_qubits[N - n :]:
        circuit.append(native_x(qubit))

    # First inductive phase: SCS_{m,n} blocks for m from N down to n + 1.
    for m in range(N, n, -1):
        scs_ops = list(build_scs_block(path_qubits, m=m, k=n))
        circuit.append(scs_ops)

    # Base-case phase: SCS_{m,m-1} blocks for m from n down to 2.
    for m in range(n, 1, -1):
        scs_ops = list(build_scs_block(path_qubits, m=m, k=m - 1))
        circuit.append(scs_ops)

    return circuit