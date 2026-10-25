# STATIC WILLOW105 TOPOLOGY

This run certifies the partition and the SWAP plan. It does not insert Divide rotations,
so it is not yet a Dicke-state circuit and it was not submitted to hardware.

- Device: `Willow105`
- Qubits: 105
- Edges: 182
- Connected: True
- Diameter: 14
- Weight k: 10
- Leaf factor: 2 (a region of size <= 20 is a leaf)
- Partition: deterministic guillotine. Score is lexicographic:
  max child size, max bounding-box span, minus cut edges, staging turns, path span, axis, coordinate.
- A cut is kept only when both sides are connected, both sides have at least k qubits,
  and a 2k staging path is constructed across one device edge.
- Routing: move the k-qubit unary anchor along a corridor inside its region,
  then odd-even transposition. Sibling regions are merged by taking the longer schedule.
- Result: PASS

## Metrics

- n: 105
- k: 10
- partition_depth: 3
- internal_nodes: 6
- leaves: 7
- min_child_size: 13
- max_child_size: 59
- max_staging_path_length: 20
- max_corridor_length: 23
- mean_corridor_length: 20.0
- parallel_routing_depth: 42
- swap_count: 470
- routes: 5
- failed_routes: 0
- structural_benchmark: 48.923174227787605
- routing_depth_over_benchmark: 0.8584888585611162
