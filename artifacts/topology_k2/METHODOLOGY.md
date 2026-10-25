# STATIC WILLOW105 TOPOLOGY

This run certifies the partition and the SWAP plan. It does not insert Divide rotations,
so it is not yet a Dicke-state circuit and it was not submitted to hardware.

- Device: `Willow105`
- Qubits: 105
- Edges: 182
- Connected: True
- Diameter: 14
- Weight k: 2
- Leaf factor: 2 (a region of size <= 4 is a leaf)
- Partition: deterministic guillotine. Score is lexicographic:
  max child size, max bounding-box span, minus cut edges, staging turns, path span, axis, coordinate.
- A cut is kept only when both sides are connected, both sides have at least k qubits,
  and a 2k staging path is constructed across one device edge.
- Routing: move the k-qubit unary anchor along a corridor inside its region,
  then odd-even transposition. Sibling regions are merged by taking the longer schedule.
- Result: PASS

## Metrics

- n: 105
- k: 2
- partition_depth: 6
- internal_nodes: 32
- leaves: 33
- min_child_size: 2
- max_child_size: 59
- max_staging_path_length: 4
- max_corridor_length: 14
- mean_corridor_length: 5.391304347826087
- parallel_routing_depth: 38
- swap_count: 141
- routes: 23
- failed_routes: 0
- structural_benchmark: 26.428491035332243
- routing_depth_over_benchmark: 1.4378422116191882
