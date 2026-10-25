# STATIC WILLOW105 TOPOLOGY

This run certifies the partition and the SWAP plan. It does not insert Divide rotations,
so it is not yet a Dicke-state circuit and it was not submitted to hardware.

- Device: `Willow105`
- Qubits: 105
- Edges: 182
- Connected: True
- Diameter: 14
- Weight k: 3
- Leaf factor: 2 (a region of size <= 6 is a leaf)
- Partition: deterministic guillotine. Score is lexicographic:
  max child size, max bounding-box span, minus cut edges, staging turns, path span, axis, coordinate.
- A cut is kept only when both sides are connected, both sides have at least k qubits,
  and a 2k staging path is constructed across one device edge.
- Routing: move the k-qubit unary anchor along a corridor inside its region,
  then odd-even transposition. Sibling regions are merged by taking the longer schedule.
- Result: PASS

## Metrics

- n: 105
- k: 3
- partition_depth: 5
- internal_nodes: 21
- leaves: 22
- min_child_size: 3
- max_child_size: 59
- max_staging_path_length: 6
- max_corridor_length: 13
- mean_corridor_length: 6.857142857142857
- parallel_routing_depth: 37
- swap_count: 164
- routes: 14
- failed_routes: 0
- structural_benchmark: 30.387849050834898
- routing_depth_over_benchmark: 1.2175919374255098
