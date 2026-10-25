# Benchmarks

`benchmark.divide_benchmark` writes a new JSON file under `artifacts/` for the isolated Divide at each requested `k`. `benchmark.rectangle_benchmark` does the same for a rectangle. Files are stamped in UTC and are not overwritten.

Levels, matching the report:

- Logical: Divide blocks, snake-sort SWAP layers, logical moments of each stage.
- Hardware: moments and CZ count after `lower_to_willow_cz`.
- Critical path: each of Divide, routing, and the leaf is lowered on its own, parallel children contribute their maximum, and the root records which of those four pieces is largest. That sum is an attribution, not a second copy of the jointly lowered moment count. The jointly lowered count is `metrics.hardware.moments`.

Baseline A in the rectangle report is the 2019 SCS stair on the same snake, lowered by the same pass. Baseline C is the existing `willow_dicke.yuan_zhang` compiler; it is the right comparison for "did the 2025 construction get cheaper, or did we only use fewer qubits," and it is expensive at `k>=3` because of the bridge. It is not invoked automatically inside `rectangle_benchmark` for that reason. Run it separately on the same patch when a number is needed.

The `k=2` Divide regression is recorded in `DIVIDE.md` and in the JSON. It is not papered over by skipping the reference column.

Measured on the primary 7 by 8 Willow patch (methodology `yz2025-2`: displaced-only block permutation), jointly lowered moments versus the 2019 snake on the same patch and the same CZ pass:

| k | 2025 moments | 2025 CZ | 2019 snake moments | 2019 snake CZ | root Divide | root permutation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 275 | 482 | 221 | 110 | 5 | 19 |
| 2 | 1223 | 3018 | 2381 | 1190 | 165 | 97 |
| 3 | 2836 | 5539 | 4713 | 2356 | 567 | 97 |
| 4 | 5676 | 10890 | 7625 | 3812 | 1265 | 97 |
| 5 | 8367 | 14568 | 11093 | 5546 | 2329 | 103 |
| 8 | 28643 | 37792 | 24593 | 12296 | 8425 | 121 |
| 10 | 38987 | 48266 | 35933 | 17966 | 15649 | 133 |

The existing bridged rectangle compiler (`willow_dicke.yuan_zhang.verify_willow`) measures 285 moments at `k=1`, 3052 at `k=2`, and 11947 at `k=3` on this same patch, and its own verifier accepts each circuit. `Willow105.validate_circuit` accepts the new compiler at `k=1` and `k=3`. Higher `k` passed the same device-edge check inside routing and the grid-CZ assertion; they were not each sent through `validate_circuit` again.

At `k=3` the root Divide is 567 moments and the root block permutation is 97. The longer child is the rest of the critical path, because it contains the next Divide. Divide blocks on sibling regions share moments, so they are not summed. The moment advantage over the 2019 snake holds through `k=5` and reverses at `k=8` and `k=10`: one root Divide is then 8425 and 15649 moments, and several of those sit in series down the critical path. CZ count is higher than the snake at every measured `k`. `k=5` Divide equivalence passed (2329 moments, 1495 CZ). The bridged reference at `k=5` was not finished; at `k=4` it was already 4051 moments.
