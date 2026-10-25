# Geometry

Willow105 is read from `cirq_google.Willow105`. The primary patch is physical rows 2 through 9 and columns 4 through 10: 56 qubits, 48 horizontal edges, 49 vertical edges, 97 couplers, all required to exist on the device.

Theorem 14 wants `n1 <= n2`. That patch is 8 by 7 in the device's row-column order, so paper row `r` is placed on physical column `4+r` and paper column `c` on physical row `2+c`. The algorithm then sees a 7 by 8 grid. The shifted candidate, rows 3 through 10 and the same columns, is checked the same way.

`abstract_patch` builds the same kind of rectangle on fresh grid qubits for the 2 by 2, 2 by 3, 2 by 4, and 3 by 3 tests. If the caller passes the long side first, the axes are swapped so the paper convention still holds.
