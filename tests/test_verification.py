"""Independent Dicke-reference and verification tests."""

import numpy as np
import pytest

from dicke_compiler import DickeTarget, dicke_state_vector


@pytest.mark.parametrize(
    "target",
    [
        DickeTarget(2, 1),
        DickeTarget(4, 2),
        DickeTarget(8, 4),
    ],
)
def test_reference_state_is_normalized_and_weight_restricted(
    target: DickeTarget,
) -> None:
    state = dicke_state_vector(target)

    assert np.linalg.norm(state) == pytest.approx(1.0)
    populated = np.flatnonzero(abs(state) > 0)
    assert all(
        int(index).bit_count() == target.num_excitations
        for index in populated
    )
