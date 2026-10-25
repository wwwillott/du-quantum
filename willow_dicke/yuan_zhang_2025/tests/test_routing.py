"""Snake-sort certificates on a small rectangle."""

from willow_dicke.yuan_zhang_2025.geometry import abstract_patch
from willow_dicke.yuan_zhang_2025.permutation import route_block_exchange


def test_row_block_exchange_is_a_matching_and_lands():
    rectangle = abstract_patch(2, 6).rectangle()
    source = tuple(rectangle.rows[0][:2])
    destination = tuple(rectangle.rows[0][4:])
    certificate = route_block_exchange(
        rectangle, source, destination, None, region_id="row-exchange"
    )
    assert certificate.correct_permutation
    assert certificate.collision_free
    assert certificate.valid_edges
    for layer in certificate.swap_layers:
        used = set()
        for left, right in layer:
            assert left not in used and right not in used
            used.add(left)
            used.add(right)


def test_identity_block_has_no_swaps():
    rectangle = abstract_patch(2, 4).rectangle()
    block = tuple(rectangle.rows[0][:2])
    certificate = route_block_exchange(rectangle, block, block, None, region_id="identity")
    assert certificate.swap_layers == []
    assert certificate.correct_permutation
