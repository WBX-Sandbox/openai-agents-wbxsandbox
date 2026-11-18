import pytest
from pricing import apply_discount, subtotal, total


def test_subtotal_uses_quantity():
    assert subtotal([(2.50, 4), (10.00, 1)]) == 20.00


def test_discount():
    assert apply_discount(100, 15) == 85.00
    assert apply_discount(19.99, 0) == 19.99


def test_discount_range():
    with pytest.raises(ValueError):
        apply_discount(10, 120)


def test_free_shipping_threshold_is_inclusive():
    assert total([(25.00, 2)]) == 50.00


def test_shipping_added_below_threshold():
    assert total([(10.00, 1)]) == 14.90
