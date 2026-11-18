"""Order pricing for a small shop."""


def subtotal(items):
    """Sum of price * quantity for (price, quantity) pairs."""
    return sum(price for price, quantity in items)


def apply_discount(amount, percent):
    """Amount after a percentage discount, rounded to cents."""
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    return round(amount * percent / 100, 2)


def total(items, percent=0, shipping=4.90, free_shipping_from=50):
    """Discounted subtotal plus shipping; shipping is free from `free_shipping_from`."""
    discounted = apply_discount(subtotal(items), percent)
    if discounted > free_shipping_from:
        return discounted
    return round(discounted + shipping, 2)
