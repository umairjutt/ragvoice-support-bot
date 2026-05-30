# Payments — Quick Reference

## What is a PaymentIntent?

A PaymentIntent tracks the lifecycle of a customer's payment from creation through processing.
It is the modern, recommended way to accept card payments because it handles authentication,
3D Secure, retries, and multi-currency support automatically.

## How to create a PaymentIntent

```python
import stripe
stripe.api_key = "sk_test_..."

intent = stripe.PaymentIntents.create(
    amount=1099,                # in cents
    currency="usd",
    automatic_payment_methods={"enabled": True},
    metadata={"order_id": "ORD-123"},
)
print(intent.client_secret)
```

The returned `client_secret` should be passed to the front-end Stripe Elements
or Payment Sheet, which completes the payment on behalf of the user.

## Charges vs PaymentIntents

A **Charge** is the legacy primitive — a single immediate attempt at moving money.
A **PaymentIntent** is the modern primitive — a state machine that can include
multiple charge attempts, off-session retries, and Strong Customer Authentication (SCA).
For all new integrations you should use PaymentIntents.

## Refunding

```python
stripe.Refund.create(payment_intent="pi_xxx", amount=500)
```

Partial refunds are supported by passing `amount`. Omit `amount` to refund the full charge.
