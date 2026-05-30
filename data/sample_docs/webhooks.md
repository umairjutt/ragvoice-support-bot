# Webhooks

## Verifying signatures

Every webhook request includes a `Stripe-Signature` header. Always verify it:

```python
event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
```

If verification fails, return HTTP 400 immediately. Do **not** process the event.

## Idempotency

Webhooks may be delivered more than once. Always use the `event.id` as an idempotency
key in your database (e.g. via `INSERT ... ON CONFLICT DO NOTHING` or Redis `SETNX`)
before doing side-effects like sending emails or shipping orders.

## Common events

- `payment_intent.succeeded` — funds captured, you can fulfill the order
- `payment_intent.payment_failed` — payment failed permanently; notify the user
- `charge.refunded` — a refund (full or partial) was processed
