---
type: tool
id: stripe.create_charge
title: Create Stripe Charge
description: Charge a payment source a given amount in a currency.
tags: [stripe, payments, charge, bill, money, write]
input_schema:
  type: object
  required: [amount, currency, source]
  properties:
    amount: {type: integer, description: "in the smallest currency unit, e.g. cents"}
    currency: {type: string, description: "ISO 4217, e.g. usd"}
    source: {type: string, description: "token or customer id"}
    description: {type: string}
interface: http
target: "POST https://api.stripe.com/v1/charges"
auth: stripe_api_key
side_effects: write
alternatives: [./stripe.create_refund.md]
timestamp: 2026-07-30T00:00:00Z
version: "1.0"
---

Use this to **bill** a card/source. Synonyms: take a payment, charge a customer.
To reverse a charge use `create_refund`. Amounts are in the smallest currency
unit (cents), a classic gotcha.
