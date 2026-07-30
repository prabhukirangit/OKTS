---
type: tool
id: stripe.create_refund
title: Create Stripe Refund
description: Refund a previously created charge, fully or partially.
tags: [stripe, payments, refund, reverse, money, write]
input_schema:
  type: object
  required: [charge]
  properties:
    charge: {type: string, description: "the charge id to refund"}
    amount: {type: integer, description: "partial refund amount in cents; omit for full"}
interface: http
target: "POST https://api.stripe.com/v1/refunds"
auth: stripe_api_key
side_effects: write
alternatives: [./stripe.create_charge.md]
timestamp: 2026-07-30T00:00:00Z
version: "1.0"
---

Use this to **reverse** a charge, fully or in part. To take a payment in the
first place use `create_charge`.
