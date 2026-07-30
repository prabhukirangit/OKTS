---
type: tool
id: slack.send_message
title: Send Slack Message
description: Post a message to a Slack channel or direct message.
tags: [slack, chat, message, send, post, notify, write]
input_schema:
  type: object
  required: [channel, text]
  properties:
    channel: {type: string, description: "channel id or name"}
    text: {type: string}
    thread_ts: {type: string, description: "reply in a thread"}
interface: mcp
target: slack-mcp
auth: slack_oauth
side_effects: write
alternatives: [./slack.update_message.md]
timestamp: 2026-07-30T00:00:00Z
version: "1.0"
---

Use this to **post** a new message to a channel or DM. Synonyms: notify, ping,
announce, message someone. To edit an already-sent message use `update_message`.
