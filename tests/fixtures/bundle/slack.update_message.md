---
type: tool
id: slack.update_message
title: Update Slack Message
description: Edit the text of a previously sent Slack message.
tags: [slack, chat, message, edit, update, write]
input_schema:
  type: object
  required: [channel, ts, text]
  properties:
    channel: {type: string}
    ts: {type: string, description: "timestamp of the message to edit"}
    text: {type: string}
interface: mcp
target: slack-mcp
auth: slack_oauth
side_effects: write
alternatives: [./slack.send_message.md]
timestamp: 2026-07-30T00:00:00Z
version: "1.0"
---

Use this to **edit** a message you already posted. To post a new one use
`send_message`.
