---
type: tool
id: github.update_issue
title: Update GitHub Issue
description: Edit the title, body, state, or labels of an existing GitHub issue.
tags: [github, issues, update, edit, write, close, reopen]
input_schema:
  type: object
  required: [repo, number]
  properties:
    repo: {type: string, description: "owner/name"}
    number: {type: integer}
    title: {type: string}
    body: {type: string}
    state: {type: string, enum: [open, closed]}
    labels: {type: array, items: {type: string}}
interface: mcp
target: github-mcp
side_effects: write
alternatives: [./github.create_issue.md]
composes_with: [./github.add_labels.md]
timestamp: 2026-07-30T00:00:00Z
version: "1.0"
---

Use this to **edit** an issue that already exists — change its title/body, close
or reopen it, or swap labels. Do not use it to open a new issue (`create_issue`).
Synonyms: close a ticket, reopen a bug, edit an issue.
