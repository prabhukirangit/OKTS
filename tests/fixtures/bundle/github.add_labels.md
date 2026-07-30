---
type: tool
id: github.add_labels
title: Add Labels to Issue
description: Attach one or more existing labels to a GitHub issue.
tags: [github, issues, labels, write, tag]
input_schema:
  type: object
  required: [repo, number, labels]
  properties:
    repo: {type: string, description: "owner/name"}
    number: {type: integer}
    labels: {type: array, items: {type: string}}
interface: mcp
target: github-mcp
side_effects: write
composes_with: [./github.create_issue.md, ./github.update_issue.md]
timestamp: 2026-07-30T00:00:00Z
version: "1.0"
---

Use this to **tag** an existing issue with labels that already exist in the repo.
Composes with `create_issue`/`update_issue`. Gotcha: creating a label is a
separate operation; unknown labels 422.
