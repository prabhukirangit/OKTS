---
type: tool
id: github.create_issue
title: Create GitHub Issue
description: Open a new issue in a GitHub repository.
tags: [github, issues, create, write, ticket, bug]
input_schema:
  type: object
  required: [repo, title]
  properties:
    repo: {type: string, description: "owner/name"}
    title: {type: string}
    body: {type: string}
    labels: {type: array, items: {type: string}}
interface: mcp
target: github-mcp
side_effects: write
alternatives: [./github.update_issue.md]
composes_with: [./github.add_labels.md]
prerequisites: [./github.get_repo.md]
timestamp: 2026-07-30T00:00:00Z
version: "1.0"
---

Use this to **create** a brand-new issue. Do not use it to comment on or edit an
existing one — that's `update_issue`. Synonyms: file a ticket, open a bug, log a
task. Gotcha: labels must already exist in the repo or the call 422s.
