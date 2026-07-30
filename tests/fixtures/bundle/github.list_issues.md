---
type: tool
id: github.list_issues
title: List GitHub Issues
description: List issues in a repository, optionally filtered by state or label.
tags: [github, issues, list, read, browse, enumerate]
input_schema:
  type: object
  required: [repo]
  properties:
    repo: {type: string, description: "owner/name"}
    state: {type: string, enum: [open, closed, all]}
    labels: {type: array, items: {type: string}}
    per_page: {type: integer}
interface: mcp
target: github-mcp
side_effects: read
alternatives: [./github.search_issues.md, ./github.find_issues_by_label.md]
timestamp: 2026-07-30T00:00:00Z
version: "1.0"
---

Use this to **enumerate** issues in one repo when you already know the repo and
just want the list, optionally narrowed by state or label. For a free-text query
across issues use `search_issues`; to fetch strictly by label use
`find_issues_by_label`. Read-only.
