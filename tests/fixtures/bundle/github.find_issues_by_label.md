---
type: tool
id: github.find_issues_by_label
title: Find GitHub Issues by Label
description: Return issues in a repository carrying a specific label.
tags: [github, issues, label, filter, read, find]
input_schema:
  type: object
  required: [repo, label]
  properties:
    repo: {type: string, description: "owner/name"}
    label: {type: string}
interface: mcp
target: github-mcp
side_effects: read
alternatives: [./github.list_issues.md, ./github.search_issues.md]
timestamp: 2026-07-30T00:00:00Z
version: "1.0"
---

Use this when you want issues **for exactly one label** in a known repo — the
narrowest of the three issue-reading tools. For arbitrary filters use
`list_issues`; for free-text use `search_issues`. This is the tool-collision
trap the graph edges exist to disambiguate.
