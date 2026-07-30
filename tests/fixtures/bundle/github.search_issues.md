---
type: tool
id: github.search_issues
title: Search GitHub Issues
description: Full-text search for issues across one or many repositories.
tags: [github, issues, search, find, query, read]
input_schema:
  type: object
  required: [query]
  properties:
    query: {type: string, description: "GitHub search qualifiers, e.g. 'repo:o/n is:open label:bug'"}
    sort: {type: string, enum: [comments, created, updated]}
    per_page: {type: integer}
interface: mcp
target: github-mcp
side_effects: read
alternatives: [./github.list_issues.md, ./github.find_issues_by_label.md]
timestamp: 2026-07-30T00:00:00Z
version: "1.0"
---

Use this for a **free-text or qualifier search** across issues, possibly spanning
repos. If you only want every issue in one known repo, `list_issues` is cheaper;
if you're filtering purely by a label, `find_issues_by_label` is more direct.
Near-duplicate warning: don't confuse with `list_issues`.
