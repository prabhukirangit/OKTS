---
type: tool
id: github.get_repo
title: Get GitHub Repository
description: Fetch metadata for a single GitHub repository.
tags: [github, repo, read, metadata]
input_schema:
  type: object
  required: [repo]
  properties:
    repo: {type: string, description: "owner/name"}
interface: mcp
target: github-mcp
side_effects: read
composes_with: [./github.create_issue.md]
timestamp: 2026-07-30T00:00:00Z
version: "1.0"
---

Use this to read a repo's metadata (default branch, visibility, topics). Often a
prerequisite before creating issues or PRs. Read-only.
