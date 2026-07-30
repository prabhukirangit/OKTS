---
hierarchy:
  github/issues:
    - github.create_issue
    - github.update_issue
    - github.list_issues
    - github.search_issues
    - github.find_issues_by_label
    - github.add_labels
  github/repos:
    - github.get_repo
  slack/chat:
    - slack.send_message
    - slack.update_message
  stripe/payments:
    - stripe.create_charge
    - stripe.create_refund
---

Category hierarchy for the test bundle. Used by the retrieval hierarchy
prefilter to scope candidate concepts before ranking.
