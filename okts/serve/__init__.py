"""Layer 4 — serving: the three meta-tools the agent actually sees.

``search_tools`` (phase 1) · ``load_tool`` (phase 2) · ``call_tool`` (phase 3).
This is the ENTIRE public surface, forever (invariant #1). Ships as an MCP
server (mode 1), an in-process SDK (mode 2), and an HTTP sidecar (mode 3).

Depends only on the ``Retriever`` and ``Dispatcher`` protocols from
``okts.core`` — never on concrete index/adapter classes. Populated by Phase 1C.
"""
