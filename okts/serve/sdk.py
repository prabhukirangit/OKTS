"""Layer 4 / integration mode 2 — in-process library / SDK.

For agents not speaking MCP: import OKTS, register ``search_tools`` /
``load_tool`` / ``call_tool`` as native functions in the framework's own tool
list. No process boundary, no protocol — just three plain callables.
"""

from __future__ import annotations

from typing import Any, Callable

from okts.core.model import Bundle
from okts.core.protocols import Dispatcher, Retriever
from okts.serve.service import OKTSService

#: The shape every framework adapter registers: three plain callables keyed by
#: their public tool name, forever exactly these three (invariant #1).
SdkTools = dict[str, Callable[..., Any]]


def build_sdk_tools(service: OKTSService) -> SdkTools:
    """Return the three meta-tools as plain callables bound to ``service``.

    Usage in any framework::

        service = OKTSService(bundle, retriever, dispatcher)
        tools = build_sdk_tools(service)
        my_framework.register_tool(tools["search_tools"])
        my_framework.register_tool(tools["load_tool"])
        my_framework.register_tool(tools["call_tool"])

    The public surface is exactly three tools, forever (invariant #1), so this
    dict has exactly three entries. ``call_tool`` here is the sync entry point;
    an async framework that wants to await an async target
    (``invocation: async``) natively can register ``service.acall_tool``
    directly in place of ``tools["call_tool"]`` — it's the same phase-3 tool,
    not a fourth one.
    """
    return {
        "search_tools": service.search_tools,
        "load_tool": service.load_tool,
        "call_tool": service.call_tool,
    }


def sdk_tools(bundle: Bundle, retriever: Retriever, dispatcher: Dispatcher) -> SdkTools:
    """One-liner convenience: build an :class:`OKTSService` and immediately
    return its three callables, for callers who don't need the service object
    itself. Equivalent to ``build_sdk_tools(OKTSService(bundle, retriever, dispatcher))``.
    """
    return build_sdk_tools(OKTSService(bundle=bundle, retriever=retriever, dispatcher=dispatcher))
