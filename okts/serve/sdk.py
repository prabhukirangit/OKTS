"""Layer 4 / integration mode 2 — in-process library / SDK.

For agents not speaking MCP: import OKTS, register ``search_tools`` /
``load_tool`` / ``call_tool`` as native functions in the framework's own tool
list. No process boundary, no protocol — just three plain callables.

Security note: the callables returned here are handed to an AGENT framework,
which typically builds each tool's schema by introspecting the callable's
signature. ``OKTSService.call_tool`` has a host-only, keyword-only ``scope``
parameter (used to satisfy pre-dispatch policies, e.g. a side-effect
confirmation gate). Exposing the raw bound method would let a signature-
introspecting framework surface ``scope`` to the model, and the agent could
self-authorize a gated call. So this module wraps ``call_tool`` in a callable
whose signature is exactly ``(id, args)`` — ``scope`` is structurally
unreachable by the agent. A host that needs a fixed scope for the session
passes it to :func:`build_sdk_tools`, where it is baked into the wrapper and
never model-controllable.
"""

from __future__ import annotations

from typing import Any, Callable

from okts.core.model import Bundle
from okts.core.protocols import Dispatcher, Retriever
from okts.serve.service import OKTSService

#: The shape every framework adapter registers: three plain callables keyed by
#: their public tool name, forever exactly these three (invariant #1).
SdkTools = dict[str, Callable[..., Any]]


def build_sdk_tools(
    service: OKTSService,
    *,
    scope: dict[str, Any] | None = None,
) -> SdkTools:
    """Return the three meta-tools as plain callables bound to ``service``.

    Usage in any framework::

        service = OKTSService(bundle, retriever, dispatcher)
        tools = build_sdk_tools(service)
        my_framework.register_tool(tools["search_tools"])
        my_framework.register_tool(tools["load_tool"])
        my_framework.register_tool(tools["call_tool"])

    The public surface is exactly three tools, forever (invariant #1), so this
    dict has exactly three entries.

    ``scope`` is optional HOST context (e.g. ``{"confirmed": True}`` to satisfy a
    side-effect gate for a trusted, non-interactive session). It is baked into
    the ``call_tool`` wrapper here — the agent-facing callable's signature is
    only ``(id, args)``, so the agent can never set or override ``scope`` and
    cannot self-authorize a policy-gated call.
    """

    def call_tool(id: str, args: dict[str, Any] | None = None) -> Any:
        """Validate args against the tool's input_schema and dispatch the call."""
        return service.call_tool(id, args, scope=scope)

    return {
        "search_tools": service.search_tools,
        "load_tool": service.load_tool,
        "call_tool": call_tool,
    }


def build_async_sdk_tools(
    service: OKTSService,
    *,
    scope: dict[str, Any] | None = None,
) -> SdkTools:
    """Async counterpart to :func:`build_sdk_tools`: ``call_tool`` is an async
    wrapper over :meth:`OKTSService.acall_tool`, for frameworks that await tool
    calls and need native async targets (``invocation: async``).

    Same security property: the agent-facing ``call_tool`` signature is only
    ``(id, args)`` — ``scope`` is host-bound here and unreachable by the agent.
    Register these in place of the sync set; the public surface is still exactly
    the same three tools (``acall_tool`` is the async variant of phase 3, not a
    fourth tool).
    """

    async def call_tool(id: str, args: dict[str, Any] | None = None) -> Any:
        """Validate args against the tool's input_schema and dispatch the call."""
        return await service.acall_tool(id, args, scope=scope)

    return {
        "search_tools": service.search_tools,
        "load_tool": service.load_tool,
        "call_tool": call_tool,
    }


def sdk_tools(bundle: Bundle, retriever: Retriever, dispatcher: Dispatcher) -> SdkTools:
    """One-liner convenience: build an :class:`OKTSService` and immediately
    return its three callables, for callers who don't need the service object
    itself. Equivalent to ``build_sdk_tools(OKTSService(bundle, retriever, dispatcher))``.
    """
    return build_sdk_tools(OKTSService(bundle=bundle, retriever=retriever, dispatcher=dispatcher))
