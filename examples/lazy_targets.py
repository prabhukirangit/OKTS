"""Example — lazy target initialization for OKTS dispatch.

OKTS core deliberately never opens upstream connections on its own: building a
bundle is fully offline, and a ``Dispatcher`` holds *caller-wired* ``targets``
and raises ``NotConfiguredError`` rather than connecting (see
``okts/serve/dispatch.py``). So the connection lifecycle is yours — which means
that when you proxy 20+ MCP servers you don't want to open all 20 sessions at
startup just so ``search_tools`` works. You want each upstream to connect **on
first ``call_tool``** and stay cached for the process.

That's a caller-side pattern, not an OKTS abstraction. This file is the
reference implementation: :class:`LazyConnectionFactory` (connect-once + cache)
and two thin adapters matching the two target shapes the dispatchers expect —
an object with ``call_tool(name, args)`` for ``McpDispatcher`` and a callable
``(concept, args, credential)`` for ``HttpDispatcher``.

Run it::

    python examples/lazy_targets.py

It's fully offline: the ``connect`` callables build dummy backends and count how
many times they fire, proving each upstream connects exactly once no matter how
many calls are dispatched.
"""

from __future__ import annotations

from typing import Any, Callable

from okts.core.model import Bundle, Interface, OKTConcept
from okts.serve.dispatch import DispatcherRegistry, HttpDispatcher, McpDispatcher
from okts.serve.mcp_server import NaiveFallbackRetriever
from okts.serve.service import OKTSService


class LazyConnectionFactory:
    """Wrap a ``connect`` callable; run it once, on first :meth:`get`, then cache
    the backend for the life of the process.

    Thread-safety is intentionally omitted for clarity — add a lock if your
    dispatcher is driven from multiple threads. ``reset`` drops the cached
    backend so the next :meth:`get` reconnects (useful after a dropped session).
    """

    def __init__(self, connect: Callable[[], Any]) -> None:
        self._connect = connect
        self._backend: Any = None

    @property
    def connected(self) -> bool:
        return self._backend is not None

    def get(self) -> Any:
        if self._backend is None:
            self._backend = self._connect()
        return self._backend

    def reset(self) -> None:
        self._backend = None


class LazyMcpTarget:
    """``McpDispatcher`` target that connects its MCP session on first use.

    ``McpDispatcher`` expects ``targets[server]`` to expose
    ``call_tool(name, args)``. This wrapper satisfies that shape while deferring
    the (expensive) session connect until the first tool call. ``connect`` must
    return an object that itself has ``call_tool(name, args)`` — e.g. a helper
    that opens a stdio ``ClientSession``.
    """

    def __init__(self, connect: Callable[[], Any]) -> None:
        self.factory = LazyConnectionFactory(connect)

    def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        return self.factory.get().call_tool(name, args)


def lazy_http_client(connect: Callable[[], Callable[..., Any]]) -> Callable[..., Any]:
    """Build an ``HttpDispatcher`` target callable ``(concept, args, credential)``
    that constructs its underlying HTTP client once, on first request.

    The returned callable carries a ``.factory`` attribute so a test/host can
    inspect whether it has connected yet."""
    factory = LazyConnectionFactory(connect)

    def _call(concept: OKTConcept, args: dict[str, Any], credential: str | None) -> Any:
        client = factory.get()
        return client(concept, args, credential)

    _call.factory = factory  # type: ignore[attr-defined]
    return _call


# ---------------------------------------------------------------------------
# offline demo: prove each upstream connects exactly once
# ---------------------------------------------------------------------------


def _demo_service() -> tuple[OKTSService, dict[str, int]]:
    """One mcp tool + one http tool, each behind a lazy target that counts
    connects. Returns the service and the connect-counter dict."""
    connects = {"mcp": 0, "http": 0}

    class _DummySession:
        def call_tool(self, name: str, args: dict[str, Any]) -> Any:
            return {"tool": name, "args": args, "via": "lazy-mcp-session"}

    def connect_mcp() -> _DummySession:
        connects["mcp"] += 1
        return _DummySession()

    def connect_http() -> Callable[..., Any]:
        connects["http"] += 1

        def _client(concept: OKTConcept, args: dict[str, Any], credential: str | None) -> Any:
            return {"tool": concept.id, "args": args, "via": "lazy-http-client"}

        return _client

    bundle = Bundle()
    bundle.add(OKTConcept(
        id="calc.add", title="Add", description="Add two numbers.",
        input_schema={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
        interface=Interface.MCP, target="calc",
    ))
    bundle.add(OKTConcept(
        id="demo.create", title="Create", description="Create a thing over HTTP.",
        input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        interface=Interface.HTTP, target="demo-api",
    ))

    registry = DispatcherRegistry()
    registry.register(Interface.MCP, McpDispatcher(targets={"calc": LazyMcpTarget(connect_mcp)}))
    registry.register(Interface.HTTP, HttpDispatcher(targets={"demo-api": lazy_http_client(connect_http)}))

    return OKTSService(bundle, NaiveFallbackRetriever(), registry), connects


def main() -> None:
    service, connects = _demo_service()
    print("before any call_tool — connects:", connects)

    for i in range(3):
        service.call_tool("calc.add", {"a": i, "b": 1})
    for i in range(2):
        service.call_tool("demo.create", {"name": f"thing-{i}"})

    print("after 3 mcp + 2 http calls — connects:", connects)
    assert connects == {"mcp": 1, "http": 1}, "each upstream should connect exactly once"
    print("OK — each upstream connected lazily, exactly once, and was reused.")


if __name__ == "__main__":
    main()
