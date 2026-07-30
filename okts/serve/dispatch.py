"""Layer 4 / phase 3 — dispatch: routing a validated call to the real source.

``OKTSService.call_tool`` hands a validated ``(concept, args)`` pair to a
``Dispatcher`` (see ``okts.core.protocols.Dispatcher``). This module provides:

- :class:`MockDispatcher` — records calls and returns a canned echo. Used by
  tests and offline runs; never touches the network or credentials.
- :class:`DispatcherRegistry` — routes by ``concept.interface`` to a
  per-interface ``Dispatcher`` (mcp/function/http/agent/search). It is itself
  a ``Dispatcher``, so it drops straight into ``OKTSService``.
- Real-ish per-interface dispatcher skeletons (:class:`McpDispatcher`,
  :class:`FunctionDispatcher`, :class:`HttpDispatcher`, :class:`AgentDispatcher`,
  :class:`SearchDispatcher`). Each reads credentials from a supplied
  ``SecretsProvider`` (defaulting to environment variables) and NEVER returns
  them. If no live backend is registered for a concept's ``target``, they
  raise :class:`NotConfiguredError` — a clear error, not an import-time or
  call-time crash.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from okts.core.model import Interface, OKTConcept
from okts.core.protocols import Dispatcher


class DispatchError(RuntimeError):
    """Base class for dispatch-time failures."""


class NotConfiguredError(DispatchError):
    """Raised when a live backend/credential isn't configured for a call.

    This is the expected, safe failure mode when a dispatcher skeleton is used
    without wiring in real backends — never a crash on import.
    """


# ---- credentials (invariant #4: never leave OKTS, never enter agent context) ----


@runtime_checkable
class SecretsProvider(Protocol):
    """Resolves a credential name (``OKTConcept.auth``) to a secret value.

    Implementations may read from env vars, a vault, a keychain, etc.
    Whatever it returns is used only inside a dispatcher and must never be
    placed on a return value handed back through ``OKTSService``.
    """

    def get(self, name: str) -> str | None:
        ...


@dataclass
class EnvSecretsProvider:
    """Default :class:`SecretsProvider`: reads ``<prefix><NAME_UPPER>`` from
    the process environment, e.g. ``auth: github_oauth`` -> ``OKTS_SECRET_GITHUB_OAUTH``.
    """

    prefix: str = "OKTS_SECRET_"

    def get(self, name: str) -> str | None:
        if not name:
            return None
        return os.environ.get(f"{self.prefix}{name.upper()}")


# ---- MockDispatcher: tests / offline runs ----


@dataclass
class MockDispatcher:
    """Records every call and returns a canned echo. Supports every concept.

    Never touches the network or real credentials — safe default for tests
    and for running the serving layer with no sources configured yet.
    """

    canned: Any = None
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def supports(self, concept: OKTConcept) -> bool:
        return True

    def dispatch(self, concept: OKTConcept, args: dict[str, Any]) -> Any:
        self.calls.append((concept.id, dict(args)))
        if self.canned is not None:
            return self.canned
        return {"mock": True, "id": concept.id, "args": dict(args)}

    async def adispatch(self, concept: OKTConcept, args: dict[str, Any]) -> Any:
        return self.dispatch(concept, args)


# ---- per-interface live dispatcher skeletons ----


class _LiveDispatcherBase:
    """Shared plumbing for the real-ish per-interface skeletons below.

    Subclasses set ``interface`` and implement ``dispatch``. A ``targets``
    map lets a caller register the actual backend client/callable per
    ``concept.target`` (falling back to ``concept.id``); nothing is
    guessed or auto-connected, so import and construction never touch the
    network. Missing target/credential configuration raises
    :class:`NotConfiguredError` rather than crashing.
    """

    interface: Interface

    def __init__(
        self,
        targets: dict[str, Any] | None = None,
        secrets: SecretsProvider | None = None,
    ) -> None:
        self.targets: dict[str, Any] = dict(targets or {})
        self.secrets: SecretsProvider = secrets or EnvSecretsProvider()

    def supports(self, concept: OKTConcept) -> bool:
        return concept.interface == self.interface

    def register_target(self, name: str, backend: Any) -> None:
        """Wire a real backend (client/callable) for ``concept.target == name``."""
        self.targets[name] = backend

    def _resolve_target(self, concept: OKTConcept) -> Any:
        key = concept.target or concept.id
        backend = self.targets.get(key)
        if backend is None:
            raise NotConfiguredError(
                f"{self.interface.value} dispatcher has no backend configured for "
                f"target {key!r} (tool {concept.id!r}); register one via "
                f"`register_target({key!r}, ...)` or the `targets=` constructor arg"
            )
        return backend

    def _resolve_credential(self, concept: OKTConcept) -> str | None:
        if not concept.auth:
            return None
        credential = self.secrets.get(concept.auth)
        if credential is None:
            raise NotConfiguredError(
                f"credential {concept.auth!r} required by tool {concept.id!r} is not "
                f"configured (set env var {EnvSecretsProvider().prefix}"
                f"{concept.auth.upper()} or supply a SecretsProvider)"
            )
        return credential

    def dispatch(self, concept: OKTConcept, args: dict[str, Any]) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def adispatch(self, concept: OKTConcept, args: dict[str, Any]) -> Any:
        """Async path shared by every live dispatcher.

        Each subclass's ``dispatch`` already returns whatever the wired target
        returns — a coroutine when that target is async (a live MCP session, an
        async function/agent/HTTP client), a plain value when it is sync. So we
        just run ``dispatch`` and await the result if it is awaitable. No
        per-subclass async code is needed.
        """
        result = self.dispatch(concept, args)
        if inspect.isawaitable(result):
            return await result
        return result


class McpDispatcher(_LiveDispatcherBase):
    """``interface: mcp`` — routes to a connected MCP client session.

    Expects ``targets[concept.target]`` to be an object exposing
    ``call_tool(name, arguments) -> Any`` (an already-authenticated MCP client
    session — a live ``mcp`` ``ClientSession`` drops in directly). Credentials,
    if any, are resolved but left for the caller-supplied client to have already
    applied at connect time — this skeleton never forwards them into the return
    value.

    The MCP adapter namespaces ids as ``<server>.<tool>`` (with ``target ==
    <server>``), but the upstream server knows the tool by its bare ``<tool>``
    name. So the dispatcher strips the ``<target>.`` prefix before calling —
    no client-side wrapper needed. Ids that don't carry that prefix (e.g.
    hand-authored concepts whose namespace differs from ``target``) are passed
    through unchanged.
    """

    interface = Interface.MCP

    @staticmethod
    def _tool_name(concept: OKTConcept) -> str:
        prefix = f"{concept.target}." if concept.target else ""
        if prefix and concept.id.startswith(prefix):
            return concept.id[len(prefix):]
        return concept.id

    def dispatch(self, concept: OKTConcept, args: dict[str, Any]) -> Any:
        client = self._resolve_target(concept)
        self._resolve_credential(concept)  # validated as configured; never returned
        call = getattr(client, "call_tool", None)
        if call is None:
            raise DispatchError(
                f"MCP client registered for target {concept.target!r} has no "
                f"call_tool(name, arguments) method"
            )
        return call(self._tool_name(concept), args)


class FunctionDispatcher(_LiveDispatcherBase):
    """``interface: function`` — calls an in-process Python callable directly.

    Expects ``targets[concept.target or concept.id]`` to be a callable that
    accepts ``**args``.
    """

    interface = Interface.FUNCTION

    def dispatch(self, concept: OKTConcept, args: dict[str, Any]) -> Any:
        fn = self._resolve_target(concept)
        self._resolve_credential(concept)
        if not callable(fn):
            raise DispatchError(
                f"function dispatcher target for tool {concept.id!r} is not callable"
            )
        return fn(**args)


class HttpDispatcher(_LiveDispatcherBase):
    """``interface: http`` — routes to a caller-supplied HTTP client callable.

    Deliberately does not embed an HTTP client (that belongs to an adapter,
    not the serving layer). Expects ``targets[concept.target]`` to be a
    callable ``(concept, args, credential) -> Any`` — e.g. a thin wrapper
    around ``requests``/``urllib`` that applies auth headers using the
    resolved credential, which this dispatcher never sees returned to it.
    """

    interface = Interface.HTTP

    def dispatch(self, concept: OKTConcept, args: dict[str, Any]) -> Any:
        client = self._resolve_target(concept)
        credential = self._resolve_credential(concept)
        if not callable(client):
            raise DispatchError(
                f"http dispatcher target for tool {concept.id!r} is not callable"
            )
        return client(concept, args, credential)


class AgentDispatcher(_LiveDispatcherBase):
    """``interface: agent`` — routes to a sub-agent invocation callable.

    Expects ``targets[concept.target or concept.id]`` to be a callable
    ``(args) -> Any`` that runs the sub-agent and returns its result.
    """

    interface = Interface.AGENT

    def dispatch(self, concept: OKTConcept, args: dict[str, Any]) -> Any:
        agent = self._resolve_target(concept)
        self._resolve_credential(concept)
        if not callable(agent):
            raise DispatchError(
                f"agent dispatcher target for tool {concept.id!r} is not callable"
            )
        return agent(args)


class SearchDispatcher(_LiveDispatcherBase):
    """``interface: search`` — routes to a search-endpoint callable.

    Expects ``targets[concept.target or concept.id]`` to be a callable
    ``(args) -> Any`` (e.g. a wrapper around a search API's query params).
    """

    interface = Interface.SEARCH

    def dispatch(self, concept: OKTConcept, args: dict[str, Any]) -> Any:
        endpoint = self._resolve_target(concept)
        self._resolve_credential(concept)
        if not callable(endpoint):
            raise DispatchError(
                f"search dispatcher target for tool {concept.id!r} is not callable"
            )
        return endpoint(args)


# ---- the fan-out registry ----


@dataclass
class DispatcherRegistry:
    """Routes ``dispatch``/``supports`` by ``concept.interface`` to a
    per-interface :class:`~okts.core.protocols.Dispatcher`.

    Implements the ``Dispatcher`` protocol itself, so one registry drops
    straight into ``OKTSService(dispatcher=registry)`` and fans calls out to
    whichever concrete dispatcher is registered for each interface. An
    interface with nothing registered (and no ``default``) raises
    :class:`NotConfiguredError` — the safe default, not a crash.
    """

    dispatchers: dict[Interface, Dispatcher] = field(default_factory=dict)
    default: Dispatcher | None = None

    def register(self, interface: Interface | str, dispatcher: Dispatcher) -> None:
        self.dispatchers[Interface(interface)] = dispatcher

    def _for(self, concept: OKTConcept) -> Dispatcher | None:
        return self.dispatchers.get(concept.interface, self.default)

    def supports(self, concept: OKTConcept) -> bool:
        d = self._for(concept)
        return d is not None and d.supports(concept)

    def dispatch(self, concept: OKTConcept, args: dict[str, Any]) -> Any:
        return self._route(concept).dispatch(concept, args)

    async def adispatch(self, concept: OKTConcept, args: dict[str, Any]) -> Any:
        d = self._route(concept)
        adispatch = getattr(d, "adispatch", None)
        if adispatch is not None:
            return await adispatch(concept, args)
        # sub-dispatcher is sync-only: run it and await if it returned a coroutine
        result = d.dispatch(concept, args)
        if inspect.isawaitable(result):
            return await result
        return result

    def _route(self, concept: OKTConcept) -> Dispatcher:
        d = self._for(concept)
        if d is None:
            raise NotConfiguredError(
                f"no dispatcher registered for interface {concept.interface.value!r} "
                f"(tool {concept.id!r}); register one via `.register(interface, dispatcher)` "
                f"or set a `default`"
            )
        return d

    @classmethod
    def mock_all(cls) -> "DispatcherRegistry":
        """Convenience: one shared :class:`MockDispatcher` for every interface.
        Handy for tests and demos that need `call_tool` to succeed offline."""
        mock = MockDispatcher()
        return cls(dispatchers={i: mock for i in Interface}, default=mock)
