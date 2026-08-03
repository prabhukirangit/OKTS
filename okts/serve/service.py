"""``OKTSService`` — the framework-agnostic heart of layer 4.

Implements the three runtime phases (CLAUDE.md "Runtime phases") as plain
Python methods with no framework dependency:

- **Phase 1 — search.** ``search_tools`` delegates to a ``Retriever`` and
  returns lightweight refs (``id``, ``title``, ``description``) — never
  schemas (invariant #2/#3).
- **Phase 2 — load.** ``load_tool`` returns one concept's ``call_view()``
  (structured ``input_schema`` + ``side_effects``).
- **Phase 3 — call.** ``call_tool`` (sync) / ``acall_tool`` (async) validate
  ``args`` against that schema with a dependency-free lightweight JSON-Schema
  checker, then dispatch via a ``Dispatcher``. Async targets (a tool whose
  ``invocation`` is ``async`` — a live MCP session, an async function/agent/HTTP
  client) are awaited on the ``acall_tool`` path; ``call_tool`` bridges them when
  no event loop is running. Credentials are applied inside the dispatcher and
  never flow back through this service (invariant #4).

Every other integration (``mcp_server``, ``sdk``, ``http_sidecar``) is a thin
wrapper around this class. It depends only on the ``Retriever`` and
``Dispatcher`` PROTOCOLS from ``okts.core`` — never on concrete index/adapter
classes (see ``okts/core/protocols.py``).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Sequence

from okts.core.model import Bundle, OKTConcept
from okts.core.protocols import Dispatcher, PreDispatchPolicy, Retriever

log = logging.getLogger(__name__)

#: Key added to every ``load_tool`` payload so a context-hygiene scrubber can
#: recognize a loaded schema and evict it once the matching call completes.
#: See ``examples/context_hygiene.py``.
SCHEMA_MARKER_KEY = "_okts"
SCHEMA_MARKER_KIND = "schema-instance"


class ToolNotFoundError(KeyError):
    """Raised when an id doesn't resolve to a concept in the served bundle.

    Covers both ``load_tool`` and ``call_tool`` — only ids present in the
    bundle are ever loadable/callable.
    """


class ArgumentValidationError(ValueError):
    """Raised when ``call_tool`` args fail validation against ``input_schema``."""


class DispatchNotSupportedError(RuntimeError):
    """Raised when the configured ``Dispatcher`` declines a concept (``supports()``
    returned ``False``) — e.g. no live backend wired for that interface."""


class PolicyDenied(RuntimeError):
    """Raised when a :class:`~okts.core.protocols.PreDispatchPolicy` blocks a call.

    A deliberate, safe refusal (side-effect gate, rate limit, allowlist miss),
    not a bug — carries a human-readable reason for the caller/host to surface."""


# ---- lightweight, dependency-free JSON-Schema arg validation ----
#
# This intentionally implements a small, predictable subset (object/string/
# integer/number/boolean/array + "required" + nested "properties"/"items"),
# not full JSON-Schema. It has no hard dependency on the `jsonschema` package
# so `call_tool` validation works offline in every environment.

_JSON_TYPE_TO_PY: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "null": type(None),
}


def _type_name(value: Any) -> str:
    return type(value).__name__


def _check_type(value: Any, json_type: str, *, path: str) -> None:
    py_type = _JSON_TYPE_TO_PY.get(json_type)
    if py_type is None:
        return  # unknown/unsupported type keyword: degrade gracefully, skip
    # bool is a subclass of int in Python; JSON Schema treats them as distinct.
    if json_type in ("integer", "number") and isinstance(value, bool):
        raise ArgumentValidationError(
            f"{path}: expected {json_type}, got boolean"
        )
    if json_type == "boolean" and not isinstance(value, bool):
        raise ArgumentValidationError(
            f"{path}: expected boolean, got {_type_name(value)}"
        )
    if not isinstance(value, py_type):
        raise ArgumentValidationError(
            f"{path}: expected {json_type}, got {_type_name(value)}"
        )


def _validate_against_schema(schema: Any, value: Any, *, path: str) -> None:
    """Recursively check ``value`` against ``schema``. Best-effort/lightweight."""
    if not isinstance(schema, dict):
        return  # not a structured schema we can check against; skip
    if "resource" in schema and "type" not in schema and "properties" not in schema:
        # A {resource: ...} pointer schema — the actual contract lives in an
        # external file this layer doesn't resolve. Nothing to check against.
        return

    json_type = schema.get("type")
    properties = schema.get("properties")

    if json_type == "object" or (json_type is None and isinstance(properties, dict)):
        if not isinstance(value, dict):
            raise ArgumentValidationError(
                f"{path}: expected object, got {_type_name(value)}"
            )
        for key in schema.get("required") or []:
            if key not in value:
                raise ArgumentValidationError(
                    f"{path}: missing required property {key!r}"
                )
        for key, subschema in (properties or {}).items():
            if key in value:
                _validate_against_schema(subschema, value[key], path=f"{path}.{key}")
        return

    if json_type == "array":
        if not isinstance(value, list):
            raise ArgumentValidationError(
                f"{path}: expected array, got {_type_name(value)}"
            )
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for i, item in enumerate(value):
                _validate_against_schema(items_schema, item, path=f"{path}[{i}]")
        return

    if isinstance(json_type, str):
        _check_type(value, json_type, path=path)
        return

    # No usable "type" (and not implicitly an object via "properties"): nothing
    # to check structurally beyond having reached this point.


class OKTSService:
    """The three meta-tools, as plain methods. Framework-agnostic.

    Args:
        bundle: the served OKT bundle (only its concepts are ever exposed).
        retriever: phase-1 ranker; must satisfy ``okts.core.protocols.Retriever``.
            ``bundle`` is indexed into it once, at construction time.
        dispatcher: phase-3 router; must satisfy ``okts.core.protocols.Dispatcher``.
        policies: optional pre-dispatch gates (``PreDispatchPolicy``) run in order
            at the single dispatch choke point, after arg-validation and before
            dispatch, for BOTH ``call_tool`` and ``acall_tool``. Empty by default
            — construction and dispatch behave exactly as before when omitted.

    Only these protocols are depended on — never concrete classes from
    ``okts.index`` or ``okts.adapters`` — so retrieval and dispatch can be
    swapped freely (e.g. ``okts.serve.dispatch.MockDispatcher`` for tests, the
    real hybrid retriever wired in by the coordinator for production).
    """

    def __init__(
        self,
        bundle: Bundle,
        retriever: Retriever,
        dispatcher: Dispatcher,
        policies: Sequence[PreDispatchPolicy] = (),
    ) -> None:
        self.bundle = bundle
        self.retriever = retriever
        self.dispatcher = dispatcher
        self.policies: tuple[PreDispatchPolicy, ...] = tuple(policies)
        self.retriever.index(self.bundle)
        log.info(
            "OKTSService ready: %d tools served via retriever=%s dispatcher=%s policies=%d",
            sum(1 for _ in bundle),
            type(retriever).__name__,
            type(dispatcher).__name__,
            len(self.policies),
        )

    # ---- phase 1: search ----

    def search_tools(self, query: str, k: int = 5, **opts: Any) -> list[dict[str, Any]]:
        """Rank concepts for ``query``. Returns lightweight refs, NEVER schemas.

        Each result is ``{"id", "title", "description"}`` — exactly
        ``SearchHit.to_ref()``. Extra ``**opts`` (e.g. retrieval-mode knobs)
        are forwarded verbatim to the retriever.
        """
        log.debug("phase 1 search_tools q=%r k=%d", query, k)
        hits = self.retriever.search(query, k=k, **opts)
        log.debug("phase 1 -> %d refs: %s", len(hits), [h.id for h in hits])
        return [hit.to_ref() for hit in hits]

    # ---- phase 2: load ----

    def load_tool(self, id: str) -> dict[str, Any]:
        """Return the structured ``input_schema`` + ``side_effects`` for ``id``.

        This is ``OKTConcept.call_view()`` plus one additive key: a
        :data:`SCHEMA_MARKER_KEY` (``"_okts"``) envelope tagging the payload as a
        loaded schema instance, so a context-hygiene scrubber can evict it from
        history once the matching ``call_tool`` completes (see
        ``examples/context_hygiene.py``). Existing consumers that read
        ``input_schema``/``side_effects`` by key are unaffected. Raises
        :class:`ToolNotFoundError` if ``id`` isn't present in the served bundle.
        """
        log.debug("phase 2 load_tool id=%r", id)
        concept = self._require_concept(id)
        view = concept.call_view()
        view[SCHEMA_MARKER_KEY] = {"kind": SCHEMA_MARKER_KIND, "for_id": concept.id}
        return view

    # ---- phase 3: call ----

    def call_tool(
        self,
        id: str,
        args: dict[str, Any] | None = None,
        *,
        scope: dict[str, Any] | None = None,
    ) -> Any:
        """Validate ``args`` against ``id``'s ``input_schema``, then dispatch (sync).

        Raises :class:`ToolNotFoundError` for an unknown/uncataloged id,
        :class:`ArgumentValidationError` for missing required args or a type
        mismatch, :class:`DispatchNotSupportedError` if the configured
        dispatcher declines the concept, and :class:`PolicyDenied` if a
        configured pre-dispatch policy blocks the call. Credentials are applied
        inside the dispatcher and never appear in the return value (invariant #4).

        ``scope`` is optional HOST context threaded to any policies (e.g.
        ``{"confirmed": True}`` to satisfy a side-effect gate). It is
        caller-supplied, never agent args — an agent cannot self-authorize.

        If the dispatcher's target is async (the tool's ``invocation`` is
        ``async`` — a live MCP session, an async function/agent/HTTP client),
        ``dispatch`` returns a coroutine. This sync entry point bridges it: it
        runs the coroutine to completion when no event loop is active. When
        called from *inside* a running event loop it cannot block, so it raises
        a clear error directing the caller to ``acall_tool`` instead — never a
        silently-unawaited coroutine.
        """
        concept, call_args = self._prepare_call(id, args, scope)
        log.debug(
            "phase 3 call_tool id=%r interface=%s invocation=%s (sync path)",
            id, concept.interface.value, concept.invocation.value,
        )
        result = self.dispatcher.dispatch(concept, call_args)
        if inspect.isawaitable(result):
            log.debug("dispatch for %r returned an awaitable; bridging on sync path", id)
            return self._run_sync(result, id)
        return result

    async def acall_tool(
        self,
        id: str,
        args: dict[str, Any] | None = None,
        *,
        scope: dict[str, Any] | None = None,
    ) -> Any:
        """Async counterpart to :meth:`call_tool`, for callers already inside an
        event loop (the MCP server, async agent frameworks).

        Uses the dispatcher's optional ``adispatch`` when present, otherwise
        awaits whatever ``dispatch`` returns. Same validation, policy, and error
        contract as :meth:`call_tool` (including the optional ``scope``).
        """
        concept, call_args = self._prepare_call(id, args, scope)
        log.debug(
            "phase 3 acall_tool id=%r interface=%s invocation=%s (async path)",
            id, concept.interface.value, concept.invocation.value,
        )
        adispatch = getattr(self.dispatcher, "adispatch", None)
        if adispatch is not None:
            log.debug("dispatching %r via dispatcher.adispatch", id)
            return await adispatch(concept, call_args)
        result = self.dispatcher.dispatch(concept, call_args)
        if inspect.isawaitable(result):
            return await result
        return result

    # ---- internal ----

    def _prepare_call(
        self,
        id: str,
        args: dict[str, Any] | None,
        scope: dict[str, Any] | None = None,
    ) -> tuple[OKTConcept, dict[str, Any]]:
        """Shared phase-3 preamble: resolve, validate args, check dispatch
        support, then run the pre-dispatch policy chain."""
        concept = self._require_concept(id)
        call_args: dict[str, Any] = dict(args or {})
        _validate_against_schema(concept.input_schema, call_args, path="args")
        if not self.dispatcher.supports(concept):
            log.warning(
                "no dispatcher backend for tool %r (interface=%s)",
                id, concept.interface.value,
            )
            raise DispatchNotSupportedError(
                f"no dispatcher backend available for tool {id!r} "
                f"(interface={concept.interface.value!r})"
            )
        call_args = self._apply_policies(concept, call_args, scope)
        return concept, call_args

    def _apply_policies(
        self,
        concept: OKTConcept,
        call_args: dict[str, Any],
        scope: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Run each policy in order; a policy may mutate args or raise
        :class:`PolicyDenied`. No-op (returns args unchanged) when none wired."""
        if not self.policies:
            return call_args
        policy_scope: dict[str, Any] = dict(scope or {})
        for policy in self.policies:
            try:
                call_args = policy.check(concept, call_args, policy_scope)
            except PolicyDenied as exc:
                log.warning(
                    "policy %s denied tool %r: %s",
                    type(policy).__name__, concept.id, exc,
                )
                raise
        return call_args

    @staticmethod
    def _run_sync(awaitable: Any, id: str) -> Any:
        """Drive an awaitable to completion from sync code, or fail clearly."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        # A loop is already running on this thread: we must not block it.
        awaitable.close()  # avoid a "coroutine was never awaited" warning
        raise RuntimeError(
            f"tool {id!r} dispatches to an async target; call_tool() cannot run it "
            f"from inside a running event loop — use `await service.acall_tool({id!r}, ...)`"
        )

    def _require_concept(self, id: str) -> OKTConcept:
        concept = self.bundle.get(id)
        if concept is None:
            raise ToolNotFoundError(f"unknown tool id: {id!r}")
        return concept
