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
from typing import Any

from okts.core.model import Bundle, OKTConcept
from okts.core.protocols import Dispatcher, Retriever


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

    Only these two protocols are depended on — never concrete classes from
    ``okts.index`` or ``okts.adapters`` — so retrieval and dispatch can be
    swapped freely (e.g. ``okts.serve.dispatch.MockDispatcher`` for tests, the
    real hybrid retriever wired in by the coordinator for production).
    """

    def __init__(self, bundle: Bundle, retriever: Retriever, dispatcher: Dispatcher) -> None:
        self.bundle = bundle
        self.retriever = retriever
        self.dispatcher = dispatcher
        self.retriever.index(self.bundle)

    # ---- phase 1: search ----

    def search_tools(self, query: str, k: int = 5, **opts: Any) -> list[dict[str, Any]]:
        """Rank concepts for ``query``. Returns lightweight refs, NEVER schemas.

        Each result is ``{"id", "title", "description"}`` — exactly
        ``SearchHit.to_ref()``. Extra ``**opts`` (e.g. retrieval-mode knobs)
        are forwarded verbatim to the retriever.
        """
        hits = self.retriever.search(query, k=k, **opts)
        return [hit.to_ref() for hit in hits]

    # ---- phase 2: load ----

    def load_tool(self, id: str) -> dict[str, Any]:
        """Return the structured ``input_schema`` + ``side_effects`` for ``id``.

        Exactly ``OKTConcept.call_view()``. Raises :class:`ToolNotFoundError`
        if ``id`` isn't present in the served bundle.
        """
        concept = self._require_concept(id)
        return concept.call_view()

    # ---- phase 3: call ----

    def call_tool(self, id: str, args: dict[str, Any] | None = None) -> Any:
        """Validate ``args`` against ``id``'s ``input_schema``, then dispatch (sync).

        Raises :class:`ToolNotFoundError` for an unknown/uncataloged id,
        :class:`ArgumentValidationError` for missing required args or a type
        mismatch, and :class:`DispatchNotSupportedError` if the configured
        dispatcher declines the concept. Credentials are applied inside the
        dispatcher and never appear in the return value (invariant #4).

        If the dispatcher's target is async (the tool's ``invocation`` is
        ``async`` — a live MCP session, an async function/agent/HTTP client),
        ``dispatch`` returns a coroutine. This sync entry point bridges it: it
        runs the coroutine to completion when no event loop is active. When
        called from *inside* a running event loop it cannot block, so it raises
        a clear error directing the caller to ``acall_tool`` instead — never a
        silently-unawaited coroutine.
        """
        concept, call_args = self._prepare_call(id, args)
        result = self.dispatcher.dispatch(concept, call_args)
        if inspect.isawaitable(result):
            return self._run_sync(result, id)
        return result

    async def acall_tool(self, id: str, args: dict[str, Any] | None = None) -> Any:
        """Async counterpart to :meth:`call_tool`, for callers already inside an
        event loop (the MCP server, async agent frameworks).

        Uses the dispatcher's optional ``adispatch`` when present, otherwise
        awaits whatever ``dispatch`` returns. Same validation and error contract
        as :meth:`call_tool`.
        """
        concept, call_args = self._prepare_call(id, args)
        adispatch = getattr(self.dispatcher, "adispatch", None)
        if adispatch is not None:
            return await adispatch(concept, call_args)
        result = self.dispatcher.dispatch(concept, call_args)
        if inspect.isawaitable(result):
            return await result
        return result

    # ---- internal ----

    def _prepare_call(
        self, id: str, args: dict[str, Any] | None
    ) -> tuple[OKTConcept, dict[str, Any]]:
        """Shared phase-3 preamble: resolve, validate args, check dispatch support."""
        concept = self._require_concept(id)
        call_args: dict[str, Any] = dict(args or {})
        _validate_against_schema(concept.input_schema, call_args, path="args")
        if not self.dispatcher.supports(concept):
            raise DispatchNotSupportedError(
                f"no dispatcher backend available for tool {id!r} "
                f"(interface={concept.interface.value!r})"
            )
        return concept, call_args

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
