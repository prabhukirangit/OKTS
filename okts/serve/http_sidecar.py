"""Layer 4 / integration mode 3 — HTTP sidecar.

For polyglot stacks: run OKTS as a standalone service and hit three JSON
endpoints. Built on stdlib ``http.server`` only — no framework dependency, so
this module imports and runs with zero extra packages.

Endpoints (all POST, JSON body in, JSON body out)::

    POST /search   {"query": str, "k": int=5}   -> {"results": [{id,title,description}, ...]}
    POST /load     {"id": str}                   -> {"id", "input_schema", "side_effects", ...}
    POST /call     {"id": str, "args": {...}}     -> {"result": <dispatcher return value>}

Errors come back as ``{"error": "..."}`` with a matching non-2xx status.
Credentials never appear in any response body — they never leave the
``Dispatcher`` (see ``okts.serve.dispatch``); this sidecar only ever sees
whatever the dispatcher chooses to return.

SECURITY: this sidecar performs NO authentication and ``POST /call`` dispatches
real tools (including writes). It binds to ``127.0.0.1`` by default for that
reason. Do NOT expose it on a public interface (``0.0.0.0``) without an
authenticating reverse proxy in front and (recommended) an ``OKTSService``
configured with pre-dispatch policies (see ``okts.serve.policy``). Request
bodies are capped at :data:`_MAX_BODY_BYTES` to bound memory use.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from okts.serve.service import (
    ArgumentValidationError,
    DispatchNotSupportedError,
    OKTSService,
    PolicyDenied,
    ToolNotFoundError,
)

_ROUTES = {"/search", "/load", "/call"}

#: Maximum accepted request-body size (bytes). Bounds memory for a single
#: request so a hostile/oversized ``Content-Length`` can't exhaust the process.
_MAX_BODY_BYTES = 1 * 1024 * 1024  # 1 MiB


class _RequestTooLarge(ArgumentValidationError):
    """Body exceeds :data:`_MAX_BODY_BYTES`."""


def _validate_body_length(length: int) -> None:
    """Raise :class:`_RequestTooLarge` for a negative or over-limit body length.

    Checked from the ``Content-Length`` header BEFORE any bytes are read, so a
    hostile length never allocates memory."""
    if length < 0 or length > _MAX_BODY_BYTES:
        raise _RequestTooLarge(
            f"request body of {length} bytes exceeds the {_MAX_BODY_BYTES}-byte limit"
        )


def _status_for(exc: Exception) -> int:
    if isinstance(exc, ToolNotFoundError):
        return 404
    if isinstance(exc, ArgumentValidationError):
        return 400
    if isinstance(exc, PolicyDenied):
        return 403
    if isinstance(exc, DispatchNotSupportedError):
        return 501
    return 500


def make_handler(service: OKTSService) -> type[BaseHTTPRequestHandler]:
    """Build a ``BaseHTTPRequestHandler`` subclass bound to ``service``.

    A fresh class per ``service`` (``http.server`` handlers are classes, not
    instances) so tests can spin up independent servers against independent
    services without global state.
    """

    class Handler(BaseHTTPRequestHandler):
        server_version = "OKTSHTTPSidecar/0.1"

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0) or 0)
            _validate_body_length(length)
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ArgumentValidationError("request body must be a JSON object")
            return data

        def do_POST(self) -> None:  # noqa: N802 -- stdlib handler API name
            # Always drain the request body first, even for an unknown route --
            # responding without reading a pending body can reset the client
            # connection before it finishes writing.
            try:
                payload = self._read_json()
            except _RequestTooLarge as exc:
                self._send_json(413, {"error": str(exc)})
                return
            except ArgumentValidationError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except json.JSONDecodeError as exc:
                self._send_json(400, {"error": f"invalid JSON body: {exc}"})
                return

            if self.path not in _ROUTES:
                self._send_json(404, {"error": f"unknown endpoint: {self.path}"})
                return
            try:
                if self.path == "/search":
                    results = service.search_tools(payload.get("query", ""), k=payload.get("k", 5))
                    self._send_json(200, {"results": results})
                elif self.path == "/load":
                    result = service.load_tool(payload["id"])
                    self._send_json(200, result)
                else:  # /call
                    result = service.call_tool(payload["id"], payload.get("args") or {})
                    self._send_json(200, {"result": result})
            except (
                ToolNotFoundError,
                ArgumentValidationError,
                DispatchNotSupportedError,
                PolicyDenied,
            ) as exc:
                # NOTE: ToolNotFoundError subclasses KeyError, so this branch
                # must be checked before the bare KeyError below.
                self._send_json(_status_for(exc), {"error": str(exc)})
            except KeyError as exc:
                self._send_json(400, {"error": f"missing required field: {exc}"})
            except Exception as exc:  # pragma: no cover - defensive catch-all
                self._send_json(500, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:  # silence default stderr access log
            pass

    return Handler


def serve(service: OKTSService, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Build a ready-to-serve HTTP sidecar. Does NOT block.

    Call ``.serve_forever()`` on the result to actually run it, or drive it
    manually (e.g. ``handle_request()`` once) in tests.
    """
    handler = make_handler(service)
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:  # pragma: no cover - thin CLI wrapper, exercised manually
    """Standalone CLI: build a service from a bundle dir and serve it."""
    import argparse

    from okts.core.bundle_io import load_bundle
    from okts.serve.dispatch import DispatcherRegistry
    from okts.serve.mcp_server import NaiveFallbackRetriever

    parser = argparse.ArgumentParser(prog="okts-http", description="Serve the three OKTS meta-tools over HTTP.")
    parser.add_argument("--bundle-dir", default="./okt-bundle")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    bundle = load_bundle(args.bundle_dir)
    service = OKTSService(bundle=bundle, retriever=NaiveFallbackRetriever(), dispatcher=DispatcherRegistry())
    httpd = serve(service, host=args.host, port=args.port)
    print(f"okts http sidecar listening on http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":  # pragma: no cover
    main()
