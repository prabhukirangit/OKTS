"""Layer 4 (serving) tests: the three meta-tools, dispatch, sdk, http sidecar.

Depends only on the ``Retriever``/``Dispatcher`` protocols from ``okts.core``.
Uses a tiny in-test stub retriever (NOT ``okts.index`` — that's being built in
parallel) and ``okts.serve.dispatch.MockDispatcher`` throughout, so everything
here runs offline with no network, no keys, and no dependency on layer 3.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import pytest

from okts.core.model import Bundle
from okts.core.protocols import SearchHit
from okts.serve.dispatch import (
    DispatcherRegistry,
    EnvSecretsProvider,
    FunctionDispatcher,
    McpDispatcher,
    MockDispatcher,
    NotConfiguredError,
)
from okts.serve.http_sidecar import serve as http_serve
from okts.serve.sdk import build_sdk_tools, sdk_tools
from okts.serve.service import (
    ArgumentValidationError,
    DispatchNotSupportedError,
    OKTSService,
    ToolNotFoundError,
)


# ---------------------------------------------------------------------------
# a tiny in-test Retriever stub (deliberately NOT okts.index)
# ---------------------------------------------------------------------------


@dataclass
class StubRetriever:
    """Naive substring-match Retriever satisfying okts.core.protocols.Retriever."""

    _bundle: Bundle | None = field(default=None, init=False, repr=False)

    def index(self, bundle: Bundle) -> None:
        self._bundle = bundle

    def search(self, query: str, k: int = 5, **opts: Any) -> list[SearchHit]:
        assert self._bundle is not None, "index() must be called before search()"
        terms = [t for t in query.lower().split() if t]
        hits: list[SearchHit] = []
        for concept in self._bundle:
            text = concept.match_text().lower()
            if terms and not any(t in text for t in terms):
                continue
            hits.append(
                SearchHit(
                    id=concept.id,
                    title=concept.title,
                    description=concept.description,
                    score=1.0,
                )
            )
        return hits[:k]


@pytest.fixture
def retriever() -> StubRetriever:
    return StubRetriever()


@pytest.fixture
def dispatcher() -> MockDispatcher:
    return MockDispatcher()


@pytest.fixture
def service(bundle, retriever, dispatcher) -> OKTSService:
    return OKTSService(bundle=bundle, retriever=retriever, dispatcher=dispatcher)


# ---------------------------------------------------------------------------
# phase 1: search_tools
# ---------------------------------------------------------------------------


def test_search_tools_returns_lightweight_refs_never_schemas(service):
    results = service.search_tools("issue", k=10)
    assert results, "expected at least one hit for 'issue'"
    for ref in results:
        assert set(ref) == {"id", "title", "description"}
        assert "input_schema" not in ref
        assert "side_effects" not in ref


def test_search_tools_respects_k(service):
    results = service.search_tools("issue", k=2)
    assert len(results) <= 2


def test_search_tools_indexes_bundle_at_construction(bundle, dispatcher):
    r = StubRetriever()
    assert r._bundle is None
    OKTSService(bundle=bundle, retriever=r, dispatcher=dispatcher)
    assert r._bundle is bundle


# ---------------------------------------------------------------------------
# phase 2: load_tool
# ---------------------------------------------------------------------------


def test_load_tool_returns_input_schema_and_side_effects(service):
    result = service.load_tool("github.create_issue")
    assert result["id"] == "github.create_issue"
    assert result["side_effects"] == "write"
    assert result["input_schema"]["required"] == ["repo", "title"]
    assert "repo" in result["input_schema"]["properties"]


def test_load_tool_unknown_id_raises(service):
    with pytest.raises(ToolNotFoundError):
        service.load_tool("nope.does_not_exist")


# ---------------------------------------------------------------------------
# phase 3: call_tool -- validation
# ---------------------------------------------------------------------------


def test_call_tool_valid_args_dispatches(service, dispatcher):
    result = service.call_tool("github.create_issue", {"repo": "o/n", "title": "Bug"})
    assert dispatcher.calls == [("github.create_issue", {"repo": "o/n", "title": "Bug"})]
    assert result["id"] == "github.create_issue"


def test_call_tool_rejects_missing_required_arg(service):
    with pytest.raises(ArgumentValidationError, match="title"):
        service.call_tool("github.create_issue", {"repo": "o/n"})


def test_call_tool_rejects_missing_all_required_args(service):
    with pytest.raises(ArgumentValidationError):
        service.call_tool("github.create_issue", {})


def test_call_tool_rejects_wrong_type(service):
    with pytest.raises(ArgumentValidationError, match="string"):
        service.call_tool("github.create_issue", {"repo": "o/n", "title": 12345})


def test_call_tool_rejects_wrong_array_item_type(service):
    with pytest.raises(ArgumentValidationError):
        service.call_tool(
            "github.create_issue",
            {"repo": "o/n", "title": "t", "labels": "not-a-list"},
        )


def test_call_tool_unknown_id_raises(service):
    with pytest.raises(ToolNotFoundError):
        service.call_tool("nope.does_not_exist", {})


def test_call_tool_only_bundle_ids_are_callable(service, bundle):
    # sanity: every id call_tool will accept must resolve in the bundle
    for concept_id in list(bundle.concepts)[:3]:
        assert bundle.get(concept_id) is not None
    with pytest.raises(ToolNotFoundError):
        service.call_tool("totally.unregistered", {"x": 1})


def test_call_tool_defaults_missing_args_to_empty_dict(bundle, retriever):
    # a concept with no required properties should accept args=None
    from okts.core.model import Interface, OKTConcept

    b = Bundle()
    for c in bundle:
        b.add(c)
    b.add(
        OKTConcept(
            id="noop.ping",
            title="Ping",
            description="no-op",
            input_schema={"type": "object", "properties": {}},
            interface=Interface.FUNCTION,
        )
    )
    svc = OKTSService(bundle=b, retriever=StubRetriever(), dispatcher=MockDispatcher())
    result = svc.call_tool("noop.ping")  # args omitted entirely
    assert result["id"] == "noop.ping"


# ---------------------------------------------------------------------------
# credentials must never leak through call_tool's return value
# ---------------------------------------------------------------------------


def test_credentials_never_appear_in_call_tool_output_mock(service):
    result = service.call_tool("slack.send_message", {"channel": "#general", "text": "hi"})
    dumped = json.dumps(result)
    assert "slack_oauth" not in dumped


def test_credentials_never_appear_in_call_tool_output_live_dispatcher(bundle, monkeypatch):
    monkeypatch.setenv("OKTS_SECRET_SLACK_OAUTH", "super-secret-token-xyz")

    captured: dict[str, Any] = {}

    class FakeMcpClient:
        def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            captured["name"] = name
            captured["arguments"] = arguments
            return {"ok": True, "ts": "123.456"}

    mcp_dispatcher = McpDispatcher(targets={"slack-mcp": FakeMcpClient()})
    registry = DispatcherRegistry()
    registry.register("mcp", mcp_dispatcher)

    svc = OKTSService(bundle=bundle, retriever=StubRetriever(), dispatcher=registry)
    result = svc.call_tool("slack.send_message", {"channel": "#general", "text": "hi"})

    dumped = json.dumps(result)
    assert "super-secret-token-xyz" not in dumped
    assert result == {"ok": True, "ts": "123.456"}
    assert captured["name"] == "slack.send_message"


def test_live_dispatcher_raises_not_configured_without_credential(bundle, monkeypatch):
    monkeypatch.delenv("OKTS_SECRET_SLACK_OAUTH", raising=False)

    class FakeMcpClient:
        def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            return {"ok": True}

    mcp_dispatcher = McpDispatcher(targets={"slack-mcp": FakeMcpClient()})
    registry = DispatcherRegistry()
    registry.register("mcp", mcp_dispatcher)
    svc = OKTSService(bundle=bundle, retriever=StubRetriever(), dispatcher=registry)

    with pytest.raises(NotConfiguredError):
        svc.call_tool("slack.send_message", {"channel": "#general", "text": "hi"})


def test_live_dispatcher_raises_not_configured_without_target(bundle):
    mcp_dispatcher = McpDispatcher()  # no targets registered at all
    registry = DispatcherRegistry()
    registry.register("mcp", mcp_dispatcher)
    svc = OKTSService(bundle=bundle, retriever=StubRetriever(), dispatcher=registry)

    with pytest.raises(NotConfiguredError):
        svc.call_tool("github.create_issue", {"repo": "o/n", "title": "t"})


# ---------------------------------------------------------------------------
# dispatch.py: MockDispatcher / DispatcherRegistry / live skeletons
# ---------------------------------------------------------------------------


def test_mock_dispatcher_records_calls_and_echoes(bundle):
    concept = bundle.get("github.create_issue")
    d = MockDispatcher()
    result = d.dispatch(concept, {"repo": "o/n", "title": "t"})
    assert d.calls == [("github.create_issue", {"repo": "o/n", "title": "t"})]
    assert result["mock"] is True
    assert d.supports(concept) is True


def test_mock_dispatcher_canned_response(bundle):
    concept = bundle.get("github.create_issue")
    d = MockDispatcher(canned={"issue_number": 42})
    assert d.dispatch(concept, {}) == {"issue_number": 42}


def test_dispatcher_registry_routes_by_interface(bundle):
    concept = bundle.get("github.create_issue")
    assert concept.interface.value == "mcp"

    mock = MockDispatcher()
    registry = DispatcherRegistry()
    registry.register("mcp", mock)

    assert registry.supports(concept) is True
    registry.dispatch(concept, {"repo": "o/n", "title": "t"})
    assert mock.calls


def test_dispatcher_registry_raises_not_configured_for_unwired_interface(bundle):
    concept = bundle.get("github.create_issue")
    registry = DispatcherRegistry()  # nothing registered, no default
    assert registry.supports(concept) is False
    with pytest.raises(NotConfiguredError):
        registry.dispatch(concept, {})


def test_dispatcher_registry_mock_all(bundle):
    registry = DispatcherRegistry.mock_all()
    for concept in bundle:
        assert registry.supports(concept) is True
        registry.dispatch(concept, {})


def test_call_tool_raises_dispatch_not_supported(bundle):
    registry = DispatcherRegistry()  # empty: supports() is False for everything
    svc = OKTSService(bundle=bundle, retriever=StubRetriever(), dispatcher=registry)
    with pytest.raises(DispatchNotSupportedError):
        svc.call_tool("github.create_issue", {"repo": "o/n", "title": "t"})


def test_function_dispatcher_calls_registered_callable(bundle):
    from okts.core.model import Interface, OKTConcept

    called_with: dict[str, Any] = {}

    def local_fn(**kwargs: Any) -> dict[str, Any]:
        called_with.update(kwargs)
        return {"ran": True}

    concept = OKTConcept(
        id="local.ping",
        title="Ping",
        description="local no-op",
        input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        interface=Interface.FUNCTION,
        target="local.ping",
    )
    d = FunctionDispatcher(targets={"local.ping": local_fn})
    result = d.dispatch(concept, {"x": 1})
    assert result == {"ran": True}
    assert called_with == {"x": 1}


def test_env_secrets_provider_reads_prefixed_env(monkeypatch):
    monkeypatch.setenv("OKTS_SECRET_GITHUB_OAUTH", "abc123")
    provider = EnvSecretsProvider()
    assert provider.get("github_oauth") == "abc123"
    assert provider.get("") is None
    assert provider.get("unset_thing") is None


# ---------------------------------------------------------------------------
# sdk.py: in-process integration (mode 2)
# ---------------------------------------------------------------------------


def test_build_sdk_tools_exposes_exactly_three_callables(service):
    tools = build_sdk_tools(service)
    assert set(tools) == {"search_tools", "load_tool", "call_tool"}
    assert all(callable(fn) for fn in tools.values())


def test_sdk_tools_callables_work_end_to_end(bundle):
    tools = sdk_tools(bundle, StubRetriever(), MockDispatcher())
    refs = tools["search_tools"]("issue", k=3)
    assert refs and set(refs[0]) == {"id", "title", "description"}

    schema = tools["load_tool"]("github.create_issue")
    assert schema["input_schema"]["required"] == ["repo", "title"]

    result = tools["call_tool"]("github.create_issue", {"repo": "o/n", "title": "t"})
    assert result["id"] == "github.create_issue"


# ---------------------------------------------------------------------------
# http_sidecar.py: mode 3 (stdlib http.server only)
# ---------------------------------------------------------------------------


def _post(base_url: str, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


@pytest.fixture
def http_base_url(bundle):
    svc = OKTSService(bundle=bundle, retriever=StubRetriever(), dispatcher=MockDispatcher())
    httpd = http_serve(svc, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_http_search_endpoint(http_base_url):
    status, body = _post(http_base_url, "/search", {"query": "issue", "k": 3})
    assert status == 200
    assert body["results"]
    assert set(body["results"][0]) == {"id", "title", "description"}


def test_http_load_endpoint(http_base_url):
    status, body = _post(http_base_url, "/load", {"id": "github.create_issue"})
    assert status == 200
    assert body["input_schema"]["required"] == ["repo", "title"]
    assert body["side_effects"] == "write"


def test_http_load_unknown_id_returns_404(http_base_url):
    status, body = _post(http_base_url, "/load", {"id": "nope.nope"})
    assert status == 404
    assert "error" in body


def test_http_call_endpoint_valid_args(http_base_url):
    status, body = _post(
        http_base_url, "/call", {"id": "github.create_issue", "args": {"repo": "o/n", "title": "t"}}
    )
    assert status == 200
    assert body["result"]["id"] == "github.create_issue"


def test_http_call_endpoint_rejects_missing_required(http_base_url):
    status, body = _post(http_base_url, "/call", {"id": "github.create_issue", "args": {}})
    assert status == 400
    assert "error" in body


def test_http_call_endpoint_credentials_never_leak(http_base_url):
    status, body = _post(
        http_base_url,
        "/call",
        {"id": "slack.send_message", "args": {"channel": "#general", "text": "hi"}},
    )
    assert status == 200
    assert "slack_oauth" not in json.dumps(body)


def test_http_unknown_route_404(http_base_url):
    status, body = _post(http_base_url, "/nope", {})
    assert status == 404


# ---------------------------------------------------------------------------
# mcp_server.py: import guard + service assembly (mode 1)
# ---------------------------------------------------------------------------


def test_mcp_server_module_imports_without_error():
    import okts.serve.mcp_server as mcp_server

    assert hasattr(mcp_server, "main")
    assert hasattr(mcp_server, "build_service")


def test_mcp_server_require_mcp_raises_clear_error_when_absent(monkeypatch):
    import okts.serve.mcp_server as mcp_server

    monkeypatch.setattr(mcp_server, "_MCP_IMPORT_ERROR", ImportError("simulated: no mcp installed"))
    with pytest.raises(RuntimeError, match="mcp"):
        mcp_server._require_mcp()


def test_build_service_uses_naive_fallback_retriever_and_empty_registry(bundle_dir):
    from okts.serve.dispatch import DispatcherRegistry
    from okts.serve.mcp_server import NaiveFallbackRetriever, build_service

    service = build_service(bundle_dir=bundle_dir)
    assert isinstance(service.retriever, NaiveFallbackRetriever)
    assert isinstance(service.dispatcher, DispatcherRegistry)

    refs = service.search_tools("issue", k=5)
    assert refs

    # an empty DispatcherRegistry declines every concept (supports() is False),
    # so call_tool raises DispatchNotSupportedError -- a clear, safe error, not
    # a crash -- before ever reaching a per-interface "not configured" dispatch.
    with pytest.raises(DispatchNotSupportedError):
        service.call_tool("github.create_issue", {"repo": "o/n", "title": "t"})


def test_build_service_accepts_injected_retriever_and_dispatcher(bundle_dir):
    from okts.serve.mcp_server import build_service

    service = build_service(bundle_dir=bundle_dir, retriever=StubRetriever(), dispatcher=MockDispatcher())
    result = service.call_tool("github.create_issue", {"repo": "o/n", "title": "t"})
    assert result["id"] == "github.create_issue"
