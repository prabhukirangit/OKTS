"""Pre-dispatch policy engine + side-effects gating (okts/serve/policy.py)."""

from __future__ import annotations

import asyncio

import pytest

from okts.core.model import Bundle, Interface, OKTConcept, SideEffects
from okts.serve.dispatch import DispatcherRegistry, MockDispatcher
from okts.serve.mcp_server import NaiveFallbackRetriever
from okts.serve.policy import (
    ArgRedactionPolicy,
    DomainAllowlistPolicy,
    RateLimitPolicy,
    SideEffectPolicy,
)
from okts.serve.service import OKTSService, PolicyDenied


def _concept(id, effect=SideEffects.WRITE, interface=Interface.FUNCTION, target="t"):
    return OKTConcept(
        id=id,
        title="T",
        description="d",
        input_schema={"type": "object", "properties": {}},
        interface=interface,
        target=target,
        side_effects=effect,
    )


def _service(*concepts, policies=()):
    bundle = Bundle()
    for c in concepts:
        bundle.add(c)
    registry = DispatcherRegistry(
        dispatchers={i: MockDispatcher() for i in Interface}
    )
    return OKTSService(bundle, NaiveFallbackRetriever(), registry, policies=policies)


# --- no policies: unchanged behavior ---------------------------------------


def test_no_policies_is_unchanged():
    svc = _service(_concept("demo.write"))
    result = svc.call_tool("demo.write", {})
    assert result["id"] == "demo.write"
    assert svc.policies == ()


# --- SideEffectPolicy: confirmation gate -----------------------------------


def test_side_effect_requires_confirmation_for_write():
    svc = _service(_concept("demo.write", SideEffects.WRITE), policies=[SideEffectPolicy()])
    with pytest.raises(PolicyDenied):
        svc.call_tool("demo.write", {})
    # host confirmation via scope unblocks it
    assert svc.call_tool("demo.write", {}, scope={"confirmed": True})["id"] == "demo.write"


def test_side_effect_allows_read_without_confirmation():
    svc = _service(_concept("demo.read", SideEffects.READ), policies=[SideEffectPolicy()])
    assert svc.call_tool("demo.read", {})["id"] == "demo.read"


def test_side_effect_read_only_blocks_writes():
    svc = _service(
        _concept("demo.write", SideEffects.WRITE),
        _concept("demo.read", SideEffects.READ),
        policies=[SideEffectPolicy(read_only=True)],
    )
    with pytest.raises(PolicyDenied):
        svc.call_tool("demo.write", {}, scope={"confirmed": True})  # confirm can't override read-only
    assert svc.call_tool("demo.read", {})["id"] == "demo.read"


def test_confirmation_flag_comes_from_scope_not_agent_args():
    # an agent putting confirmed=True in its ARGS must NOT self-authorize
    svc = _service(_concept("demo.destroy", SideEffects.DESTRUCTIVE), policies=[SideEffectPolicy()])
    with pytest.raises(PolicyDenied):
        svc.call_tool("demo.destroy", {"confirmed": True})


# --- ArgRedactionPolicy: mutate --------------------------------------------


def test_arg_redaction_deny_strips_key():
    svc = _service(_concept("demo.read", SideEffects.READ), policies=[ArgRedactionPolicy(deny=frozenset({"__admin"}))])
    result = svc.call_tool("demo.read", {"__admin": True, "q": 1})
    assert result["args"] == {"q": 1}


def test_arg_redaction_allow_keeps_only_listed():
    svc = _service(_concept("demo.read", SideEffects.READ), policies=[ArgRedactionPolicy(allow=frozenset({"q"}))])
    result = svc.call_tool("demo.read", {"q": 1, "secret": "x"})
    assert result["args"] == {"q": 1}


# --- DomainAllowlistPolicy: egress scoping ---------------------------------


def test_domain_allowlist_denies_off_list_target():
    concept = _concept(
        "api.charge", interface=Interface.HTTP, target="POST https://evil.example/v1/charges"
    )
    svc = _service(concept, policies=[DomainAllowlistPolicy(allowed_hosts=frozenset({"api.stripe.com"}))])
    with pytest.raises(PolicyDenied):
        svc.call_tool("api.charge", {})


def test_domain_allowlist_allows_listed_target_and_ignores_non_http():
    ok = _concept("api.charge", interface=Interface.HTTP, target="POST https://api.stripe.com/v1/charges")
    fn = _concept("demo.read", SideEffects.READ, interface=Interface.FUNCTION)
    svc = _service(ok, fn, policies=[DomainAllowlistPolicy(allowed_hosts=frozenset({"api.stripe.com"}))])
    assert svc.call_tool("api.charge", {})["id"] == "api.charge"
    assert svc.call_tool("demo.read", {})["id"] == "demo.read"  # function iface untouched


def test_domain_allowlist_checks_url_bearing_arg():
    concept = _concept("search.web", SideEffects.READ, interface=Interface.SEARCH, target="search-api")
    svc = _service(concept, policies=[DomainAllowlistPolicy(allowed_hosts=frozenset({"safe.example"}))])
    # target has no host, but a url arg points off-list -> denied (SSRF vector)
    with pytest.raises(PolicyDenied):
        svc.call_tool("search.web", {"url": "https://evil.example/x"})


def test_domain_allowlist_fails_closed_when_host_unresolvable():
    # OpenAPI-style target "POST /path" has no host -> the guard must DENY, not
    # silently allow (that would be false egress protection).
    concept = _concept("api.charge", interface=Interface.HTTP, target="POST /v1/charges")
    svc = _service(concept, policies=[DomainAllowlistPolicy(allowed_hosts=frozenset({"api.stripe.com"}))])
    with pytest.raises(PolicyDenied):
        svc.call_tool("api.charge", {})


def test_domain_allowlist_host_resolver_enables_openapi_target():
    concept = _concept("api.charge", interface=Interface.HTTP, target="POST /v1/charges")
    good = DomainAllowlistPolicy(
        allowed_hosts=frozenset({"api.stripe.com"}),
        host_resolver=lambda c, a: "api.stripe.com",
    )
    assert _service(concept, policies=[good]).call_tool("api.charge", {})["id"] == "api.charge"

    bad = DomainAllowlistPolicy(
        allowed_hosts=frozenset({"api.stripe.com"}),
        host_resolver=lambda c, a: "evil.example",
    )
    with pytest.raises(PolicyDenied):
        _service(concept, policies=[bad]).call_tool("api.charge", {})


def test_domain_allowlist_denies_scheme_less_url_arg():
    # a url-bearing arg with no parseable host must be refused, not passed through
    concept = _concept("search.web", SideEffects.READ, interface=Interface.SEARCH, target="POST https://safe.example/s")
    svc = _service(concept, policies=[DomainAllowlistPolicy(allowed_hosts=frozenset({"safe.example"}))])
    with pytest.raises(PolicyDenied):
        svc.call_tool("search.web", {"url": "evil.example/steal"})


def test_domain_allowlist_can_opt_out_of_fail_closed():
    concept = _concept("api.charge", interface=Interface.HTTP, target="POST /v1/charges")
    policy = DomainAllowlistPolicy(
        allowed_hosts=frozenset({"api.stripe.com"}), require_resolvable_host=False
    )
    assert _service(concept, policies=[policy]).call_tool("api.charge", {})["id"] == "api.charge"


# --- RateLimitPolicy -------------------------------------------------------


def test_rate_limit_denies_past_the_window():
    clock = [0.0]
    policy = RateLimitPolicy(max_calls=2, window_seconds=10.0, _now=lambda: clock[0])
    svc = _service(_concept("demo.read", SideEffects.READ), policies=[policy])
    assert svc.call_tool("demo.read", {})["id"] == "demo.read"
    assert svc.call_tool("demo.read", {})["id"] == "demo.read"
    with pytest.raises(PolicyDenied):
        svc.call_tool("demo.read", {})
    # advance past the window -> allowed again
    clock[0] = 20.0
    assert svc.call_tool("demo.read", {})["id"] == "demo.read"


# --- policies apply on the async path too ----------------------------------


def test_policy_denies_on_async_path():
    svc = _service(_concept("demo.write", SideEffects.WRITE), policies=[SideEffectPolicy()])
    with pytest.raises(PolicyDenied):
        asyncio.run(svc.acall_tool("demo.write", {}))
    assert asyncio.run(svc.acall_tool("demo.write", {}, scope={"confirmed": True}))["id"] == "demo.write"


# --- policy chain: order + mutation threading ------------------------------


def test_policy_chain_threads_mutation_then_gates():
    svc = _service(
        _concept("demo.write", SideEffects.WRITE),
        policies=[ArgRedactionPolicy(deny=frozenset({"drop"})), SideEffectPolicy()],
    )
    result = svc.call_tool("demo.write", {"drop": 1, "keep": 2}, scope={"confirmed": True})
    assert result["args"] == {"keep": 2}
