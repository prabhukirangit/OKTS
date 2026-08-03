"""Layer 4 / phase 3 — pre-dispatch policies (``PreDispatchPolicy``).

These are the **policy-based gating** answer to "how do you keep a central
``call_tool`` proxy safe" — deliberately NOT value-level SSRF/SQL sanitization,
which is a downstream-tool property OKTS cannot generically enforce (attempting
it at the proxy is false security). Instead each policy is a small,
composable gate run at the single dispatch choke point
(``OKTSService._prepare_call``), after arg-validation and before dispatch, that
can **allow**, **mutate**, or **deny** (raise :class:`~okts.serve.service.PolicyDenied`).

All are opt-in: an ``OKTSService`` with no ``policies=`` behaves exactly as
before. Wire them per deployment::

    from okts.serve.policy import SideEffectPolicy
    service = OKTSService(bundle, retriever, dispatcher,
                          policies=[SideEffectPolicy(read_only=True)])

Policies MUST NOT put a credential/secret onto the returned args (invariant #4)
and should log by id/name, never by value.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from okts.core.model import Interface, OKTConcept, SideEffects
from okts.serve.service import PolicyDenied

__all__ = [
    "SideEffectPolicy",
    "RateLimitPolicy",
    "ArgRedactionPolicy",
    "DomainAllowlistPolicy",
]

log = logging.getLogger(__name__)


@dataclass
class SideEffectPolicy:
    """Gate a call on its declared ``side_effects`` — the metadata OKTS already
    carries (``read`` | ``write`` | ``destructive``) but never enforced.

    - ``read_only``: when True, only ``read`` tools may run; anything else is
      denied. The safe mode for an untrusted/exploratory agent.
    - ``require_confirmation``: effect classes that need an explicit host
      opt-in. A matching call is denied unless the host passed
      ``scope={"confirmation_key": True}`` (default key ``"confirmed"``). The
      flag comes from the HOST via ``call_tool(..., scope=...)``, never from the
      agent's args — so an agent cannot self-authorize a destructive call.
    """

    read_only: bool = False
    require_confirmation: frozenset[SideEffects] = frozenset(
        {SideEffects.WRITE, SideEffects.DESTRUCTIVE}
    )
    confirmation_key: str = "confirmed"

    def check(self, concept: OKTConcept, args: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
        effect = concept.side_effects
        if self.read_only and effect != SideEffects.READ:
            raise PolicyDenied(
                f"read-only mode: tool {concept.id!r} has side_effects={effect.value!r} "
                f"and may not run"
            )
        if effect in self.require_confirmation and not scope.get(self.confirmation_key):
            raise PolicyDenied(
                f"tool {concept.id!r} has side_effects={effect.value!r} and requires "
                f"host confirmation; pass scope={{{self.confirmation_key!r}: True}} to allow it"
            )
        return args


@dataclass
class RateLimitPolicy:
    """Fixed-window in-memory rate limit, keyed by ``concept.id``.

    At most ``max_calls`` per ``window_seconds`` per tool; the next call in the
    window is denied. In-process only (a demo/skeleton — swap in Redis etc. for
    a multi-process deployment). ``_now`` is injectable for deterministic tests.
    """

    max_calls: int = 60
    window_seconds: float = 60.0
    _now: Any = time.monotonic
    _hits: dict[str, list[float]] = field(default_factory=dict)

    def check(self, concept: OKTConcept, args: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
        now = self._now()
        cutoff = now - self.window_seconds
        recent = [t for t in self._hits.get(concept.id, ()) if t > cutoff]
        if len(recent) >= self.max_calls:
            raise PolicyDenied(
                f"rate limit exceeded for tool {concept.id!r} "
                f"({self.max_calls}/{self.window_seconds:g}s)"
            )
        recent.append(now)
        self._hits[concept.id] = recent
        return args


@dataclass
class ArgRedactionPolicy:
    """A **mutate** policy: strip disallowed keys from ``args`` before dispatch.

    Two modes (choose one): ``deny`` drops any listed key; ``allow`` keeps ONLY
    listed keys and drops the rest. Useful to defensively remove fields an
    upstream tool must never receive (e.g. an injected ``__admin`` flag), or to
    pin a tool to a known parameter set. Never raises — it only removes keys.
    """

    deny: frozenset[str] = frozenset()
    allow: Optional[frozenset[str]] = None

    def check(self, concept: OKTConcept, args: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(args)
        if self.allow is not None:
            dropped = [k for k in cleaned if k not in self.allow]
            cleaned = {k: v for k, v in cleaned.items() if k in self.allow}
        else:
            dropped = [k for k in cleaned if k in self.deny]
            for k in dropped:
                cleaned.pop(k, None)
        if dropped:
            log.debug("ArgRedactionPolicy stripped keys %s from tool %r", dropped, concept.id)
        return cleaned


@dataclass
class DomainAllowlistPolicy:
    """Egress allowlist for network-bound tools (``http`` / ``search``).

    Denies a call whose destination host is not in ``allowed_hosts``. **Fails
    closed**: if the destination host cannot be determined it DENIES rather than
    allowing — a guard that silently checks nothing is worse than none.

    Hosts are gathered from three places:

    1. ``concept.target`` — usable only when it carries a full URL (e.g. a
       hand-authored ``"POST https://api.x/…"`` target or a search endpoint URL).
    2. ``host_resolver(concept, args)`` — a caller-supplied hook returning the
       real destination host. **This is required for OpenAPI-adapted tools**,
       whose ``target`` is ``"METHOD /path"`` with NO host (the base URL lives in
       your wired HTTP client, which OKTS never sees). Wire it to your client's
       base-URL config so this policy can actually constrain those tools.
    3. ``url_arg_keys`` — argument keys that carry a URL (the residual SSRF
       vector where a tool legitimately takes a URL). A key that is present but
       whose value has no parseable host is treated as a violation (denied).

    With ``require_resolvable_host=True`` (default), an in-scope call for which
    NO host could be determined from any source is denied. Set it False only if
    you accept that unresolvable targets pass through unchecked.

    This is coarse egress *scoping*, not request inspection.
    """

    allowed_hosts: frozenset[str] = frozenset()
    url_arg_keys: tuple[str, ...] = ("url", "endpoint")
    interfaces: frozenset[Interface] = frozenset({Interface.HTTP, Interface.SEARCH})
    host_resolver: Optional[Callable[[OKTConcept, dict[str, Any]], Optional[str]]] = None
    require_resolvable_host: bool = True

    def check(self, concept: OKTConcept, args: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
        if concept.interface not in self.interfaces:
            return args

        allowed = {h.lower() for h in self.allowed_hosts}
        hosts: list[str] = []

        target_host = _host_of(concept.target or "")
        if target_host:
            hosts.append(target_host)

        if self.host_resolver is not None:
            resolved = self.host_resolver(concept, args)
            if resolved:
                hosts.append(resolved.strip().lower())

        for key in self.url_arg_keys:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                host = _host_of(value)
                if not host:
                    # a URL-bearing arg with no parseable host (scheme-less,
                    # protocol-relative, malformed) — refuse rather than let an
                    # unconstrained destination through.
                    raise PolicyDenied(
                        f"tool {concept.id!r} argument {key!r}={value!r} has no parseable "
                        f"host; refusing (egress allowlist cannot verify it)"
                    )
                hosts.append(host)

        for host in hosts:
            if host not in allowed:
                raise PolicyDenied(
                    f"egress to host {host!r} is not on the allowlist for tool {concept.id!r}"
                )

        if self.require_resolvable_host and not hosts:
            raise PolicyDenied(
                f"egress host for tool {concept.id!r} could not be determined "
                f"(target={concept.target!r}); refusing. Supply a host_resolver "
                f"(e.g. from your HTTP client's base URL) or a full-URL target, "
                f"or set require_resolvable_host=False to allow unresolvable targets."
            )
        return args


def _host_of(value: str) -> str:
    """Extract a lowercased hostname from a target/URL string. Handles a bare
    URL, an OpenAPI ``"METHOD https://host/path"`` target, or returns ``""`` when
    no scheme-qualified host is present (scheme-less/relative/empty)."""
    token = value.strip().rsplit(" ", 1)[-1].strip()  # drop a leading HTTP verb
    if "://" not in token:
        return ""
    return (urlparse(token).hostname or "").lower()
