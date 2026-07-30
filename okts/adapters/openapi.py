"""Layer 1 adapter: OpenAPI/REST spec -> OKT concepts.

One concept per ``operationId`` (synthesized from ``<method>_<path>`` if the
spec omits it). Path/query/header parameters and the JSON ``requestBody`` are
merged into a single ``input_schema`` object. ``path`` + ``method`` become
``target`` (e.g. ``"POST /v1/charges"``); ``interface: http``; ``auth`` is
resolved from the operation's (or the spec's global) ``security`` requirement
against ``components.securitySchemes``.
"""

from __future__ import annotations

import re
from typing import Any

from okts.core.model import Interface, OKTConcept, SideEffects

__all__ = ["openapi_to_okt"]

_HTTP_METHODS = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}


def _synthesize_title(concept_id: str) -> str:
    tail = concept_id.rsplit(".", 1)[-1]
    words = [w for w in re.split(r"[_\-]+", tail) if w]
    return " ".join(w.capitalize() for w in words) or concept_id


def _synthesize_operation_id(method: str, path: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_")
    return f"{method}_{slug}"


def _side_effects_for_method(method: str) -> SideEffects:
    if method in ("get", "head", "options"):
        return SideEffects.READ
    if method == "delete":
        return SideEffects.DESTRUCTIVE
    return SideEffects.WRITE


def _build_input_schema(operation: dict[str, Any], path_level_params: list[Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param in [*path_level_params, *(operation.get("parameters") or [])]:
        if not isinstance(param, dict) or "name" not in param:
            continue
        pname = param["name"]
        pschema = dict(param.get("schema") or {"type": "string"})
        if param.get("description") and "description" not in pschema:
            pschema["description"] = param["description"]
        properties[pname] = pschema
        if param.get("required"):
            required.append(pname)

    request_body = operation.get("requestBody")
    if isinstance(request_body, dict):
        content = request_body.get("content") or {}
        body_schema = None
        for media_type in ("application/json", *content.keys()):
            if media_type in content:
                body_schema = (content[media_type] or {}).get("schema")
                break
        if isinstance(body_schema, dict):
            if body_schema.get("type") in (None, "object") and isinstance(body_schema.get("properties"), dict):
                for k, v in body_schema["properties"].items():
                    properties.setdefault(k, v)
                for r in body_schema.get("required") or []:
                    if r not in required:
                        required.append(r)
            else:
                properties["body"] = body_schema
                if request_body.get("required"):
                    required.append("body")

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _resolve_auth(
    operation: dict[str, Any],
    global_security: Any,
    security_schemes: dict[str, Any],
) -> str | None:
    sec = operation.get("security", global_security)
    if not sec:
        return None
    for entry in sec:
        if isinstance(entry, dict) and entry:
            name = next(iter(entry))
            return name
    return None


def openapi_to_okt(spec: dict[str, Any], *, auth: str | None = None) -> list[OKTConcept]:
    """Convert an OpenAPI 3.x spec dict into one :class:`OKTConcept` per operation."""
    concepts: list[OKTConcept] = []
    paths = spec.get("paths") or {}
    security_schemes = ((spec.get("components") or {}).get("securitySchemes")) or {}
    global_security = spec.get("security")

    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        path_level_params = path_item.get("parameters") or []

        for method, operation in path_item.items():
            method_l = method.lower()
            if method_l not in _HTTP_METHODS or not isinstance(operation, dict):
                continue

            op_id = operation.get("operationId") or _synthesize_operation_id(method_l, path)
            input_schema = _build_input_schema(operation, path_level_params)
            description = (
                operation.get("description")
                or operation.get("summary")
                or f"{method_l.upper()} {path}"
            )
            title = operation.get("summary") or _synthesize_title(op_id)
            tags = list(operation.get("tags") or [])
            auth_name = auth or _resolve_auth(operation, global_security, security_schemes)

            concepts.append(
                OKTConcept(
                    id=op_id,
                    title=title,
                    description=description,
                    tags=tags,
                    input_schema=input_schema,
                    interface=Interface.HTTP,
                    target=f"{method_l.upper()} {path}",
                    auth=auth_name,
                    side_effects=_side_effects_for_method(method_l),
                )
            )
    return concepts
