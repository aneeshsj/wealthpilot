"""MCP client used by the underwriting graph.

The default local transport is an in-process MCP session. This keeps the graph
synchronous while still exercising protocol initialization, tool calls, and
resource reads; a separate stdio server remains available by configuration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

try:
    from mcp import Client, StdioServerParameters
    _MCP_HIGH_LEVEL_CLIENT = True
except ImportError:  # MCP 1.x
    from mcp import ClientSession, StdioServerParameters  # type: ignore[no-redef]
    from mcp.client.stdio import stdio_client  # type: ignore[no-redef]
    _MCP_HIGH_LEVEL_CLIENT = False

try:
    from mcp.shared.memory import create_connected_server_and_client_session
except ImportError:  # pragma: no cover - only older SDKs lack in-process transport
    create_connected_server_and_client_session = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def _result_payload(result: Any) -> dict[str, Any]:
    if getattr(result, "isError", False):
        return {"error": _content_to_text(getattr(result, "content", []))}
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    content = getattr(result, "content", [])
    text = _content_to_text(content)
    try:
        decoded = json.loads(text)
        return decoded if isinstance(decoded, dict) else {"content": decoded}
    except (TypeError, json.JSONDecodeError):
        return {"content": text}


async def _request(kind: str, **kwargs: Any) -> dict[str, Any]:
    """Issue one MCP request through the configured protocol transport.

    The local-first default uses MCP's in-process memory transport. It still
    performs the full MCP initialize/list/call or read-resource exchange, but
    avoids a fragile subprocess boundary for a single-process demo. Stdio is
    retained as an explicit deployment option via MCP_TRANSPORT=stdio.
    """
    transport = os.getenv("MCP_TRANSPORT", "in_process").strip().lower()
    if transport == "in_process":
        if create_connected_server_and_client_session is None:
            raise RuntimeError("The installed MCP SDK does not support in-process transport")
        from src import mcp_server

        server = getattr(mcp_server.mcp, "_mcp_server", None)
        if server is None:
            server = getattr(mcp_server.mcp, "_lowlevel_server", mcp_server.mcp)
        async with create_connected_server_and_client_session(server) as client:
            if kind == "resource":
                result = await client.read_resource(kwargs["uri"])
                contents = getattr(result, "contents", [])
                return {"uri": kwargs["uri"], "content": _content_to_text(contents), "mcp_transport": "in_process"}
            result = await client.call_tool(kwargs["name"], kwargs.get("arguments", {}))
            payload = _result_payload(result)
            payload["mcp_transport"] = "in_process"
            return payload

    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    if _MCP_HIGH_LEVEL_CLIENT:
        async with Client(server) as client:  # type: ignore[name-defined]
            if kind == "resource":
                result = await client.read_resource(kwargs["uri"])
                contents = getattr(result, "contents", [])
                return {"uri": kwargs["uri"], "content": _content_to_text(contents)}
            result = await client.call_tool(kwargs["name"], kwargs.get("arguments", {}))
            return _result_payload(result)
    async with stdio_client(server) as (read, write):  # type: ignore[name-defined]
        async with ClientSession(read, write) as client:  # type: ignore[name-defined]
            await client.initialize()
            if kind == "resource":
                result = await client.read_resource(kwargs["uri"])
                contents = getattr(result, "contents", [])
                return {"uri": kwargs["uri"], "content": _content_to_text(contents)}
            result = await client.call_tool(kwargs["name"], kwargs.get("arguments", {}))
            return _result_payload(result)


def _run(coro: Any, *, fallback: Any = None) -> dict[str, Any]:
    try:
        async def bounded() -> dict[str, Any]:
            return await asyncio.wait_for(coro, timeout=2.0)

        return asyncio.run(bounded())
    except (TimeoutError, asyncio.TimeoutError) as exc:
        logger.error("MCP request timed out", extra={"error": str(exc)})
        return {"error": "MCP request timed out"}
    except Exception as exc:  # noqa: BLE001 - external transport boundary
        # Graph nodes run in a worker thread, but keep a clear failure contract
        # if a caller accidentally invokes this from an active event loop.
        logger.exception("MCP request could not be executed: %s", exc)
        return {"error": f"MCP request failed: {exc}"}


def read_mcp_policy_resource(resource_uri: str) -> dict[str, Any]:
    """Read a policy resource through the MCP client protocol."""
    return _run(_request("resource", uri=resource_uri))


def call_mcp_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a typed WealthPilot MCP tool through the MCP client protocol."""
    return _run(_request("tool", name=name, arguments=arguments))


__all__ = ["call_mcp_tool", "read_mcp_policy_resource"]
