"""Protocol-level MCP client checks; local Python function calls are not accepted."""

from __future__ import annotations

from src.mcp_client import call_mcp_tool, read_mcp_policy_resource


def test_mcp_tool_call_uses_protocol_transport(monkeypatch) -> None:
    monkeypatch.setenv("MCP_TRANSPORT", "in_process")
    result = call_mcp_tool(
        "lookup_credit_bureau",
        {"pan_or_entity_id": "ABCDE1234F", "business_name": "Apex Gears", "vintage_years": 4.0},
    )
    assert result["bureau_score"] == 785
    assert result["mcp_transport"] == "in_process"
    assert "mcp_transport_fallback" not in result


def test_mcp_resource_read_uses_protocol_transport(monkeypatch) -> None:
    monkeypatch.setenv("MCP_TRANSPORT", "in_process")
    result = read_mcp_policy_resource("policy://fair-lending-standard")
    assert "Fair Lending" in result["content"]
    assert result["mcp_transport"] == "in_process"

