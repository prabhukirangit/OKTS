# mcp_company_db.py
"""Shared upstream MCP server for the company-analysis examples.

A tiny FastMCP stdio server exposing one tool, ``get_internal_metrics``, over a
mock confidential database. Both examples point at THIS server:

- ``company_analysis_basic.py``  — binds this tool (plus native tools) directly.
- ``company_analysis_okts.py``   — ingests it into an OKT bundle and serves it
  behind the three meta-tools.

Run standalone (it just waits on stdio): ``python examples/mcp_company_db.py``.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Company Internal DB")

# Mock database
DB = {
    "ACME CORP": {
        "ARR": "$45M",
        "churn_rate": "2.1%",
        "key_risk": "High reliance on single enterprise client",
        "internal_notes": "Considering acquisition of competitor Beta Co.",
    },
    "BETA CO": {
        "ARR": "$12M",
        "churn_rate": "5.4%",
        "key_risk": "High burn rate",
        "internal_notes": "Open to acquisition offers.",
    },
}


@mcp.tool()
def get_internal_metrics(company_name: str) -> str:
    """Retrieve confidential internal company financial metrics and strategy notes.

    Args:
        company_name: The target company name (e.g., 'Acme Corp', 'Beta Co')
    """
    key = company_name.upper().strip()
    if key in DB:
        data = DB[key]
        return (
            f"Company: {company_name}\n"
            f"ARR: {data['ARR']}\n"
            f"Churn: {data['churn_rate']}\n"
            f"Key Risk: {data['key_risk']}\n"
            f"Notes: {data['internal_notes']}"
        )
    return f"No internal record found for company '{company_name}'."


if __name__ == "__main__":
    mcp.run(transport="stdio")
