"""A tiny, real MCP server over stdio — spawned by the `live` MCP tests.

Not imported by the test process: the live test launches it as a subprocess
(`python tests/fixtures/live_mcp_server.py`) and connects to it over stdio via
`okts.adapters.mcp.load_mcp_tools_live`, exercising the genuine `mcp` SDK path
end to end. Requires the optional `mcp` extra; the test skips if it's absent.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test-live")


@mcp.tool()
def echo(text: str) -> str:
    """Echo back the given text unchanged."""
    return text


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return their sum."""
    return a + b


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
