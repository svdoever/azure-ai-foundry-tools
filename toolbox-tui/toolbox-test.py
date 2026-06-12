import asyncio
import httpx
import json

from azure.identity import DefaultAzureCredential
from copilot import CopilotClient
from copilot.tools import Tool, ToolInvocation, ToolResult

# ── Configuration ─────────────────────────────────────────────────────────────

endpoint = "https://svdofoundryproject-resource.services.ai.azure.com/api/projects/svdofoundryproject"
toolbox_name = "test"
toolbox_version = "2"
model_deployment = "gpt-5.4"
azure_api_version = "2024-12-01-preview"

from urllib.parse import urlparse
_parsed = urlparse(endpoint)
toolbox_url = f"{endpoint.rstrip('/')}/toolboxes/{toolbox_name}/versions/{toolbox_version}/mcp?api-version=v1"
# OpenAI-compatible inference base for the Foundry (AI Services) resource
model_base_url = f"{_parsed.scheme}://{_parsed.netloc}"

# ── Reusable functions (can be pulled into a hosted agent main.py) ────────────

def _get_toolbox_token() -> str:
    """Get a bearer token for the toolbox MCP endpoint."""
    credential = DefaultAzureCredential()
    return credential.get_token("https://ai.azure.com/.default").token


def _get_model_token() -> str:
    """Get a bearer token for the Foundry model inference endpoint."""
    credential = DefaultAzureCredential()
    return credential.get_token("https://cognitiveservices.azure.com/.default").token


def _get_toolbox_headers(token: str) -> dict:
    """Headers required for toolbox MCP calls."""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "Foundry-Features": "Toolboxes=V1Preview",
    }


class McpBridge:
    """HTTP-based MCP client that connects to a Foundry toolbox MCP endpoint."""

    def __init__(self, endpoint: str, token: str):
        self.endpoint = endpoint
        self.headers = _get_toolbox_headers(token)
        self._session_id: str | None = None
        self._client = httpx.AsyncClient(timeout=60.0)
        self._req_id = 0

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _request_headers(self) -> dict:
        headers = dict(self.headers)
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        return headers

    async def initialize(self) -> str:
        """Send MCP initialize + notifications/initialized."""
        resp = await self._client.post(
            self.endpoint, headers=self.headers,
            json={
                "jsonrpc": "2.0", "id": self._next_id(), "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "copilot-toolbox-bridge", "version": "1.0.0"},
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()
        self._session_id = resp.headers.get("mcp-session-id")

        await self._client.post(
            self.endpoint, headers=self._request_headers(),
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        return data.get("result", {}).get("serverInfo", {}).get("name", "unknown")

    async def list_tools(self) -> list[dict]:
        """Call tools/list and return the tools array."""
        resp = await self._client.post(
            self.endpoint, headers=self._request_headers(),
            json={"jsonrpc": "2.0", "id": self._next_id(), "method": "tools/list", "params": {}},
        )
        resp.raise_for_status()
        return resp.json().get("result", {}).get("tools", [])

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Call tools/call and return the text result."""
        resp = await self._client.post(
            self.endpoint, headers=self._request_headers(),
            json={
                "jsonrpc": "2.0", "id": self._next_id(), "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
        )
        resp.raise_for_status()
        result = resp.json().get("result", {})
        content = result.get("content", [])
        texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
        return "\n".join(t for t in texts if t) or json.dumps(result)

    async def close(self):
        await self._client.aclose()


def _make_copilot_tools(bridge: McpBridge, mcp_tools: list[dict]) -> list[Tool]:
    """Convert MCP tool definitions into Copilot SDK Tool objects."""
    tools = []
    for mcp_tool in mcp_tools:
        mcp_name = mcp_tool["name"]
        # Copilot SDK rejects tool names with dots/hyphens
        sdk_name = mcp_name.replace(".", "_").replace("-", "_")
        desc = mcp_tool.get("description", f"MCP tool: {mcp_name}")
        schema = mcp_tool.get("inputSchema", {"type": "object", "properties": {}})

        def _make_handler(original_name):
            async def handler(invocation: ToolInvocation) -> ToolResult:
                args = invocation.arguments if isinstance(invocation.arguments, dict) else {}
                try:
                    result_text = await bridge.call_tool(original_name, args)
                    return ToolResult(text_result_for_llm=result_text)
                except Exception as e:
                    return ToolResult(text_result_for_llm="", result_type="error", error=str(e))
            return handler

        tools.append(Tool(
            name=sdk_name,
            description=desc,
            parameters=schema,
            handler=_make_handler(mcp_name),
            skip_permission=True,
        ))
    return tools


# [START copilot_sdk_toolbox]
_client = None
_bridge = None

async def create_agent_with_toolbox():
    """Start CopilotClient and connect to a Foundry toolbox via MCP."""
    global _client, _bridge

    _client = CopilotClient()
    await _client.start()

    token = _get_toolbox_token()
    _bridge = McpBridge(toolbox_url, token)
    await _bridge.initialize()


async def call_agent_with_toolbox(user_input: str):
    """Send a message to a Copilot session with toolbox tools and print the response."""
    mcp_tools = await _bridge.list_tools()
    tools = _make_copilot_tools(_bridge, mcp_tools)

    provider = {
        "type": "azure",
        "base_url": model_base_url,
        "bearer_token": _get_model_token(),
        "model_id": model_deployment,
        "azure": {"api_version": azure_api_version},
    }

    session = await _client.create_session(
        on_permission_request=lambda req, ctx: {"kind": "approved"},
        tools=tools,
        model=model_deployment,
        provider=provider,
    )
    try:
        event = await session.send_and_wait(user_input, timeout=120.0)
        print(event.data.content if event else "(no response)")
    finally:
        await session.disconnect()


async def close_agent():
    """Shut down the Copilot client and MCP bridge."""
    if _client:
        await _client.stop()
    if _bridge:
        await _bridge.close()
# [END copilot_sdk_toolbox]


# ── Script entry point ────────────────────────────────────────────────────
async def main():
    await create_agent_with_toolbox()
    try:
        await call_agent_with_toolbox("What tools are available?")
    finally:
        await close_agent()

asyncio.run(main())
