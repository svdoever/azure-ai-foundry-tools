"""Interactive TUI for exploring and testing a Foundry toolbox.

Tabs:
  * Tools  – pick a toolbox + version, then browse its tools. When the toolbox
             has "Tool search" enabled it exposes only `tool_search` +
             `call_tool`, so you search for tools and run them via `call_tool`;
             otherwise the real tools are listed and called directly. Either
             way, selecting a tool builds a parameter form. Also lists the
             Copilot built-in tools.
  * Skills – read the skills defined in the toolbox.
  * Agent  – ask the LLM (which has access to all the tools) a question.

Run with:  python toolbox_tui.py
"""

import asyncio
import json
from pathlib import Path
from urllib.parse import urlparse

import httpx
from azure.identity import DefaultAzureCredential
from copilot import CopilotClient
from copilot.tools import Tool, ToolInvocation, ToolResult

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Markdown,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
    TextArea,
)

# ── Configuration ────────────────────────────────────────────────────────────

data_plane_api_version = "v1"

_CONFIG_FILE = Path(__file__).with_name("toolbox-tui.config")

_DEFAULT_CONFIG: dict = {
    "resource": "project-resource",
    "project": "project",
    "model_deployment": "gpt-5.4",
    "azure_api_version": "2024-12-01-preview",
    "toolbox_name": "",
    "toolbox_version": "",
}


def _load_config() -> dict:
    """Load config from file, falling back to defaults for any missing keys."""
    if _CONFIG_FILE.exists():
        try:
            data = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            return {**_DEFAULT_CONFIG, **data}
        except Exception:  # noqa: BLE001
            pass
    return dict(_DEFAULT_CONFIG)


def _save_config(cfg: dict) -> None:
    """Persist config dict to the config file (best-effort)."""
    try:
        _CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _endpoint_from(resource: str, project: str) -> str:
    return f"https://{resource}.services.ai.azure.com/api/projects/{project}"


def _model_base_from(resource: str) -> str:
    return f"https://{resource}.services.ai.azure.com"


def _build_toolbox_url(name: str, version: str, endpoint: str) -> str:
    """Build the MCP endpoint URL for a specific toolbox name + version."""
    return (
        f"{endpoint.rstrip('/')}/toolboxes/{name}/versions/{version}"
        f"/mcp?api-version={data_plane_api_version}"
    )


# ── Auth helpers ──────────────────────────────────────────────────────────────

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


async def _list_toolboxes(token: str, endpoint: str) -> list[dict]:
    """List the toolboxes available in the project (data-plane API)."""
    url = f"{endpoint.rstrip('/')}/toolboxes?api-version={data_plane_api_version}"
    headers = {"Authorization": f"Bearer {token}", "Foundry-Features": "Toolboxes=V1Preview"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json().get("data", [])


async def _list_toolbox_versions(token: str, name: str, endpoint: str) -> list[dict]:
    """List the versions of a toolbox (newest first), via the data-plane API."""
    url = (
        f"{endpoint.rstrip('/')}/toolboxes/{name}/versions"
        f"?api-version={data_plane_api_version}"
    )
    headers = {"Authorization": f"Bearer {token}", "Foundry-Features": "Toolboxes=V1Preview"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json().get("data", [])


# ── MCP bridge ────────────────────────────────────────────────────────────────

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

    @staticmethod
    def _raise_for_rpc_error(data: dict, what: str) -> None:
        """Raise if a JSON-RPC response carries an error object."""
        err = data.get("error")
        if err:
            msg = err.get("message", err) if isinstance(err, dict) else err
            raise RuntimeError(str(msg))

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
        data = resp.json()
        self._raise_for_rpc_error(data, "tools/list")
        return data.get("result", {}).get("tools", [])

    async def list_skills(self) -> list[dict]:
        """Read the toolbox skill discovery index (skill:// resources).

        Returns a list of {name, description, url} dicts.
        """
        resp = await self._client.post(
            self.endpoint, headers=self._request_headers(),
            json={"jsonrpc": "2.0", "id": self._next_id(),
                  "method": "resources/read", "params": {"uri": "skill://index.json"}},
        )
        resp.raise_for_status()
        contents = resp.json().get("result", {}).get("contents", [])
        for c in contents:
            text = c.get("text")
            if text:
                return json.loads(text).get("skills", [])
        return []

    async def read_resource(self, uri: str) -> str:
        """Read a single resource (e.g. a skill's SKILL.md) and return its text."""
        resp = await self._client.post(
            self.endpoint, headers=self._request_headers(),
            json={"jsonrpc": "2.0", "id": self._next_id(),
                  "method": "resources/read", "params": {"uri": uri}},
        )
        resp.raise_for_status()
        contents = resp.json().get("result", {}).get("contents", [])
        texts = [c.get("text", "") for c in contents if c.get("text")]
        return "\n\n".join(texts) or json.dumps(resp.json().get("result", {}), indent=2)

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
        data = resp.json()
        self._raise_for_rpc_error(data, f"tools/call {name}")
        result = data.get("result", {})
        content = result.get("content", [])
        texts: list[str] = []
        for c in content:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "text" and c.get("text"):
                texts.append(c["text"])
            elif c.get("type") == "resource":
                res = c.get("resource", {})
                if isinstance(res, dict) and res.get("text"):
                    texts.append(res["text"])
        return "\n".join(texts) or json.dumps(result, indent=2)

    async def search_tools(self, query: str, limit: int = 5) -> list[dict]:
        """Run the toolbox `tool_search` and return the discovered tool defs.

        The toolbox does not expose its underlying tools through ``tools/list``;
        it returns only ``tool_search`` + ``call_tool``. ``tool_search`` does a
        keyword search and returns a JSON payload ``{"tools": [...]}`` where each
        entry has ``name``/``title``/``description``/``inputSchema``. Those tools
        are then invoked indirectly via :meth:`call_discovered_tool`.
        """
        raw = await self.call_tool("tool_search", {"query": query, "limit": limit})
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # tool_search returns a plain-text message on error / no matches.
            raise RuntimeError(raw.strip() or "tool_search returned no usable result")
        return data.get("tools", [])

    async def call_discovered_tool(self, name: str, arguments: dict) -> str:
        """Invoke a tool discovered via :meth:`search_tools` through ``call_tool``."""
        return await self.call_tool("call_tool", {"name": name, "arguments": arguments})

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


# ── Form helpers ──────────────────────────────────────────────────────────────

def _coerce_value(raw: str, prop_schema: dict):
    """Convert a raw string from an Input into the JSON type the schema expects."""
    json_type = prop_schema.get("type")
    raw = raw.strip()
    if raw == "":
        return None
    if json_type in ("integer", "number"):
        try:
            return int(raw) if json_type == "integer" else float(raw)
        except ValueError:
            return raw
    if json_type in ("object", "array"):
        return json.loads(raw)
    return raw


def _parse_tool_json(text: str) -> list[dict]:
    """Extract a JSON array of {name, description} tools from an agent reply."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    tools = []
    for item in data:
        if isinstance(item, dict) and "name" in item:
            tools.append({"name": str(item["name"]),
                          "description": str(item.get("description", ""))})
        elif isinstance(item, str):
            tools.append({"name": item, "description": ""})
    return tools


def _format_list_error(msg: str) -> str:
    """Turn a raw tools/list error string into a compact human-readable warning.

    The Foundry MCP server embeds a JSON blob in the error message when one or
    more nested tool sources fail to enumerate. We extract just the per-source
    failures and show them on one line each.
    """
    # Try to pull out the embedded JSON: "... {\"errors\":[...]}"
    brace = msg.find("{")
    if brace != -1:
        try:
            data = json.loads(msg[brace:])
            errors = data.get("errors", [])
            if errors:
                lines = []
                for e in errors:
                    name = e.get("name", "?")
                    etype = e.get("type", "")
                    inner = e.get("error", {})
                    code = inner.get("code", "")
                    detail = inner.get("message", str(inner))
                    # Truncate long detail strings
                    if len(detail) > 120:
                        detail = detail[:117] + "…"
                    lines.append(f"  [b]{name}[/b] ({etype}): {code} — {detail}")
                prefix = msg[:brace].strip().rstrip("{").strip()
                return (
                    f"[b yellow]⚠ Partial enumeration failure:[/b yellow] {prefix}\n"
                    + "\n".join(lines)
                )
        except (json.JSONDecodeError, AttributeError):
            pass
    # Fallback: truncate the raw message
    if len(msg) > 200:
        msg = msg[:197] + "…"
    return f"[b yellow]⚠ Couldn't enumerate all tools:[/b yellow] {msg}"


# ── TUI application ───────────────────────────────────────────────────────────

class ToolboxApp(App):
    """Interactive overview + tester for a Foundry toolbox."""

    CSS = """
    #connect-bar {
        height: auto;
        margin: 0 0 1 0;
    }
    #toolbox-select, #toolbox-input {
        width: 26;
        margin: 0 1 0 0;
    }
    #version-select, #version-input {
        width: 18;
        margin: 0 1 0 0;
    }
    #connect-btn {
        width: 12;
    }
    #reload-btn {
        width: 5;
        min-width: 5;
    }
    .json-field {
        height: 5;
        border: tall $accent;
    }
    .string-field {
        height: 8;
        border: tall $accent;
    }
    .string-array-field {
        height: 5;
        border: tall $accent;
    }
    #tool-source {
        width: 36;
        margin: 0 1 1 0;
    }
    #search-bar {
        height: auto;
        margin: 0 0 1 0;
    }
    #search-query {
        width: 1fr;
    }
    #search-limit-label {
        width: auto;
        margin: 0 0 0 1;
        content-align: left middle;
    }
    #search-limit {
        width: 8;
        margin: 0 1 0 0;
    }
    #tool-list, #skill-list {
        width: 38;
        border: round $accent;
    }
    #tool-detail, #skill-detail {
        border: round $accent;
        padding: 0 1;
    }
    #detail-header {
        margin-bottom: 1;
    }
    .field-label {
        margin-top: 1;
        text-style: bold;
    }
    .field-help {
        color: $text-muted;
    }
    #result-tabs, #agent-result-tabs {
        height: 1fr;
        border: round $success;
    }
    #result-tabs > TabbedContent > ContentSwitcher,
    #agent-result-tabs > TabbedContent > ContentSwitcher {
        height: 1fr;
    }
    #result-raw, #result-json,
    #agent-result-raw, #agent-result-json {
        height: 1fr;
        padding: 0 1;
    }
    #result-md-scroll, #agent-result-md-scroll {
        height: 1fr;
        padding: 0 1;
    }
    #result-md, #agent-result-md {
        height: auto;
    }
    #skill-content {
        border: round $success;
        height: 1fr;
        padding: 0 1;
    }
    #form-actions, #agent-actions {
        height: auto;
        margin-top: 1;
    }
    Switch {
        height: auto;
    }
    #agent-input {
        height: 5;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+r", "refresh", "Refresh"),
        ("[", "list_shrink", "List ◀"),
        ("]", "list_grow", "List ▶"),
    ]

    def __init__(self) -> None:
        super().__init__()
        _cfg = _load_config()
        self.resource: str = _cfg["resource"]
        self.project: str = _cfg["project"]
        self.model_deployment: str = _cfg["model_deployment"]
        self.azure_api_version: str = _cfg["azure_api_version"]
        self.endpoint: str = _endpoint_from(self.resource, self.project)
        self.model_base_url: str = _model_base_from(self.resource)
        self.bridge: McpBridge | None = None
        self.client: CopilotClient | None = None
        self.token: str | None = None
        self.mcp_tools: list[dict] = []
        self.tools_by_name: dict[str, dict] = {}
        self.discovered_tools: list[dict] = []
        self.discovered_by_name: dict[str, dict] = {}
        self.copilot_tools: list[dict] = []
        self.skills: list[dict] = []
        self.skills_by_name: dict[str, dict] = {}
        self.current_tool: dict | None = None
        self.tool_source: str = "toolbox"
        self.server_name: str = "connecting…"
        self.toolbox_name: str = _cfg["toolbox_name"]
        self.toolbox_version: str = _cfg["toolbox_version"]
        self.search_mode: bool = False
        self.toolboxes: list[dict] = []
        self.versions: list[dict] = []
        self._loading: bool = False
        self.agent_session = None
        self.agent_history: list[tuple[str, str]] = []
        self._list_width: int = 38

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="tools-tab"):
            with TabPane("Tools", id="tools-tab"):
                with Vertical():
                    with Horizontal(id="connect-bar"):
                        yield Select([(self.toolbox_name, self.toolbox_name)], value=self.toolbox_name,
                                     allow_blank=False, id="toolbox-select")
                        yield Input(value=self.toolbox_name, placeholder="toolbox name",
                                    id="toolbox-input")
                        yield Select([("(latest)", "")], value="", allow_blank=False,
                                     id="version-select")
                        yield Input(placeholder="version", id="version-input")
                        yield Button("Connect", id="connect-btn", variant="primary")
                        yield Button("⟳", id="reload-btn", tooltip="Reload toolboxes & versions, reconnect at latest")
                    yield Select(
                        [("Toolbox tools", "toolbox"),
                         ("Copilot built-in tools", "copilot")],
                        value="toolbox", allow_blank=False, id="tool-source",
                    )
                    with Horizontal(id="search-bar"):
                        yield Input(
                            placeholder="Describe the capability you need, or a question…",
                            id="search-query",
                        )
                        yield Label("max:", id="search-limit-label")
                        yield Input(value="5", id="search-limit", type="integer")
                        yield Button("Search", id="search-btn", variant="primary")
                    with Horizontal():
                        yield ListView(id="tool-list")
                        with VerticalScroll(id="tool-detail"):
                            yield Static("Connecting to toolbox\u2026", id="detail-header")
                            yield Markdown("", id="detail-desc")
                            yield Vertical(id="form-fields")
                            with Horizontal(id="form-actions"):
                                yield Button("Execute", id="execute-btn", variant="primary", disabled=True)
                                yield Button("Clear", id="clear-btn")
                            yield Label("Result", classes="field-label")
                            with TabbedContent(id="result-tabs", initial="result-tab-raw"):
                                with TabPane("Raw", id="result-tab-raw"):
                                    yield TextArea("", id="result-raw", read_only=True)
                                with TabPane("Markdown", id="result-tab-md"):
                                    with VerticalScroll(id="result-md-scroll"):
                                        yield Markdown("", id="result-md")
                                with TabPane("JSON", id="result-tab-json"):
                                    yield TextArea("", id="result-json", read_only=True)
            with TabPane("Skills", id="skills-tab"):
                with Horizontal():
                    yield ListView(id="skill-list")
                    with VerticalScroll(id="skill-detail"):
                        yield Static("Loading skills…", id="skill-header")
                        yield TextArea("", id="skill-content", read_only=True)
            with TabPane("Agent", id="agent-tab"):
                with Vertical():
                    yield Label("Ask the agent (it can use every tool):", classes="field-label")
                    yield Input(placeholder="What tools are available?", id="agent-input")
                    with Horizontal(id="agent-actions"):
                        yield Button("Send", id="agent-send-btn", variant="primary")
                        yield Button("Clear thread", id="agent-clear-btn")
                    yield Label("Response", classes="field-label")
                    with TabbedContent(id="agent-result-tabs", initial="agent-result-tab-raw"):
                        with TabPane("Raw", id="agent-result-tab-raw"):
                            yield TextArea("", id="agent-result-raw", read_only=True)
                        with TabPane("Markdown", id="agent-result-tab-md"):
                            with VerticalScroll(id="agent-result-md-scroll"):
                                yield Markdown("", id="agent-result-md")
                        with TabPane("JSON", id="agent-result-tab-json"):
                            yield TextArea("", id="agent-result-json", read_only=True)
            with TabPane("Settings", id="settings-tab"):
                with Vertical():
                    yield Label("Foundry resource name", classes="field-label")
                    yield Static("The Azure AI Services resource (hostname prefix).",
                                 classes="field-help")
                    yield Input(id="cfg-resource", placeholder="my-resource")
                    yield Label("Project name", classes="field-label")
                    yield Static("The Foundry project name.", classes="field-help")
                    yield Input(id="cfg-project", placeholder="my-project")
                    yield Label("Model deployment", classes="field-label")
                    yield Input(id="cfg-model", placeholder="gpt-4o")
                    yield Label("Azure OpenAI API version", classes="field-label")
                    yield Input(id="cfg-apiversion", placeholder="2024-12-01-preview")
                    yield Button("Apply & reconnect", id="cfg-apply-btn", variant="primary")
                    yield Static("", id="cfg-status")
        yield Footer()

    async def on_mount(self) -> None:
        self.title = "Foundry Toolbox Tester"
        self.sub_title = f"{self.toolbox_name} v{self.toolbox_version or 'latest'}"
        self.query_one("#toolbox-input").display = False
        self.query_one("#version-input").display = False
        # Populate settings fields from loaded config
        self.query_one("#cfg-resource", Input).value = self.resource
        self.query_one("#cfg-project", Input).value = self.project
        self.query_one("#cfg-model", Input).value = self.model_deployment
        self.query_one("#cfg-apiversion", Input).value = self.azure_api_version
        self.load_toolboxes()

    # ── Connection / loading ──────────────────────────────────────────────

    @work(exclusive=True, group="connect")
    async def connect_and_load(self) -> None:
        header = self.query_one("#detail-header", Static)
        version = self.toolbox_version or (
            str(self.versions[0]["version"]) if self.versions else ""
        )
        self.toolbox_version = version
        self.sub_title = f"{self.toolbox_name} v{version or 'latest'}"
        header.update(f"Connecting to [b]{self.toolbox_name}[/b] v{version or 'latest'}…")

        try:
            if self.bridge is not None:
                await self.bridge.close()
            # Reset the agent session — it's bound to the previous toolbox/version
            await self._reset_agent_session()
            self.agent_history = []
            if self.token is None:
                self.token = _get_toolbox_token()
            self.bridge = McpBridge(_build_toolbox_url(self.toolbox_name, version, self.endpoint), self.token)
            self.server_name = await self.bridge.initialize()
        except Exception as e:  # noqa: BLE001
            header.update(f"[b red]Failed to connect:[/b red] {e}")
            return

        list_error: str | None = None
        try:
            self.mcp_tools = await self.bridge.list_tools()
        except Exception as e:  # noqa: BLE001
            self.mcp_tools = []
            list_error = str(e)

        self.tools_by_name = {t["name"]: t for t in self.mcp_tools}
        # reset per-connection state
        self.discovered_tools = []
        self.discovered_by_name = {}
        self.current_tool = None
        await self.query_one("#form-fields", Vertical).remove_children()
        self.query_one("#execute-btn", Button).disabled = True
        self._clear_result()

        # Detect whether "Tool search" is enabled for this toolbox: when it is,
        # the MCP surface is just tool_search + call_tool; otherwise the real
        # tools are returned directly by tools/list.
        names = {t["name"] for t in self.mcp_tools}
        self.search_mode = "tool_search" in names and "call_tool" in names
        self._update_search_bar()
        await self._populate_tool_list()

        if self.search_mode:
            self.sub_title = f"{self.toolbox_name} v{version} · tool_search + call_tool"
            header.update(
                f"[b]Connected to {self.toolbox_name} v{version}.[/b]\n"
                "[b]Tool search is enabled[/b] — the toolbox exposes only "
                "[b]tool_search[/b] + [b]call_tool[/b].\n"
                "Type a capability or question above and press [b]Search[/b] to "
                "discover tools, then select one to fill in its parameters and run it."
            )
        else:
            self.sub_title = f"{self.toolbox_name} v{version} · {len(self.mcp_tools)} tools"
            header.update(
                f"[b]Connected to {self.toolbox_name} v{version}.[/b]\n"
                f"[b]{len(self.mcp_tools)} tool(s)[/b] exposed directly. "
                "Select one on the left to fill in its parameters and run it."
            )
        if list_error:
            warn = _format_list_error(list_error)
            # Append to (don't replace) the success line already set above.
            current = (
                f"[b]Connected to {self.toolbox_name} v{version}.[/b]\n"
                f"{warn}"
            )
            header.update(current)
        self.load_skills()

    async def _populate_tool_list(self) -> None:
        """Fill #tool-list according to the currently selected source."""
        list_view = self.query_one("#tool-list", ListView)
        await list_view.clear()
        if self.tool_source == "toolbox":
            source = self.discovered_tools if self.search_mode else self.mcp_tools
        else:
            source = self.copilot_tools
        for t in source:
            name = t["name"]
            title = t.get("title")
            label = f"{name}" if not title or title == name else f"{name}  [dim]({title})[/dim]"
            item = ListItem(Label(label))
            item.tool_name = name  # type: ignore[attr-defined]
            await list_view.append(item)

    @work(exclusive=True, group="search")
    async def search_tools_worker(self, query: str, limit: int) -> None:
        header = self.query_one("#detail-header", Static)
        await self.query_one("#form-fields", Vertical).remove_children()
        self.query_one("#execute-btn", Button).disabled = True
        self._clear_result()
        header.update(f"Searching for tools matching [b]{query}[/b]…")
        try:
            self.discovered_tools = await self.bridge.search_tools(query, limit)
        except Exception as e:  # noqa: BLE001
            self.discovered_tools = []
            self.discovered_by_name = {}
            await self._populate_tool_list()
            header.update(f"[b yellow]No results.[/b yellow]\n{e}")
            return
        self.discovered_by_name = {t["name"]: t for t in self.discovered_tools}
        await self._populate_tool_list()
        if self.discovered_tools:
            header.update(
                f"[b]{len(self.discovered_tools)} tool(s)[/b] found for [b]{query}[/b].\n"
                "Select one on the left to fill in its parameters and execute it."
            )
        else:
            header.update(f"No tools matched [b]{query}[/b]. Try a broader query.")

    @work(exclusive=True, group="skills")
    async def load_skills(self) -> None:
        header = self.query_one("#skill-header", Static)
        list_view = self.query_one("#skill-list", ListView)
        # Clear stale content from the previous version immediately
        await list_view.clear()
        self.query_one("#skill-content", TextArea).text = ""
        header.update("Loading skills…")
        try:
            self.skills = await self.bridge.list_skills()
        except Exception as e:  # noqa: BLE001
            header.update(f"[b red]Failed to load skills:[/b red] {e}")
            return
        self.skills_by_name = {s["name"]: s for s in self.skills}
        for s in self.skills:
            item = ListItem(Label(s["name"]))
            item.skill_name = s["name"]  # type: ignore[attr-defined]
            await list_view.append(item)
        if self.skills:
            header.update(
                f"[b]{len(self.skills)} skill(s)[/b] defined in this toolbox.\n"
                "Select one to read its definition."
            )
        else:
            header.update(
                f"[b]No skills[/b] are defined in [b]{self.toolbox_name} v{self.toolbox_version}[/b].\n"
                "Add a skill to this toolbox version in the Foundry portal to see it here."
            )

    @work(exclusive=True, group="copilot-tools")
    async def load_copilot_tools(self) -> None:
        """Enumerate the Copilot CLI built-in tools by asking the agent.

        The Copilot runtime exposes no structured tool-listing API, so we ask
        the model to report its own tools as JSON and cache the result.
        """
        header = self.query_one("#detail-header", Static)
        header.update("Asking the agent to enumerate its built-in tools…")
        prompt = (
            "List every tool you have available in this session. "
            "Respond with ONLY a JSON array, no prose, no code fences. "
            'Each element must be {"name": "<tool name>", "description": "<one line>"}.'
        )
        try:
            text = await self._run_agent_turn(prompt)
            self.copilot_tools = _parse_tool_json(text)
        except Exception as e:  # noqa: BLE001
            header.update(f"[b red]Failed to list Copilot tools:[/b red] {e}")
            return
        await self._populate_tool_list()
        header.update(
            f"[b]{len(self.copilot_tools)} Copilot built-in tools[/b] "
            "(as reported by the agent).\n"
            "These run inside the Copilot runtime — use the [b]Agent[/b] tab to exercise them."
        )

    def action_refresh(self) -> None:
        self.connect_and_load()

    def action_list_shrink(self) -> None:
        """Shrink the tool/skill list pane by 4 columns."""
        self._list_width = max(16, self._list_width - 4)
        for w_id in ("#tool-list", "#skill-list"):
            try:
                self.query_one(w_id).styles.width = self._list_width
            except Exception:  # noqa: BLE001
                pass

    def action_list_grow(self) -> None:
        """Grow the tool/skill list pane by 4 columns."""
        self._list_width = min(80, self._list_width + 4)
        for w_id in ("#tool-list", "#skill-list"):
            try:
                self.query_one(w_id).styles.width = self._list_width
            except Exception:  # noqa: BLE001
                pass

    # ── Toolbox / version selection ──────────────────────────────────

    def _clear_loading(self) -> None:
        self._loading = False

    def _effective_toolbox(self) -> str:
        inp = self.query_one("#toolbox-input", Input)
        if inp.display:
            return inp.value.strip() or self.toolbox_name
        return str(self.query_one("#toolbox-select", Select).value)

    def _effective_version(self) -> str:
        inp = self.query_one("#version-input", Input)
        if inp.display:
            return inp.value.strip()
        value = self.query_one("#version-select", Select).value
        return "" if value in (Select.BLANK, None) else str(value)

    @work(exclusive=True, group="toolboxes")
    async def load_toolboxes(self) -> None:
        """List the project's toolboxes; fall back to manual entry if it fails."""
        header = self.query_one("#detail-header", Static)
        try:
            if self.token is None:
                self.token = _get_toolbox_token()
            self.toolboxes = await _list_toolboxes(self.token, self.endpoint)
        except Exception as e:  # noqa: BLE001
            self.toolboxes = []
            self.query_one("#toolbox-select").display = False
            self.query_one("#toolbox-input").display = True
            header.update(
                "[b yellow]Couldn't list toolboxes.[/b yellow]\n"
                f"{e}\nType a toolbox name and press [b]Connect[/b]."
            )
            await self._load_versions(self.toolbox_name, connect=True)
            return

        if not self.toolboxes:
            self.query_one("#toolbox-select").display = False
            self.query_one("#toolbox-input").display = True
            await self._load_versions(self.toolbox_name, connect=True)
            return

        names = [t["name"] for t in self.toolboxes]
        if self.toolbox_name not in names:
            self.toolbox_name = names[0]
        self._loading = True
        select = self.query_one("#toolbox-select", Select)
        select.set_options([(n, n) for n in names])
        select.value = self.toolbox_name
        self.call_after_refresh(self._clear_loading)
        await self._load_versions(self.toolbox_name, connect=True)

    async def _load_versions(self, name: str, connect: bool) -> None:
        """Populate the version picker for *name* (latest first) and optionally connect."""
        sel = self.query_one("#version-select", Select)
        inp = self.query_one("#version-input", Input)
        try:
            if self.token is None:
                self.token = _get_toolbox_token()
            self.versions = await _list_toolbox_versions(self.token, name, self.endpoint)
        except Exception:  # noqa: BLE001
            self.versions = []

        if self.versions:
            vers = [str(v["version"]) for v in self.versions]  # newest first
            chosen = self.toolbox_version if self.toolbox_version in vers else vers[0]
            self._loading = True
            sel.set_options(
                [(f"v{v}" + (" (latest)" if i == 0 else ""), v) for i, v in enumerate(vers)]
            )
            sel.value = chosen
            self.call_after_refresh(self._clear_loading)
            sel.display = True
            inp.display = False
            self.toolbox_version = chosen
        else:
            sel.display = False
            inp.display = True
            if not inp.value.strip():
                inp.value = self.toolbox_version or "1"
            self.toolbox_version = inp.value.strip()

        if connect:
            self.connect_and_load()

    @on(Select.Changed, "#toolbox-select")
    def on_toolbox_changed(self, event: Select.Changed) -> None:
        if self._loading:
            return
        self.toolbox_name = str(event.value)
        self.toolbox_version = ""  # reset → pick the latest version of the new toolbox
        self._save_current_config()
        self.run_worker(
            self._load_versions(self.toolbox_name, connect=True),
            group="versions", exclusive=True,
        )

    @on(Select.Changed, "#version-select")
    def on_version_changed(self, event: Select.Changed) -> None:
        if self._loading:
            return
        self.toolbox_version = "" if event.value in (Select.BLANK, None) else str(event.value)
        self._save_current_config()
        self.connect_and_load()

    @on(Button.Pressed, "#connect-btn")
    @on(Input.Submitted, "#toolbox-input")
    @on(Input.Submitted, "#version-input")
    def on_connect(self) -> None:
        name = self._effective_toolbox()
        if name != self.toolbox_name:
            # toolbox changed via manual entry → reload its versions, then connect
            self.toolbox_name = name
            self.toolbox_version = self._effective_version()
            self.run_worker(
                self._load_versions(name, connect=True),
                group="versions", exclusive=True,
            )
            return
        self.toolbox_version = self._effective_version()
        self.connect_and_load()

    @on(Button.Pressed, "#reload-btn")
    def on_reload(self) -> None:
        """Re-fetch the toolbox + version lists, keep the current toolbox, use latest version."""
        self.toolbox_version = ""   # empty → pick latest after reload
        self.token = None           # refresh the auth token too
        self.load_toolboxes()

    # ── Config helpers ─────────────────────────────────────────────

    def _current_config(self) -> dict:
        return {
            "resource": self.resource,
            "project": self.project,
            "model_deployment": self.model_deployment,
            "azure_api_version": self.azure_api_version,
            "toolbox_name": self.toolbox_name,
            "toolbox_version": self.toolbox_version,
        }

    def _save_current_config(self) -> None:
        _save_config(self._current_config())

    @on(Button.Pressed, "#cfg-apply-btn")
    def on_cfg_apply(self) -> None:
        resource = self.query_one("#cfg-resource", Input).value.strip()
        project = self.query_one("#cfg-project", Input).value.strip()
        model = self.query_one("#cfg-model", Input).value.strip()
        apiversion = self.query_one("#cfg-apiversion", Input).value.strip()
        status = self.query_one("#cfg-status", Static)
        if not resource or not project:
            status.update("[b red]Resource and project are required.[/b red]")
            return
        changed = (
            resource != self.resource
            or project != self.project
            or model != self.model_deployment
            or apiversion != self.azure_api_version
        )
        self.resource = resource
        self.project = project
        self.model_deployment = model or self.model_deployment
        self.azure_api_version = apiversion or self.azure_api_version
        self.endpoint = _endpoint_from(self.resource, self.project)
        self.model_base_url = _model_base_from(self.resource)
        self.token = None   # force new token for the new resource
        self._save_current_config()
        status.update(f"[b green]Saved.[/b green] Endpoint: {self.endpoint}")
        if changed:
            self.toolbox_version = ""
            self.load_toolboxes()

    # ── Result panel helpers ──────────────────────────────────────────────

    def _clear_result(self) -> None:
        self.query_one("#result-md", Markdown).update("")
        self.query_one("#result-raw", TextArea).text = ""
        self.query_one("#result-json", TextArea).text = ""

    def _set_result(self, text: str) -> None:
        """Populate all three result tabs from *text*, auto-detecting JSON."""
        # Unescape literal \n / \t sequences that some tools return as Python
        # repr-style escaped strings instead of real whitespace.
        def _unescape(s: str) -> str:
            try:
                return s.encode("raw_unicode_escape").decode("unicode_escape")
            except (UnicodeDecodeError, ValueError):
                return s

        if "\\n" in text and "\n" not in text:
            text = _unescape(text)

        self.query_one("#result-raw", TextArea).text = text
        # Attempt JSON pretty-print
        stripped = text.strip()
        json_text = ""
        if stripped.startswith(("{", "[")):
            try:
                json_text = json.dumps(json.loads(stripped), indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        self.query_one("#result-json", TextArea).text = json_text or "(not valid JSON)"
        # Markdown: use the text as-is; wrap JSON in a fenced code block.
        if json_text:
            md = f"```json\n{json_text}\n```"
        else:
            md = text
        self.query_one("#result-md", Markdown).update(md)
        # Auto-switch: JSON tab for JSON, Raw for everything else
        tabs = self.query_one("#result-tabs", TabbedContent)
        if json_text:
            tabs.active = "result-tab-json"
        else:
            tabs.active = "result-tab-raw"

    def _update_search_bar(self) -> None:
        """Show the search bar only for a search-enabled toolbox."""
        self.query_one("#search-bar").display = (
            self.tool_source == "toolbox" and self.search_mode
        )

    # ── Tool search ───────────────────────────────────────────────────────

    @on(Button.Pressed, "#search-btn")
    @on(Input.Submitted, "#search-query")
    def on_search(self) -> None:
        query = self.query_one("#search-query", Input).value.strip()
        header = self.query_one("#detail-header", Static)
        if not query:
            header.update("[b]Enter a query[/b] above to search for tools.")
            return
        limit_raw = self.query_one("#search-limit", Input).value.strip() or "5"
        try:
            limit = max(1, min(10, int(limit_raw)))
        except ValueError:
            limit = 5
        self.search_tools_worker(query, limit)

    # ── Source switch ─────────────────────────────────────────────────────

    @on(Select.Changed, "#tool-source")
    async def on_source_changed(self, event: Select.Changed) -> None:
        self.tool_source = str(event.value)
        self.current_tool = None
        await self.query_one("#form-fields", Vertical).remove_children()
        self.query_one("#execute-btn", Button).disabled = True
        self._clear_result()
        self._update_search_bar()
        header = self.query_one("#detail-header", Static)

        if self.tool_source == "copilot":
            if not self.copilot_tools:
                self.load_copilot_tools()
            else:
                await self._populate_tool_list()
                header.update(
                    "Copilot built-in tools — use the [b]Agent[/b] tab to exercise them."
                )
            return

        await self._populate_tool_list()
        if self.search_mode:
            header.update(
                "[b]Tool search is enabled.[/b] Type a capability or question and press "
                "[b]Search[/b] to discover tools, then select one to run it."
            )
        else:
            header.update(
                "Select a toolbox tool on the left to fill in its parameters and run it."
            )

    # ── Tool selection → build form ───────────────────────────────────────

    @on(ListView.Selected, "#tool-list")
    async def on_tool_selected(self, event: ListView.Selected) -> None:
        name = getattr(event.item, "tool_name", None)
        if not name:
            return
        if self.tool_source == "toolbox":
            if self.search_mode:
                self.current_tool = self.discovered_by_name.get(name)
            else:
                self.current_tool = self.tools_by_name.get(name)
            await self._build_form(self.current_tool)
        else:
            self.current_tool = None
            tool = next((t for t in self.copilot_tools if t["name"] == name), {})
            await self.query_one("#form-fields", Vertical).remove_children()
            self.query_one("#execute-btn", Button).disabled = True
            header = self.query_one("#detail-header", Static)
            header.update(
                f"[b]{name}[/b]  [dim](Copilot built-in)[/dim]\n"
                f"{tool.get('description', '(no description)')}\n\n"
                "[i]Run this via the Agent tab; it cannot be called directly.[/i]"
            )

    @on(ListView.Selected, "#skill-list")
    async def on_skill_selected(self, event: ListView.Selected) -> None:
        name = getattr(event.item, "skill_name", None)
        if not name:
            return
        self.show_skill(name)

    @work(exclusive=True, group="skill-read")
    async def show_skill(self, name: str) -> None:
        skill = self.skills_by_name.get(name, {})
        header = self.query_one("#skill-header", Static)
        content = self.query_one("#skill-content", TextArea)
        header.update(f"[b]{name}[/b]\n{skill.get('description', '')}")
        url = skill.get("url")
        if not url:
            content.text = "(no SKILL.md url in the index)"
            return
        content.text = "Loading…"
        try:
            content.text = await self.bridge.read_resource(url)
        except Exception as e:  # noqa: BLE001
            content.text = f"[error] {e}"

    async def _build_form(self, tool: dict) -> None:
        header = self.query_one("#detail-header", Static)
        desc_widget = self.query_one("#detail-desc", Markdown)
        fields = self.query_one("#form-fields", Vertical)
        await fields.remove_children()

        desc = tool.get("description", "")
        header.update(f"[b]{tool['name']}[/b]")
        await desc_widget.update(desc)

        schema = tool.get("inputSchema", {}) or {}
        properties = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])

        if not properties:
            await fields.mount(Static("[i]This tool takes no parameters.[/i]"))
        for prop_name, prop_schema in properties.items():
            prop_schema = prop_schema or {}
            json_type = prop_schema.get("type", "string")
            if isinstance(json_type, list):  # e.g. ["string", "null"]
                json_type = next((t for t in json_type if t != "null"), "string")
            items_type = (prop_schema.get("items") or {}).get("type", "")
            is_string_array = json_type == "array" and items_type == "string"
            enum = prop_schema.get("enum")
            default = prop_schema.get("default")

            req_mark = " [b red]*[/b red]" if prop_name in required else ""
            hint = "one item per line" if is_string_array else json_type
            label_text = f"{prop_name}{req_mark}  [dim]({hint})[/dim]"
            await fields.mount(Label(label_text, classes="field-label"))

            prop_desc = prop_schema.get("description")
            if prop_desc:
                await fields.mount(Static(prop_desc, classes="field-help"))

            widget_id = f"field-{prop_name}"
            if enum:
                options = [(str(v), str(v)) for v in enum]
                select = Select(
                    options, id=widget_id, allow_blank=prop_name not in required,
                )
                if default is not None and str(default) in [o[1] for o in options]:
                    select.value = str(default)
                select.prop_name = prop_name  # type: ignore[attr-defined]
                await fields.mount(select)
            elif json_type == "boolean":
                switch = Switch(value=bool(default), id=widget_id)
                switch.prop_name = prop_name  # type: ignore[attr-defined]
                await fields.mount(switch)
            elif is_string_array:
                # array of strings: one entry per line
                init = "\n".join(default) if isinstance(default, list) else ""
                area = TextArea(init, id=widget_id, classes="string-array-field")
                area.prop_name = prop_name  # type: ignore[attr-defined]
                area.is_string_array = True  # type: ignore[attr-defined]
                await fields.mount(area)
            elif json_type == "array":
                text = json.dumps(default) if default is not None else ""
                area = TextArea(text, id=widget_id, classes="json-field")
                area.prop_name = prop_name  # type: ignore[attr-defined]
                await fields.mount(area)
            elif json_type == "object":
                text = json.dumps(default, indent=2) if default is not None else ""
                area = TextArea(text, id=widget_id, classes="json-field")
                area.prop_name = prop_name  # type: ignore[attr-defined]
                await fields.mount(area)
            elif json_type in ("integer", "number"):
                input_type = "integer" if json_type == "integer" else "number"
                inp = Input(
                    value="" if default is None else str(default),
                    placeholder=str(prop_schema["examples"][0]) if prop_schema.get("examples") else "",
                    id=widget_id,
                    type=input_type,
                )
                inp.prop_name = prop_name  # type: ignore[attr-defined]
                await fields.mount(inp)
            else:
                # string (and anything else) → multiline scrollable TextArea
                init_text = "" if default is None else str(default)
                area = TextArea(init_text, id=widget_id, classes="string-field")
                area.prop_name = prop_name  # type: ignore[attr-defined]
                await fields.mount(area)

        self.query_one("#execute-btn", Button).disabled = False

    def _collect_arguments(self) -> dict:
        """Read the current form widgets into a JSON-ready argument dict."""
        if not self.current_tool:
            return {}
        schema = self.current_tool.get("inputSchema", {}) or {}
        properties = schema.get("properties", {}) or {}
        args: dict = {}
        fields = self.query_one("#form-fields", Vertical)
        for widget in fields.query(Input):
            prop = getattr(widget, "prop_name", None)
            if prop is None:
                continue
            value = _coerce_value(widget.value, properties.get(prop, {}))
            if value is not None:
                args[prop] = value
        for widget in fields.query(TextArea):
            prop = getattr(widget, "prop_name", None)
            if prop is None:
                continue
            if getattr(widget, "is_string_array", False):
                # one item per line → list of non-empty strings
                lines = [l for l in widget.text.splitlines() if l.strip()]
                if lines:
                    args[prop] = lines
            else:
                value = _coerce_value(widget.text, properties.get(prop, {}))
                if value is not None:
                    args[prop] = value
        for widget in fields.query(Select):
            prop = getattr(widget, "prop_name", None)
            if prop is None:
                continue
            if widget.value not in (Select.BLANK, None):
                args[prop] = _coerce_value(str(widget.value), properties.get(prop, {}))
        for widget in fields.query(Switch):
            prop = getattr(widget, "prop_name", None)
            if prop is None:
                continue
            args[prop] = widget.value
        return args

    # ── Buttons ───────────────────────────────────────────────────────────

    @on(Button.Pressed, "#clear-btn")
    def on_clear(self) -> None:
        fields = self.query_one("#form-fields", Vertical)
        for widget in fields.query(Input):
            widget.value = ""
        for widget in fields.query(TextArea):
            widget.text = ""
        for widget in fields.query(Switch):
            widget.value = False
        self._clear_result()

    @on(Button.Pressed, "#execute-btn")
    def on_execute(self) -> None:
        if not self.current_tool or not self.bridge:
            return
        try:
            args = self._collect_arguments()
        except json.JSONDecodeError as e:
            self._set_result(f"Invalid JSON in a field: {e}")
            return
        via_call_tool = self.tool_source == "toolbox" and self.search_mode
        self.execute_tool(self.current_tool["name"], args, via_call_tool)

    @work(exclusive=True, group="tool-call")
    async def execute_tool(self, name: str, args: dict, via_call_tool: bool = False) -> None:
        label = f"call_tool → {name}" if via_call_tool else name
        self._set_result(f"Calling {label}…\narguments: {json.dumps(args)}\n")
        try:
            if via_call_tool:
                result = await self.bridge.call_discovered_tool(name, args)
            else:
                result = await self.bridge.call_tool(name, args)
        except Exception as e:  # noqa: BLE001
            self._set_result(f"[error] {e}")
            return
        self._set_result(result)

    # ── Agent tab ─────────────────────────────────────────────────────────

    @on(Button.Pressed, "#agent-send-btn")
    @on(Input.Submitted, "#agent-input")
    def on_agent_send(self) -> None:
        prompt = self.query_one("#agent-input", Input).value.strip()
        if not prompt:
            return
        self.query_one("#agent-input", Input).value = ""
        self.ask_agent(prompt)

    @on(Button.Pressed, "#agent-clear-btn")
    def on_agent_clear(self) -> None:
        self.agent_history = []
        self.run_worker(self._reset_agent_session(), group="agent", exclusive=True)
        self._set_agent_result("")

    async def _reset_agent_session(self) -> None:
        if self.agent_session is not None:
            try:
                await self.agent_session.disconnect()
            except Exception:  # noqa: BLE001
                pass
            self.agent_session = None

    @work(exclusive=True, group="agent")
    async def ask_agent(self, prompt: str) -> None:
        self.agent_history.append(("user", prompt))
        self._render_agent_history("Thinking…")
        try:
            reply = await self._run_agent_turn(prompt)
            self.agent_history.append(("assistant", reply))
            self._render_agent_history()
        except Exception as e:  # noqa: BLE001
            self.agent_history.append(("assistant", f"[error] {e}"))
            self._render_agent_history()

    def _set_agent_result(self, text: str) -> None:
        """Populate the agent response tabs with a single text block."""
        if text and "\\n" in text and "\n" not in text:
            try:
                text = text.encode("raw_unicode_escape").decode("unicode_escape")
            except (UnicodeDecodeError, ValueError):
                pass
        self.query_one("#agent-result-raw", TextArea).text = text
        stripped = text.strip()
        json_text = ""
        if stripped.startswith(("{", "[")):
            try:
                json_text = json.dumps(json.loads(stripped), indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        self.query_one("#agent-result-json", TextArea).text = json_text or "(not valid JSON)"
        self.query_one("#agent-result-md", Markdown).update(
            f"```json\n{json_text}\n```" if json_text else text
        )
        tabs = self.query_one("#agent-result-tabs", TabbedContent)
        tabs.active = "agent-result-tab-json" if json_text else "agent-result-tab-raw"

    def _render_agent_history(self, pending: str | None = None) -> None:
        """Re-render the full conversation thread into the result tabs."""
        raw_lines: list[str] = []
        md_lines: list[str] = []
        sep = "\u2500" * 60
        for role, text in self.agent_history:
            if "\\n" in text and "\n" not in text:
                try:
                    text = text.encode("raw_unicode_escape").decode("unicode_escape")
                except (UnicodeDecodeError, ValueError):
                    pass
            if role == "user":
                raw_lines.append(f"You: {text}\n{sep}")
                md_lines.append(f"**You:** {text}\n\n---")
            else:
                raw_lines.append(f"Assistant:\n{text}\n{sep}")
                md_lines.append(f"**Assistant:**\n\n{text}\n\n---")
        if pending:
            raw_lines.append(pending)
            md_lines.append(f"*{pending}*")
        full_raw = "\n".join(raw_lines)
        full_md = "\n".join(md_lines)
        self.query_one("#agent-result-raw", TextArea).text = full_raw
        self.query_one("#agent-result-json", TextArea).text = "(conversation thread — not JSON)"
        self.query_one("#agent-result-md", Markdown).update(full_md)
        tabs = self.query_one("#agent-result-tabs", TabbedContent)
        tabs.active = "agent-result-tab-md"

    async def _run_agent_turn(self, prompt: str) -> str:
        """Send *prompt* on the persistent session, reusing conversation history."""
        if self.client is None:
            self.client = CopilotClient()
            await self.client.start()
        if not self.mcp_tools:
            self.mcp_tools = await self.bridge.list_tools()

        tools = _make_copilot_tools(self.bridge, self.mcp_tools)
        provider = {
            "type": "azure",
            "base_url": self.model_base_url,
            "bearer_token": _get_model_token(),
            "model_id": self.model_deployment,
            "azure": {"api_version": self.azure_api_version},
        }
        # Create a new session only if we don't have one yet
        if self.agent_session is None:
            self.agent_session = await self.client.create_session(
                on_permission_request=lambda req, ctx: {"kind": "approved"},
                tools=tools,
                model=self.model_deployment,
                provider=provider,
            )
        event = await self.agent_session.send_and_wait(prompt, timeout=120.0)
        return event.data.content if event else "(no response)"

    # ── Cleanup ───────────────────────────────────────────────────────────

    async def on_unmount(self) -> None:
        if self.client:
            await self.client.stop()
        if self.bridge:
            await self.bridge.close()


if __name__ == "__main__":
    ToolboxApp().run()
