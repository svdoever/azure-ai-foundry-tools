# toolbox-tui

An interactive terminal UI (TUI) for exploring and testing [Azure AI Foundry](https://ai.azure.com) toolboxes.

## Overview

`toolbox-tui.py` connects to an Azure AI Foundry project, lists its toolboxes, and lets you interactively browse tools, read skills, and chat with an LLM that has access to all of the toolbox's tools — all from your terminal.

## Features

| Tab | What it does |
|-----|--------------|
| **Tools** | Pick a toolbox and version, then browse and invoke its tools. Automatically handles both regular toolboxes (direct `tools/list`) and "Tool Search" toolboxes (search via `tool_search`, invoke via `call_tool`). Selecting a tool builds a parameter form from its JSON Schema. Also lists Copilot built-in tools. |
| **Skills** | Reads the skill discovery index (`skill://index.json`) exposed by the toolbox and renders each skill's `SKILL.md` as Markdown. |
| **Agent** | Send a natural-language question to an Azure OpenAI model. The model has access to all toolbox tools and can call them autonomously to answer your question. |

## Prerequisites

- Python 3.11+
- An Azure AI Foundry project with at least one toolbox
- Azure credentials resolvable by `DefaultAzureCredential` (e.g. `az login`)

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `toolbox-tui.config` (created automatically on first run) or set values directly in the TUI settings panel:

| Key | Description |
|-----|-------------|
| `resource` | Azure AI Services resource name (e.g. `myproject-resource`) |
| `project` | Foundry project name (e.g. myproject) |
| `model_deployment` | Azure OpenAI deployment name used for the Agent tab  (e.g. gpt-5.4) |
| `azure_api_version` | Azure OpenAI API version (e.g. "2024-12-01-preview") |
| `toolbox_name` | Default toolbox name to pre-select |
| `toolbox_version` | Default toolbox version to pre-select |

## Running

```bash
# Windows
start.cmd

# PowerShell
.\start.ps1

# Linux / macOS
./start.sh

# Or directly
python toolbox-tui.py
```

## Architecture

- Uses [Textual](https://textual.textualize.io/) for the TUI.
- Communicates with the Foundry data-plane API over HTTP to list toolboxes and versions.
- Speaks the [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) JSON-RPC protocol over HTTP to initialize sessions, list/call tools, and read skill resources.
- Uses `azure-identity` `DefaultAzureCredential` for authentication — no API keys required.
- Wraps MCP tool definitions as Copilot SDK `Tool` objects so the Agent tab can call them transparently.
