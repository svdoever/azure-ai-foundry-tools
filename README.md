# Azure AI Foundry Tools

A collection of developer utilities for working with [Azure AI Foundry](https://ai.azure.com) — the portal and platform for building, deploying, and managing AI agents and models on Azure.

## Why this repo?

Azure AI Foundry exposes a rich data-plane API and the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for toolboxes, but there is no built-in terminal-based developer experience for quickly exploring what a project contains, testing tools interactively, or chatting with an agent that has live access to your toolbox.

This repo fills that gap with lightweight, dependency-minimal scripts you can run locally during development — no portal required.

## Tools

| Tool | Description |
|------|-------------|
| [`toolbox-tui`](toolbox-tui/) | Interactive terminal UI (TUI) for browsing and testing Foundry toolboxes. Lets you list toolboxes and versions, invoke individual tools with a generated parameter form, read skills, and chat with an LLM that has access to all toolbox tools. |

## Getting started

Each tool lives in its own subdirectory with its own `README.md` and `requirements.txt`. See the individual READMEs for setup and usage instructions.

## Contributing

Contributions are welcome. Please open an issue or pull request.

## License

[MIT](LICENSE)
