# MCP Memory

A Model Context Protocol (MCP) server implementing memory solutions for data-rich applications with efficient knowledge graph capabilities.

## Overview

This MCP server implements a memory solution for data-rich applications that involve searching information from many sources including uploaded files. It uses HippoRAG internally to manage memory through an efficient knowledge graph.

## Features

- **Session-based Memory**: Create and manage memory for specific chat sessions
- **Efficient Knowledge Graph**: Uses HippoRAG for advanced memory management
- **Multiple Transport Support**: Works with both stdio and SSE transports
- **Search Capabilities**: Search information from various sources including uploaded files

## Installation

Install from PyPI:

```bash
pip install mcp-mem
```

Or install from source:

```bash
git clone https://github.com/ddkang1/mcp-mem.git
cd mcp-mem
pip install -e .
```

## Usage

You can run the MCP server directly:

```bash
mcp-mem
```

By default, it uses stdio transport. To use SSE transport:

```bash
mcp-mem --sse
```

You can also specify host and port for SSE transport:

```bash
mcp-mem --sse --host 127.0.0.1 --port 3001
```

## Configuration

To use this tool with Claude in Windsurf, add the following configuration to your MCP config file:

```json
"memory": {
    "command": "/path/to/mcp-mem",
    "args": [],
    "type": "stdio",
    "pollingInterval": 30000,
    "startupTimeout": 30000,
    "restartOnFailure": true
}
```

The `command` field should point to the directory where you installed the python package using pip.

## Available Tools

The MCP server provides the following tools:

- **create_memory**: Create a new memory for a given chat session
- **store_memory**: Add memory to a specific session
- **retrieve_memory**: Retrieve memory from a specific session

## Development

### Installation for Development

```bash
git clone https://github.com/ddkang1/mcp-mem.git
cd mcp-mem
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Style

This project uses Black for formatting, isort for import sorting, and flake8 for linting:

```bash
black src tests
isort src tests
flake8 src tests
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.