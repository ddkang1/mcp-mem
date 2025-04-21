# MCP Memory

A Model Context Protocol (MCP) server implementing memory solutions for data-rich applications with efficient knowledge graph capabilities.

## Overview

This MCP server implements a memory solution for data-rich applications that involve searching information from many sources including uploaded files. It uses HippoRAG internally to manage memory through an efficient knowledge graph. HippoRAG is a required dependency for this package.

## Features

- **Session-based Memory**: Create and manage memory for specific chat sessions
- **Efficient Knowledge Graph**: Uses HippoRAG for memory management
- **Multiple Transport Support**: Works with both stdio and SSE transports
- **Search Capabilities**: Search information from various sources including uploaded files
- **Automatic Resource Management**: TTL-based cleanup for both sessions and memory instances

## Installation

Install from PyPI:

```bash
pip install mcp-mem hipporag
```

Or install from source:

```bash
git clone https://github.com/ddkang1/mcp-mem.git
cd mcp-mem
pip install -e .
pip install hipporag
```

Note: HippoRAG is a required dependency for mcp-mem to function.

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

### Basic Configuration

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

### Environment Variable Configuration

You can configure the LLM and embedding models used by mcp-mem through environment variables:

- `EMBEDDING_MODEL_NAME`: Name of the embedding model to use (default: "text-embedding-3-large")
- `EMBEDDING_BASE_URL`: Base URL for the embedding API (optional)
- `LLM_NAME`: Name of the LLM model to use (default: "gpt-4o-mini")
- `LLM_BASE_URL`: Base URL for the LLM API (optional)
- `OPENAI_API_KEY`: OpenAI API key (required)

### Memory Management Configuration

The server includes automatic resource management features:

- **Session TTL**: Automatically removes session directories after a specified number of days of inactivity.
  Set using the `session_ttl_days` configuration parameter (default: None - disabled).

- **Instance TTL**: Automatically offloads HippoRAG instances from memory after a specified period of inactivity.
  Set using the `instance_ttl_minutes` configuration parameter (default: 30 minutes).
  
  This feature helps manage memory usage by unloading inactive instances while preserving the underlying data.
  When an offloaded instance is accessed again, it will be automatically reloaded from disk.

Example usage:

```bash
EMBEDDING_MODEL_NAME="your-model" LLM_NAME="your-llm" mcp-mem
```

For convenience, you can use the provided example script:

```bash
./examples/run_with_env_vars.sh
```

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