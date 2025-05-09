# MCP Memory Server

A Model Context Protocol (MCP) server that provides permanent external memory capabilities for LLMs using LightRAG.

## Overview

MCP Memory Server enables LLMs to store and retrieve information without context window limitations. It provides a permanent, external memory system that preserves complete information without summarization or information loss.

The server uses LightRAG, a powerful Retrieval-Augmented Generation system that combines vector search with knowledge graph capabilities for more effective information retrieval.

## Features

- **Permanent Storage**: Store unlimited amounts of information that persists across sessions
- **Graph-Enhanced Retrieval**: Leverage LightRAG's knowledge graph capabilities for more contextual retrieval
- **Session Management**: Organize memories by session ID for different use cases
- **TTL-Based Cleanup**: Automatically clean up old sessions and offload inactive instances
- **MCP Protocol**: Seamless integration with any MCP-compatible client
- **Flexible Integration**: Support for both direct integration and API-based approaches

## Installation

```bash
pip install mcp-mem
```

## Integration Options

MCP Memory Server supports two integration approaches:

### 1. Direct Integration

Direct integration instantiates LightRAG objects directly in the same process as the MCP server. This approach is simpler to set up and has lower latency, but requires all dependencies to be installed locally.

```bash
# Run with direct integration (default)
mcp-mem --integration-type direct
```

### 2. API-Based Integration

API-based integration connects to a separate LightRAG API server. This approach allows for better resource isolation and centralized knowledge management, but requires a separate LightRAG server to be running.

```bash
# Run with API integration
mcp-mem --integration-type api --lightrag-api-url http://localhost:8000 --lightrag-api-key your_api_key
```

## Usage

### Starting the Server

```bash
# Using stdio transport (for integration with MCP clients)
mcp-mem --transport stdio

# Using SSE transport (for HTTP-based access)
mcp-mem --transport sse --host 127.0.0.1 --port 8000
```

### Environment Variables

Configure the server using these environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key (required for embeddings and LLM)
- `OPENAI_API_BASE`: Custom base URL for OpenAI API (optional)
- `LIGHTRAG_API_BASE_URL`: Base URL for LightRAG API (for API integration)
- `LIGHTRAG_API_KEY`: API key for LightRAG API (for API integration)

### Command Line Options

```
usage: mcp-mem [-h] [--transport {stdio,sse}] [--host HOST] [--port PORT] [--debug]
               [--integration-type {direct,api}] [--lightrag-api-url LIGHTRAG_API_URL]
               [--lightrag-api-key LIGHTRAG_API_KEY]

Run MCP Memory server

optional arguments:
  -h, --help            show this help message and exit
  --transport {stdio,sse}
                        Transport protocol to use (stdio or sse)
  --host HOST           Host to bind to (for SSE transport)
  --port PORT           Port to listen on (for SSE transport)
  --debug               Enable debug mode with verbose logging
  --integration-type {direct,api}
                        Integration type (direct or api)
  --lightrag-api-url LIGHTRAG_API_URL
                        LightRAG API base URL (for API integration)
  --lightrag-api-key LIGHTRAG_API_KEY
                        LightRAG API key (for API integration)
```

## MCP Tools

### store_memory

Store content permanently in the external memory system.

```json
{
  "session_id": "user123",
  "content": "The complete content to store"
}
```

### retrieve_memory

Retrieve previously stored content based on a query.

```json
{
  "session_id": "user123",
  "query": "What information do we have about climate change?",
  "limit": 5
}
```

### configure_memory

Configure the memory system.

```json
{
  "integration_type": "api",
  "lightrag_api_base_url": "http://localhost:8000",
  "lightrag_api_key": "your_api_key",
  "embedding_provider": "openai",
  "embedding_model_name": "text-embedding-3-large",
  "llm_provider": "openai",
  "llm_model_name": "gpt-4o-mini"
}
```

## How It Works

MCP Memory Server uses LightRAG to provide sophisticated memory capabilities:

1. **Storage**: Content is processed through LightRAG's pipeline:
   - Text is split into manageable chunks
   - Entities and relationships are extracted to build a knowledge graph
   - Vector embeddings are created for semantic search
   - Original content is preserved without loss

2. **Retrieval**: When queried, the system:
   - Uses hybrid search combining vector similarity and graph traversal
   - Finds the most relevant information based on the query
   - Returns complete, unaltered content as it was stored

3. **Graph-Enhanced Context**: LightRAG's knowledge graph:
   - Captures relationships between entities in the stored content
   - Enables more contextual retrieval by traversing related information
   - Improves retrieval quality beyond simple vector similarity

## Integration Comparison

### Direct Integration

**Advantages:**
- Simpler deployment (single process)
- Lower latency (no network overhead)
- Direct access to all LightRAG features
- Each MCP server has its own LightRAG instances

**Best for:**
- Simpler deployments with fewer moving parts
- Minimizing latency
- Session isolation for privacy or multi-tenant scenarios
- Resource-constrained environments
- Smaller-scale applications

### API-Based Integration

**Advantages:**
- LightRAG runs as a dedicated service
- Can scale LightRAG independently from the MCP server
- Multiple clients/applications can access the same LightRAG instance
- Knowledge is shared across all clients
- Better resource isolation

**Best for:**
- Centralized knowledge base shared across multiple applications
- Microservices architecture
- Independent scaling of components
- High resource requirements that benefit from isolation
- Supporting multiple programming languages/frameworks

## Examples

See the `examples` directory for example scripts:

- `basic_usage.py`: Demonstrates how to use the MCP Memory server with both direct and API-based integration

## Use Cases

- **Document Interaction**: Talk to multiple documents without context limitations
- **Knowledge Management**: Build and query persistent knowledge bases
- **Agent Memory**: Provide agents with permanent memory capabilities
- **Long-term Conversations**: Maintain conversation history across multiple sessions

## License

MIT