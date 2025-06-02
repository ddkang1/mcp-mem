# MCP Memory (Agentic, Metadata-Aware Backend)

This package provides a Model Context Protocol (MCP) memory server for agentic workflows, supporting persistent, shared, and metadata-rich memory for multi-agent systems.

## Features

- Session-based, persistent memory for agentic workflows
- Store and retrieve content with rich metadata (agent, type, tags, timestamp)
- Advanced querying and filtering (by agent, type, tags, time range)
- Summarization tool for context window management
- Tools for updating and deleting memory entries
- Designed for multi-agent, collaborative, and context-aware AI applications

## Available Tools

### `store_memory`
Store a memory entry (with optional metadata) in persistent, shared session memory.

**Parameters:**
- `session_id` (str): Unique session or workflow ID
- `content` (str): Text content to store
- `metadata` (dict, optional): Additional metadata (e.g., tags, custom fields)
- `source_agent` (str, optional): Name of the agent/tool storing this entry
- `message_type` (str, optional): Type of message (e.g., 'user', 'tool', 'summary')
- `timestamp` (str, optional): ISO8601 timestamp (auto-set if not provided)

**Example:**
```python
await store_memory(
    session_id="workflow-abc",
    content="User approved deployment to production.",
    source_agent="approval_agent",
    message_type="decision",
    metadata={"tags": ["approval", "prod"]}
)
```

### `retrieve_memory`
Retrieve memory entries with advanced filtering.

**Parameters:**
- `session_id` (str): Unique session or workflow ID
- `query` (str): Query string (keyword, phrase, or question)
- `limit` (int, optional): Max results
- `source_agent` (str, optional): Filter by agent/tool
- `message_type` (str, optional): Filter by message type
- `tags` (list of str, optional): Filter by tags
- `start_time` (str, optional): ISO8601 start time
- `end_time` (str, optional): ISO8601 end time

**Example:**
```python
result = await retrieve_memory(
    session_id="workflow-abc",
    query="deployment approval",
    source_agent="approval_agent",
    tags=["prod"],
    start_time="2024-01-01T00:00:00Z"
)
print(result["result"])
```

### `summarize_memory`
Summarize session memory to preserve key context for agentic workflows.

**Parameters:**
- `session_id` (str): Unique session or workflow ID
- `query` (str, optional): Focus summary on specific topics
- `max_tokens` (int, optional): Max tokens for summary

**Example:**
```python
result = await summarize_memory(
    session_id="workflow-abc",
    query="deployment decisions",
    max_tokens=256
)
print(result["summary"])
```

### `delete_memory`
Delete memory entries by ID or advanced filters.

**Parameters:**
- `session_id` (str): Unique session or workflow ID
- `entry_id` (str, optional): ID of entry to delete
- `query` (str, optional): Query string to match entries
- `source_agent` (str, optional): Filter by agent/tool
- `message_type` (str, optional): Filter by message type
- `tags` (list of str, optional): Filter by tags
- `start_time` (str, optional): ISO8601 start time
- `end_time` (str, optional): ISO8601 end time

**Example:**
```python
result = await delete_memory(
    session_id="workflow-abc",
    query="outdated info",
    source_agent="old_agent"
)
print(result["deleted_count"])
```

### `update_memory`
Update memory entry content or metadata by entry ID.

**Parameters:**
- `session_id` (str): Unique session or workflow ID
- `entry_id` (str): ID of entry to update
- `new_content` (str, optional): New content
- `new_metadata` (dict, optional): New metadata to merge/replace

**Example:**
```python
result = await update_memory(
    session_id="workflow-abc",
    entry_id="5",
    new_content="Corrected info."
)
print(result["status"])
```

## Agentic Memory for Multi-Agent Workflows

This server is designed for agentic, multi-agent, and collaborative AI workflows. It supports:
- Persistent, shared memory across agents
- Rich metadata for provenance, type, and context
- Advanced querying and summarization for scalable, context-aware applications

## License

MIT