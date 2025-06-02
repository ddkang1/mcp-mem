# Example Usage of the MCP Memory Tool (LightRAG Backend)

This document demonstrates how to use the MCP Memory tool with LightRAG as the backend.

## Example 1: Creating and Using Session Memory

When an LLM needs to maintain memory across interactions:

```python
from mcp_mem.server import store_memory, retrieve_memory

# Store memory
await store_memory(session_id="user123", content="User is building a web application using React and Node.js")

# Retrieve memory
result = await retrieve_memory(session_id="user123", query="React")
print(result["result"])
```

## Example 2: Session Cleanup

Session data is automatically cleaned up based on TTL settings in the config.

## Example 3: Retrieving Memory

When an LLM needs to recall information from previous interactions:

```
Human: What was I working on again?

[LLM uses the retrieve_memory tool]
retrieve_memory(session_id="user123")
Result: {
  "session_id": "user123",
  "status": "success",
  "query": null,
  "memories": [
    {
      "id": "mem123",
      "content": "User is building a web application using React and Node.js",
      "timestamp": "2025-04-18T22:01:00",
      "metadata": {"topic": "project", "technologies": ["React", "Node.js"]}
    }
  ]
}

LLM: You were working on building a web application using React and Node.js. Would you like to continue discussing that project?
```

## Example 4: Searching Memory

When an LLM needs to find specific information:

```
Human: What technologies am I using for my project?

[LLM uses the retrieve_memory tool with a query]
retrieve_memory(session_id="user123", query="technologies")
Result: {
  "session_id": "user123",
  "status": "success",
  "query": "technologies",
  "memories": [
    {
      "id": "mem123",
      "content": "User is building a web application using React and Node.js",
      "timestamp": "2025-04-18T22:01:00",
      "metadata": {"topic": "project", "technologies": ["React", "Node.js"]}
    }
  ]
}

LLM: For your project, you're using React for the frontend and Node.js for the backend.
```

## Integration with MCP Configuration

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
