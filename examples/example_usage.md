# Example Usage of the MCP Memory Tool

This document demonstrates how to use the MCP Memory tool effectively in various scenarios.

## Basic Setup

First, install the package:

```bash
pip install mcp-mem
```

Then, you can run the server:

```bash
mcp-mem
```

## Example 1: Creating and Using Session Memory

When an LLM needs to maintain memory across interactions:

```
Human: Let me tell you about my project. I'm building a web application using React and Node.js.

LLM: I'll remember that you're building a web application using React and Node.js.

[LLM uses the create_memory tool]
create_memory(session_id="user123")
Result: {"session_id": "user123", "status": "created", "message": "Memory for session user123 created successfully"}

[LLM uses the store_memory tool]
store_memory(session_id="user123", content="User is building a web application using React and Node.js", metadata={"topic": "project", "technologies": ["React", "Node.js"]})
Result: {"session_id": "user123", "memory_id": "mem123", "status": "stored", "message": "Memory stored successfully in session user123"}
```

## Example 2: Retrieving Memory

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

## Example 3: Searching Memory

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
