#!/bin/bash
# Example script to run mcp-mem with environment variables for LLM and embedding configurations

# Set environment variables for LLM and embedding configurations
export EMBEDDING_MODEL_NAME="text-embedding-3-large"
export EMBEDDING_BASE_URL="https://api.example.com/embeddings"
export LLM_NAME="gpt-4o-mini"
export LLM_BASE_URL="https://api.example.com/llm"
export OPENAI_API_KEY="your-openai-api-key"  # Required for HippoRAG to function properly

# Run mcp-mem
echo "Starting mcp-mem with custom LLM and embedding configurations..."
python -m mcp_mem

# Alternatively, you can use the run_mcp.py script
# python run_mcp.py