#!/usr/bin/env python3
"""
Test script to verify that environment variables are properly read by mcp-mem.
"""

import os
import sys
import json
from mcp_mem.config import get_config

def main():
    """Print the current HippoRAG configuration with environment variables."""
    # Get the current configuration
    config = get_config()
    
    # Print the HippoRAG configuration
    print("Current HippoRAG configuration:")
    print(json.dumps(config.hipporag_config, indent=2))
    
    # Print environment variables
    print("\nEnvironment variables:")
    for var in ["EMBEDDING_MODEL_NAME", "EMBEDDING_BASE_URL", "LLM_NAME", "LLM_BASE_URL", "OPENAI_API_KEY"]:
        value = os.environ.get(var, 'Not set')
        # Mask API key for security
        if var == "OPENAI_API_KEY" and value != 'Not set':
            value = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
        print(f"{var}: {value}")

if __name__ == "__main__":
    main()