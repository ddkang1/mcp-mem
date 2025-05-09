#!/usr/bin/env python3
"""Setup script for mcp-mem."""

from setuptools import setup, find_packages

setup(
    name="mcp-mem",
    version="0.2.0",
    description="MCP server for permanent external memory using LightRAG",
    author="MCP Team",
    author_email="info@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastmcp>=0.2.0",
        "lightrag>=0.1.0",  # Direct integration
        "aiohttp>=3.8.0",   # For API client
        "openai>=1.0.0",    # For OpenAI embeddings and LLM
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "mcp-mem=mcp_mem.server:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)