# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-04-21

### Added
- TTL-based automatic cleanup for HippoRAG instances
- New `HippoRAGInstanceManager` class for managing instances with TTL
- New `instance_ttl_minutes` configuration parameter (default: 30 minutes)
- Improved memory management by offloading inactive instances while preserving data

### Changed
- Refactored server.py to use the new instance manager
- Updated documentation to reflect new TTL features

## [0.2.0] - 2025-04-21

### Changed
- **BREAKING**: Removed all fallback mechanisms and backward compatibilities
- **BREAKING**: HippoRAG is now strictly required with no fallback to basic memory storage
- **BREAKING**: Removed `use_hipporag` configuration option
- Updated tests to reflect the removal of fallback mechanisms
- Updated documentation to clarify HippoRAG dependency requirements

## [0.1.0] - 2025-04-19

### Added
- Initial release of mcp-mem
- Session-based memory management
- Integration with HippoRAG for knowledge graph capabilities
- Basic memory storage and retrieval
- Memory search functionality
- Configuration system
- Support for both stdio and SSE transports