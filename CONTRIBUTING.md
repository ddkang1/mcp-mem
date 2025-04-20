# Contributing to MCP Memory

Thank you for considering contributing to MCP Memory! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with the following information:

- A clear, descriptive title
- A detailed description of the issue
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Any relevant logs or error messages
- Your environment (OS, Python version, etc.)

### Suggesting Features

We welcome feature suggestions! Please create an issue on GitHub with:

- A clear, descriptive title
- A detailed description of the proposed feature
- Any relevant examples or use cases
- If applicable, references to similar features in other projects

### Pull Requests

1. Fork the repository
2. Create a new branch for your changes
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Update documentation if necessary
6. Submit a pull request

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mcp-mem.git
   cd mcp-mem
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. Run tests:
   ```bash
   pytest
   ```

## Code Style

This project uses:
- [Black](https://black.readthedocs.io/) for code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [flake8](https://flake8.pycqa.org/) for linting

You can run these tools with:
```bash
black src tests
isort src tests
flake8 src tests
```

## Testing

Please ensure that your code includes appropriate tests. Run the test suite with:
```bash
pytest
```

## Documentation

Please update documentation when making changes to the code. This includes:
- Docstrings for new functions, classes, and methods
- Updates to README.md if necessary
- Examples in the examples directory if applicable

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.