# Testing Guide

This document describes how to run tests for the emitter project.

## Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

The testing dependencies are already included in `requirements.txt`:
- `pytest>=7.4.0`
- `pytest-asyncio>=0.21.0`

## Running Tests

### Run All Tests

From the project root directory:

```bash
pytest tests/ -v
```

### Run Specific Test Files

```bash
# Test computation module
pytest tests/test_computation.py -v

# Test memory store
pytest tests/test_memory_store.py -v

# Test pattern index
pytest tests/test_pattern_index.py -v

# Test metrics computer
pytest tests/test_metrics_computer.py -v

# Test pattern generator
pytest tests/test_pattern_generator.py -v

# Test end-to-end (requires running server)
pytest tests/test_e2e.py -v
```

### Run Tests by Category

```bash
# Run only unit tests (exclude e2e)
pytest tests/ -v --ignore=tests/test_e2e.py

# Run with coverage report
pytest tests/ --cov=backend --cov-report=html

# Run tests matching a pattern
pytest tests/ -v -k "test_amplitude"
```

### Run Tests with Verbose Output

```bash
pytest tests/ -v -s
```

The `-s` flag shows print statements and other stdout output.

## Test Structure

```
tests/
├── test_computation.py      # Tests for FFT, Goertzel, windowing
├── test_memory_store.py     # Tests for SQLite memory storage
├── test_pattern_index.py    # Tests for inverted pattern index
├── test_metrics_computer.py # Tests for feature extraction
├── test_pattern_generator.py # Tests for pattern classification
└── test_e2e.py              # End-to-end tests (requires server)
```

## Test Categories

### Unit Tests

These tests run without external dependencies:

- **test_computation.py**: Tests signal processing functions
  - Window functions (Hann, rectangular)
  - Goertzel amplitude estimation
  - FFT-based amplitude estimation
  - Multi-channel FFT computation

- **test_memory_store.py**: Tests SQLite storage layer
  - CRUD operations for memory records
  - Search by text and time range
  - Filtering and pagination

- **test_pattern_index.py**: Tests inverted index
  - Adding/removing patterns
  - AND/OR search semantics
  - Persistence (save/load)

- **test_metrics_computer.py**: Tests feature extraction
  - Time-domain features (mean, RMS, crest factor)
  - Spectral features (dominant frequency, centroid)
  - Cross-channel metrics (correlation, phase)

- **test_pattern_generator.py**: Tests pattern classification
  - Frequency band classification
  - Amplitude level classification
  - Temporal pattern classification
  - Adaptive threshold learning

### End-to-End Tests

The `test_e2e.py` file requires a running server:

```bash
# Start the server first
uvicorn app:app --reload --port 8000 --app-dir backend

# Then run e2e tests
pytest tests/test_e2e.py -v
```

## Writing New Tests

### Test File Template

```python
import pytest
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from your_module import YourClass


@pytest.fixture
def sample_data():
    """Create sample test data."""
    return {"key": "value"}


class TestYourClass:
    """Tests for YourClass."""
    
    def test_basic_operation(self, sample_data):
        """Should perform basic operation."""
        result = YourClass().method(sample_data)
        assert result is not None
```

### Testing Async Code

```python
import pytest

@pytest.mark.asyncio
async def test_async_operation():
    """Test async function."""
    result = await some_async_function()
    assert result == expected
```

## Continuous Integration

Tests are designed to run in CI/CD pipelines. Example GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --ignore=tests/test_e2e.py
```

## Troubleshooting

### Import Errors

If you see import errors like `ModuleNotFoundError`, ensure you're running tests from the project root:

```bash
cd /path/to/emitter
pytest tests/ -v
```

### FAISS Not Installed

Some tests may require FAISS. Install with:

```bash
pip install faiss-cpu
```

### Slow Tests

Some tests involving large FFTs or embeddings may be slow. Use markers to skip them:

```bash
pytest tests/ -v -m "not slow"
```

## Coverage

To generate a coverage report:

```bash
# Install coverage
pip install pytest-cov

# Run with coverage
pytest tests/ --cov=backend --cov-report=html --ignore=tests/test_e2e.py

# Open the report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```
