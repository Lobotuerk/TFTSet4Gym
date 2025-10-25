# TFT Set 4 Gym Tests

This directory contains comprehensive tests for the TFT Set 4 Gym environment.

## Running Tests

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Using pytest (Recommended)

Run all tests:
```bash
pytest
```

Run specific test categories:
```bash
# Run only fast unit tests
pytest -m "not slow"

# Run only benchmark tests
pytest -m benchmark

# Run only parallel environment tests
pytest -m parallel

# Run only player functionality tests
pytest -m player

# Run only drop rate tests
pytest -m droprate

# Run only minion/PVE tests
pytest -m minion
```

Run tests with verbose output:
```bash
pytest -v
```

Run a specific test file:
```bash
pytest test_parallel_env.py
pytest PlayerTests.py
pytest DropRateTests.py
```

### Using the test runner script

```bash
# Run all tests
python run_tests.py all

# Run only unit tests (fast)
python run_tests.py unit

# Run only benchmark tests
python run_tests.py benchmark

# Run only parallel environment tests
python run_tests.py parallel
```

### Legacy test execution

Individual test files can still be run directly:
```bash
python PlayerTests.py
python DropRateTests.py
python MinionTests.py
python test_parallel_env.py
python benchmark_test.py
python verify_env.py
```

## Test Categories

### Unit Tests (Fast)
- **PlayerTests.py**: Tests player functionality like champion buying, leveling, income calculation
- **MinionTests.py**: Tests PVE combat and minion interactions

### Integration Tests  
- **test_parallel_env.py**: Comprehensive PettingZoo parallel environment API compliance tests
- **verify_env.py**: Environment verification and SB3 compatibility tests

### Performance Tests
- **benchmark_test.py**: Performance benchmarks for environment operations (marked as `slow`)

### Data Validation Tests
- **DropRateTests.py**: Statistical validation of champion drop rates (marked as `slow`)

## Test Markers

The tests use pytest markers for categorization:

- `@pytest.mark.slow`: Long-running tests (benchmarks, statistical tests)
- `@pytest.mark.unit`: Fast unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.benchmark`: Performance benchmark tests
- `@pytest.mark.parallel`: Parallel environment tests
- `@pytest.mark.player`: Player functionality tests
- `@pytest.mark.droprate`: Drop rate validation tests
- `@pytest.mark.minion`: Minion/PVE tests

## Configuration

Test configuration is managed in `pytest.ini`. Key settings:

- Test discovery patterns
- Timeout settings for slow tests  
- Logging configuration
- Warning filters

## Common Issues

1. **Import Errors**: Make sure the package is properly installed or the path is correctly set
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Slow Tests**: Use `-m "not slow"` to skip benchmark and statistical tests during development
4. **Type Errors**: Some legacy test functions may have type annotation issues but should still run correctly

## Writing New Tests

When adding new tests:

1. Use `test_` prefix for test functions
2. Add appropriate pytest markers
3. Include docstrings describing what the test validates
4. Follow existing patterns for setup and teardown
5. Consider performance - mark slow tests appropriately

Example:
```python
@pytest.mark.player
def test_new_player_feature():
    """Test description here."""
    player = setup()
    # Test implementation
    assert expected_condition
```