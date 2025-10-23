#!/usr/bin/env python3
"""
Test runner for TFT Set 4 Gym
Run all tests related to the TFT simulator environment.
"""

import sys
import os
import pytest

# Add the package root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

if __name__ == "__main__":
    # Run all tests in the tests directory
    pytest.main([os.path.dirname(__file__)])