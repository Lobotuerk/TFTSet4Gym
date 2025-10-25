#!/usr/bin/env python3
"""
Test runner for TFT Set 4 Gym
Run all tests related to the TFT simulator environment using pytest.
"""

import sys
import os
import pytest

# Add the package root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_all_tests():
    """Run all tests using pytest."""
    test_dir = os.path.dirname(__file__)
    return pytest.main([
        test_dir,
        "-v",
        "--tb=short",
        "--color=yes"
    ])


def run_unit_tests():
    """Run only unit tests (fast tests)."""
    test_dir = os.path.dirname(__file__)
    return pytest.main([
        test_dir,
        "-v",
        "-m", "not slow",
        "--tb=short",
        "--color=yes"
    ])


def run_benchmark_tests():
    """Run only benchmark tests."""
    test_dir = os.path.dirname(__file__)
    return pytest.main([
        test_dir,
        "-v",
        "-m", "benchmark",
        "--tb=short", 
        "--color=yes"
    ])


def run_parallel_tests():
    """Run only parallel environment tests."""
    test_dir = os.path.dirname(__file__)
    return pytest.main([
        test_dir,
        "-v",
        "-m", "parallel",
        "--tb=short",
        "--color=yes"
    ])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type == "unit":
            sys.exit(run_unit_tests())
        elif test_type == "benchmark":
            sys.exit(run_benchmark_tests())
        elif test_type == "parallel":
            sys.exit(run_parallel_tests())
        elif test_type == "all":
            sys.exit(run_all_tests())
        else:
            print("Usage: python run_tests.py [unit|benchmark|parallel|all]")
            print("  unit      - Run fast unit tests only")
            print("  benchmark - Run benchmark tests only")
            print("  parallel  - Run parallel environment tests only")
            print("  all       - Run all tests (default)")
            sys.exit(1)
    else:
        # Default: run all tests
        sys.exit(run_all_tests())