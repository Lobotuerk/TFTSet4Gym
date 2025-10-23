"""
TFT Set 4 Gymnasium Environment

A PettingZoo-compatible environment for Teamfight Tactics Set 4.
This package provides a complete simulation of TFT mechanics including:
- Champion abilities and interactions
- Items and synergies
- Combat simulation
- Multi-agent environment interface
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tft-set4-gym",
    version="0.1.0",
    author="Lobotuerk",
    author_email="",
    description="A PettingZoo-compatible environment for Teamfight Tactics Set 4",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Lobotuerk/TFT-Set4-Gym",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Simulation",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tft-demo=tft_set4_gym.demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)