#!/usr/bin/env python3
"""Setup script for Franka Golf."""

from setuptools import setup, find_packages

setup(
    name="franka-golf",
    version="0.1.0",
    description="Franka Golf - ArenaX Labs ML Hiring Challenge",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.2.0",
        "gymnasium>=0.29.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    entry_points={
        "console_scripts": [
            "franka-golf=main:cli",
        ],
    },
    python_requires=">=3.9",
) 