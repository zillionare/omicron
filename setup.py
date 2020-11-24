#!/usr/bin/env python
"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "pyemit >= 0.4.0",
    "cfg4py == 0.8.0",
    "arrow ~= 0.15.5",
    "numpy ~= 1.19.4",
    "aioredis == 1.3.1",
    "numba == 0.49.1",
    "SQLAlchemy == 1.3.20",
    "gino == 1.0.1",
    "asyncpg == 0.21.0",
    "aiohttp >= 3.7",
    "sh == 1.14.1",
]

setup_requirements = []

setup(
    author="Aaron Yang",
    author_email="code@jieyu.ai",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
    ],
    description="Core module for Zillionare",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="omicron",
    name="zillionare-omicron",
    packages=find_packages(include=["omicron", "omicron.*"]),
    setup_requires=setup_requirements,
    url="https://github.com/zillionare/omicron",
    version="0.3.0",
    zip_safe=False,
)
