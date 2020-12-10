#!/usr/bin/env python
"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()


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
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="omicron",
    name="zillionare-omicron",
    packages=find_packages(include=["omicron", "omicron.*"]),
    url="https://github.com/zillionare/omicron",
    zip_safe=False,
)
