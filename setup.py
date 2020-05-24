#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0',
                'pyemit>=0.4.0',
                'cfg4py>=0.4.0',
                'arrow==0.15.5',
                'numpy>=1.18.1',
                'aioredis==1.3.1',
                'numba==0.49.1'
                ]

setup_requirements = []

test_requirements = ['omega']

setup(
    author="Aaron Yang",
    author_email='code@jieyu.ai',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Core module for Zillionare",
    entry_points={
        'console_scripts': [
            'omicron=omicron.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='omicron',
    name='zillionare-omicron',
    packages=find_packages(include=['omicron', 'omicron.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/zillionare/omicron',
    version='0.1.2',
    zip_safe=False,
)
