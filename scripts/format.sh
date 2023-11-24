#!/bin/sh -ex

# Sort imports one per line, so autoflake can remove unused imports
#isort --force-single-line-imports tidecv scripts tests

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place tidecv scripts --exclude=__init__.py
black tidecv scripts tests
isort tidecv scripts tests
