#!/bin/sh -ex

mypy tidecv
flake8 tidecv tests
black tidecv tests --check
isort tidecv tests scripts --check-only
