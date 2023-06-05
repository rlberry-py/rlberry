#!/usr/bin/env bash

# A script for running mypy,
# with all its dependencies installed.

set -o errexit

# Change directory to the project root directory.
mypy --config-file mypy.ini --follow-imports silent
