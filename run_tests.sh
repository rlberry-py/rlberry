#!/bin/bash

# disable JIT to get complete coverage report
export NUMBA_DISABLE_JIT=1

# run pytest 
pytest --cov=rlberry --cov-report html:cov_html
