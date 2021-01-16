#!/bin/bash

cd docs/
sphinx-apidoc -o source/ ../rlberry
make html
cd ..

# Useful: https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/