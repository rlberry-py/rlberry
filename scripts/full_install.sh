#!/bin/bash

# Install everything!

pip install -e .[default]
pip install -e .[jax_agents]
pip install -e .[torch_agents]

pip install pytest
pip install pytest-cov
conda install -c conda-forge jupyterlab
