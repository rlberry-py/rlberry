"""
To run the experiment:

$ python examples/demo_experiment/run.py examples/demo_experiment/params_experiment.yaml

To see more options:

$ python examples/demo_experiment/run.py
"""

from rlberry.experiment import main, doc
from docopt import docopt

arguments = docopt(doc)
main(arguments)
