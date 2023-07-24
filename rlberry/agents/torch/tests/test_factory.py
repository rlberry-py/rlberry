import pytest

from rlberry.agents.torch.utils.training import model_factory


@pytest.mark.parametrize(
    "ntype",
    [
        "MultiLayerPerceptron",
        "ConvolutionalNetwork",
        "DuelingNetwork",
        "Table",
    ],
)
def test_dqn_agent(ntype):
    if ntype == "MultiLayerPerceptron":
        nkwargs = {"in_size": 5, "layer_sizes": [5, 5]}
    elif ntype == "ConvolutionalNetwork":
        nkwargs = dict(in_channels=10, in_height=20, in_width=30, out_size=15)
    elif ntype == "DuelingNetwork":
        nkwargs = {"in_size": 5, "out_size": 3}
    elif ntype == "Table":
        nkwargs = dict(state_size=5, action_size=3)
    network = model_factory(ntype, **nkwargs)
