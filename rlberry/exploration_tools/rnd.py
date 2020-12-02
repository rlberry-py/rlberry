from rlberry.exploration_tools.uncertainty_estimator \
    import UncertaintyEstimator
from rlberry.agents.utils.torch_models import ConvolutionalNetwork, MultiLayerPerceptron
import torch

# choose device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RandomNetworkDistillation(UncertaintyEstimator):
    """
    References
    ----------
    Burda Yuri, Harrison Edwards, Amos Storkey, and Oleg Klimov. 2018.
    "Exploration by random network distillation."
    In International Conference on Learning Representations.
    """

    def __init__(self, observation_space, action_space, **kwargs):
        UncertaintyEstimator.__init__(self, observation_space, action_space)

        self.random_embedding = None
        self.predicted_embedding = None
        self.rnd_optimizer = None

        if len(self.observation_space.shape) == 3:
            H, W, C = self.observation_space.shape
            self.random_target_network = ConvolutionalNetwork(in_channels=C, in_width=W, in_height=H,
                                                              activation="ELU").to(device=device)
            self.predictor_network = ConvolutionalNetwork(in_channels=C, in_width=W, in_height=H, activation="ELU").to(
                device=device)
        elif len(self.observation_space.shape) == 2:
            H, W = self.observation_space.shape
            self.random_target_network = ConvolutionalNetwork(in_channels=1, in_width=W, in_height=H,
                                                              activation="ELU").to(device=device)
            self.predictor_network = ConvolutionalNetwork(in_channels=1, in_width=W, in_height=H, activation="ELU").to(
                device=device)
        elif len(self.observation_space.shape) == 1:
            self.random_target_network = MultiLayerPerceptron(in_size=self.observation_space.shape[0],
                                                              activation="RELU").to(device=device)
            self.predictor_network = MultiLayerPerceptron(in_size=self.observation_space.shape[0],
                                                          activation="RELU").to(device=device)
        else:
            raise ValueError("Incompatible observation shape: {}".format(self.observation_space.shape))

    def optimizer(self, learning_rate):
        return torch.optim.Adam(self.predictor_network.parameters(),
                                lr=learning_rate,
                                betas=(0.9, 0.999))

    def update(self, state, action, next_state, reward, **kwargs):
        self.random_embedding = self.random_target_network(torch.from_numpy(state).unsqueeze(0).to(device))
        self.predicted_embedding = self.predictor_network.forward(torch.from_numpy(state).unsqueeze(0).to(device))

    def measure(self, state, action, **kwargs):
        return torch.norm(self.predicted_embedding.detach() - self.random_embedding.detach(), p=2)
