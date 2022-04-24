""" 
 ==============================
 Demo: demo_supervised_learning 
 ==============================

 Using rlberry to solve a simple regression problem.

 This example requires scikit-learn to be installed.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from rlberry.agents import Agent
from rlberry.manager import AgentManager
from rlberry.manager import plot_writer_data
from rlberry.utils.torch import choose_device
from rlberry.agents.torch.utils.models import MultiLayerPerceptron
from sklearn.model_selection import KFold


class SimpleRegressionAgent(Agent):
    name = "simple-regression"

    def __init__(
        self,
        X,
        y,
        train_indices,
        val_indices,
        learning_rate=1e-3,
        batch_size=32,
        torch_device="cpu",
        **kwargs,
    ):
        Agent.__init__(self, **kwargs)
        self.X_train, self.y_train = X[train_indices], y[train_indices]
        self.X_val, self.y_val = X[val_indices], y[val_indices]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = choose_device(torch_device)

        self.net = MultiLayerPerceptron(in_size=1, layer_sizes=(64, 64), out_size=1).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.total_steps = 0

    def fit(self, budget: int, **kwargs):
        del kwargs
        for _ in range(budget):
            self.total_steps += 1
            batch_indices = self.rng.choice(self.X_train.shape[0], size=self.batch_size)
            X_batch = torch.tensor(self.X_train[batch_indices, None]).to(self.device)
            y_batch = torch.tensor(self.y_train[batch_indices, None]).to(self.device)

            net_out = self.net(X_batch)
            assert net_out.shape == y_batch.shape
            loss = torch.square(net_out - y_batch).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log to writer (e.g. tensorboard)
            if self.writer:
                self.writer.add_scalar(
                    "training_loss", loss.detach().cpu().item(), self.total_steps
                )

                # validation loss every few steps
                if (self.total_steps % 50) == 0:
                    validation_loss = self.eval()
                    self.writer.add_scalar(
                        "validation_loss", validation_loss, self.total_steps
                    )

    def eval(self, **kwargs):
        del kwargs
        # Compute loss on validation set
        y_hat = self.predict(self.X_val[:, None])[:, 0]
        validation_loss = np.mean((y_hat - self.y_val) ** 2)
        return validation_loss

    def predict(self, x):
        """Convenience method, not required by rlberry."""
        y_hat = self.net(torch.tensor(x).to(self.device)).detach().cpu().numpy()
        return y_hat


if __name__ == "__main__":
    # Create a toy dataset for regression
    rng = np.random.default_rng(123)
    n_samples = 1000
    X = np.random.uniform(low=-1.0, high=1.0, size=(n_samples,))
    y = np.cos(2 * np.pi * X) + 0.1 * rng.normal(size=(n_samples,))

    # Create indices for train-validation splits
    n_splits = 4
    kf = KFold(n_splits=n_splits, random_state=None, shuffle=False)
    all_train_indices = []
    all_val_indices = []
    init_kwargs_per_instance = []
    for train_indices, val_indices in kf.split(X):
        init_kwargs_per_instance.append(
            dict(train_indices=train_indices, val_indices=val_indices)
        )

    # Train all splits in parallel using AgentManager, and
    # visualize with tensorboard.
    manager = AgentManager(
        agent_class=SimpleRegressionAgent,
        n_fit=n_splits,
        max_workers=1,
        parallelization="process",
        fit_budget=1000,
        enable_tensorboard=True,
        init_kwargs=dict(X=X, y=y),
        init_kwargs_per_instance=init_kwargs_per_instance,
    )
    manager.fit()

    # Plot losses
    fig, axes = plt.subplots(1, 2)
    plot_writer_data(manager, tag="training_loss", show=False, ax=axes[0])
    plot_writer_data(manager, tag="validation_loss", show=False, ax=axes[1])

    # Path to tensorboard
    print(
        f"To visualize tensorboard, run: \n"
        f"   tensorboard --logdir {manager.tensorboard_dir}"
    )

    # Visualize function learned by one instance
    plt.figure()
    trained_instance = manager.get_agent_instances()[0]
    x_range = np.linspace(-1.0, 1.0, 500)
    y_hat = trained_instance.predict(x_range[:, None])[:, 0]
    plt.plot(x_range, y_hat, label="learned function", linewidth=2)
    plt.plot(X, y, ".", label="data", alpha=0.25)
    plt.legend()
    plt.show()
