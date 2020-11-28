# Contribution guidelines for rlberry

Currently, we are accepting the following forms of contributions:

-   Bug reports (open an [Issue](https://github.com/rlberry-py/rlberry/issues/new?assignees=&labels=&template=bug_report.md&title=) indicating your system information, the current behavior, the expected behavior, a standalone code to reproduce the issue and other information as needed)
-   Pull requests for bug fixes
-   Documentation improvements
-   New environments
-   New agents


# Guidelines for new agents


* Create a folder for the agent `rlberry/agents/agent_name`.
* Create `rlberry/agents/agent_name/__init__.py`.
* Write a test to check that the agent is running `rlberry/agents/test_agent_name.py`.

### Agent code template

The template below gives the general structure that the Agent code must follow. See more options in the  abstract `Agent` class (`rlberry/agents/agent.py`).

```python

class MyAgent(Agent):
    name = "MyAgent"
    fit_info = ('episode_rewards',)   # tuple of strings containing the keys in the dictionary returned by fit()

    def __init__(self,
                 env,
                 param_1,
                 param_2,
                 param_n,
                 **kwargs):
        Agent.__init__(self, env, **kwargs)
    
    def fit(self, **kwargs):
        """
        ** Must be implemented. **

        Trains the agent.
        """
        # code to train the agent
        # ...

        # information about training, e.g., episode_rewards,
        # an array with the rewards gathered in each episode
        info = {'episode_rewards': episode_rewards}
        return info

    def policy(self, observation, **kwargs):
        """
        ** Must be implemented. **

        Returns an action, given the observation
        """
        # for example
        action = self.policy_network(observation)
        return action


    @classmethod
    def sample_parameters(cls, trial):
        """
        ** Optional **

        Sample hyperparameters for hyperparam optimization using
        Optuna (https://optuna.org/).

        Parameters
        ----------
        trial: optuna.trial
        """
        # for example
        batch_size = trial.suggest_categorical('batch_size',
                                               [1, 4, 8, 16, 32, 64])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1)
        return {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                }
```