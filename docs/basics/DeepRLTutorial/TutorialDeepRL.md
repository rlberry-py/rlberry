(TutorialDeepRL)=

Quickstart for Deep Reinforcement Learning in rlberry
=====================================================

In this tutorial, we will focus on Deep Reinforcement Learning with the
**Advantage Actor-Critic** algorithm.

Imports
-------

```python
from rlberry.envs import gym_make
from rlberry.manager import plot_writer_data, ExperimentManager, evaluate_agents
from rlberry.agents.stable_baselines import StableBaselinesAgent
from stable_baselines3 import PPO
```

Reminder of the RL setting
--------------------------

We will consider a MDP $M = (\mathcal{S}, \mathcal{A}, p, r, \gamma)$
with:

-   $\mathcal{S}$ the state space,
-   $\mathcal{A}$ the action space,
-   $p(x^\prime \mid x, a)$ the transition probability,
-   $r(x, a, x^\prime)$ the reward of the transition $(x, a, x^\prime)$,
-   $\gamma \in [0,1)$ is the discount factor.

A policy $\pi$ is a mapping from the state space $\mathcal{S}$ to the
probability of selecting each action. The action value function of a
policy is the overall expected reward from a state action.
$Q^\pi(s, a) = \mathbb{E}_{\tau \sim \pi}\big[R(\tau) \mid s_0=s, a_0=a\big]$
where $\tau$ is an episode
$(s_0, a_0, r_0, s_1, a_1, r_1, s_2, ..., s_T, a_T, r_T)$ with the
actions drawn from $\pi(s)$; $R(\tau)$ is the random variable defined as
the cumulative sum of the discounted reward.

The goal is to maximize the cumulative sum of discount rewards:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\big[R(\tau) \big]$$

Gymnasium Environment
---------------------

In this tutorial we are going to use the [Gymnasium library (previously
OpenAI's Gym)](https://gymnasium.farama.org/api/env/). This library
provides a large number of environments to test RL algorithm.

We will focus only on the **Acrobot-v1** environment, although you can
experimenting with other environments such as **CartPole-v1**
or **MountainCar-v0**. The following table presents some basic
components of the three environments, such as the dimensions of their
observation and action spaces and the rewards occurring at each step.

 | Env Info              | CartPole-v1 | Acrobot-v1                  | MountainCar-v0 |
 |:----------------------|:------------|:----------------------------|:---------------|
 | **Observation Space** |  Box(4)     |   Box(6)                    | Box(2)         |
 | **Action Space**      |  Discrete(2)|   Discrete(3)               | Discrete(3)    |
 | **Rewards**           |  1 per step |   -1 if not terminal else 0 | -1 per step    |


Running A2C on **Acrobot-v1**
-----------------------

<span>&#9888;</span> **warning :** depending on the seed, you may get different results, and if you're (un)lucky, your default agent may learn and be better than the tuned agent. <span>&#9888;</span>

In the next example we use default parameters PPO and we use rlberry to train and evaluate our PPO agent. The
default networks are:

```python
"""
The ExperimentManager class is a compact way of experimenting with a deepRL agent.
"""
default_xp = ExperimentManager(
    StableBaselinesAgent,  # The Agent class.
    (gym_make, dict(id="Acrobot-v1")),  # The Environment to solve.
    fit_budget=1e5,  # The number of interactions
    # between the agent and the
    # environment during training.
    init_kwargs=dict(algo_cls=PPO),  # Init value for StableBaselinesAgent
    eval_kwargs=dict(eval_horizon=500),  # The number of interactions
    # between the agent and the
    # environment during evaluations.
    n_fit=3,  # The number of agents to train.
    # Usually, it is good to do more
    # than 1 because the training is
    # stochastic.
    agent_name="PPO default",  # The agent's name.
)

print("Training ...")
default_xp.fit()  # Trains the agent on fit_budget steps!


# Plot the training data:
_ = plot_writer_data(
    [default_xp],
    tag="rollout/ep_rew_mean",
    title="Training Episode Cumulative Rewards",
    show=True,
)
```

```none
Training ...
[INFO] 15:57: Running ExperimentManager fit() for PPO default with n_fit = 3 and max_workers = None.
[INFO] 15:57:                                   agent_name  worker  time/iterations  max_global_step
                                               PPO default    0           1               2048
[INFO] 15:57: [PPO default[worker: 2]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = -500.0 | rollout/ep_len_mean = 500.0 | time/fps = 727 | time/time_elapsed = 2 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
[INFO] 15:57: [PPO default[worker: 0]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = -500.0 | rollout/ep_len_mean = 500.0 | time/fps = 705 | time/time_elapsed = 2 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
[INFO] 15:57: [PPO default[worker: 1]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = -500.0 | rollout/ep_len_mean = 500.0 | time/fps = 719 | time/time_elapsed = 2 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
...
...
...
[INFO] 16:01: [PPO default[worker: 1]] | max_global_step = 100352 | time/iterations = 48 | rollout/ep_rew_mean = -79.26 | rollout/ep_len_mean = 80.26 | time/fps = 398 | time/time_elapsed = 246 | time/total_timesteps = 98304 | train/learning_rate = 0.0003 | train/entropy_loss = -0.1558212914969772 | train/policy_gradient_loss = -0.001575780064376886 | train/value_loss = 9.463472159206868 | train/approx_kl = 0.0019107917323708534 | train/clip_fraction = 0.018701171875 | train/loss = 10.015292167663574 | train/explained_variance = 0.9502505771815777 | train/n_updates = 470 | train/clip_range = 0.2 |
[INFO] 16:01: [PPO default[worker: 2]] | max_global_step = 100352 | time/iterations = 48 | rollout/ep_rew_mean = -81.55 | rollout/ep_len_mean = 82.55 | time/fps = 398 | time/time_elapsed = 246 | time/total_timesteps = 98304 | train/learning_rate = 0.0003 | train/entropy_loss = -0.1251153098186478 | train/policy_gradient_loss = -0.003398912865668535 | train/value_loss = 8.9318665035069 | train/approx_kl = 0.0023679514415562153 | train/clip_fraction = 0.022314453125 | train/loss = 4.039802551269531 | train/explained_variance = 0.9598834589123726 | train/n_updates = 470 | train/clip_range = 0.2 |
```

</br>

```{image} output_5_3.png
:align: center
```

```python
print("Evaluating ...")
_ = evaluate_agents(
    [default_xp], n_simulations=50, show=True
)  # Evaluate the trained agent on
# 10 simulations of 500 steps each.
```


```none
Evaluating ...
[INFO] 16:03: Evaluating PPO default...
[INFO] Evaluation:..................................................  Evaluation finished
```

</br>


```{image} output_6_3.png
:align: center
```

Let's try to change the hyperparameters and see if we can
beat our previous result. This time we use the recommanded hyperparameters from SB3.


```python
tuned_xp = ExperimentManager(
    StableBaselinesAgent,  # The Agent class.
    (gym_make, dict(id="Acrobot-v1")),  # The Environment to solve.
    init_kwargs=dict(  # Where to put the agent's hyperparameters
        algo_cls=PPO,
        # gradient descent steps.
        # descent steps.
        ent_coef=0.00,  # How much to force exploration.
        normalize_advantage=True,  # normalize the advantage
        gae_lambda=0.94,  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        n_epochs=4,  # Number of epoch when optimizing the surrogate loss
        n_steps=256,  # The number of steps to run for the environment per update
    ),
    fit_budget=1e5,  # The number of interactions
    # between the agent and the
    # environment during training.
    eval_kwargs=dict(eval_horizon=500),  # The number of interactions
    # between the agent and the
    # environment during evaluations.
    n_fit=3,  # The number of agents to train.
    # Usually, it is good to do more
    # than 1 because the training is
    # stochastic.
    agent_name="PPO tuned",  # The agent's name.
)


print("Training ...")
tuned_xp.fit()  # Trains the agent on fit_budget steps!


# Plot the training data:
_ = plot_writer_data(
    [default_xp, tuned_xp],
    tag="rollout/ep_rew_mean",
    title="Training Episode Cumulative Rewards",
    show=True,
)
```

```none
[INFO] 16:06: Running ExperimentManager fit() for PPO tuned with n_fit = 3 and max_workers = None.
```

</br>

```none
Training ...
```

</br>

```none
[INFO] 16:07: [PPO tuned[worker: 1]] | max_global_step = 2048 | time/iterations = 7 | time/fps = 609 | time/time_elapsed = 2 | time/total_timesteps = 1792 | train/learning_rate = 0.0003 | train/entropy_loss = -1.0974667742848396 | train/policy_gradient_loss = -0.0009369864128530025 | train/value_loss = 162.15267086029053 | train/approx_kl = 0.00016404897905886173 | train/clip_fraction = 0.0 | train/loss = 70.76416778564453 | train/explained_variance = 0.038179218769073486 | train/n_updates = 24 | train/clip_range = 0.2 | rollout/ep_rew_mean = -500.0 | rollout/ep_len_mean = 500.0 |
[INFO] 16:07: [PPO tuned[worker: 0]] | max_global_step = 2048 | time/iterations = 7 | time/fps = 608 | time/time_elapsed = 2 | time/total_timesteps = 1792 | train/learning_rate = 0.0003 | train/entropy_loss = -1.097521685063839 | train/policy_gradient_loss = -0.0021453166846185923 | train/value_loss = 159.90410614013672 | train/approx_kl = 0.00030635856091976166 | train/clip_fraction = 0.0 | train/loss = 72.51579284667969 | train/explained_variance = 0.00991511344909668 | train/n_updates = 24 | train/clip_range = 0.2 | rollout/ep_rew_mean = -500.0 | rollout/ep_len_mean = 500.0 |
[INFO] 16:07: [PPO tuned[worker: 2]] | max_global_step = 2048 | time/iterations = 7 | time/fps = 604 | time/time_elapsed = 2 | time/total_timesteps = 1792 | train/learning_rate = 0.0003 | train/entropy_loss = -1.0938222110271454 | train/policy_gradient_loss = -0.002581374952569604 | train/value_loss = 167.01866149902344 | train/approx_kl = 0.0009766086004674435 | train/clip_fraction = 0.0 | train/loss = 77.16150665283203 | train/explained_variance = -0.26804113388061523 | train/n_updates = 24 | train/clip_range = 0.2 | rollout/ep_rew_mean = -500.0 | rollout/ep_len_mean = 500.0 |
[INFO] 16:07: [PPO tuned[worker: 1]] | max_global_step = 3840 | time/iterations = 14 | time/fps = 598 | time/time_elapsed = 5 | time/total_timesteps = 3584 | train/learning_rate = 0.0003 | train/entropy_loss = -1.0899569168686867 | train/policy_gradient_loss = -0.002536407206207514 | train/value_loss = 165.09891891479492 | train/approx_kl = 0.0010566194541752338 | train/clip_fraction = 0.0 | train/loss = 79.24039459228516 | train/explained_variance = -0.2665156126022339 | train/n_updates = 52 | train/clip_range = 0.2 | rollout/ep_rew_mean = -493.7142857142857 | rollout/ep_len_mean = 493.85714285714283 |
...
...
...
[INFO] 16:10: ... trained!
[INFO] 16:10: Saved ExperimentManager(PPO tuned) using pickle.
[INFO] 16:10: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO tuned_2024-04-17_16-06-57_5f0b4f99/manager_obj.pickle'
```

</br>


```{image} output_9_3.png
:align: center
```

<span>&#9728;</span> : For more information on plots and visualization, you can check [here (in construction)](visualization_page)
Here, we can see that modifying the hyperparameters has accelerated learning...


```python
print("Evaluating ...")

# Evaluating and comparing the agents :
_ = evaluate_agents([default_xp, tuned_xp], n_simulations=50, show=True)
```

</br>

```none
Evaluating ...
```

</br>

```none
[INFO] 16:11: Evaluating PPO default...
[INFO] Evaluation:..................................................  Evaluation finished
[INFO] 16:11: Evaluating PPO tuned...
[INFO] Evaluation:..................................................  Evaluation finished
```

</br>

```{image} output_10_3.png
:align: center
```
