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
    fit_budget=1e6,  # The number of interactions
    # between the agent and the
    # environment during training.
    init_kwargs=dict(algo_cls=PPO),  # Init value for StableBaselinesAgent
    eval_kwargs=dict(eval_horizon=500),  # The number of interactions
    # between the agent and the
    # environment during evaluations.
    n_fit=1,  # The number of agents to train.
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
[INFO] 17:13: Running ExperimentManager fit() for PPO default with n_fit = 1 and max_workers = None.
```

</br>

```none
Training ...
```

</br>

```none
[INFO] 17:13: Running ExperimentManager fit() for PPO default with n_fit = 1 and max_workers = None.
[INFO] 17:13: [PPO default[worker: 0]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = -500.0 | rollout/ep_len_mean = 500.0 | time/fps = 1692 | time/time_elapsed = 2 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -1.0929798234254122 | train/policy_gradient_loss = -0.006189992171130143 | train/value_loss = 141.715856385231 | train/approx_kl = 0.007915161550045013 | train/clip_fraction = 0.064501953125 | train/loss = 18.01680564880371 | train/explained_variance = -0.006626486778259277 | train/n_updates = 10 | train/clip_range = 0.2 |
[INFO] 17:13: [PPO default[worker: 0]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = -426.7368421052632 | rollout/ep_len_mean = 427.1578947368421 | time/fps = 1500 | time/time_elapsed = 5 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -1.0738775245845318 | train/policy_gradient_loss = -0.007528258884849493 | train/value_loss = 97.8167538881302 | train/approx_kl = 0.007092837244272232 | train/clip_fraction = 0.06171875 | train/loss = 22.329425811767578 | train/explained_variance = 0.16747123003005981 | train/n_updates = 30 | train/clip_range = 0.2 |
[INFO] 17:13: [PPO default[worker: 0]] | max_global_step = 14336 | time/iterations = 6 | rollout/ep_rew_mean = -328.2432432432432 | rollout/ep_len_mean = 328.9189189189189 | time/fps = 1422 | time/time_elapsed = 8 | time/total_timesteps = 12288 | train/learning_rate = 0.0003 | train/entropy_loss = -0.9952785328030587 | train/policy_gradient_loss = -0.004915896706006606 | train/value_loss = 76.84451106190681 | train/approx_kl = 0.009993761777877808 | train/clip_fraction = 0.063134765625 | train/loss = 16.79842185974121 | train/explained_variance = 0.18965238332748413 | train/n_updates = 50 | train/clip_range = 0.2 |
[INFO] 17:14: [PPO default[worker: 0]] | max_global_step = 20480 | time/iterations = 9 | rollout/ep_rew_mean = -260.77142857142854 | rollout/ep_len_mean = 261.5857142857143 | time/fps = 1424 | time/time_elapsed = 12 | time/total_timesteps = 18432 | train/learning_rate = 0.0003 | train/entropy_loss = -0.9102774120867252 | train/policy_gradient_loss = -0.004213768222689396 | train/value_loss = 62.170489662885664 | train/approx_kl = 0.006206408608704805 | train/clip_fraction = 0.018017578125 | train/loss = 12.47692584991455 | train/explained_variance = 0.5796919465065002 | train/n_updates = 80 | train/clip_range = 0.2 |
[INFO] 17:14: [PPO default[worker: 0]] | max_global_step = 26624 | time/iterations = 12 | rollout/ep_rew_mean = -209.51 | rollout/ep_len_mean = 210.45 | time/fps = 1466 | time/time_elapsed = 16 | time/total_timesteps = 24576 | train/learning_rate = 0.0003 | train/entropy_loss = -0.7815314799547195 | train/policy_gradient_loss = -0.007312664366327226 | train/value_loss = 43.975503134727475 | train/approx_kl = 0.008047394454479218 | train/clip_fraction = 0.068896484375 | train/loss = 15.753148078918457 | train/explained_variance = 0.8005904108285904 | train/n_updates = 110 | train/clip_range = 0.2 |
[INFO] 17:14: [PPO default[worker: 0]] | max_global_step = 32768 | time/iterations = 15 | rollout/ep_rew_mean = -144.47 | rollout/ep_len_mean = 145.47 | time/fps = 1495 | time/time_elapsed = 20 | time/total_timesteps = 30720 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6775180252268911 | train/policy_gradient_loss = -0.006114470241300296 | train/value_loss = 34.544698759913445 | train/approx_kl = 0.008602862246334553 | train/clip_fraction = 0.081396484375 | train/loss = 18.569143295288086 | train/explained_variance = 0.8845842033624649 | train/n_updates = 140 | train/clip_range = 0.2 |
[INFO] 17:14: [PPO default[worker: 0]] | max_global_step = 38912 | time/iterations = 18 | rollout/ep_rew_mean = -110.58 | rollout/ep_len_mean = 111.58 | time/fps = 1515 | time/time_elapsed = 24 | time/total_timesteps = 36864 | train/learning_rate = 0.0003 | train/entropy_loss = -0.49321664227172735 | train/policy_gradient_loss = -0.00420049975218717 | train/value_loss = 31.819509637355804 | train/approx_kl = 0.004260750487446785 | train/clip_fraction = 0.04560546875 | train/loss = 12.861420631408691 | train/explained_variance = 0.8872770071029663 | train/n_updates = 170 | train/clip_range = 0.2 |
[INFO] 17:14: [PPO default[worker: 0]] | max_global_step = 45056 | time/iterations = 21 | rollout/ep_rew_mean = -100.74 | rollout/ep_len_mean = 101.74 | time/fps = 1532 | time/time_elapsed = 28 | time/total_timesteps = 43008 | train/learning_rate = 0.0003 | train/entropy_loss = -0.4429807928390801 | train/policy_gradient_loss = -0.0051260963329696095 | train/value_loss = 22.604873842000963 | train/approx_kl = 0.004053293261677027 | train/clip_fraction = 0.046728515625 | train/loss = 9.488809585571289 | train/explained_variance = 0.8918172493577003 | train/n_updates = 200 | train/clip_range = 0.2 |
[INFO] 17:14: [PPO default[worker: 0]] | max_global_step = 51200 | time/iterations = 24 | rollout/ep_rew_mean = -87.7 | rollout/ep_len_mean = 88.7 | time/fps = 1545 | time/time_elapsed = 31 | time/total_timesteps = 49152 | train/learning_rate = 0.0003 | train/entropy_loss = -0.3213156964164227 | train/policy_gradient_loss = -0.002665774476190563 | train/value_loss = 20.599636045098304 | train/approx_kl = 0.0024083973839879036 | train/clip_fraction = 0.017236328125 | train/loss = 12.99824333190918 | train/explained_variance = 0.8812666162848473 | train/n_updates = 230 | train/clip_range = 0.2 |
[INFO] 17:14: [PPO default[worker: 0]] | max_global_step = 57344 | time/iterations = 27 | rollout/ep_rew_mean = -88.24 | rollout/ep_len_mean = 89.24 | time/fps = 1555 | time/time_elapsed = 35 | time/total_timesteps = 55296 | train/learning_rate = 0.0003 | train/entropy_loss = -0.31546892756596207 | train/policy_gradient_loss = -0.009636750411300455 | train/value_loss = 17.506983835995197 | train/approx_kl = 0.008459953591227531 | train/clip_fraction = 0.054296875 | train/loss = 7.022050380706787 | train/explained_variance = 0.8970916345715523 | train/n_updates = 260 | train/clip_range = 0.2 |
[INFO] 17:14: [PPO default[worker: 0]] | max_global_step = 63488 | time/iterations = 30 | rollout/ep_rew_mean = -96.3 | rollout/ep_len_mean = 97.29 | time/fps = 1563 | time/time_elapsed = 39 | time/total_timesteps = 61440 | train/learning_rate = 0.0003 | train/entropy_loss = -0.28947532204911114 | train/policy_gradient_loss = -0.001286450280167628 | train/value_loss = 35.65989404916763 | train/approx_kl = 0.0011013790499418974 | train/clip_fraction = 0.009375 | train/loss = 18.42564582824707 | train/explained_variance = 0.8290886282920837 | train/n_updates = 290 | train/clip_range = 0.2 |
[INFO] 17:14: [PPO default[worker: 0]] | max_global_step = 69632 | time/iterations = 33 | rollout/ep_rew_mean = -100.76 | rollout/ep_len_mean = 101.74 | time/fps = 1571 | time/time_elapsed = 43 | time/total_timesteps = 67584 | train/learning_rate = 0.0003 | train/entropy_loss = -0.22212261543609202 | train/policy_gradient_loss = -0.0013434989712550304 | train/value_loss = 19.162969033420087 | train/approx_kl = 0.0013764428440481424 | train/clip_fraction = 0.0099609375 | train/loss = 7.710111618041992 | train/explained_variance = 0.8366725593805313 | train/n_updates = 320 | train/clip_range = 0.2 |
[INFO] 17:14: [PPO default[worker: 0]] | max_global_step = 75776 | time/iterations = 36 | rollout/ep_rew_mean = -99.14 | rollout/ep_len_mean = 100.11 | time/fps = 1574 | time/time_elapsed = 46 | time/total_timesteps = 73728 | train/learning_rate = 0.0003 | train/entropy_loss = -0.22284777173772455 | train/policy_gradient_loss = -0.003595446604595054 | train/value_loss = 47.21005566716194 | train/approx_kl = 0.0020544053986668587 | train/clip_fraction = 0.0236328125 | train/loss = 19.401063919067383 | train/explained_variance = 0.7258298695087433 | train/n_updates = 350 | train/clip_range = 0.2 |
[INFO] 17:14: [PPO default[worker: 0]] | max_global_step = 81920 | time/iterations = 39 | rollout/ep_rew_mean = -83.72 | rollout/ep_len_mean = 84.72 | time/fps = 1579 | time/time_elapsed = 50 | time/total_timesteps = 79872 | train/learning_rate = 0.0003 | train/entropy_loss = -0.1912181852152571 | train/policy_gradient_loss = -0.0029860971349989994 | train/value_loss = 20.7523592710495 | train/approx_kl = 0.002668976318091154 | train/clip_fraction = 0.018408203125 | train/loss = 10.850214004516602 | train/explained_variance = 0.9057487025856972 | train/n_updates = 380 | train/clip_range = 0.2 |
[INFO] 17:14: [PPO default[worker: 0]] | max_global_step = 88064 | time/iterations = 42 | rollout/ep_rew_mean = -84.51 | rollout/ep_len_mean = 85.51 | time/fps = 1584 | time/time_elapsed = 54 | time/total_timesteps = 86016 | train/learning_rate = 0.0003 | train/entropy_loss = -0.16024661988485606 | train/policy_gradient_loss = -0.0008719900164578575 | train/value_loss = 17.229137426614763 | train/approx_kl = 0.0011531587224453688 | train/clip_fraction = 0.006982421875 | train/loss = 9.205939292907715 | train/explained_variance = 0.9104878678917885 | train/n_updates = 410 | train/clip_range = 0.2 |
[INFO] 17:14: [PPO default[worker: 0]] | max_global_step = 94208 | time/iterations = 45 | rollout/ep_rew_mean = -92.42 | rollout/ep_len_mean = 93.42 | time/fps = 1589 | time/time_elapsed = 57 | time/total_timesteps = 92160 | train/learning_rate = 0.0003 | train/entropy_loss = -0.12848295361036435 | train/policy_gradient_loss = -0.002640917531971354 | train/value_loss = 33.56368041336536 | train/approx_kl = 0.001574347261339426 | train/clip_fraction = 0.01591796875 | train/loss = 22.291500091552734 | train/explained_variance = 0.8023407310247421 | train/n_updates = 440 | train/clip_range = 0.2 |
[INFO] 17:14: [PPO default[worker: 0]] | max_global_step = 100352 | time/iterations = 48 | rollout/ep_rew_mean = -87.38 | rollout/ep_len_mean = 88.37 | time/fps = 1593 | time/time_elapsed = 61 | time/total_timesteps = 98304 | train/learning_rate = 0.0003 | train/entropy_loss = -0.23304079584777354 | train/policy_gradient_loss = -0.0029938832129118966 | train/value_loss = 15.94501298815012 | train/approx_kl = 0.0073814853094518185 | train/clip_fraction = 0.043359375 | train/loss = 5.659388542175293 | train/explained_variance = 0.7538405954837799 | train/n_updates = 470 | train/clip_range = 0.2 |
[INFO] 17:14: ... trained!
[INFO] 17:14: Saved ExperimentManager(PPO default) using pickle.
[INFO] 17:14: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO default_2024-04-16_17-13-48_3ce30c7c/manager_obj.pickle'
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
[INFO] 17:15: Evaluating PPO default...
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
        batch_size=256,  # Number of interactions used to
        # estimate the policy gradient
        # for each policy update.
    ),
    fit_budget=1e6,  # The number of interactions
    # between the agent and the
    # environment during training.
    eval_kwargs=dict(eval_horizon=500),  # The number of interactions
    # between the agent and the
    # environment during evaluations.
    n_fit=1,  # The number of agents to train.
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
[INFO] 17:15: Running ExperimentManager fit() for PPO tuned with n_fit = 1 and max_workers = None.
```

</br>

```none
Training ...
```

</br>

```none
[INFO] 17:15: [PPO tuned[worker: 0]] | max_global_step = 6400 | time/iterations = 24 | time/fps = 2105 | time/time_elapsed = 2 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -1.0572556108236313 | train/policy_gradient_loss = -0.00047643095604144037 | train/value_loss = 128.2477970123291 | train/approx_kl = 5.645095370709896e-05 | train/clip_fraction = 0.0 | train/loss = 55.283321380615234 | train/explained_variance = 0.20806163549423218 | train/n_updates = 92 | train/clip_range = 0.2 | rollout/ep_rew_mean = -333.1111111111111 | rollout/ep_len_mean = 333.8333333333333 |
[INFO] 17:15: [PPO tuned[worker: 0]] | max_global_step = 12544 | time/iterations = 48 | time/fps = 2068 | time/time_elapsed = 5 | time/total_timesteps = 12288 | train/learning_rate = 0.0003 | train/entropy_loss = -0.7988822720944881 | train/policy_gradient_loss = -0.001968120865058154 | train/value_loss = 89.75015926361084 | train/approx_kl = 0.0006263391114771366 | train/clip_fraction = 0.0 | train/loss = 43.21318054199219 | train/explained_variance = 0.33033323287963867 | train/n_updates = 188 | train/clip_range = 0.2 | rollout/ep_rew_mean = -218.23214285714286 | rollout/ep_len_mean = 219.14285714285714 |
[INFO] 17:15: [PPO tuned[worker: 0]] | max_global_step = 18432 | time/iterations = 71 | time/fps = 2007 | time/time_elapsed = 9 | time/total_timesteps = 18176 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6998001970350742 | train/policy_gradient_loss = -0.0007780164014548063 | train/value_loss = 91.61138582229614 | train/approx_kl = 0.0002921083942055702 | train/clip_fraction = 0.0 | train/loss = 49.43030548095703 | train/explained_variance = 0.199457049369812 | train/n_updates = 280 | train/clip_range = 0.2 | rollout/ep_rew_mean = -175.49 | rollout/ep_len_mean = 176.45 |
[INFO] 17:15: [PPO tuned[worker: 0]] | max_global_step = 24320 | time/iterations = 94 | time/fps = 1988 | time/time_elapsed = 12 | time/total_timesteps = 24064 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6855811513960361 | train/policy_gradient_loss = -0.00044799927854910493 | train/value_loss = 84.82663869857788 | train/approx_kl = 0.0006827195174992085 | train/clip_fraction = 0.0 | train/loss = 40.594425201416016 | train/explained_variance = 0.40675830841064453 | train/n_updates = 372 | train/clip_range = 0.2 | rollout/ep_rew_mean = -120.54 | rollout/ep_len_mean = 121.54 |
[INFO] 17:15: [PPO tuned[worker: 0]] | max_global_step = 30720 | time/iterations = 119 | time/fps = 2000 | time/time_elapsed = 15 | time/total_timesteps = 30464 | train/learning_rate = 0.0003 | train/entropy_loss = -0.4461955167353153 | train/policy_gradient_loss = -0.0019316497491672635 | train/value_loss = 39.96290922164917 | train/approx_kl = 0.00022931606508791447 | train/clip_fraction = 0.0 | train/loss = 18.5803279876709 | train/explained_variance = 0.7989254891872406 | train/n_updates = 472 | train/clip_range = 0.2 | rollout/ep_rew_mean = -100.35 | rollout/ep_len_mean = 101.35 |
[INFO] 17:15: [PPO tuned[worker: 0]] | max_global_step = 36864 | time/iterations = 143 | time/fps = 1995 | time/time_elapsed = 18 | time/total_timesteps = 36608 | train/learning_rate = 0.0003 | train/entropy_loss = -0.3954697884619236 | train/policy_gradient_loss = -0.0026701100287027657 | train/value_loss = 31.354408621788025 | train/approx_kl = 0.0005203159525990486 | train/clip_fraction = 0.001953125 | train/loss = 13.910658836364746 | train/explained_variance = 0.8476790189743042 | train/n_updates = 568 | train/clip_range = 0.2 | rollout/ep_rew_mean = -94.41 | rollout/ep_len_mean = 95.41 |
[INFO] 17:15: [PPO tuned[worker: 0]] | max_global_step = 43264 | time/iterations = 168 | time/fps = 2005 | time/time_elapsed = 21 | time/total_timesteps = 43008 | train/learning_rate = 0.0003 | train/entropy_loss = -0.45925334468483925 | train/policy_gradient_loss = -0.00039937475230544806 | train/value_loss = 33.857659459114075 | train/approx_kl = 0.0001619535032659769 | train/clip_fraction = 0.0 | train/loss = 16.692773818969727 | train/explained_variance = 0.8541684299707413 | train/n_updates = 668 | train/clip_range = 0.2 | rollout/ep_rew_mean = -97.28 | rollout/ep_len_mean = 98.28 |
[INFO] 17:16: [PPO tuned[worker: 0]] | max_global_step = 49408 | time/iterations = 192 | time/fps = 2003 | time/time_elapsed = 24 | time/total_timesteps = 49152 | train/learning_rate = 0.0003 | train/entropy_loss = -0.34735729917883873 | train/policy_gradient_loss = -0.0035068761208094656 | train/value_loss = 24.374780774116516 | train/approx_kl = 0.0010997226927429438 | train/clip_fraction = 0.00390625 | train/loss = 7.9426116943359375 | train/explained_variance = 0.894627682864666 | train/n_updates = 764 | train/clip_range = 0.2 | rollout/ep_rew_mean = -86.75 | rollout/ep_len_mean = 87.75 |
[INFO] 17:16: [PPO tuned[worker: 0]] | max_global_step = 55552 | time/iterations = 216 | time/fps = 2004 | time/time_elapsed = 27 | time/total_timesteps = 55296 | train/learning_rate = 0.0003 | train/entropy_loss = -0.25039983075112104 | train/policy_gradient_loss = -0.002326533489394933 | train/value_loss = 16.638584196567535 | train/approx_kl = 0.0007837177254259586 | train/clip_fraction = 0.0029296875 | train/loss = 8.219216346740723 | train/explained_variance = 0.9247641786932945 | train/n_updates = 860 | train/clip_range = 0.2 | rollout/ep_rew_mean = -83.17 | rollout/ep_len_mean = 84.17 |
[INFO] 17:16: [PPO tuned[worker: 0]] | max_global_step = 61696 | time/iterations = 240 | time/fps = 2004 | time/time_elapsed = 30 | time/total_timesteps = 61440 | train/learning_rate = 0.0003 | train/entropy_loss = -0.20490748342126608 | train/policy_gradient_loss = -0.002675430674571544 | train/value_loss = 11.756703406572342 | train/approx_kl = 0.0009058164432644844 | train/clip_fraction = 0.00390625 | train/loss = 4.831850528717041 | train/explained_variance = 0.9516513273119926 | train/n_updates = 956 | train/clip_range = 0.2 | rollout/ep_rew_mean = -85.84 | rollout/ep_len_mean = 86.84 |
[INFO] 17:16: [PPO tuned[worker: 0]] | max_global_step = 67840 | time/iterations = 264 | time/fps = 2001 | time/time_elapsed = 33 | time/total_timesteps = 67584 | train/learning_rate = 0.0003 | train/entropy_loss = -0.1371078807860613 | train/policy_gradient_loss = -0.0002851566532626748 | train/value_loss = 39.02770984172821 | train/approx_kl = 0.0003623703960329294 | train/clip_fraction = 0.001953125 | train/loss = 11.338791847229004 | train/explained_variance = 0.8494161516427994 | train/n_updates = 1052 | train/clip_range = 0.2 | rollout/ep_rew_mean = -85.28 | rollout/ep_len_mean = 86.28 |
[INFO] 17:16: [PPO tuned[worker: 0]] | max_global_step = 73984 | time/iterations = 288 | time/fps = 1999 | time/time_elapsed = 36 | time/total_timesteps = 73728 | train/learning_rate = 0.0003 | train/entropy_loss = -0.09845160157419741 | train/policy_gradient_loss = 1.6727717593312263e-06 | train/value_loss = 62.19053077697754 | train/approx_kl = 1.8619466572999954e-06 | train/clip_fraction = 0.0 | train/loss = 22.528833389282227 | train/explained_variance = 0.8270737081766129 | train/n_updates = 1148 | train/clip_range = 0.2 | rollout/ep_rew_mean = -88.41 | rollout/ep_len_mean = 89.41 |
[INFO] 17:16: [PPO tuned[worker: 0]] | max_global_step = 80384 | time/iterations = 313 | time/fps = 2006 | time/time_elapsed = 39 | time/total_timesteps = 80128 | train/learning_rate = 0.0003 | train/entropy_loss = -0.18064112542197108 | train/policy_gradient_loss = -0.00035066844429820776 | train/value_loss = 11.814534723758698 | train/approx_kl = 0.00025136792100965977 | train/clip_fraction = 0.0009765625 | train/loss = 4.615758419036865 | train/explained_variance = 0.947364155203104 | train/n_updates = 1248 | train/clip_range = 0.2 | rollout/ep_rew_mean = -85.55 | rollout/ep_len_mean = 86.55 |
[INFO] 17:16: [PPO tuned[worker: 0]] | max_global_step = 86784 | time/iterations = 338 | time/fps = 2012 | time/time_elapsed = 42 | time/total_timesteps = 86528 | train/learning_rate = 0.0003 | train/entropy_loss = -0.16239579999819398 | train/policy_gradient_loss = -0.0017573575023561716 | train/value_loss = 11.899139255285263 | train/approx_kl = 0.0012513825204223394 | train/clip_fraction = 0.0048828125 | train/loss = 6.563343524932861 | train/explained_variance = 0.9511667862534523 | train/n_updates = 1348 | train/clip_range = 0.2 | rollout/ep_rew_mean = -85.92 | rollout/ep_len_mean = 86.92 |
[INFO] 17:16: [PPO tuned[worker: 0]] | max_global_step = 93184 | time/iterations = 363 | time/fps = 2018 | time/time_elapsed = 46 | time/total_timesteps = 92928 | train/learning_rate = 0.0003 | train/entropy_loss = -0.13055721437558532 | train/policy_gradient_loss = -0.0035327630466781557 | train/value_loss = 11.075345873832703 | train/approx_kl = 0.0011467819567769766 | train/clip_fraction = 0.0048828125 | train/loss = 6.420516490936279 | train/explained_variance = 0.9501275159418583 | train/n_updates = 1448 | train/clip_range = 0.2 | rollout/ep_rew_mean = -86.08 | rollout/ep_len_mean = 87.08 |
[INFO] 17:16: [PPO tuned[worker: 0]] | max_global_step = 99584 | time/iterations = 388 | time/fps = 2023 | time/time_elapsed = 49 | time/total_timesteps = 99328 | train/learning_rate = 0.0003 | train/entropy_loss = -0.10564398556016386 | train/policy_gradient_loss = -0.0005617931601591408 | train/value_loss = 20.7921279668808 | train/approx_kl = 0.00010642432607710361 | train/clip_fraction = 0.0 | train/loss = 8.198343276977539 | train/explained_variance = 0.9145599380135536 | train/n_updates = 1548 | train/clip_range = 0.2 | rollout/ep_rew_mean = -90.74 | rollout/ep_len_mean = 91.73 |
[INFO] 17:16: ... trained!
[INFO] 17:16: Saved ExperimentManager(PPO tuned) using pickle.
[INFO] 17:16: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO tuned_2024-04-16_17-15-35_f1c48474/manager_obj.pickle'
```

</br>


```{image} output_9_3.png
:align: center
```

<span>&#9728;</span> : For more information on plots and visualization, you can check [here (in construction)](visualization_page)

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
[INFO] 17:16: Evaluating PPO default...
[INFO] Evaluation:..................................................  Evaluation finished
[INFO] 17:16: Evaluating PPO tuned...
[INFO] Evaluation:..................................................  Evaluation finished
```

</br>

```{image} output_10_3.png
:align: center
```
