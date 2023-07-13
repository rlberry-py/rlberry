
# Comparison of Agents

The performance of one execution of a Deep RL algorithm is random so that independent executions are needed to assess it precisely.
In this section we use multiple hypothesis testing to assert that we used enough fits to be able to say that the agents are indeed differents and that the perceived differences are not just a result of randomness.


## Quick reminder on hypothesis testing

### Two sample testing

In its most simple form, a statistical test is aimed at deciding, whether a given collection of data $X_1,\dots,X_N$ adheres to some hypothesis $H_0$ (called the null hypothesis), or if it is a better fit for an alternative hypothesis $H_1$.

Consider two samples $X_1,\dots,X_N$ and $Y_1,\dots,Y_N$ and do a two-sample test deciding whether the mean of the distribution of the $X_i$'s is equal to the mean of the distribution of the $Y_i$'s:

\begin{equation*}
H_0 : \mathbb{E}[X] = \mathbb{E}[Y] \quad \text{vs}\quad H_1: \mathbb{E}[X] \neq \mathbb{E}[Y]
\end{equation*}

In both cases, the result of a test is either accept $H_0$ or reject $H_0$. This answer is not a ground truth: there is some probability that we make an error. However, this  probability of error is often controlled and can be decomposed in type I error and type II errors (often denoted $\alpha$ and $\beta$ respectively).

<center>

|                 | $H_0$ is true         | $H_0$ is false        |
|-----------------|-----------------------|-----------------------|
| We accept $H_0$ | No error              | type II error $\beta$ |
| We reject $H_0$ | type I error $\alpha$ | No error              |

</center>

Note that the problem is not symmetric: failing to reject the null hypothesis does not mean that the null hypothesis is true.  It can be that there is not enough data to reject $H_0$.

### Multiple testing

When doing simultaneously several statistical tests, one must be careful that the error of each test accumulate and if one is not cautious, the overall error may become non-negligible. As a consequence, multiple strategies have been developed to deal with multiple testing problem.

To deal with the multiple testing problem, the first step is to define what is an error. There are several definitions of error in multiple testing, one possibility is the Family-wise error which is defined as the probability to make at least one false rejection (at least one type I error):

$$\mathrm{FWE} = \mathbb{P}_{H_j, j \in \textbf{I}}\left(\exists j \in \textbf{I}:\quad  \text{reject }H_j \right),$$

where $\mathbb{P}_{H_j, j \in \textbf{I}}$ is used to denote the probability when $\textbf{I}$ is the set of indices of the hypotheses that are actually true (and $\textbf{I}^c$ the set of hypotheses that are actually false).

## Multiple agent comparison in rlberry

We compute the performances of one agent as follows:

```python
import numpy as np
from rlberry.envs import gym_make
from rlberry.agents.torch import A2CAgent
from rlberry.manager import AgentManager, evaluate_agents

env_ctor = gym_make
env_kwargs = dict(id="CartPole-v1")

n_simulations = 50
n_fit = 8

rbagent = AgentManager(
    A2CAgent,
    (env_ctor, env_kwargs),
    agent_name="A2CAgent",
    fit_budget=3e4,
    eval_kwargs=dict(eval_horizon=500),
    n_fit=n_fit,
)

rbagent.fit()  # get 5 trained agents
performances = [
    np.mean(rbagent.eval_agents(n_simulations, agent_id=idx)) for idx in range(8)
]
```

We begin by training all the agents (here $8$ agents). Then, we evaluate each trained agent `n_simulations` times (here $50$). The performance of one trained agent is then the mean of its evaluations. We do this for each agent that we trained, obtaining `n_fit` evaluation performances. These `n_fit` numbers are the random numbers that will be used to do hypothesis testing.

The evaluation and statistical hypothesis testing is handled through the function {class}`~rlberry.manager.compare_agents`.

For example we may compare PPO, A2C and DQNAgent on Cartpole with the following code.

``` python
from rlberry.agents.torch import A2CAgent, PPOAgent, DQNAgent
from rlberry.manager.comparison import compare_agents

agents = [
    AgentManager(
        A2CAgent,
        (env_ctor, env_kwargs),
        agent_name="A2CAgent",
        fit_budget=3e4,
        eval_kwargs=dict(eval_horizon=500),
        n_fit=n_fit,
    ),
    AgentManager(
        PPOAgent,
        (env_ctor, env_kwargs),
        agent_name="PPOAgent",
        fit_budget=3e4,
        eval_kwargs=dict(eval_horizon=500),
        n_fit=n_fit,
    ),
    AgentManager(
        DQNAgent,
        (env_ctor, env_kwargs),
        agent_name="DQNAgent",
        fit_budget=3e4,
        eval_kwargs=dict(eval_horizon=500),
        n_fit=n_fit,
    ),
]

for agent in agents:
    agent.fit()

print(compare_agents(agents))
```

```
       Agent1 vs Agent2  mean Agent1  mean Agent2   mean diff    std diff     p-val significance
0  A2CAgent vs PPOAgent   212.673875   423.125500 -210.451625  143.458479  0.001939           **
1  A2CAgent vs DQNAgent   212.673875   443.296625 -230.622750  152.642416  0.000790          ***
2  PPOAgent vs DQNAgent   423.125500   443.296625  -20.171125  102.137556  0.923614
```

The results of `compare_agents(agents)` show the p-values if the method is `tukey_hsd` and in all the cases it shows a significance level. If there is at least one "*" in the significance it means that the decision of the hypothesis test is "reject $H_0$". In our case, we see that A2C seems significantly worst than both PPO and DQN but the difference between PPO and DQN is not statistically significant. Remark that no significance (which is to say, decision to accept $H_0$) does not necessarily mean that the algorithms perform the same, it can be that there is not enough data.

*Remark*: the comparison we do here is a black-box comparison in the sense that we don't care how the algorithms were tuned or how many training steps are used, we suppose that the user already tuned these parameters adequately for a fair comparison.
