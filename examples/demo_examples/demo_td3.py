from rlberry.envs import gym_make
from rlberry.manager import AgentManager, plot_writer_data
from rlberry.agents.torch.td3 import nets as td3nets
from rlberry.agents.torch.td3.td3 import TD3Agent


def q_net_constructor(env):
    return td3nets.ED3MLPCritic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        n_heads=2,
        shared_hidden_sizes=(),
        head_hidden_sizes=(256, 256),
    )


def pi_net_constructor(env):
    return td3nets.ED3MLPActor(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_sizes=(256, 256),
    )


if __name__ == "__main__":
    env = (gym_make, dict(id="CartPole-v0"))
    # env = (gym_make, dict(id='CartPole-v1'))
    # env = (gym_make, dict(id='BipedalWalker-v3'))
    # env = (gym_make, dict(id='Pendulum-v1'))
    # env = (gym_make, dict(id="LunarLander-v2"))
    # env = (gym_make, dict(id='LunarLanderContinuous-v2'))
    params = dict(
        q_net_constructor=q_net_constructor,
        pi_net_constructor=pi_net_constructor,
        learning_starts=5_000,
        train_interval=-1,
        gamma=0.98,
    )
    fit_kwargs = dict(
        fit_budget=150_000,
    )

    manager = AgentManager(
        TD3Agent,
        train_env=env,
        fit_kwargs=fit_kwargs,
        init_kwargs=params,
        n_fit=1,
        output_dir="temp/",
        parallelization="process",
    )
    manager.fit()

    plot_writer_data(manager, tag="eval_rewards", show=False)
    plot_writer_data(manager, tag="buffer_size", show=False)
    plot_writer_data(manager, tag="q_loss", xtag="total_updates", show=False)
    plot_writer_data(manager, tag="policy_loss", xtag="total_updates", show=False)
    plot_writer_data(manager, tag="policy_reg_loss", xtag="total_updates", show=False)
    plot_writer_data(manager, tag="total_episodes", show=False)
    plot_writer_data(manager, tag="dw_time_elapsed", show=True)

    # render
    agent = manager.get_agent_instances()[0]
    env = agent.eval_env
    for _ in range(3):
        state = env.reset()
        done = False
        while not done:
            action = agent.policy(state, evaluation=True)
            state, _, done, _ = env.step(action)
            env.render()
    env.close()
