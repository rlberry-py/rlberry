from rlberry.envs import Wrapper


class WriterWrapper(Wrapper):
    """
    Wrapper for environment to automatically record reward or action in writer.

    Parameters
    ----------

    env : gym.Env or tuple (constructor, kwargs)
        Environment used to fit the agent.

    writer : object, default: None
        Writer object (e.g. tensorboard SummaryWriter).

    write_scalar : string in {"reward", "action", "action_and_reward"},
                    default = "reward"
        Scalar that will be recorded in the writer.

    """

    def __init__(self, env, writer, write_scalar="regret"):
        Wrapper.__init__(self, env)
        self.writer = writer
        self.write_scalar = write_scalar
        self.iteration_ = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        self.iteration_ += 1
        if self.write_scalar == "reward":
            self.writer.add_scalar('reward', reward, self.iteration_)
        elif self.write_scalar == "action":
            self.writer.add_scalar('action', action, self.iteration_)
        elif self.write_scalar == "action_and_reward":
            self.writer.add_scalar('reward', action, self.iteration_)
            self.writer.add_scalar('action', action, self.iteration_)
        else:
            raise ValueError("write_scalar %s is not known" %(self.write_scalar))

        return observation, reward, done, info
