from rlberry.envs.interface import Model
import rlberry.spaces as spaces
import logging
import numpy as np

import requests

logger = logging.getLogger(__name__)
TIMEOUT = 2

mirrors_ubuntu = np.array(
    [
        "https://ubuntu.lafibre.info/ubuntu/",
        "https://mirror.ubuntu.ikoula.com/",
        "http://ubuntu.mirrors.ovh.net/ubuntu/",
        "http://miroir.univ-lorraine.fr/ubuntu/",
        "http://ubuntu.univ-nantes.fr/ubuntu/",
        "https://ftp.u-picardie.fr/mirror/ubuntu/ubuntu/",
        "http://ubuntu.univ-reims.fr/ubuntu/",
        "http://www-ftp.lip6.fr/pub/linux/distributions/Ubuntu/archive/",
    ]
)


def get_time(url):
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        return resp.elapsed.total_seconds()
    except:
        return np.inf


class MirrorBandit(Model):
    """
    Real environment for bandit problems.
    The reward is the response time for French servers meant to download ubuntu.

    On action i, gives a negative waiting time to reach url i in mirror_ubuntu.

    WARNING : if there is a timeout when querying the mirror, will result in
    a negative infinite reward.

    Parameters
    ----------

    url_ids : list of int or None,
        list of ids used to select a subset of the url list provided in the source.
        if None, all the urls are selected (i.e. 8 arms bandits).
    """

    name = "MirrorEnv"

    def __init__(self, url_ids=None, **kwargs):
        Model.__init__(self, **kwargs)
        if url_ids:
            self.url_list = mirrors_ubuntu[url_ids]
        else:
            self.url_list = mirrors_ubuntu

        self.n_arms = len(self.url_list)
        self.action_space = spaces.Discrete(self.n_arms)

    def step(self, action):
        """
        Sample the reward associated to the action.
        """
        # test that the action exists
        assert action < self.n_arms

        reward = -get_time(self.url_list[action])
        done = True
        return 0, reward, done, {}

    def reset(self):
        """
        Reset the environment to a default state.
        """
        return 0
