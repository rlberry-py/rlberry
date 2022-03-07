# This script needs requests
# pip install requests

from rlberry.envs.interface import Model
import rlberry.spaces as spaces
import logging
import subprocess
import numpy as np
import pandas as pd

import requests

logger = logging.getLogger(__name__)
TIMEOUT = 2

mirrorlist = "mirrorlist"
repo = "core"
arch = "x86_64"


def get_time(url):
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        return resp.elapsed.total_seconds()
    except:
        return np.inf


def process_url(url):
    url = url.replace("$repo", repo)
    url = url.replace("$arch", arch).strip()
    return url


class MirrorBandit(Model):
    """
    Real environment for bandit problems.

    WARNING : if there is a timeout when querying the mirror, will result in
    an infinite reward.
    """

    name = ""

    def __init__(self, n_url=5, **kwargs):
        Model.__init__(self, **kwargs)
        servers = pd.read_csv("mirrorlist", skiprows=6, sep="=", names=["type", "url"])
        servers["url"] = servers["url"].apply(process_url)
        self.url_list = servers["url"].values[:n_url]

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
