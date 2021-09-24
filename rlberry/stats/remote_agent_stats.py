import pathlib
from typing import Any, Mapping
from rlberry.network import interface
from rlberry.network.client import BerryClient


class RemoteAgentStats:
    """
    Class to define a client that handles an AgentStats instance in a remote BerryServer.

    Parameters
    ----------
    client: BerryClient
        Client instance, to communicate with a BerryServer.
    **kwargs:
        Parameters for AgentStats instance.
        Some parameters (as agent_class, train_env, eval_env) can be defined using a ResourceRequest.
    """
    def __init__(
        self,
        client: BerryClient,
        **kwargs: Mapping[str, Any],
    ):
        self._client = client

        # Create a remote AgentStats object and keep reference to the filename
        # in the server where the object was saved.
        msg = self._client.send(
            interface.Message.create(
                command=interface.Command.CREATE_AGENT_STATS_INSTANCE,
                params=kwargs,
                data=None,
            )
        )
        self._remote_agent_stats_filename = pathlib.Path(
            msg.info['filename']
        )

    @property
    def remote_file(self):
        return str(self._remote_agent_stats_filename)

    def fit(self):
        msg = self._client.send(
            interface.Message.create(
                command=interface.Command.FIT_AGENT_STATS,
                params=dict(filename=self.remote_file),
                data=None,
            )
        )
        del msg

    def eval(self):
        msg = self._client.send(
            interface.Message.create(
                command=interface.Command.EVAL_AGENT_STATS,
                params=dict(filename=self.remote_file),
                data=None,
            )
        )
        out = msg.data['output']
        return out
