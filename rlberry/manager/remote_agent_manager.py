import pathlib
from typing import Any, Mapping
from rlberry.network import interface
from rlberry.network.client import BerryClient


class RemoteAgentManager:
    """
    Class to define a client that handles an AgentManager instance in a remote BerryServer.

    Parameters
    ----------
    client: BerryClient
        Client instance, to communicate with a BerryServer.
    **kwargs:
        Parameters for AgentManager instance.
        Some parameters (as agent_class, train_env, eval_env) can be defined using a ResourceRequest.
    """
    def __init__(
        self,
        client: BerryClient,
        **kwargs: Mapping[str, Any],
    ):
        self._client = client

        # Create a remote AgentManager object and keep reference to the filename
        # in the server where the object was saved.
        msg = self._client.send(
            interface.Message.create(
                command=interface.Command.AGENT_MANAGER_CREATE_INSTANCE,
                params=kwargs,
                data=None,
            )
        )
        self._remote_agent_manager_filename = pathlib.Path(
            msg.info['filename']
        )

    @property
    def remote_file(self):
        return str(self._remote_agent_manager_filename)

    def fit(self):
        msg = self._client.send(
            interface.Message.create(
                command=interface.Command.AGENT_MANAGER_FIT,
                params=dict(filename=self.remote_file),
                data=None,
            )
        )
        if msg.command == interface.Command.RAISE_EXCEPTION:
            raise Exception(msg.message)

    def eval_agents(self):
        msg = self._client.send(
            interface.Message.create(
                command=interface.Command.AGENT_MANAGER_EVAL,
                params=dict(filename=self.remote_file),
                data=None,
            )
        )
        if msg.command == interface.Command.RAISE_EXCEPTION:
            raise Exception(msg.message)
        out = msg.data['output']
        return out

    def clear_output_dir(self):
        msg = self._client.send(
            interface.Message.create(
                command=interface.Command.AGENT_MANAGER_CLEAR_OUTPUT_DIR,
                params=dict(filename=self.remote_file),
                data=None,
            )
        )
        if msg.command == interface.Command.RAISE_EXCEPTION:
            raise Exception(msg.message)

    def clear_handlers(self):
        msg = self._client.send(
            interface.Message.create(
                command=interface.Command.AGENT_MANAGER_CLEAR_HANDLERS,
                params=dict(filename=self.remote_file),
                data=None,
            )
        )
        if msg.command == interface.Command.RAISE_EXCEPTION:
            raise Exception(msg.message)

    def set_writer(self, idx, writer_fn, writer_kwargs=None):
        """Note: Use ResourceRequest for writer_fn."""
        params = dict(
            idx=idx,
            writer_fn=writer_fn,
            writer_kwargs=writer_kwargs
        )
        msg = self._client.send(
            interface.Message.create(
                command=interface.Command.AGENT_MANAGER_SET_WRITER,
                params=dict(filename=self.remote_file, kwargs=params),
                data=None,
            )
        )
        if msg.command == interface.Command.RAISE_EXCEPTION:
            raise Exception(msg.message)

    def optimize_hyperparams(self, **kwargs):
        msg = self._client.send(
            interface.Message.create(
                command=interface.Command.AGENT_MANAGER_OPTIMIZE_HYPERPARAMS,
                params=dict(filename=self.remote_file, kwargs=kwargs),
                data=None,
            )
        )
        if msg.command == interface.Command.RAISE_EXCEPTION:
            raise Exception(msg.message)
        best_params_dict = msg.data
        return best_params_dict
