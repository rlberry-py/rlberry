import base64
import dill
import io
import logging
import pandas as pd
import pathlib
import pickle
import zipfile
from typing import Any, Mapping, Optional
from rlberry.network import interface
from rlberry.network.client import BerryClient


logger = logging.getLogger(__name__)


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
        if client:
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
            if msg.command == interface.Command.RAISE_EXCEPTION:
                raise Exception(msg.message)

            self._remote_agent_manager_filename = pathlib.Path(
                msg.info['filename']
            )

            # get useful attributes
            self.agent_name = msg.info['agent_name']
            self.output_dir = pathlib.Path(msg.info['output_dir'])  # to save locally

    def set_client(self, client: BerryClient):
        self._client = client

    @property
    def remote_file(self):
        return str(self._remote_agent_manager_filename)

    def get_writer_data(self):
        """
        * Calls get_writer_data() in the remote AgentManager and returns the result locally.
        * If tensorboard data is available in the remote AgentManager, the data is zipped,
        received locally and unzipped.
        """
        msg = self._client.send(
            interface.Message.create(
                command=interface.Command.AGENT_MANAGER_GET_WRITER_DATA,
                params=dict(filename=self.remote_file),
            )
        )
        if msg.command == interface.Command.RAISE_EXCEPTION:
            raise Exception(msg.message)
        raw_data = msg.data['writer_data']
        writer_data = dict()
        for idx in raw_data:
            csv_content = raw_data[idx]
            writer_data[idx] = pd.read_csv(io.StringIO(csv_content), sep=',')

        # check if tensorboard data was received
        # If so, read file and unzip it.
        tensorboard_bin_data = msg.data['tensorboard_bin_data']
        if tensorboard_bin_data is not None:
            tensorboard_bin_data = base64.b64decode(tensorboard_bin_data.encode('ascii'))
            zip_file = open(self.output_dir / 'tensorboard_data.zip', "wb")
            zip_file.write(tensorboard_bin_data)
            zip_file.close()
            with zipfile.ZipFile(self.output_dir / 'tensorboard_data.zip', 'r') as zip_ref:
                zip_ref.extractall(self.output_dir)
        return writer_data

    def fit(self, budget=None, **kwargs):
        msg = self._client.send(
            interface.Message.create(
                command=interface.Command.AGENT_MANAGER_FIT,
                params=dict(
                    filename=self.remote_file,
                    budget=budget,
                    extra_params=kwargs),
                data=None,
            )
        )
        if msg.command == interface.Command.RAISE_EXCEPTION:
            raise Exception(msg.message)

    def eval_agents(self, n_simulations: Optional[int] = None):
        msg = self._client.send(
            interface.Message.create(
                command=interface.Command.AGENT_MANAGER_EVAL,
                params=dict(
                    filename=self.remote_file,
                    n_simulations=n_simulations),
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

    def save(self):
        """
        Save RemoteAgentManager data to self.output_dir.

        Returns
        -------
        filename where the AgentManager object was saved.
        """
        # use self.output_dir
        output_dir = self.output_dir

        # create dir if it does not exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # save
        filename = pathlib.Path('remote_manager_obj').with_suffix('.pickle')
        filename = output_dir / filename
        filename.parent.mkdir(parents=True, exist_ok=True)
        try:
            with filename.open("wb") as ff:
                pickle.dump(self.__dict__, ff)
            logger.info("Saved RemoteAgentManager({}) using pickle.".format(self.agent_name))
        except Exception:
            try:
                with filename.open("wb") as ff:
                    dill.dump(self.__dict__, ff)
                logger.info("Saved RemoteAgentManager({}) using dill.".format(self.agent_name))
            except Exception as ex:
                logger.warning("[RemoteAgentManager] Instance cannot be pickled: " + str(ex))

        return filename

    @classmethod
    def load(cls, filename):
        filename = pathlib.Path(filename).with_suffix('.pickle')

        obj = cls(None)
        try:
            with filename.open('rb') as ff:
                tmp_dict = pickle.load(ff)
            logger.info("Loaded RemoteAgentManager using pickle.")
        except Exception:
            with filename.open('rb') as ff:
                tmp_dict = dill.load(ff)
            logger.info("Loaded RemoteAgentManager using dill.")

        obj.__dict__.clear()
        obj.__dict__.update(tmp_dict)
        return obj
