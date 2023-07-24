import pathlib
from rlberry.network import interface
from rlberry.manager import ExperimentManager
from rlberry import metadata_utils
import rlberry.utils.io
import base64


def execute_message(
    message: interface.Message, resources: interface.Resources
) -> interface.Message:
    response = interface.Message.create(command=interface.Command.ECHO)
    # LIST_RESOURCES
    if message.command == interface.Command.LIST_RESOURCES:
        info = {}
        for rr in resources:
            info[rr] = resources[rr]["description"]
        response = interface.Message.create(info=info)
    # AGENT_MANAGER_CREATE_INSTANCE
    elif message.command == interface.Command.AGENT_MANAGER_CREATE_INSTANCE:
        params = message.params
        base_dir = pathlib.Path(metadata_utils.RLBERRY_DEFAULT_DATA_DIR)
        if "output_dir" in params:
            params["output_dir"] = base_dir / "server_data" / params["output_dir"]
        else:
            params["output_dir"] = base_dir / "server_data/"
        experiment_manager = ExperimentManager(**params)
        filename = str(experiment_manager.save())
        response = interface.Message.create(
            info=dict(
                filename=filename,
                agent_name=experiment_manager.agent_name,
                output_dir=str(experiment_manager.output_dir).replace(
                    "server_data/", "client_data/"
                ),
            )
        )
        del experiment_manager
    # AGENT_MANAGER_FIT
    elif message.command == interface.Command.AGENT_MANAGER_FIT:
        filename = message.params["filename"]
        budget = message.params["budget"]
        extra_params = message.params["extra_params"]
        experiment_manager = ExperimentManager.load(filename)
        experiment_manager.fit(budget, **extra_params)
        experiment_manager.save()
        response = interface.Message.create(command=interface.Command.ECHO)
        del experiment_manager
    # AGENT_MANAGER_EVAL
    elif message.command == interface.Command.AGENT_MANAGER_EVAL:
        filename = message.params["filename"]
        experiment_manager = ExperimentManager.load(filename)
        eval_output = experiment_manager.eval_agents(message.params["n_simulations"])
        response = interface.Message.create(data=dict(output=eval_output))
        del experiment_manager
    # AGENT_MANAGER_CLEAR_OUTPUT_DIR
    elif message.command == interface.Command.AGENT_MANAGER_CLEAR_OUTPUT_DIR:
        filename = message.params["filename"]
        experiment_manager = ExperimentManager.load(filename)
        experiment_manager.clear_output_dir()
        response = interface.Message.create(
            message=f"Cleared output dir: {experiment_manager.output_dir}"
        )
        del experiment_manager
    # AGENT_MANAGER_CLEAR_HANDLERS
    elif message.command == interface.Command.AGENT_MANAGER_CLEAR_HANDLERS:
        filename = message.params["filename"]
        experiment_manager = ExperimentManager.load(filename)
        experiment_manager.clear_handlers()
        experiment_manager.save()
        response = interface.Message.create(message=f"Cleared handlers: {filename}")
        del experiment_manager
    # AGENT_MANAGER_SET_WRITER
    elif message.command == interface.Command.AGENT_MANAGER_SET_WRITER:
        filename = message.params["filename"]
        experiment_manager = ExperimentManager.load(filename)
        experiment_manager.set_writer(**message.params["kwargs"])
        experiment_manager.save()
        del experiment_manager
    # AGENT_MANAGER_OPTIMIZE_HYPERPARAMS
    elif message.command == interface.Command.AGENT_MANAGER_OPTIMIZE_HYPERPARAMS:
        filename = message.params["filename"]
        experiment_manager = ExperimentManager.load(filename)
        best_params_dict = experiment_manager.optimize_hyperparams(
            **message.params["kwargs"]
        )
        experiment_manager.save()
        del experiment_manager
        response = interface.Message.create(data=best_params_dict)
    # AGENT_MANAGER_GET_WRITER_DATA
    elif message.command == interface.Command.AGENT_MANAGER_GET_WRITER_DATA:
        # writer scalar data
        filename = message.params["filename"]
        experiment_manager = ExperimentManager.load(filename)
        writer_data = experiment_manager.get_writer_data()
        writer_data = writer_data or dict()
        for idx in writer_data:
            writer_data[idx] = writer_data[idx].to_csv(index=False)
        # tensoboard data
        tensorboard_bin_data = None
        if experiment_manager.tensorboard_dir is not None:
            tensorboard_zip_file = rlberry.utils.io.zipdir(
                experiment_manager.tensorboard_dir,
                experiment_manager.output_dir / "tensorboard_data.zip",
            )
            if tensorboard_zip_file is not None:
                tensorboard_bin_data = open(tensorboard_zip_file, "rb").read()
                tensorboard_bin_data = base64.b64encode(tensorboard_bin_data).decode(
                    "ascii"
                )
        response = interface.Message.create(
            data=dict(
                writer_data=writer_data, tensorboard_bin_data=tensorboard_bin_data
            )
        )
        del experiment_manager
    # end
    return response
