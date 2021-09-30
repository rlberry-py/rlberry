from rlberry.network import interface
from rlberry.manager import AgentManager


def execute_message(
        message: interface.Message,
        resources: interface.Resources) -> interface.Message:
    response = interface.Message.create(command=interface.Command.ECHO)
    # LIST_RESOURCES
    if message.command == interface.Command.LIST_RESOURCES:
        info = {}
        for rr in resources:
            info[rr] = resources[rr]['description']
        response = interface.Message.create(info=info)
    # AGENT_MANAGER_CREATE_INSTANCE
    elif message.command == interface.Command.AGENT_MANAGER_CREATE_INSTANCE:
        params = message.params
        if 'output_dir' in params:
            params['output_dir'] = 'client_data' / params['output_dir']
        else:
            params['output_dir'] = 'client_data/'
        agent_manager = AgentManager(**params)
        filename = str(agent_manager.save())
        response = interface.Message.create(
            info=dict(
                filename=filename,
                agent_name=agent_manager.agent_name,
                output_dir=str(agent_manager.output_dir).replace('client_data/', 'remote_data/')
            )
        )
        del agent_manager
    # AGENT_MANAGER_FIT
    elif message.command == interface.Command.AGENT_MANAGER_FIT:
        filename = message.params['filename']
        agent_manager = AgentManager.load(filename)
        agent_manager.fit()
        agent_manager.save()
        response = interface.Message.create(command=interface.Command.ECHO)
        del agent_manager
    # AGENT_MANAGER_EVAL
    elif message.command == interface.Command.AGENT_MANAGER_EVAL:
        filename = message.params['filename']
        agent_manager = AgentManager.load(filename)
        eval_output = agent_manager.eval_agents(message.params['n_simulations'])
        response = interface.Message.create(data=dict(output=eval_output))
        del agent_manager
    # AGENT_MANAGER_CLEAR_OUTPUT_DIR
    elif message.command == interface.Command.AGENT_MANAGER_CLEAR_OUTPUT_DIR:
        filename = message.params['filename']
        agent_manager = AgentManager.load(filename)
        agent_manager.clear_output_dir()
        response = interface.Message.create(message=f'Cleared output dir: {agent_manager.output_dir}')
        del agent_manager
    # AGENT_MANAGER_CLEAR_HANDLERS
    elif message.command == interface.Command.AGENT_MANAGER_CLEAR_HANDLERS:
        filename = message.params['filename']
        agent_manager = AgentManager.load(filename)
        agent_manager.clear_handlers()
        agent_manager.save()
        response = interface.Message.create(message=f'Cleared handlers: {filename}')
        del agent_manager
    # AGENT_MANAGER_SET_WRITER
    elif message.command == interface.Command.AGENT_MANAGER_SET_WRITER:
        filename = message.params['filename']
        agent_manager = AgentManager.load(filename)
        agent_manager.set_writer(**message.params['kwargs'])
        agent_manager.save()
        del agent_manager
    # AGENT_MANAGER_OPTIMIZE_HYPERPARAMS
    elif message.command == interface.Command.AGENT_MANAGER_OPTIMIZE_HYPERPARAMS:
        filename = message.params['filename']
        agent_manager = AgentManager.load(filename)
        best_params_dict = agent_manager.optimize_hyperparams(**message.params['kwargs'])
        agent_manager.save()
        del agent_manager
        response = interface.Message.create(data=best_params_dict)
    # AGENT_MANAGER_GET_WRITER_DATA
    elif message.command == interface.Command.AGENT_MANAGER_GET_WRITER_DATA:
        filename = message.params['filename']
        agent_manager = AgentManager.load(filename)
        writer_data = agent_manager.get_writer_data()
        writer_data = writer_data or dict()
        for idx in writer_data:
            writer_data[idx] = writer_data[idx].to_csv(index=False)
        del agent_manager
        response = interface.Message.create(data=writer_data)
    # end
    return response
