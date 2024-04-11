import requests
from tempfile import mkdtemp
import os
import shutil


# # TODO : convert external benchmark to DataFrame that match the input of rlberry.manager.comparaison.py -> compare_agents_data()
# # TODO : Download the external benchmark to a specific folder (or new rlberrygithub?), except if they are stable (huggingface/github)

# benchmark_list = {
#     "Google Atari bucket": "https://console.cloud.google.com/storage/brow",
#     "SB3 zoo": "https://github.com/DLR-RM/rl-baselines3-zoo/tree/master/logs/benchmark",
#     "cleanrl": "https://wandb.ai/openrlbenchmark/openrlbenchmark/reportlist",
# }


# def import_from_google_atari_bucket():
#     """import benchmark from Google Atari bucket

#     Parameters
#     -----------
#     x_vec : numpy.ndarray
#         numpy 1d array to be searched in the bins
#     bins : list
#         list of numpy 1d array, bins[d] = bins of the d-th dimension


#     Returns
#     --------
#     index (int) corresponding to the position of x in the partition
#     defined by the bins.
#     """
#     print("TODO")


# def import_from_cleanrl():
#     print("TODO")


# def import_from_hugingface():
#     print("TODO")


def download_benchmark_from_SB3_zoo(
    agent_name, environment_name, overwrite, output_dir=None
):
    """
    Download folder from pre-trained Reinforcement Learning agents using the rl-baselines3-zoo and Stable Baselines3.
    https://github.com/DLR-RM/rl-trained-agents

    Parameters
    -----------
    agent_name : str
        agent name for benchmark to download
    environment_name : list
        environment name for benchmark to download
    overwrite : bool
        how to manage if the combination agent_name/environment_name exist :
        True : delete the previous folder, then download
        False : raise an error
    output_dir : str
        root path where to download files. (default=None : create temp folder)

    Returns
    --------
    Return the path containing the downloaded files   (output_dir/agent_name/environment_name)
    """
    if not output_dir:
        output_dir = mkdtemp()

    GITHUB_URL = "https://raw.githubusercontent.com/DLR-RM/rl-trained-agents/master/"
    base_url = GITHUB_URL + agent_name + "/" + environment_name + "/"

    output_folder = os.path.join(output_dir, agent_name, environment_name)
    environment_base_name = environment_name.split("_")[0]

    if os.path.exists(output_folder):
        if not overwrite:
            raise FileExistsError(
                "The 'overwrite' bool is false, and the combination %s / %s already exist"
                % (agent_name, environment_name)
            )
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # download CSVs
    url_content = None
    i = 0
    while url_content != b"404: Not Found":
        file_name_to_download = str(i) + ".monitor.csv"
        url_csv_to_download = base_url + file_name_to_download

        req = requests.get(url_csv_to_download)
        url_content = req.content

        if url_content != b"404: Not Found":
            csv_file = open(os.path.join(output_folder, file_name_to_download), "wb")
            csv_file.write(url_content)
            csv_file.close()
        else:
            break
        i = i + 1

    # download zip
    file_name_to_download = environment_base_name + ".zip"
    url_zip_to_download = base_url + file_name_to_download
    req = requests.get(url_zip_to_download)
    url_content = req.content
    csv_file = open(os.path.join(output_folder, file_name_to_download), "wb")
    csv_file.write(url_content)
    csv_file.close()

    # download evaluations.npz
    file_name_to_download = "evaluations.npz"
    url_zip_to_download = base_url + file_name_to_download
    req = requests.get(url_zip_to_download)
    url_content = req.content
    csv_file = open(os.path.join(output_folder, file_name_to_download), "wb")
    csv_file.write(url_content)
    csv_file.close()

    # hyperparameter and config
    config_folder = output_folder + "/" + environment_base_name
    base_url_config = base_url + environment_base_name + "/"

    os.makedirs(config_folder)
    file_name_to_download = "args.yml"
    url_zip_to_download = base_url_config + file_name_to_download
    req = requests.get(url_zip_to_download)
    url_content = req.content
    csv_file = open(os.path.join(config_folder, file_name_to_download), "wb")
    csv_file.write(url_content)
    csv_file.close()

    file_name_to_download = "config.yml"
    url_zip_to_download = base_url_config + file_name_to_download
    req = requests.get(url_zip_to_download)
    url_content = req.content
    csv_file = open(os.path.join(config_folder, file_name_to_download), "wb")
    csv_file.write(url_content)
    csv_file.close()

    file_name_to_download = "vecnormalize.pkl"
    url_zip_to_download = base_url_config + file_name_to_download
    req = requests.get(url_zip_to_download)
    url_content = req.content
    csv_file = open(os.path.join(config_folder, file_name_to_download), "wb")
    csv_file.write(url_content)
    csv_file.close()

    return output_folder
