import time


# This code (below) was copied from
# https://gist.github.com/pdashford/2e4bcd4fc2343e2fd03efe4da17f577d
# and modified by "leynier" to work with Python 3, type hints, correct format and
# simplified the code to our needs.


import base64
import getopt
import os
import shutil
import sys
from typing import Optional

from github import Github, GithubException
from github.ContentFile import ContentFile
from github.Repository import Repository


def get_sha_for_tag(repository: Repository, tag: str) -> str:
    """
    Returns a commit PyGithub object for the specified repository and tag.
    """
    branches = repository.get_branches()
    matched_branches = [match for match in branches if match.name == tag]
    if matched_branches:
        return matched_branches[0].commit.sha

    tags = repository.get_tags()
    matched_tags = [match for match in tags if match.name == tag]
    if not matched_tags:
        raise ValueError("No Tag or Branch exists with that name")
    return matched_tags[0].commit.sha


def download_directory(
    repository: Repository,
    sha: str,
    server_path: str,
    download_path: str = "download_benchmark",
) -> None:
    """
    Download all contents at server_path with commit tag sha in
    the repository.
    """

    full_download_path = os.path.join(download_path, server_path)

    if os.path.exists(full_download_path):
        shutil.rmtree(full_download_path)

    os.makedirs(full_download_path)
    contents = repository.get_dir_contents(server_path, ref=sha)

    for content in contents:
        print("Processing %s" % content.path)
        if content.type == "dir":
            os.makedirs(os.path.join(download_path, content.path))
            download_directory(repository, sha, content.path)
        else:
            try:
                path = content.path
                time.sleep(2)
                file_content = repository.get_contents(path, ref=sha)
                if not isinstance(file_content, ContentFile):
                    raise ValueError("Expected ContentFile")
                file_out = open(os.path.join(download_path, content.path), "w+")
                if file_content.content:
                    file_data = base64.b64decode(file_content.content)
                    file_out.write(file_data.decode("utf-8"))
                file_out.close()
            except (GithubException, IOError, ValueError) as exc:
                print("Error processing %s: %s", content.path, exc)


def usage():
    """
    Prints the usage command lines
    """
    print("usage: gh-download --repo=repo --branch=branch --folder=folder")


def main(argv):
    """
    Main function block
    """
    try:
        opts, _ = getopt.getopt(argv, "r:b:f:", ["repo=", "branch=", "folder="])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(2)
    repo: Optional[str] = None
    branch: Optional[str] = None
    folder: Optional[str] = None
    for opt, arg in opts:
        if opt in ("-r", "--repo"):
            repo = arg
        elif opt in ("-b", "--branch"):
            branch = arg
        elif opt in ("-f", "--folder"):
            folder = arg

    if not repo:
        print("Repo is required")
        usage()
        sys.exit(2)
    if not branch:
        print("Branch is required")
        usage()
        sys.exit(2)
    if not folder:
        print("Folder is required")
        usage()
        sys.exit(2)


# End of code copied code from
# https://gist.github.com/pdashford/2e4bcd4fc2343e2fd03efe4da17f577d
# and modified by "leynier" to work with Python 3, type hints, correct format and
# simplified the code to our needs.


# TODO : convert external benchmark to DataFrame that match the input of rlberry.manager.comparaison.py -> compare_agents_data()
# TODO : Download the external benchmark to a specific folder (or new rlberrygithub?), except if they are stable (huggingface/github)

benchmark_list = {
    "Google Atari bucket": "https://console.cloud.google.com/storage/brow",
    "SB3 zoo": "https://github.com/DLR-RM/rl-baselines3-zoo/tree/master/logs/benchmark",
    "cleanrl": "https://wandb.ai/openrlbenchmark/openrlbenchmark/reportlist",
}


def import_from_google_atari_bucket():
    """import benchmark from Google Atari bucket

    Parameters
    -----------
    x_vec : numpy.ndarray
        numpy 1d array to be searched in the bins
    bins : list
        list of numpy 1d array, bins[d] = bins of the d-th dimension


    Returns
    --------
    index (int) corresponding to the position of x in the partition
    defined by the bins.
    """
    print("TODO")


def download_benchmark_from_SB3_zoo(
    agent_name, environment_name, download_path="download_benchmark"
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

    Returns
    --------
    TODO

    """
    github = Github(None)
    repository = github.get_repo("DLR-RM/rl-trained-agents")
    sha = get_sha_for_tag(repository, "master")
    download_directory(
        repository, sha, str(agent_name + "/" + environment_name), download_path
    )


def import_from_cleanrl():
    print("TODO")


def import_from_hugingface():
    print("TODO")


if __name__ == "__main__":
    """
    Entry point
    """
    download_benchmark_from_SB3_zoo("ppo", "Acrobot-v1_1")
