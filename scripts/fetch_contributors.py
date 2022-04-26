"""
Partially copied from https://github.com/scikit-learn/scikit-learn/blob/main/build_tools/generate_authors_table.py
This script generates an html table of contributors, with names and avatars.
The table should be updated for each new inclusion in the teams.
Generating the table requires admin rights.
"""
import requests
import time
from pathlib import Path
from os import path


LOGO_URL = "https://avatars.githubusercontent.com/u/72948299?v=4"
REPO_FOLDER = Path(path.abspath(__file__)).parent.parent


MEMBERS = [
    "sauxpa",
    "TimotheeMathieu",
    "omardrwch",
    "xuedong",
    "eleurent",
    "yfletberliac",
    "mmcenta",
    "menardprr",
    "riccardodv",
    "AleShi94",
]


def get(url):
    for sleep_time in [10, 30, 0]:
        reply = requests.get(url)
        api_limit = (
            "message" in reply.json()
            and "API rate limit exceeded" in reply.json()["message"]
        )
        if not api_limit:
            break
        print("API rate limit exceeded, waiting..")
        time.sleep(sleep_time)

    reply.raise_for_status()
    return reply


def get_contributors():
    """Get the list of contributor profiles. Require admin rights."""

    members = set(MEMBERS)

    # remove CI bots
    members -= {"dependabot"}

    # get profiles from GitHub
    members = [get_profile(login) for login in members]

    # sort by last name
    members = sorted(members, key=key)

    return members


def get_profile(login):
    """Get the GitHub profile from login"""
    print("get profile for %s" % (login,))
    try:
        profile = get("https://api.github.com/users/%s" % login).json()
    except requests.exceptions.HTTPError:
        return dict(name=login, avatar_url=LOGO_URL, html_url="")

    if profile["name"] is None:
        profile["name"] = profile["login"]

    return profile


def key(profile):
    """Get a sorting key based on the lower case last name, then firstname"""
    components = profile["name"].lower().split(" ")
    return " ".join([components[-1]] + components[:-1])


def generate_table(contributors):
    lines = [
        ".. raw :: html\n",
        "    <!-- Generated by fetrch_contributors.py -->",
        '    <div class="sk-authors-container">',
        "    <style>",
        "      img.avatar {border-radius: 10px;}",
        "    </style>",
    ]
    for contributor in contributors:
        lines.append("    <div>")
        lines.append(
            "    <a href='%s'><img src='%s' class='avatar' /></a> <br />"
            % (contributor["html_url"], contributor["avatar_url"])
        )
        lines.append("    <p>%s</p>" % (contributor["name"],))
        lines.append("    </div>")
    lines.append("    </div>")
    return "\n".join(lines)


def generate_list(contributors):
    lines = []
    for contributor in contributors:
        lines.append("- %s" % (contributor["name"],))
    return "\n".join(lines)


if __name__ == "__main__":

    members = get_contributors()

    with open(REPO_FOLDER / "docs" / "contributors.rst", "w+") as rst_file:
        rst_file.write(generate_table(members))
