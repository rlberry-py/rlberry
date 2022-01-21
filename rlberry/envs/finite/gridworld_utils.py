import numpy as np

WALL_SYMBOL = "#"
REWARD_TERMINAL_SYMBOL = "r"
REWARD_SYMBOL = "R"
TERMINAL_STATE_SYMBOL = "T"
INITIAL_STATE_SYMBOL = "I"


# spaces are ignored
DEFAULT_LAYOUT = """
IOOOO # OOOOO  O OOOOR
OOOOO # OOOOO  # OOOOO
OOOOO O OOOOO  # OOOOO
OOOOO # OOOOO  # OOOOO
IOOOO # OOOOO  # OOOOr
"""


def _preprocess_layout(layout):
    layout = layout.replace(" ", "")  # remove spaces
    # remove first and last line breaks
    if layout[0] == "\n":
        layout = layout[1:]
    if layout[-1] == "\n":
        layout = layout[:-1]

    # make sure all lines have the same length
    lines = layout.split("\n")
    len_lines = [len(line) for line in lines]
    max_len = np.max(len_lines)
    # below, also reverse lines (so that render is not inversed in the y-direction)
    adjusted_lines = [
        line.ljust(max_len, "O") for line in reversed(lines)
    ]  # fill with empty state
    layout = "\n".join(adjusted_lines)
    return layout


def get_layout_info(layout):
    layout = _preprocess_layout(layout)
    lines = layout.split("\n")
    nrows = len(lines)
    ncols = len(lines[0])
    walls = []
    initial_states = []
    terminal_states = []
    reward_at = dict()
    for rr in range(nrows):
        line = lines[rr]
        for cc in range(ncols):
            symbol = line[cc]
            state_coord = (rr, cc)
            if symbol == WALL_SYMBOL:
                walls.append(state_coord)
            if symbol == TERMINAL_STATE_SYMBOL or symbol == REWARD_TERMINAL_SYMBOL:
                terminal_states.append(state_coord)
            if symbol == REWARD_SYMBOL or symbol == REWARD_TERMINAL_SYMBOL:
                reward_at[state_coord] = 1.0
            if symbol == INITIAL_STATE_SYMBOL:
                initial_states.append(state_coord)
    info = dict(
        nrows=nrows,
        ncols=ncols,
        initial_states=tuple(initial_states),
        terminal_states=tuple(terminal_states),
        walls=tuple(walls),
        reward_at=reward_at,
    )
    return info
