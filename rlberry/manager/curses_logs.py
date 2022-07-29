import sys
import re
import curses
import numpy as np

import rlberry

from .screen import update_screen
from .screen import (
    clear_columns,
    initialize_keep_count,
    initialize_text,
)

from queue import Queue
from multiprocessing.managers import SyncManager

logger = rlberry.logger


Y_PROCESS = 8


def get_screen_layout(n_fit, maxyx):
    y, x = maxyx
    n_cols = min(n_fit, 3)
    size_col = x // n_cols

    screen_layout = {
        "_screen": {"title": "rlberry_logs", "color": 256},
        "total": {
            "position": (1, 4),
            "text": "n_fit: 0",
            "text_color": 6,
            "color": 1,
            "regex": r"^msg:total fits:" + "(?P<value>\d+)$",
        },
        "finished": {
            "position": (2, 4),
            "text": "fits finished: -",
            "text_color": 6,
            "color": 1,
            "keep_count": True,
            "regex": r"^msg:Process.* finished",
        },
        "errors": {
            "position": (1, 45),
            "text": "Other messages: -",
            "text_color": 6,
            "clear": True,
            "regex": r"^(?!msg:Process).*" + "(?P<value>.*)$",
        },
    }
    for id_n in range(n_cols):
        screen_layout["Process" + str(id_n)] = {
            "position": (Y_PROCESS, 2 + id_n * size_col),
            "text": "Process " + str(id_n),
            "text_color": 6,
        }

        screen_layout["Process" + str(id_n) + "_messages"] = {
            "position": (Y_PROCESS, 2 + id_n * size_col),
            "list": True,
            "keep_count": True,
            "color": 0,
            "regex": r"^msg:Process" + str(id_n) + " " + "(?P<value>.*)$",
            "_count": 0,
        }
    screen_layout["progress"] = {
        "position": (4, 4),
        "text": "Progress:[" + (" " * 40) + "]",
        "text_color": 6,
    }
    screen_layout["desc"] = {
        "position": (6, 1),
        "text": "Showing the first three processes:",
        "text_color": 0,
    }
    screen_layout["progress_tracker"] = {
        "position": (4, 13),
        "text": "",
        "keep_bount": True,
        "regex": r"^msg:Process:progress(?P<value>.*)$",
        "_count": 0,
        "color": 2,
    }
    screen_layout["_counter_"] = {
        "position": (4, 14),
        "categories": ["progress_tracker"],
        "counter_text": "|",
        "width": 40,
        "color": 2,
    }

    return screen_layout


MAX_SIZE_HEADER = 15
N_cols = 3


def make_headers(dict_message, screen, screen_layout, n_fit):
    headers = []
    for key in dict_message.keys():
        if key == "max_global_step":
            key = "step"
        if len(key) > MAX_SIZE_HEADER:
            headers.append(key[:MAX_SIZE_HEADER])
        else:
            headers.append(key)

    headers = tuple(headers)
    headers_msg = ("%12s  " * len(headers)) % headers
    beginning_message = "msg:Process"
    for process in range(min(N_cols, n_fit)):
        update_screen(
            beginning_message + str(process) + headers_msg, screen, screen_layout
        )
    return headers


def listener_process(queue, screen, screen_layout, n_fit, fit_budget):
    """Listener process is a target for a multiprocess process
    that runs and listens to a queue for logging events.

    Arguments:
        queue (multiprocessing.manager.Queue): queue to monitor
        configurer (func): configures loggers
        log_name (str): name of the log to use

    Returns:
        None
    """
    update_screen("msg:total fits:" + str(n_fit), screen, screen_layout)
    print("test")
    # get headers
    formatter = logger.handlers[0]
    headers_set = False
    beginning_message = "msg:Process"

    global_steps = np.zeros(n_fit)
    steps_per_update = n_fit * fit_budget / 40
    n_updates = 1

    in_progress = True

    process_columns = {
        str(id_p): False for id_p in range(N_cols)
    }  # Whether the processes in the columns are finished

    while in_progress:
        log = queue.get()
        if log is not None:
            formatter = logger.handlers[0]
            record = formatter.format(log)
            process, dict_message = _parse_log(record)
            if process is not None:
                if not headers_set:
                    headers = make_headers(dict_message, screen, screen_layout, n_fit)
                    headers_set = True
                # update progress bar
                process_num = int(process[len(beginning_message) :])
                global_steps[process_num] = int(dict_message["max_global_step"])
                if (
                    np.sum(global_steps) - n_updates * steps_per_update
                ) > steps_per_update:
                    n_updates += 1
                    update_screen("msg:Process:progress", screen, screen_layout)

                # format message
                message = process + ("%12s  " * len(headers)) % tuple(
                    [str(dict_message[key]) for key in dict_message]
                )

            else:
                message = dict_message["message"]

                # Test to see if finished process
                is_finished = re.match(r".*msg:Process.* finished", message)
                if is_finished:
                    process = re.search(r"(?<=Process)\w+\s+", message).group(0).strip()
                    if process in process_columns.keys():
                        process_columns[process] = True

                # Test if this is a new process beginning and there is space
                begins = re.match(r".*msg:Process.* started", message)
                is_available = [process_columns[k] for k in process_columns.keys()]
                if begins and np.any(is_available):
                    maxy, maxx = screen.getmaxyx()
                    n_cols = min(n_fit, 3)
                    size_col = maxx // n_cols

                    begins_id = (
                        re.search(r"(?<=Process)\w+\s+", message).group(0).strip()
                    )
                    vacant_col = np.arange(N_cols)[is_available][0]
                    key_vacant_col = list(process_columns.keys())[vacant_col]

                    x_process = screen_layout["Process" + str(key_vacant_col)][
                        "position"
                    ][1]

                    del screen_layout["Process" + str(key_vacant_col)]
                    del screen_layout["Process" + str(key_vacant_col) + "_messages"]

                    screen_layout["Process" + begins_id] = {
                        "position": (Y_PROCESS, x_process),
                        "text": "Process " + begins_id,
                        "text_color": 6,
                    }

                    screen_layout["Process" + begins_id + "_messages"] = {
                        "position": (Y_PROCESS, x_process),
                        "list": True,
                        "keep_count": True,
                        "color": 0,
                        "regex": r"^msg:Process" + begins_id + " " + "(?P<value>.*)$",
                        "_count": 1,
                    }
                    # initialize_screen(screen, screen_layout)
                    initialize_text(
                        n_cols, "Process" + begins_id, screen_layout, screen
                    )
                    initialize_keep_count(
                        "Process" + begins_id + "_messages", n_cols, screen_layout
                    )
                    clear_columns(Y_PROCESS + 2, screen, screen_layout, maxy)

                    del process_columns[key_vacant_col]
                    process_columns[begins_id] = True

            update_screen(message, screen, screen_layout)
        else:
            in_progress = False


class MyQueue(Queue):
    def get_attribute(self, name):
        return getattr(self, name)


class MyManager(SyncManager):
    pass


def get_manager():
    MyManager.register("Queue", MyQueue)
    m = MyManager()
    m.start()
    return m


def _parse_log(record):
    is_process_msg = re.match(r".*max_global_step.*", record)
    # print(record)
    record = record.strip()
    beginning_message = "msg:Process"
    if is_process_msg:

        process = beginning_message + re.search(
            r"(?<=Process)\w+\s+", record.strip()
        ).group(0)

        categories = record.strip().split("|")

        dict_message = {
            cat.split("=")[0].strip(): cat.split("=")[1].strip()
            for cat in categories[1:-1]  # the first and last are styling strings
        }

    else:
        process = None
        if record[: len(beginning_message)] == beginning_message:
            record = record[: len(beginning_message)]
        dict_message = {"message": record}

    return process, dict_message


# redirect stdout to logger
# https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message != "\n":
            self.level(message)

    def flush(self):
        pass


sys.stdout = LoggerWriter(logger.info)


# curses wrapper to have a better handle of the exit process.
def curses_wrapper(func, *args, **kwds):
    """Wrapper function that initializes curses and calls another function,
    restoring normal keyboard/screen behavior on error.
    The callable object 'func' is then passed the main window 'stdscr'
    as its first argument, followed by any other arguments passed to
    wrapper().
    """

    try:
        # Initialize curses
        stdscr = curses.initscr()

        # Turn off echoing of keys, and enter cbreak mode,
        # where no buffering is performed on keyboard input
        curses.noecho()
        curses.cbreak()

        # In keypad mode, escape sequences for special keys
        # (like the cursor keys) will be interpreted and
        # a special value like curses.KEY_LEFT will be returned
        stdscr.keypad(1)

        # Start color, too.  Harmless if the terminal doesn't have
        # color; user can test with has_color() later on.  The try/catch
        # works around a minor bit of over-conscientiousness in the curses
        # module -- the error return from C start_color() is ignorable.
        try:
            curses.start_color()
        except:
            pass

        return func(stdscr, *args, **kwds)

    finally:
        stdscr.addstr(
            0, stdscr.getmaxyx()[1] // 2 - 10, "Press q to quit !", curses.color_pair(1)
        )
        stdscr.refresh()
        while True:

            if stdscr.getch() == ord("q"):
                break

        if "stdscr" in locals():
            stdscr.keypad(0)
            curses.echo()
            curses.nocbreak()
            curses.endwin()  #  don't flush window to keep all messages.
