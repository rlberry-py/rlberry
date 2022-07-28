# Copyright (c) 2021 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import sys
import curses
import rlberry

logger = rlberry.logger


VALUE_LENGTH = 103


def initialize_colors():
    """initialize colors"""
    curses.start_color()
    curses.use_default_colors()
    for index in range(0, curses.COLORS):
        curses.init_pair(index, index, -1)


def initialize_counter(offsets, screen_layout):
    """initialize _counter_ category

    '_counter_': {
        0: {
            '_count': 0,
            '_modulus_count': 0
        },
        1: {
            '_count': 0,
            '_modulus_count': 0
        }
    }
    """
    for offset in range(0, offsets):
        screen_layout["_counter_"][offset] = {}
        screen_layout["_counter_"][offset]["_count"] = 0
        if "modulus" in screen_layout["_counter_"]:
            screen_layout["_counter_"][offset]["_modulus_count"] = 0


def initialize_text(offsets, category, screen_layout, screen):
    """initialize screen for categories containing text"""
    category_data = screen_layout[category]
    if category_data.get("table"):
        for offset in range(0, offsets):
            screen.addstr(
                get_category_y_pos(category, offset, screen_layout),
                get_category_x_pos(category, offset, screen_layout),
                category_data["text"],
                curses.color_pair(category_data["text_color"]),
            )
    else:
        screen.addstr(
            category_data["position"][0],
            category_data["position"][1],
            category_data["text"],
            curses.color_pair(category_data["text_color"]),
        )


def initialize_keep_count(category, offsets, screen_layout):
    """initialize category keep_count

    per process:
        'category1': {
            0: {
                '_count': 0
            },
            1: {
                '_count': 0
            }
        }
    per execution:
        'category1' : {
            '_count': 0
        }
    """
    if screen_layout[category].get("table"):
        for offset in range(0, offsets):
            screen_layout[category][offset] = {}
            screen_layout[category][offset]["_count"] = 0
    else:
        screen_layout[category]["_count"] = 0


def update_screen_status(
    screen, state, config, running=None, queued=None, completed=None, data=None
):
    """update screen status"""
    height, width = screen.getmaxyx()

    color = config["color"]

    if state == "initialize":
        text = config["title"]
        screen.addstr(0, 0, " " * (width - 1), curses.color_pair(color))
        screen.addstr(0, width - len(text) - 1, text, curses.color_pair(color))
    elif state == "finalize":
        text = "[Press q to exit]"
        screen.addstr(0, 1, text, curses.color_pair(color))
    elif state == "get-process-data":
        if data:
            text = (
                f"{data.splitlines()[0].strip().capitalize()}... this may take awhile"
            )
            y_pos = int((height // 2) - 2)
            x_pos = int((width // 2) - (len(text) // 2) - len(text) % 2)
            screen.addstr(y_pos, x_pos, text, curses.color_pair(color))
        else:
            y_pos = int((height // 2) - 2)
            x_pos = 0
            screen.move(y_pos, x_pos)
            screen.clrtoeol()

    if state in ("initialize", "process-update"):
        if config.get("show_process_status"):
            if running is None:
                running = 0
            if queued is None:
                queued = 0
            if completed is None:
                completed = 0
            zfill = config["zfill"]
            rtext = f"  Running: {str(running).zfill(zfill)}"
            screen.addstr(height - 4, 1, rtext, curses.color_pair(color))
            qtext = f"   Queued: {str(queued).zfill(zfill)}"
            screen.addstr(height - 3, 1, qtext, curses.color_pair(color))
            ctext = f"Completed: {str(completed).zfill(zfill)}"
            screen.addstr(height - 2, 1, ctext, curses.color_pair(color))

    screen.refresh()


def initialize_screen(screen, screen_layout):
    """initialize screen"""
    logger.debug("initializing screen")

    # Make separations between columns
    process_cols = [
        screen_layout[category]["position"][1]
        for category in screen_layout
        if category[-8:] == "messages"
    ]
    y = screen_layout["Process1"]["position"][0]
    for col in process_cols[1:]:
        screen.vline(y, col - 1, "|", screen.getmaxyx()[0] - y)
    screen.refresh()

    set_screen_defaults(screen_layout)
    validate_screen_size(screen, screen_layout)

    process_cols = [
        screen_layout[category]["position"][1]
        for category in screen_layout
        if category[-8:] == "messages"
    ]
    y = screen_layout["Process1"]["position"][0]
    for col in process_cols[1:]:
        screen.vline(y, col - 1, "|", screen.getmaxyx()[0] - y)
    initialize_colors()
    curses.curs_set(0)
    update_screen_status(screen, "initialize", screen_layout["_screen"])


def initialize_screen_offsets(screen, screen_layout, offsets, processes_to_start):
    """initialize screen offsets"""
    logger.debug("initializing screen offsets")

    set_screen_defaults_processes(offsets, processes_to_start, screen_layout)
    validate_screen_layout_processes(offsets, screen_layout)

    for category, data in screen_layout.items():
        if category == "_counter_":
            initialize_counter(offsets, screen_layout)
        if data.get("text"):
            initialize_text(offsets, category, screen_layout, screen)
        if data.get("list") and not data.get("keep_count"):
            # list requires keep_count to be set
            data["keep_count"] = True
        if data.get("keep_count"):
            initialize_keep_count(category, offsets, screen_layout)

    update_screen_status(screen, "process-update", screen_layout["_screen"])


def get_category_values(message, offset, screen_layout, maxy, screen):
    """return list of tuples consisting of categories and their values from screen layout that match message"""
    category_values = []
    for category, data in screen_layout.items():
        regex = data.get("regex")
        if regex:

            match = re.match(regex, message)
            if match:

                value = None
                if match.groups():
                    value = match.group("value")
                    length = len(value)
                    max_length = data.get("length", VALUE_LENGTH)

                    if length > max_length:
                        value = f"{value[0:max_length - 3]}..."

                    if data.get("right_justify"):
                        spaces = " " * (max_length - length)
                        value = f"{spaces}{value}"

                original_value = value
                if screen_layout[category].get("keep_count"):
                    value = get_category_count(
                        category, offset, screen_layout, maxy, screen
                    )

                if screen_layout[category].get("replace_text"):
                    value = screen_layout[category]["replace_text"]

                if screen_layout[category].get("list"):
                    value = original_value

                category_values.append((category, value))

    return category_values


def sanitize_message(message):
    """return tuple consisting of offset and message"""
    regex = r"#(?P<offset>\d+)-.*"
    match = re.match(regex, message)
    if match:
        offset = match.group("offset")
        filtered_message = re.sub(rf"#{offset}-", "", message)
        return int(offset), filtered_message
    return 0, message


def get_position(text):
    """return position where count should start after text"""
    if ":" in text:
        return text.index(":") + 1
    elif text == len(text) * "-":
        return -1
    return len(text) + 1


def process_clear(category, y_pos, x_pos, screen_layout, screen):
    """process clear directive"""
    if screen_layout[category].get("clear"):
        if screen_layout[category].get("table"):
            orientation = screen_layout.get("table", {}).get(
                "orientation", "wrap_around"
            )
            if orientation == "horizontal":
                padding = screen_layout["table"]["padding"]
                if "padding" in screen_layout[category]:
                    padding = screen_layout[category]["padding"]
                value = " " * padding
                screen.addstr(y_pos, x_pos, value)
                return
        screen.move(y_pos, x_pos)
        screen.clrtoeol()


def process_counter(offset, category, value, screen_layout, screen):
    """process counter directive"""
    if "_counter_" not in screen_layout:
        return

    if category in screen_layout["_counter_"]["categories"]:
        position = screen_layout["_counter_"]["position"]
        x_pos = position[1] + screen_layout["_counter_"][offset]["_count"]
        y_pos = position[0] + offset
        counter_value = screen_layout["_counter_"]["counter_text"]
        color = screen_layout[category]["color"]
        screen_layout["_counter_"][offset]["_count"] += 1
        if "modulus" in screen_layout["_counter_"]:
            if (
                screen_layout["_counter_"][offset]["_count"]
                % screen_layout["_counter_"]["modulus"]
                == 0
            ):
                # increments the progress bar
                x_pos = (
                    position[1] + screen_layout["_counter_"][offset]["_modulus_count"]
                )
                color = screen_layout["_counter_"]["color"]
                screen_layout["_counter_"][offset]["_modulus_count"] += 1
                x_pos = x_pos + 1 if "regex" in screen_layout["_counter_"] else x_pos
                screen.addstr(y_pos, x_pos, counter_value, curses.color_pair(color))

        else:
            # increments the counter
            if screen_layout["_counter_"].get("width"):
                width = screen_layout["_counter_"]["width"]
                count = screen_layout["_counter_"][offset]["_count"]
                if count % width == 0:
                    screen_layout["_counter_"]["position"] = (
                        position[0] + 1,
                        position[1],
                    )
                    screen_layout["_counter_"][offset]["_count"] = 0
            screen.addstr(y_pos, x_pos, counter_value, curses.color_pair(color))

    elif category == "_counter_":
        # regex infers progress bar
        # this sets up the progress bar boundary
        position = screen_layout["_counter_"]["position"]
        color = screen_layout[category]["color"]
        span = int(value) / screen_layout["_counter_"]["modulus"]
        span_text = " " * int(span)
        progress_value = f"[{span_text}]"
        screen.addstr(
            position[0] + offset, position[1], progress_value, curses.color_pair(color)
        )


def get_category_color(category, message, screen_layout):
    """return color for category in screen layout"""
    color = screen_layout[category].get("color", 0)
    for effect in screen_layout[category].get("effects", []):
        match = re.match(effect["regex"], message)
        if match:
            color = effect["color"]
            break
    return color


def clear_columns(y, screen, screen_layout, maxy):
    line = y
    while line < maxy:
        screen.move(line, 1)
        screen.clrtoeol()
        line += 1

    for category in screen_layout:
        if category[-8:] == "messages":
            screen_layout[category]["_count"] = 1


def get_category_count(category, offset, screen_layout, maxy, screen):
    """return count for category in screen layout"""
    if screen_layout[category].get("table"):
        screen_layout[category][offset]["_count"] += 1
        return str(screen_layout[category][offset]["_count"])

    if (
        screen_layout[category]["_count"] + screen_layout[category]["position"][0]
        > maxy - 2
    ):
        y = screen_layout[category]["position"][0]
        clear_columns(y, screen, screen_layout, maxy)

    screen_layout[category]["_count"] += 1

    return str(screen_layout[category]["_count"])


def get_category_x_pos(category, offset, screen_layout):
    """return x pos for category in screen layout"""
    x_pos = screen_layout[category]["position"][1]
    if screen_layout[category].get("text", ""):
        x_pos = x_pos + get_position(screen_layout[category]["text"]) + 1
    if screen_layout[category].get("table"):
        if screen_layout.get("table"):
            orientation = screen_layout["table"].get("orientation", "wrap_around")
            if orientation == "wrap_around":
                rows = screen_layout["table"]["rows"]
                width = screen_layout["table"]["width"]
                if offset >= rows:
                    x_pos += int(offset / rows) * width
            else:
                # orientation is horizontal
                padding = screen_layout["table"]["padding"]
                if "padding" in screen_layout[category]:
                    # padding is overriden if specified in category
                    padding = screen_layout[category].get("padding")
                x_pos += offset * padding
            # logger.debug(f'table offset {offset} x_pos is {x_pos}')
    return x_pos


def get_category_y_pos(category, offset, screen_layout):
    """return y pos for category in screen layout"""
    y_pos = screen_layout[category]["position"][0]
    # table and list are mutually exclusive - can't be set at the same time for a given category
    # list must include keep_count
    if screen_layout[category].get("table"):
        y_pos += offset
        if screen_layout.get("table"):
            orientation = screen_layout["table"].get("orientation", "wrap_around")
            if orientation == "wrap_around":
                rows = screen_layout["table"]["rows"]
                if offset >= rows:
                    y_pos -= int(offset / rows) * rows
            else:
                # orientation is horizontal
                y_pos -= offset
            # logger.debug(f'table offset {offset} y_pos is {y_pos}')
    elif screen_layout[category].get("list"):
        y_pos += screen_layout[category]["_count"]
    return y_pos


def update_screen(message, screen, screen_layout):
    """update screen with message as dictated by screen layout

    gets list of categories from screen layout that match the message (via regex)
    iterates through the matching categories and executes display as dictated by
    the category
    """
    offset, sanitized_message = sanitize_message(message)
    maxy, maxx = screen.getmaxyx()
    category_values = get_category_values(
        sanitized_message, offset, screen_layout, maxy, screen
    )
    sanitized_message = sanitized_message.strip()
    if len(sanitized_message) > 0:
        try:
            for (category, value) in category_values:
                y_pos = get_category_y_pos(category, offset, screen_layout)
                x_pos = get_category_x_pos(category, offset, screen_layout)
                color = get_category_color(category, sanitized_message, screen_layout)
                process_clear(category, y_pos, x_pos, screen_layout, screen)
                screen.addstr(y_pos, x_pos, value, curses.color_pair(color))
                process_counter(offset, category, value, screen_layout, screen)

                screen.refresh()

        except Exception as exception:  # curses.error as exception:
            logger.error(f"error occurred when updating screen: {exception}")


def get_table_position(screen_layout):
    """return first position of table encountered within screen layout"""
    for _, data in screen_layout.items():
        if data.get("table"):
            return data["position"]
    return None


def get_positions_to_update(screen_layout, table_position, delta):
    """return dict of items representing categories and updated positions within screen layout"""
    positions = {}
    for category, data in screen_layout.items():
        (y_pos, x_pos) = data.get("position", (0, 0))
        if y_pos > table_position:
            positions[category] = (y_pos - delta, x_pos)
    return positions


def update_positions(screen_layout, positions):
    """update positions in screen layout"""
    for category, position in positions.items():
        screen_layout[category]["position"] = position


def squash_table(screen_layout, delta):
    """squash table"""
    logger.debug(f"squashing table by {delta} positions")
    table_position = get_table_position(screen_layout)
    positions = get_positions_to_update(screen_layout, table_position[0], delta)
    logger.debug(f"the following positions will be updated:\n{positions}")
    update_positions(screen_layout, positions)


def set_screen_defaults(screen_layout):
    """set screen defaults"""
    logger.debug("setting screen defaults")

    if "_screen" not in screen_layout:
        screen_layout["_screen"] = {}

    if "title" not in screen_layout["_screen"]:
        screen_layout["_screen"]["title"] = sys.argv[0]

    if "color" not in screen_layout["_screen"]:
        screen_layout["_screen"]["color"] = 11

    if "blink" not in screen_layout["_screen"]:
        screen_layout["_screen"]["blink"] = True


def set_screen_defaults_processes(processes, processes_to_start, screen_layout):
    """set screen defaults"""
    logger.debug("setting screen defaults processes")

    if "zfill" not in screen_layout["_screen"]:
        screen_layout["_screen"]["zfill"] = len(str(processes))

    if "show_process_status" not in screen_layout["_screen"]:
        screen_layout["_screen"]["show_process_status"] = processes_to_start < processes


def validate_screen_layout_processes(processes, screen_layout):
    """validate screen layout"""
    logger.debug("validating screen layout processes")

    table = screen_layout.get("table")
    if not table:
        return

    orientation = table.get("orientation", "wrap_around")
    if orientation == "wrap_around":
        entries = table.get("rows", 0) * table.get("cols", 0)
        if processes > entries:
            raise Exception(
                f"table definition of {entries} entries not sufficient for {processes} processes"
            )

        if table.get("squash"):
            rows = table.get("rows", 0)
            if processes < rows:
                squash_table(screen_layout, rows - processes)


def validate_screen_size(screen, screen_layout):
    """validate current screen size is large enough for screen layout"""
    logger.debug("validating screen size")

    screen_height, screen_width = screen.getmaxyx()
    max_y_pos = 0
    max_x_pos = 0
    for _, data in screen_layout.items():
        position = data.get("position")
        if position:
            y_pos = position[0]
            x_pos = position[1]
            if x_pos > max_x_pos:
                max_x_pos = x_pos
            if y_pos > max_y_pos:
                max_y_pos = y_pos

    if max_y_pos > screen_height:
        raise Exception(
            "the screen is not large enough for the configured layout - make the screen taller"
        )

    if max_x_pos > screen_width:
        raise Exception(
            "the screen is not large enough for the configured layout - make the screen wider"
        )
