"""
Interface that allows 2D rendering.
"""

from abc import ABC, abstractmethod
from rlberry.rendering.render2d import Render2D

class RenderInterface2D(ABC):
    def __init__(self):
        self._rendering_enabled           = False
        self._rendering_type              = "2d"
        self._state_history_for_rendering = []
        self._refresh_interval            = 50   # in milliseconds
        self._clipping_area               = (-1.0, 1.0, -1.0, 1.0)

    @abstractmethod
    def get_scene(self, state):
        """
        Return scene (list of shapes) representing a given state
        """
        pass

    @abstractmethod
    def get_background(self):
        """
        Returne a scene (list of shapes) representing the background
        """
        pass

    def append_state_for_rendering(self, state):
        self._state_history_for_rendering.append(state)

    def is_render_enabled(self):
        return self._rendering_enabled

    def enable_rendering(self):
        self._rendering_enabled = True

    def disable_rendering(self):
        self._rendering_enabled = False

    def set_refresh_interval(self, interval):
        self._refresh_interval = interval

    def clear_render_buffer(self):
        self._state_history_for_rendering = []

    def set_clipping_area(self, area):
        self._clipping_area = area

    def render(self):
        """
        Function to render an environment that implements the interface.
        """
        if self.is_render_enabled():
            # background
            background = self.get_background()

            # data: convert states to scenes
            data = []
            for state in self._state_history_for_rendering:
                scene = self.get_scene(state)
                data.append(scene)

            # render
            renderer = Render2D()
            renderer.window_name = self.id
            renderer.set_refresh_interval(self._refresh_interval)
            renderer.set_clipping_area(self._clipping_area)
            renderer.set_data(data)
            renderer.set_background(background)
            renderer.run_graphics()
            return 0
        else:
            print("Rendering not enabled for the environment.")
            return 1
