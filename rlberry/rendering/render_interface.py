"""
Interface that allows 2D rendering.
"""

import logging
from abc import ABC, abstractmethod

from rlberry.rendering.opengl_render2d import OpenGLRender2D
from rlberry.rendering.pygame_render2d import PyGameRender2D
from rlberry.rendering.utils import video_write

logger = logging.getLogger(__name__)


class RenderInterface(ABC):
    """
    Common interface for rendering in rlberry.
    """

    def __init__(self):
        self._rendering_enabled = False

    def is_render_enabled(self):
        return self._rendering_enabled

    def enable_rendering(self):
        self._rendering_enabled = True

    def disable_rendering(self):
        self._rendering_enabled = False

    def save_video(self, filename, **kwargs):
        """
        Save video file.
        """
        pass

    @abstractmethod
    def render(self, **kwargs):
        """
        Display on screen.
        """
        pass


class RenderInterface2D(RenderInterface):
    """
    Interface for 2D rendering in rlberry.
    """

    def __init__(self):
        RenderInterface.__init__(self)
        self._rendering_enabled = False
        self._rendering_type = "2d"
        self._state_history_for_rendering = []
        self._refresh_interval = 50   # in milliseconds
        self._clipping_area = (-1.0, 1.0, -1.0, 1.0)  # (left,right,bottom,top)

        # rendering type, either 'pygame' or 'opengl'
        self.renderer_type = 'opengl'

    def get_renderer(self):
        if self.renderer_type == 'opengl':
            return OpenGLRender2D()
        elif self.renderer_type == 'pygame':
            return PyGameRender2D()
        else:
            raise NotImplementedError("Unknown renderer type.")

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

    def set_refresh_interval(self, interval):
        self._refresh_interval = interval

    def clear_render_buffer(self):
        self._state_history_for_rendering = []

    def set_clipping_area(self, area):
        self._clipping_area = area

    def _get_background_and_scenes(self):
        # background
        background = self.get_background()

        # data: convert states to scenes
        scenes = []
        for state in self._state_history_for_rendering:
            scene = self.get_scene(state)
            scenes.append(scene)
        return background, scenes

    def render(self, loop=True, **kwargs):
        """
        Function to render an environment that implements the interface.
        """

        if self.is_render_enabled():
            # background and data
            background, data = self._get_background_and_scenes()

            if len(data) == 0:
                logger.info("No data to render.")
                return

            # render
            renderer = self.get_renderer()

            renderer.window_name = self.name
            renderer.set_refresh_interval(self._refresh_interval)
            renderer.set_clipping_area(self._clipping_area)
            renderer.set_data(data)
            renderer.set_background(background)
            renderer.run_graphics(loop)
            return 0
        else:
            logger.info("Rendering not enabled for the environment.")
            return 1

    def save_video(self, filename, framerate=25, **kwargs):

        # background and data
        background, data = self._get_background_and_scenes()

        if len(data) == 0:
            logger.info("No data to save.")
            return

        # get video data from renderer
        renderer = self.get_renderer()
        renderer.window_name = self.name
        renderer.set_refresh_interval(self._refresh_interval)
        renderer.set_clipping_area(self._clipping_area)
        renderer.set_data(data)
        renderer.set_background(background)

        video_data = renderer.get_video_data()
        video_write(filename, video_data, framerate=framerate)
