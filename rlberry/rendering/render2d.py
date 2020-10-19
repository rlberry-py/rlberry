"""
OpenGL code for 2D rendering, using pygame.
"""

from rlberry.rendering import GeometricPrimitive, Scene

from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

_IMPORT_SUCESSFUL = True
try:
    import pygame as pg
    from pygame.locals import *

    from OpenGL.GL import *
    from OpenGL.GLU import *
except:
    _IMPORT_SUCESSFUL = False
    

class Render2D:
    def __init__(self):
        # parameters
        self.window_width  = 640
        self.window_height = 640
        self.background_color = (0.6, 0.75, 1.0)
        self.refresh_interval = 50
        self.window_name   = "rlxp render"
        self.clipping_area = (-1.0, 1.0, -1.0, 1.0)

        # time counter
        self.time_count = 0

        # background scene
        self.background = Scene()
        # data to be rendered (list of scenes)
        self.data = []

    def set_window_name(self, name):
        self.window_name = name

    def set_refresh_interval(self, interval):
        self.refresh_interval = interval

    def set_clipping_area(self, area):
        """
        The clipping area is tuple with elements (left, right, bottom, top)
        Default = (-1.0, 1.0, -1.0, 1.0)
        """
        self.clipping_area = area
        base_size = max(self.window_width, self.window_height)
        width_range  = area[1] - area[0]
        height_range = area[3] - area[2]
        base_range = max(width_range, height_range)
        width_range  /= base_range
        height_range /= base_range
        self.window_width  = int( base_size*width_range  )
        self.window_height = int( base_size*height_range )

    def set_data(self, data):
        self.data = data

    def set_background(self, background):
        self.background = background

    def initGL(self):
        """
        initialize GL
        """
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(self.clipping_area[0], self.clipping_area[1],
                   self.clipping_area[2], self.clipping_area[3])

    def display(self):
        """
        Callback function, handler for window re-paint
        """
        # Set background color (clear background)
        glClearColor(self.background_color[0], self.background_color[1], self.background_color[2], 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # Display background
        for shape in self.background.shapes:
            self.draw_geometric2d(shape)

        # Display objects
        if len(self.data) > 0:
            idx = self.time_count % len(self.data)
            for shape in self.data[idx].shapes:
                self.draw_geometric2d(shape)

        self.time_count += 1
        glFlush()

    def draw_geometric2d(self, shape):
        """
        Draw a 2D shape, of type GeometricPrimitive
        """
        if   shape.type == "GL_POINTS":
            glBegin(GL_POINTS)
        elif shape.type == "GL_LINES":
            glBegin(GL_LINES)
        elif shape.type == "GL_LINE_STRIP":
            glBegin(GL_LINE_STRIP)
        elif shape.type == "GL_LINE_LOOP":
            glBegin(GL_LINE_LOOP)
        elif shape.type == "GL_POLYGON":
            glBegin(GL_POLYGON)
        elif shape.type == "GL_TRIANGLES":
            glBegin(GL_TRIANGLES)
        elif shape.type == "GL_TRIANGLE_STRIP":
            glBegin(GL_TRIANGLE_STRIP)
        elif shape.type == "GL_TRIANGLE_FAN":
            glBegin(GL_TRIANGLE_FAN)
        elif shape.type == "GL_QUADS":
            glBegin(GL_QUADS)
        elif shape.type == "GL_QUAD_STRIP":
            glBegin(GL_QUAD_STRIP)
        else:
            print("Invalid type for geometric primitive!")
            raise NameError

        # set color
        glColor3f(shape.color[0], shape.color[1], shape.color[2])

        # create vertices
        for vertex in shape.vertices:
            glVertex2f(vertex[0], vertex[1])
        glEnd()


    def run_graphics(self):
        global _IMPORT_SUCESSFUL

        if _IMPORT_SUCESSFUL:
            pg.init()
            display = (self.window_width, self.window_height)
            pg.display.set_mode(display, DOUBLEBUF|OPENGL)
            pg.display.set_caption(self.window_name)
            self.initGL()
            while True:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        pg.quit()
                        return
                #
                self.display()
                #
                pg.display.flip()
                pg.time.wait(self.refresh_interval)
        else:
            print("Error: not possible to render the environment, pygame or pyopengl not installed.")

if __name__=='__main__':
    background = Scene()
    shape = GeometricPrimitive("GL_QUADS")
    shape.add_vertex((0.0, 0.0))
    shape.add_vertex((0.0, 1.0))
    shape.add_vertex((1.0, 1.0))
    shape.add_vertex((1.0, 0.0))
    shape.set_color((1.0, 0.0, 0.0))
    background.add_shape(shape)

    render = Render2D()
    render.set_background(background)
    render.run_graphics()
    render.run_graphics()
