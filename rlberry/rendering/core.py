"""
Provide classes for geometric primitives in OpenGL and scenes.
"""


class Scene:
    """
    Class representing a scene, which is a vector of GeometricPrimitive objects
    """

    def __init__(self):
        self.shapes = []

    def add_shape(self, shape):
        self.shapes.append(shape)


class GeometricPrimitive:
    """
    Class representing an OpenGL geometric primitive.

     Primitive type (GL_LINE_LOOP by defaut)

     If using OpenGLRender2D, one of the following:
           POINTS
           LINES
           LINE_STRIP
           LINE_LOOP
           POLYGON
           TRIANGLES
           TRIANGLE_STRIP
           TRIANGLE_FAN
           QUADS
           QUAD_STRIP

    If using PyGameRender2D:
            POLYGON


    TODO: Add support to more pygame shapes,
    see https://www.pygame.org/docs/ref/draw.html
    """

    def __init__(self, primitive_type="GL_LINE_LOOP"):
        # primitive type
        self.type = primitive_type
        # color in RGB
        self.color = (0.25, 0.25, 0.25)
        # list of vertices. each vertex is a tuple with coordinates in space
        self.vertices = []

    def add_vertex(self, vertex):
        self.vertices.append(vertex)

    def set_color(self, color):
        self.color = color
