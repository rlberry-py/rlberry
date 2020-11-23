import numpy as np
from rlberry.rendering import GeometricPrimitive


def bar_shape(p0, p1, width):
    shape = GeometricPrimitive("QUADS")

    x0, y0 = p0
    x1, y1 = p1

    direction = np.array([x1-x0, y1-y0])
    norm = np.sqrt((direction*direction).sum())
    direction = direction/norm

    # get vector perpendicular to direction
    u_vec = np.zeros(2)
    u_vec[0] = -direction[1]
    u_vec[1] = direction[0]

    u_vec = u_vec*width/2

    shape.add_vertex((x0+u_vec[0], y0+u_vec[1]))
    shape.add_vertex((x0-u_vec[0], y0-u_vec[1]))
    shape.add_vertex((x1-u_vec[0], y1-u_vec[1]))
    shape.add_vertex((x1+u_vec[0], y1+u_vec[1]))
    return shape


def circle_shape(center, radius, n_points=50):
    shape = GeometricPrimitive("POLYGON")

    x0, y0 = center
    theta = np.linspace(0.0, 2*np.pi, n_points)
    for tt in theta:
        xx = radius*np.cos(tt)
        yy = radius*np.sin(tt)
        shape.add_vertex((x0+xx, y0+yy))

    return shape
