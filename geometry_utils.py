# coding: utf-8
"""Geometry utility functions."""

import numpy as np

def l2_distance(x_1, y_1, x_2, y_2):
    """Euclidean distance between 2 points"""
    return np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)


def make_circle(center_x, center_y, radius, res=30):
    """Returns a list of pairs representing coordinates of a circle.
       center_x, center_y, radius define the circle. The circle is
       approximated using a polygon with number of points = res."""
    points = []
    for i in range(res):
        ang = 2 * np.pi * i / res
        points.append((center_x + np.cos(ang) * radius, center_y + np.sin(ang) * radius))
    return points
        

def shift_then_scale_point(point, shift_x, shift_y, scale_x, scale_y):
    """Shift a point (a pair i.e. (x, y) coordinates) and then scale it."""
    (x, y) = point
    return ((shift_x + x) * scale_x, (shift_y + y) * scale_y)


def shift_then_scale_points(points, shift_x, shift_y, scale_x, scale_y):
    """Shift and THEN scale a list of points (each point is a tuple (x,y) coordinates)."""
    return [shift_then_scale_point(p, shift_x, shift_y, scale_x, scale_y) for p in points]
