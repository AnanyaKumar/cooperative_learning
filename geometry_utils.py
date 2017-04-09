# coding: utf-8
"""Geometry utility functions."""

import numpy as np

def l2_distance(x_1, y_1, x_2, y_2):
    return np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)


def make_circle(pos_x, pos_y, radius, res=30, filled=False):
    points = []
    for i in range(res):
        ang = 2*np.pi*i / res
        points.append((pos_x + np.cos(ang)*radius, pos_y + np.sin(ang)*radius))
    return points
        

def shift_then_scale_point(point, shift_x, shift_y, scale_x, scale_y):
    (x, y) = point
    return ((shift_x + x) * scale_x, (shift_y + y) * scale_y)


def shift_then_scale_points(points, shift_x, shift_y, scale_x, scale_y):
    return [shift_then_scale_point(p, shift_x, shift_y, scale_x, scale_y) for p in points]
