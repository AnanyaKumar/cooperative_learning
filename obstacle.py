# coding: utf-8
"""Define the Obstacle class."""

import numpy as np
import geometry_utils as geom

class Obstacle:
    def __init__(self, pos_x, pos_y, radius):
        self.pos_x = float(pos_x)
        self.pos_y = float(pos_y)
        self.radius = float(radius)

    def check_collision(self, car):
        return geom.l2_distance(self.pos_x, self.pos_y, car.pos_x, car.pos_y) <= self.radius + car.radius

    def type(self):
        """Get type of object, for error checking and validation"""
        return "Obstacle"
