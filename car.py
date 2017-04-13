# coding: utf-8
"""Define the Car class."""

import numpy as np
import geometry_utils as geom

class Car:
    def __init__(self, pos_x, pos_y, radius):
        self.start_x = float(pos_x) # Starting x co-ordinate.
        self.start_y = float(pos_y) # Starting y co-ordinate.
        self.radius = radius
        self.reset()
        
    def reset(self):
        """Reset state of the car"""
        self.pos_x = self.start_x # Current x-coordinate.
        self.pos_y = self.start_y # Current y-coordinate.
        self.max_x = self.start_x # Max x-coordinate car has gone.
        self.vel_x = 0.0 # Velocity x-component.
        self.vel_y = 0.0 # Velocity y-component.
        self.is_alive = True # Is the car alive? It dies when it hits something.

    def die(self):
        """Ask the car to die, dead cars cannot move or die."""
        assert self.is_alive # Can't kill a car that's dead.
        self.is_alive = False
        # We send the car far away to the left so that it doesn't interfere with moving cars.
        # However, this shouldn't matter because we ignore dead cars.
        self.pos_x = self.start_x - np.random.randint(10**2, 10**3)
        self.pos_y = self.start_y
        self.vel_x = 0.0
        self.vel_y = 0.0

    def move(self, acc_x, acc_y, time):
        """Move the car for specified time with specified acceleration vector."""
        assert self.is_alive
        self.pos_x = self.pos_x + self.vel_x * time + 0.5 * acc_x * (time ** 2)
        self.pos_y = self.pos_y + self.vel_y * time + 0.5 * acc_y * (time ** 2)
        self.vel_x = self.vel_x + acc_x * time
        self.vel_y = self.vel_y + acc_y * time
        self.max_x = max(self.pos_x, self.max_x)

    def get_reward(self):
        """Return the total reward this car has received."""
        if self.is_alive:
            return self.max_x - self.start_x
        else:
            # TODO: Decide if we actually want to subtract 100. Maybe it's too high. Furthermore,
            # even without subtracting the cars should learn to make progress because colliding
            # with another car or hitting a road boundary kills the car and prevents it from
            # making future progress.
            return self.max_x - self.start_x

    def dist(self, obj):
        return geom.l2_distance(self.pos_x, self.pos_y, obj.pos_x, obj.pos_y)

    def check_collision(self, obj):
        return self.dist(obj) <= self.radius + obj.radius

    def type(self):
        """Get type of object, for error checking and validation"""
        return "Car"
