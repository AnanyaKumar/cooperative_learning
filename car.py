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

    def move(self, acc_x, acc_y, time, max_vel):
        """Move the car for specified time with specified acceleration vector."""
        assert self.is_alive
        new_vel_x = self.vel_x + acc_x * time
        new_vel_y = self.vel_y + acc_y * time

        if new_vel_x < max_vel:
            self.pos_x = self.pos_x + self.vel_x * time + 0.5 * acc_x * (time ** 2)
            self.vel_x = new_vel_x
        else:
            dt = (max_vel - self.vel_x) / acc_x     # barring weird floating point issues, dt < time
            self.pos_x = self.pos_x + self.vel_x * dt + 0.5 * acc_x * (dt ** 2) + max_vel * (time - dt)
            self.vel_x = max_vel

        if new_vel_y < max_vel:
            self.pos_y = self.pos_y + self.vel_y * time + 0.5 * acc_y * (time ** 2)
            self.vel_y = new_vel_y
        else:
            dt = (max_vel - self.vel_y) / acc_y     # barring weird floating point issues, dt < time
            self.pos_y = self.pos_y + self.vel_y * dt + 0.5 * acc_y * (dt ** 2) + max_vel * (time - dt)
            self.vel_y = max_vel


        self.max_x = max(self.pos_x, self.max_x)

    def get_reward(self):
        """Return the total reward this car has received."""
        if self.is_alive:
            return self.max_x - self.start_x
        else:
            # TODO: Code seems to be very brittle on the penalty of dying.
            return self.max_x - self.start_x - 0.05

    def dist(self, obj):
        return geom.l2_distance(self.pos_x, self.pos_y, obj.pos_x, obj.pos_y)

    def check_collision(self, obj):
        return self.dist(obj) <= self.radius + obj.radius

    def type(self):
        """Get type of object, for error checking and validation"""
        return "Car"
