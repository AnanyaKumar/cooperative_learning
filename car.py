# coding: utf-8
"""Define the Car class."""

import numpy as np

class Car:
    def __init__(self, pos_x, pos_y):
        self.start_x = float(pos_x)
        self.start_y = float(pos_y)
        self.reset()
        
    def reset(self):
        self.pos_x = self.start_x
        self.pos_y = self.start_y
        self.max_x = self.start_x
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.is_alive = True

    def die(self):
        assert self.is_alive
        self.is_alive = False
        self.pos_x = self.start_x -np.random.randint(10**2, 10**3)
        self.pos_y = self.start_y -np.random.randint(10**2, 10**3)
        self.vel_x = 0.0
        self.vel_y = 0.0

    def move(self, acc_x, acc_y, time):
        assert self.is_alive
        self.pos_x = self.pos_x + self.vel_x * time + 0.5 * acc_x * (time ** 2)
        self.pos_y = self.pos_y + self.vel_y * time + 0.5 * acc_y * (time ** 2)
        self.vel_x = self.vel_x + acc_x * time
        self.vel_y = self.vel_y + acc_y * time
        self.max_x = max(self.pos_x, self.max_x)

    def get_reward(self):
        if self.is_alive:
            return self.max_x - self.start_x
        else:
            return self.max_x - self.start_x - 100

    def type(self):
        return "Car"
