# coding: utf-8
"""Define the cooperative game environment."""

from gym import Env, spaces
from gym.envs.registration import register
import numpy as np
from gym.envs.classic_control import rendering
from car import Car
import geometry_utils

class CoopEnv(Env):
    """Implement the cooperative game environment described in our proposal.
    """

    metadata = {'render.modes': ['human']}

    def _setup_simple_lane(self, car_radius=1, y_gap=0.5, num_cars_y=1, road_length=10):
        """Setup a simple lane, the cars want to start from the left and go to the right"""
        # y-coordinate of the bottom lane
        self._bottom_lane = 0
        # y-coordinate of the top lane
        self._top_lane = 2 * num_cars_y * car_radius + (num_cars_y + 1) * y_gap
        # List of cars
        self._car_radius = car_radius
        cars_y = [(i+1) * y_gap + (2 * i + 1) * car_radius for i in range(num_cars_y)]
        self._cars = [Car(0.0, y) for y in cars_y]
        self._road_length = road_length

    def __init__(self, num_cars_y):
        self._setup_simple_lane(num_cars_y=num_cars_y)
        self._max_accel = 1
        self._max_steps = 1000
        self._time_delta = 0.05
        # Actions are the acceleration in the x direction and y direction for each car.
        self.action_space = spaces.Tuple((
            spaces.Box(low=-self._max_accel, high=self._max_accel, shape=(num_cars_y,)), # x acceleration
            spaces.Box(low=-self._max_accel, high=self._max_accel, shape=(num_cars_y,)))) # y acceleration
        self.viewer = None
        self._reset()

    def _reset(self):
        for c in self._cars:
            c.reset()
        self._num_steps = 0
        self._reward = 0.0

    def _step(self, actions):
        """Execute the specified list of actions.
        """
        # The first dimension is the car.
        assert self.action_space.contains(actions)

        # Move cars.
        for i in range(len(self._cars)):
            # Only consider alive cars.
            if self._cars[i].is_alive:
                self._cars[i].move(actions[0][i], actions[1][i], self._time_delta)

        # Check collisions.
        for i in range(len(self._cars)):
            # Skip dead cars.
            if not(self._cars[i].is_alive):
                continue
            # Check if car i collided with any car.
            collided = False
            for j in range(len(self._cars)):
                if i <= j or not(self._cars[j].is_alive):
                    continue
                if geometry_utils.l2_distance(self._cars[i], self._cars[j]) < 2 * self._car_radius:
                    collided = True
                    self._cars[j].die()
            # Kill the car if it collided with another car or if it's outside lane boundary.
            if (collided or
                self._cars[i].pos_y < self._bottom_lane + self._car_radius or
                self._cars[i].pos_y > self._top_lane - self._car_radius):
                self._cars[i].die()

        # Get rewards
        new_reward = sum([c.get_reward() for c in self._cars])
        step_reward = new_reward - self._reward
        self._reward = new_reward

        # Update number of steps.
        self._num_steps += 1

        # Check if we are in terminal state.
        is_terminal = False
        if self._num_steps >= self._max_steps:
            is_terminal = True
        if sum([c.is_alive for c in self._cars]) == 0:
            is_terminal = True

        # No debug information right now.
        debug_info = None

        return list(self._cars), step_reward, is_terminal, debug_info


    def _render(self, mode='human', close=False):
        # TODO: un-hardcode this. We want to make sure the cars look roughly circular, and the things that matter
        # fit on screen.
        screen_width = 1200
        screen_height = 400
        shift_y = 0.5
        shift_x = 1
        scale_y = float(screen_height) / float(self._top_lane - self._bottom_lane + 2 * shift_y)
        scale_x = float(screen_width) / float(self._road_length + 2 * shift_x)

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

        # Draw the car
        for c in self._cars:
            circle = geometry_utils.make_circle(c.pos_x, c.pos_y, self._car_radius)
            circle = geometry_utils.shift_then_scale_points(circle, shift_x, shift_y, scale_x, scale_y)
            self.viewer.add_onetime(rendering.PolyLine(circle, True))

        # Draw the lanes
        bottom_lane = [(-shift_x, self._bottom_lane), (self._road_length + shift_x, self._bottom_lane)]
        bottom_lane = geometry_utils.shift_then_scale_points(bottom_lane, shift_x, shift_y, scale_x, scale_y)
        self.viewer.add_onetime(rendering.PolyLine(bottom_lane, True))
        top_lane = [(-shift_x, self._top_lane), (self._road_length + shift_x, self._top_lane)]
        top_lane = geometry_utils.shift_then_scale_points(top_lane, shift_x, shift_y, scale_x, scale_y)
        self.viewer.add_onetime(rendering.PolyLine(top_lane, True))

        self.viewer.render()

    def _seed(self, seed=None):
        """Set the random seed.

        Parameters
        ----------
        seed: int or None
          Random seed used by numpy.random and random.
        """
        np.random.seed(seed)


register(
    id='coop-v0',
    entry_point='coop_env:CoopEnv',
    kwargs={'num_cars_y': 1})
