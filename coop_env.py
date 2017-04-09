# coding: utf-8
"""Define the cooperative game environment."""

from gym import Env, spaces
from gym.envs.registration import register
import numpy as np

class Car:
    def __init__(self, pos_x, pos_y):
        self.start_x = float(pos_x)
        self.start_y = float(pos_y)
        self.reset()
        
    def reset(self):
        self.pos_x = self.start_x
        self.pos_y = self.start_y
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

    def type(self):
        return "Car"


def l2_distance(x_1, y_1, x_2, y_2):
    return np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)


class CoopEnv(Env):
    """Implement the cooperative game environment described in our proposal.
    """

    metadata = {'render.modes': ['human']}

    def _setup_simple_lane(self, car_radius=1, y_gap=0.5, num_cars_y=1, road_length=100):
        """Setup a simple lane, the cars want to start from the left and go to the right"""
        # y-coordinate of the bottom lane
        self._bottom_lane = 0
        # y-coordinate of the top lane
        self._top_lane = 2 * num_cars_y * car_radius + (num_cars_y + 1) * y_gap
        # List of cars
        self._car_radius = car_radius
        cars_x = [(i+1) * y_gap + (2 * i + 1) * car_radius for i in range(num_cars_y)]
        self._cars = [Car(x, 0.0) for x in cars_x]

    def __init__(self, num_cars_y):
        self._setup_simple_lane(num_cars_y=num_cars_y)
        self._max_accel = 1
        self._max_steps = 100
        self._time_delta = 0.05
        # Actions are the acceleration in the x direction and y direction for each car.
        self.action_space = spaces.Tuple((
            spaces.Box(low=-self._max_accel, high=self._max_accel, shape=(num_cars_y,)), # x acceleration
            spaces.Box(low=-self._max_accel, high=self._max_accel, shape=(num_cars_y,)))) # y acceleration
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
        print actions

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
                if i == j or not(self._cars[j].is_alive):
                    continue
                if l2_distance(self._cars[i], self._cars[j]) < 2 * self._car_radius:
                    collided = True
                    self._cars[j].die()
            # Kill the car if it collided with another car.
            if collided:
                self._cars[i].die()

        # Get rewards
        new_reward = sum([c.pos_x - c.start_x for c in self._cars])
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
        pass

    def _seed(self, seed=None):
        """Set the random seed.

        Parameters
        ----------
        seed: int or None
          Random seed used by numpy.random and random.
        """
        np.random.seed(seed)

    def query_model(self, state, action):
        """Return the possible transition outcomes for a state-action pair.

        This should be in the same format at the provided environments
        in section 2.

        Parameters
        ----------
        state
          State used in query. Should be in the same format at
          the states returned by reset and step.
        action: int
          The action used in query.

        Returns
        -------
        [(prob, nextstate, reward, is_terminal), ...]
          List of possible outcomes
        """
        pass

register(
    id='coop-v0',
    entry_point='coop_env:CoopEnv',
    kwargs={'num_cars_y': 1})
