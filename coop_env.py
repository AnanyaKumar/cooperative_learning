# coding: utf-8
"""Define the cooperative game environment."""

from gym import Env, spaces
from gym.envs.registration import register
import numpy as np
from gym.envs.classic_control import rendering
from car import Car
from obstacle import Obstacle
import geometry_utils

class CoopEnv(Env):
    """Implement the cooperative game environment described in our proposal."""

    metadata = {'render.modes': ['human']}

    def _setup_simple_lane(self, car_radius=0.1, num_cars_y=1, road_length=3):
        """Setup a simple lane, the cars want to start from the left and go to the right"""
        # TODO: add randomly generated obstacle(s) that don't intersect and are within the lane.
        # TODO: make a more complex environment, where we have a bunch of rows of cars.
        # y-coordinate of the bottom lane.
        self._bottom_lane = 0
        # y-coordinate of the top lane.
        self._top_lane = 1
        # List of cars.
        self._car_radius = car_radius
        cars_y = [float(i+1)/float(num_cars_y+1) for i in range(num_cars_y)]
        self._cars = [Car(0.0, y, car_radius) for y in cars_y]
        self._road_length = road_length

    def __init__(self, obstacles=[], num_cars_y=1, max_accel=0.1, max_velocity=0.5, max_steps=10, time_delta=1, time_gran=5):
        # TODO: add support for max velocity, and make sure cars don't go above this.
        self._obstacles = obstacles
        self._setup_simple_lane(num_cars_y=num_cars_y)
        self._max_accel = max_accel
        self._max_velocity = max_velocity
        self._max_steps = max_steps
        self._time_delta = time_delta
        self._time_gran = time_gran
        # Actions are the acceleration in the x direction and y direction for each car.
        self.action_space = spaces.Tuple((
            spaces.Box(low=-self._max_accel, high=self._max_accel, shape=(num_cars_y,)), # x acceleration
            spaces.Box(low=-self._max_accel, high=self._max_accel, shape=(num_cars_y,)))) # y acceleration
        # TODO: add observation space (do we really need this though?)
        self.viewer = None
        self._reset()

    def _reset(self):
        for c in self._cars:
            c.reset()
        self._num_steps = 0
        self._reward = 0.0
        return (list(self._cars), list(self._obstacles))

    def _kill_collided_cars(self):
        """Check collisions between cars and cars and obstacles, kill cars that collide with anything"""
        # TODO: Improve collision computation strategy if needed. Right now we just check if the cars
        # collide at the end of each time step, which is OK if the cars don't move too fast and our
        # time steps are small, so it's fine for a first pass. In reality, we can actually compute
        # collisions by solving a cubic equation (numpy probably does it pretty fast). Or there
        # are plenty of approximation schemes that run fast and are better.

        # list of cars to kill at the end of the loop
        kill_list = []
        for i in range(len(self._cars)):
            if not(self._cars[i].is_alive):
                continue
            collided = False
            # Check if car i collided with an obstacle
            for obs in self._obstacles:
                if self._cars[i].check_collision(obs):
                    collided = True
            # Check if car i collided with any car.
            for j in range(len(self._cars)):
                if i == j or not(self._cars[j].is_alive):
                    continue
                if self._cars[i].check_collision(self._cars[j]):
                    collided = True
            # Kill the car if it collided with another car or if it's outside lane boundary.
            if (collided or
                self._cars[i].pos_y < self._bottom_lane + self._car_radius or
                self._cars[i].pos_y > self._top_lane - self._car_radius):
                kill_list.append(self._cars[i])

        # It is important that we first decide what cars to kill, and then kill them. This
        # is because collisions are not transitive. E.g. suppose A collides with B, B collides
        # with C. If we kill B, then A and C aren't colliding. But we want to kill all of them.
        for car in kill_list:
            car.die()

    def _get_total_reward(self):
        """Get the total reward (from the start of the episode)"""
        return sum([c.get_reward() for c in self._cars])

    def _step(self, actions):
        """Execute the specified list of actions.
            action: numpy.array, shape=[num_cars, 2]
        """
        assert self.action_space.contains(actions)

        # Clip acceleration
        accel = np.clip(actions, a_min=-self._max_accel, a_max=self._max_accel)

        # Move cars.
        for j in range(self._time_gran):
            for i in range(len(self._cars)):
                # Only consider alive cars.
                if self._cars[i].is_alive:
                    self._cars[i].move(accel[0][i], accel[1][i], float(self._time_delta) / self._time_gran, self._max_velocity)

            # Check collisions.
            self._kill_collided_cars()

        # Compute rewards.
        new_reward = self._get_total_reward()
        step_reward = new_reward - self._reward
        self._reward = new_reward

        # Update number of steps.
        self._num_steps += 1

        # Check if we are in a terminal state.
        is_terminal = False
        if self._num_steps >= self._max_steps:
            is_terminal = True
        if sum([c.is_alive for c in self._cars]) == 0:
            is_terminal = True

        # No debug information right now.
        debug_info = None

        return (list(self._cars), list(self._obstacles)), step_reward, is_terminal, debug_info


    def _render(self, mode='human', close=False):
        screen_width = 1200
        screen_height = 400
        # Shift the coordinates so that we can see above the top lane, below the bottom lane,
        # left of the start position of the cars.
        shift_y = 0.005
        shift_x = 0.02
        # The scaling is defined in terms of the height, apply the same scaling so that the
        # rendering doesn't look weird.
        # TODO: make sure we can see all obstacles (and maybe all cars).
        scale_y = float(screen_height) / float(self._top_lane - self._bottom_lane + 2 * shift_y)
        scale_x = scale_y

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

        # Draw the obstacles:
        for obs in self._obstacles:
            circle = geometry_utils.make_circle(obs.pos_x, obs.pos_y, obs.radius)
            circle = geometry_utils.shift_then_scale_points(circle, shift_x, shift_y, scale_x, scale_y)
            self.viewer.add_onetime(rendering.FilledPolygon(circle))

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

    def get_max_accel(self):
        return self._max_accel

obstacle_list1 = [Obstacle(1.0,.5,0.1)]

register(
    id='coop1car1obs-v0',
    entry_point='coop_env:CoopEnv',
    kwargs={'num_cars_y': 1, 'obstacles': obstacle_list1})

register(
    id='coop2cars1obs-v0',
    entry_point='coop_env:CoopEnv',
    kwargs={'num_cars_y': 2, 'obstacles': obstacle_list1})

register(
    id='coop1car-v0',
    entry_point='coop_env:CoopEnv',
    kwargs={'num_cars_y': 1, 'obstacles': []})

register(
    id='coop4cars-v0',
    entry_point='coop_env:CoopEnv',
    kwargs={'num_cars_y': 4, 'obstacles': []})
