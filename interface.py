# coding: utf-8
"""Methods useful for processing the state."""

import numpy as np

def build_nn_input(car_list, k):
	"""Takes a list of car objects and returns neural net state information for each car.

	Args:
		car_list: list of at least k+1 car objects.
		k: Each car decides what action to take based on its information and the information of
			the closest k cars.

	Returns:
		A numpy array states, where states[i] is the state for the ith car. len(states) is the
			number of cars. states[i] is an array of size 4*k+4. The last 4 floats are the ith
			car's x position, y position, x velocity, y velocity. The first 4k floats are 4 floats
			for each neighboring car. We keep the relative position and velocity for each of the k
			nearest cars.
	"""
	assert k < len(car_list)
	nn_input = []
	for this_car in car_list:
		sorted_car_list = sorted([(this_car.dist(car), car) for car in car_list], key=(lambda x:x[0]))
		# Remove the first car, as it should be itself
		sorted_car_list = [x[1] for x in sorted_car_list[1:k+1]]
		param_list = [(car.pos_x - this_car.pos_x, car.pos_y - this_car.pos_y,
			car.vel_x - this_car.vel_x, car.vel_y - this_car.vel_y) for car in sorted_car_list]
		param_list.append((this_car.pos_x, this_car.pos_y, this_car.vel_x, this_car.vel_y))
		nn_input.append(np.array(param_list).flatten())
	return np.array(nn_input)

def clip_output(controls, max_accel):
	controls_x, controls_y = controls
	return (np.clip(controls_x, -max_accel, max_accel), np.clip(controls_y, -max_accel, max_accel))

def build_nn_output(normal_list, std_x=1, std_y=1):
	"""Returns controls corresponding to neural net output means (using normal distribution).

	Args:
		normal_list: numpy array containing [mean_x, mean_y] 2d np arrays specifying mean of control values.
		max_accel: all controls should be between -max_accel and max_accel.
   	std_x: standard deviation for the x control.
   	std_y: standard deviation for the y control.

  Returns:
  	(lx, ly): lx is a numpy array of controls in x direction, ly is a numpy array of controls
  		in the y direction. Each control is just a float.
	"""
	assert(type(normal_list) == np.ndarray)
	assert(type(normal_list[0]) == np.ndarray)
	assert(len(normal_list[0]) == 2)
	controls_x = [np.random.normal(mean_x, abs(std_x)) for (mean_x, _) in normal_list]
	controls_y = [np.random.normal(mean_x, abs(std_y)) for (_, mean_y) in normal_list]
	return (np.array(controls_x), np.array(controls_y))
