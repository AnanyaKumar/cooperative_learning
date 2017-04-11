# coding: utf-8
"""Methods useful for processing the state."""

import numpy as np

def build_nn_input(car_list, k):
	nn_input = []
	for this_car in car_list:
		sorted_car_list = sorted([(this_car.dist(car), car) for car in car_list], key=(lambda x:x[0]))
		# Remove the first car, as it should be itself
		sorted_car_list = [x[1] for x in sorted_car_list[1:k+1]]
		param_list = [(car.pos_x - this_car.pos_x, car.pos_y - this_car.pos_y,
			car.vel_x - this_car.vel_x, car.vel_y - this_car.vel_y) for car in sorted_car_list]
		nn_input.append(np.array(param_list).flatten())
	return np.array(nn_input)

def build_nn_output(normal_list):
	return [np.random.normal(mean, std) for (mean, std) in normal_list]
