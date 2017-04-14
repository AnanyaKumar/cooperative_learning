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
		param_list.append((this_car.pos_x, this_car.pos_y, this_car.vel_x, this_car.vel_y))
		nn_input.append(np.array(param_list).flatten())
	return np.array(nn_input)

def build_nn_output(normal_list, std_x=1, std_y=1):
	return np.array(([np.random.normal(mean_x, abs(std_x)) for (mean_x, _) in normal_list],
		[np.random.normal(mean_y, abs(std_y)) for (_, mean_y) in normal_list]))
