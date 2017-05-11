# coding: utf-8
"""Methods useful for processing the state."""

import numpy as np
from PIL import Image, ImageDraw
import geometry_utils

def build_nn_input(state, k, l):
	"""Takes a list of car objects and returns neural net state information for each car.

	Args: state: State from the environment. Consists of a list of
			cars and obstacles.  k: Each car decides what action
			to take based on its information and the information
			of the closest k cars, and the closest l obstacles.

	Returns: A numpy array states, where states[i] is the state
		for the ith car. len(states) is the number of
		cars. states[i] is an array of size 4*k + 3*l +4. The
		last 4 floats are the ith car's x position, y
		position, x velocity, y velocity. The first 4k floats
		are 4 floats for each neighboring car. The next 3l
		floats are the relative (x,y) position and radius of
		the l closest obstacles.

		"""
	
	car_list, obstacle_list = state
	assert k < len(car_list)
	assert l <= len(obstacle_list)
	nn_input = []
	for this_car in car_list:
		sorted_car_list = sorted([(this_car.dist(car), car) for car in car_list], key=(lambda x:x[0]))
		sorted_obstacle_list = sorted([(this_car.dist(obs), obs) for obs in obstacle_list], key=(lambda x:x[0]))
		# Remove the first car, as it should be itself
		sorted_car_list = [x[1] for x in sorted_car_list[1:k+1]]
		sorted_obstacle_list = [x[1] for x in sorted_obstacle_list[0:l]]
		param_list = [[car.pos_x - this_car.pos_x, car.pos_y - this_car.pos_y,
		car.vel_x - this_car.vel_x, car.vel_y - this_car.vel_y] for car in sorted_car_list]
		param_list += [[obs.pos_x - this_car.pos_x, obs.pos_y - this_car.pos_y, obs.radius] for obs in sorted_obstacle_list]
		param_list.append([this_car.pos_x, this_car.pos_y, this_car.vel_x, this_car.vel_y])
		nn_input.append(np.array(sum(param_list, [])))

	return np.array(nn_input)

def get_nn_input_dim(k, l):
		return 4 * k + 3 * l + 4

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
	controls_y = [np.random.normal(mean_y, abs(std_y)) for (_, mean_y) in normal_list]
	return (np.array(controls_x), np.array(controls_y))

def build_cnn_input(lists, bot_lane=0, top_lane=1, scale=(50.0, 50.0), size=(40,40)):
	car_list, obs_list = lists
	center = (size[0]/scale[0]/2, size[1]/scale[1]/2)
	lane_width = size[0] / scale[0]
	lane_height = size[1] / scale[1]
	inputs = []
	for i in range(len(car_list)):
		this_car = car_list[i]
		im = Image.new('1', size, color=0)
		draw = ImageDraw.Draw(im)
		param_list = [(car.pos_x - this_car.pos_x, car.pos_y - this_car.pos_y) for car in car_list]

		bb = ((this_car.pos_x - lane_width / 2, bot_lane), (this_car.pos_x + lane_width / 2, bot_lane - lane_height))
		bb = geometry_utils.shift_then_scale_points(bb, -this_car.pos_x+center[0], -this_car.pos_y+center[1], scale[0], scale[1])
		draw.rectangle(bb, fill=(1))

		bb = ((this_car.pos_x - lane_width / 2, top_lane), (this_car.pos_x + lane_width / 2, top_lane + lane_height))
		bb = geometry_utils.shift_then_scale_points(bb, -this_car.pos_x+center[0], -this_car.pos_y+center[1], scale[0], scale[1])
		draw.rectangle(bb, fill=(1))

		for j in range(len(car_list+obs_list)):
			if i == j:
				# Don't draw yourself (not that it really matters)
				continue

			car = car_list[j]
			bb = ((car.pos_x - car.radius, car.pos_y - car.radius), (car.pos_x + car.radius, car.pos_y + car.radius))
			bb = geometry_utils.shift_then_scale_points(bb, -this_car.pos_x+center[0], -this_car.pos_y+center[1], scale[0], scale[1])
			draw.ellipse(bb, fill=(1))

		inputs.append(np.array(im.getdata()).reshape((size[0], size[1], 1)))
		if i==0:
			im.show()

	return np.array(inputs)
