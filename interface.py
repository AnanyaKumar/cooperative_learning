# coding: utf-8
"""Methods useful for processing the state."""

import numpy as np
from PIL import Image, ImageDraw
import geometry_utils

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

def build_cnn_input(car_list, top_lane, scale=(10.0, 10.0), size=(100,100)):
	center = (size[0]/scale[0]/2, size[1]/scale[1]/2)
	lane_width = size[0] / scale[0]
	lane_height = size[1] / scale[1]
	inputs = []
	for i in range(len(car_list)):
		this_car = car_list[i]
		im = Image.new('1', size, color=1)
		draw = ImageDraw.Draw(im)
		param_list = [(car.pos_x - this_car.pos_x, car.pos_y - this_car.pos_y) for car in car_list]

		bot_lane = 0
		bb = ((this_car.pos_x - lane_width / 2, bot_lane), (this_car.pos_x + lane_width / 2, bot_lane - lane_height))
		bb = geometry_utils.shift_then_scale_points(bb, -this_car.pos_x+center[0], -this_car.pos_y+center[1], scale[0], scale[1])
		draw.rectangle(bb, fill=(0))

		bb = ((this_car.pos_x - lane_width / 2, top_lane), (this_car.pos_x + lane_width / 2, top_lane + lane_height))
		bb = geometry_utils.shift_then_scale_points(bb, -this_car.pos_x+center[0], -this_car.pos_y+center[1], scale[0], scale[1])
		draw.rectangle(bb, fill=(0))

		for j in range(len(car_list)):
			if i == j:
				# Don't draw yourself (not that it really matters)
				continue

			car = car_list[j]
			bb = ((car.pos_x - car.radius, car.pos_y - car.radius), (car.pos_x + car.radius, car.pos_y + car.radius))
			bb = geometry_utils.shift_then_scale_points(bb, -this_car.pos_x+center[0], -this_car.pos_y+center[1], scale[0], scale[1])
			draw.ellipse(bb, fill=(0))

		inputs.append(np.array(im.getdata()).reshape(size))
	return inputs
