import random
import math
import matplotlib.pyplot as plt
import numpy as np

def add_noise(node, radius_noise=.2, angle_noise=.2):
	r = math.sqrt(node[0]**2 + node[1]**2)
	x_angle = math.acos(node[0] / r)
	y_angle = math.asin(node[1] / r)
	return (
		r * (1 + np.random.normal() * radius_noise) * math.cos(x_angle * (1 + np.random.normal() * angle_noise)), 
		r * (1 + np.random.normal() * radius_noise) * math.sin(y_angle * (1 + np.random.normal() * angle_noise))
	)

def coords2vectors(coords, space_size, angle_noise=.3, radius_noise=.3, noise_ratio=0):
	vectors = np.zeros((len(coords), len(coords), 2))
	noise = noise_ratio > 0
	pairs = []
	for i in range(len(coords)):
		for j in range(len(coords)):
			if i != j:
				pairs.append((i, j))

	pairs = random.sample(pairs, int(noise_ratio * (len(coords)**2 - len(coords))))
	for i in range(len(coords)):
		for j in range(len(coords)):
			vectors[i][j] = coords[j] - coords[i] 
			if noise and (i, j) in pairs:
				vectors[i][j] = add_noise(vectors[i][j], space_size, angle_noise=angle_noise, radius_noise=radius_noise)
	return vectors 

def generate_random_node(size):
	return [random.randint(0, size), random.randint(0, size)]

def add_noise(node, space_size, radius_noise=.2, angle_noise=.2):
	r = math.sqrt(node[0]**2 + node[1]**2)
	x_angle = math.acos(node[0] / r)
	y_angle = math.asin(node[1] / r)
	return (
		min(space_size, max(0, r * (1 + np.random.normal() * radius_noise) * math.cos(x_angle * (1 + np.random.normal() * angle_noise)))), 
		min(space_size, max(0, r * (1 + np.random.normal() * radius_noise) * math.sin(y_angle * (1 + np.random.normal() * angle_noise))))
	)

def generate_coords(space_size, num_nodes):
	true_nodes = []

	for i in range(num_nodes):
		node = generate_random_node(space_size)
		true_nodes.append(node)
		
	true_nodes = np.array(true_nodes)
	return true_nodes

def draw_vectors(space_size, coords, vectors):
	#plt.quiver([0, 0, 0], [0, 0, 0], [1, -2, 4], [1, 2, -7], angles='xy', scale_units='xy', scale=1)
	origin_x = []
	origin_y = []
	vector_x = []
	vector_y = []

	for i in range(len(vectors)):
		for j in range(len(vectors)):
			if i != j:
				origin_x.append(coords[i][0])
				origin_y.append(coords[i][1])
				vector_x.append(vectors[i][j][0])
				vector_y.append(vectors[i][j][1])


	print(origin_x)
	print(origin_y)
	print(vector_x)
	print(vector_y)
	plt.scatter(origin_x, origin_y)
	#plt.quiver(origin_x, origin_y, vector_x, vector_x, angles='xy', scale_units='xy', scale=1)

	for i in range(len(vector_x)):
		plt.axes().arrow(origin_x[i], origin_y[i], vector_x[i], vector_y[i], head_width=0.05,head_length=0.1)

	plt.axis('equal')  #<-- set the axes to the same scale
	plt.xlim(-space_size, space_size)
	plt.ylim(-space_size, space_size)
	plt.grid(b=True, which='major') #<-- plot grid lines
	plt.show()

if __name__ == "__main__":
	space_size = 10
	num_nodes = 4
	true_nodes = generate_coords(space_size, num_nodes)
	true_vectors = coords2vectors(true_nodes, space_size)
	print(true_nodes)
	print(true_vectors)
	noisy_vectors = coords2vectors(true_nodes, space_size, noise_ratio=.4)
	draw_vectors(space_size, true_nodes, true_vectors)

