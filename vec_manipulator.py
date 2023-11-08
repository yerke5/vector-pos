import copy
import numpy as np
import math
import vec_helper as vh
import vec_generator as vg
import random
import collections 

def add_noise(node, space_size, radius_noise=.2, angle_noise=.2, mean_radius_noise=0, mean_angle_noise=0):
	r = math.sqrt(node[0]**2 + node[1]**2)
	theta = math.acos(node[0] / r)
	if node[1] < 0:
		theta *= -1
	
	return (
		r * (1 + np.random.normal(loc=mean_radius_noise, scale=radius_noise)) * np.cos(theta * (1 + np.random.normal(loc=mean_angle_noise, scale=angle_noise))),#r * (1 + np.random.normal() * radius_noise) * math.cos(x_angle * (1 + np.random.normal() * angle_noise)), 
		r * (1 + np.random.normal(loc=mean_radius_noise, scale=radius_noise)) * np.sin(theta * (1 + np.random.normal(loc=mean_angle_noise, scale=angle_noise)))#r * (1 + np.random.normal() * radius_noise) * math.sin(y_angle * (1 + np.random.normal() * angle_noise))
	)

def drop_unseen_vectors(vectors, orientations, max_angle=40, max_range=30, verbose=False):
	filtered = copy.deepcopy(vectors)
	if verbose:
		print('Orientations:', orientations)
	
	for i in range(len(vectors)):
		for j in range(len(vectors)):
			if i != j:
				if np.sqrt(np.sum(vectors[i][j]**2)) > max_range:
					filtered[i][j] = np.nan 
				else:
					curr_angle = math.acos(np.dot(orientations[i], vectors[i][j]) / (vh.get_magnitude(orientations[i]) * vh.get_magnitude(vectors[i][j]))) * 180 / math.pi
					if verbose:
						print(i, '->', j, ':', curr_angle, 'valid:', curr_angle <= max_angle)
					if curr_angle > max_angle:
						filtered[i][j] = np.nan 

	return filtered

def inject_noise(original, noise_ratio, space_size, coords, angle_noise, radius_noise, num_anchors=0, mean_angle_noise=0, mean_radius_noise=0, mean_angle_noise_low=None, mean_radius_noise_low=None, angle_noise_low=None, radius_noise_low=None):
	noisy_pairs = []
	for i in range(len(coords) - num_anchors):
		for j in range(len(coords) - num_anchors): # the last num_anchors nodes are anchors
			if i != j and not vh.is_missing(original[i][j]):
				noisy_pairs.append((i, j))

	vectors = copy.deepcopy(original)
	noisy_pairs = random.sample(noisy_pairs, int(noise_ratio * (len(vh.matrix2indices(original)))))
	
	for i in range(len(coords)):
		for j in range(len(coords)):
			if i != j:
				if (i, j) in noisy_pairs:
					noisy_vector = list(add_noise(vectors[i][j], space_size, angle_noise=angle_noise, radius_noise=radius_noise, mean_angle_noise=mean_angle_noise, mean_radius_noise=mean_radius_noise))
					# DON"T FORGET ABOUT PUTTING BOUNDS HERE
					# noisy_vector[0] = min(space_size - coords[i][0], noisy_vector[0])
					# noisy_vector[1] = min(space_size - coords[i][1], noisy_vector[1])
					# noisy_vector[0] = max(-coords[i][0], noisy_vector[0])
					# noisy_vector[1] = max(-coords[i][1], noisy_vector[1])
					vectors[i][j] = noisy_vector
				elif mean_angle_noise_low is not None and mean_radius_noise_low is not None:
					noisy_vector = list(add_noise(vectors[i][j], space_size, angle_noise=angle_noise_low, radius_noise=radius_noise_low, mean_angle_noise=mean_angle_noise_low, mean_radius_noise=mean_radius_noise_low))
					vectors[i][j] = noisy_vector
	return vectors

	idx = np.random.choice(len(coords) - num_anchors, int((len(coords) - num_anchors) * noise_ratio), replace=False)  
	coords[idx] *= (1 + np.random.normal() * noise_level)
	return coords