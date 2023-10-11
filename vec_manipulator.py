import copy
import numpy as np
import math
import vec_helper as vh
import vec_generator as vg
import random
import collections 

def add_noise(node, space_size, radius_noise=.2, angle_noise=.2):
	r = math.sqrt(node[0]**2 + node[1]**2)
	x_angle = math.acos(node[0] / r)
	y_angle = math.asin(node[1] / r)
	return (
		r * (1 + np.random.normal() * radius_noise) * math.cos(x_angle * (1 + np.random.normal() * angle_noise)), 
		r * (1 + np.random.normal() * radius_noise) * math.sin(y_angle * (1 + np.random.normal() * angle_noise))
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

def remove_inconsistent_vectors(vectors, max_infer_components, min_diffs_ratio, paths):
	nodes = set([i for i in range(len(vectors))])
	temp = copy.deepcopy(vectors)
	triangles = dict()
	for i in range(len(vectors)):
		for j in range(len(vectors)):
			if i != j:
				# consider all intermediaries
				min_diff = float('inf')
				min_k = None
				for k in range(len(vectors)):
					if j != k and i != k:
						diff = np.sum((vectors[i][j] - (vectors[i][k] + vectors[k][j]))**2)
						if min_k is None or diff < min_diff:
							min_diff = diff
							min_k = k
				
				triangles[(i, j)] = [(i, min_k), (min_k, j)]
	
	to_delete = set()
	intermediaries = set(sum(triangles.values(), []))

	#print(triangles)
	#print(intermediaries)
	for v in triangles:
		if v not in intermediaries and is_inferable(v[0], v[1], temp, max_infer_components):
			temp[v[0]][v[1]] = np.nan
			to_delete.add(v)

	#print("Vectors deleted after the first heuristic:", ", ".join(sorted(["M" + str(x[0] + 1) + '-' + str(x[1] + 1) for x in to_delete])))
	unchecked = set()
	for i in range(len(temp)):
		for j in range(len(temp)):
			if i != j and not vh.is_missing(temp[i][j]):
				unchecked.add((i, j))

	# restore diffs for generated vectors 
	required_num_diffs = max(1, int(min_diffs_ratio) * (len(vectors) - 1))
	restored = set()
	while unchecked:
		i, j = unchecked.pop()
		perms = set()
		for path in paths[(i, j)]:
			if len(path) > 2 and vh.is_valid(path, temp):
				perms.add(path[1])
		
		diffs = dict()
		for path in paths[(i, j)]:
			if len(path) > 2 and path[1] not in perms:
				diffs[path[1]] = np.sum((temp[i][j] - (temp[i][path[1]] + temp[path[1]][j]))**2)

		diffs = sorted(diffs.items(), key=lambda item: item[1])
		for k in range(required_num_diffs - len(perms)):
			temp[i][diffs[k][0]] = vectors[i][diffs[k][0]].copy()
			temp[diffs[k][0]][j] = vectors[diffs[k][0]][j].copy()
			restored.add((i, diffs[k][0]))
			restored.add((diffs[k][0], j))
			unchecked.add((i, diffs[k][0]))
			unchecked.add((diffs[k][0], j))

	#print("Restored", ", ".join(["M" + str(x[0] + 1) + str(x[1] + 1) for x in restored]))
	return temp

def get_consistent_vectors_simple(vectors):
	nodes = set([i for i in range(len(vectors))])
	temp = np.zeros((len(vectors), len(vectors), 2))
	temp[:, :] = np.nan 
	for _ in range(len(nodes)):
		i = nodes.pop()
		min_diff = float('inf')
		min_k = None 
		for j in range(len(temp)):
			if i != j and vh.is_missing(temp[i][j]):
				for k in range(len(temp)):
					if j != k: 
						diff = np.sum((temp[i][j] - (temp[i][k] + temp[k][j]))**2)
						if min_k is None or diff < min_diff:
							min_k = k
							min_diff = diff 
					else:
						temp[j][k] = 0 

				temp[i][j] = vectors[i][j].copy()
				temp[k][j] = vectors[k][j].copy()
				temp[i][k] = vectors[i][k].copy()
			elif i == j:
				temp[i][j] = 0
	return temp

def get_consistent_vectors2(vectors, paths, max_infer_components):
	nodes = set([i for i in range(len(vectors))])
	temp = np.zeros((len(vectors), len(vectors), 2))
	temp[:, :] = np.nan
	triangles = dict()
	for i in range(len(vectors)):
		for j in range(len(vectors)):
			if i != j:
				# consider all intermediaries
				for k in range(len(vectors)):
					if j != k and i != k:
						diff = np.sum((vectors[i][j] - (vectors[i][k] + vectors[k][j]))**2)
						triangles[(i, k, j)] = diff 
			else:
				temp[i][j] = 0

	diffs = list(sorted(triangles.items(), key=lambda item: item[1]))
	#print('Diffs:', diffs)
	for_inference = []
	for ((i, k, j), diff) in diffs:
		if diff == 0:
			#print('Curr diff:', diff, i, k, j)
			temp[i][k] = vectors[i][k].copy()
			temp[i][j] = vectors[i][j].copy()
			temp[j][k] = vectors[j][k].copy()
		else:
			inferred = infer_vectors(temp, paths, max_infer_components, enforce_inference=True)
			#print("Current chromosome:", temp)
			#print("Deduced chromosome:", inferred)
			if inferred is None:
				for_inference += [(i, j), (j, k), (i, k)]
				temp[i][k] = vectors[i][k].copy()
				temp[i][j] = vectors[i][j].copy()
				temp[j][k] = vectors[j][k].copy()
			else:
				#print('BREAKING')
				break 

	print('Added vectors for deducability:', ', '.join(sorted(["M" + str(x[0]) + '-' + str(x[1]) for x in for_inference])))
	return temp

def inject_noise(original, noise_ratio, space_size, coords, angle_noise, radius_noise, num_anchors=0):
	#print("R:", noise_ratio, "delta_r:", angle_noise)
	pairs = []
	for i in range(len(coords) - num_anchors):
		for j in range(len(coords) - num_anchors): # the last num_anchors nodes are anchors
			if i != j and not vh.is_missing(original[i][j]):
				pairs.append((i, j))

	vectors = copy.deepcopy(original)
	pairs = random.sample(pairs, int(noise_ratio * (len(vh.matrix2indices(original)))))
	
	for (i, j) in pairs:
		noisy_vector = list(add_noise(vectors[i][j], space_size, angle_noise=angle_noise, radius_noise=radius_noise))
		# DON"T FORGET ABOUT PUTTING BOUNDS HERE
		# noisy_vector[0] = min(space_size - coords[i][0], noisy_vector[0])
		# noisy_vector[1] = min(space_size - coords[i][1], noisy_vector[1])
		# noisy_vector[0] = max(-coords[i][0], noisy_vector[0])
		# noisy_vector[1] = max(-coords[i][1], noisy_vector[1])
		vectors[i][j] = noisy_vector
	return vectors

def remove_inconsistent_vectors2(vectors):
	nodes = set([i for i in range(len(vectors))])
	temp = np.zeros((len(vectors), len(vectors), 2))
	temp[:, :] = np.nan
	diffs = dict()
	for i in range(len(vectors)):
		for j in range(len(vectors)):
			if i != j:
				# consider all intermediaries
				min_diff = float('inf')
				min_k = None
				for k in range(len(vectors)):
					if j != k and i != k:
						diff = np.sum((vectors[i][j] - (vectors[i][k] + vectors[k][j]))**2)
						if min_k is None or diff < min_diff:
							min_diff = diff
							min_k = k
				
				if ((i, j) not in diffs or min_diff < diffs[(i, j)]):
					diffs[(i, min_k)] = min_diff
					diffs[(min_k, j)] = min_diff
					diffs[(i, j)] = min_diff

					temp[i][j] = vectors[i][j].copy()
					temp[i][min_k] = vectors[i][min_k].copy()
					temp[min_k][j] = vectors[min_k][j].copy()
			else:
				temp[i][j] = [0, 0]
	print(sorted(diffs.items(), key=lambda item: (item[0][0], item[0][1])))
	return temp

def inject_noise_to_coords(coords, noise_ratio, noise_level, num_anchors=0):
	idx = np.random.choice(len(coords) - num_anchors, int((len(coords) - num_anchors) * noise_ratio), replace=False)  
	coords[idx] *= (1 + np.random.normal() * noise_level)
	return coords