import copy
import numpy as np
import math
import vec_helper as vh
import vec_generator as vg
import random

def deduce_vectors(measured, paths, num_deduction_components, verbose=False, enforce_deduction=False):
	#print('Incomplete measured:\n', measured)
	deduced = copy.deepcopy(measured)
	for i in range(len(measured)):
		for j in range(len(measured)):
			if i != j and vh.is_missing(measured[i][j]):
				#print('Deducing', (i, j))
				deduced[i][j] = deduce_vector(i, j, deduced, paths[(i, j)], num_deduction_components, verbose=verbose)

				if enforce_deduction and vh.is_missing(deduced[i][j]):
					return None
	return deduced 

def deduce_vector(i, j, measured, paths, num_deduction_components, space_size=None, verbose=True):
	#paths = vh.generate_paths(len(measured), id_pairs=[(i, j)])
	
	# remove pairs
	paths = [x for x in paths if len(x) > 2]

	#print('Filtered paths:', paths)
	if len(paths) < 2: # used to be 2
		return None  

	components = []
	cids = []
	vecstrs = []
	for path in paths:
		#print('Trying path', path)
		new_vector = np.array([0.0, 0.0])
		cid = []
		vecstr = []
		broke = False
		for k in range(1, len(path)): 
			if not vh.is_missing(measured[path[k - 1]][path[k]]):
				#print('Found', path[k - 1], '-', path[k])
				new_vector += measured[path[k - 1]][path[k]]
				cid.append("M" + str(path[k - 1] + 1) + str(path[k] + 1))
				vecstr.append(str(measured[path[k - 1]][path[k]]))
				#j += 1
			else:
				broke = True 
				break

		if not broke:#j == len(path) - 1:
			#print('Success! The new vector is', new_vector)
			components.append(new_vector.copy())
			cids.append(", ".join(cid))
			vecstrs.append("(" + " + ".join(vecstr) + ")")
			#print('Components:', components)
			#print('CIDs:', cids)

	#print('Components:', components)
	if len(components) < num_deduction_components: # used to be 2 
		return None  

	if verbose:
		#print("[DEDUCED] M" + str(i + 1) + str(j + 1) + '(g):', 'sum(' + '; '.join(cids) + ') / ' + str(len(cids)))
		print("[DEDUCED]     M" + str(i + 1) + str(j + 1) + '(g):', ' + '.join(vecstrs) + ') / ' + str(len(cids)) + ' =', np.mean(components, axis=0))
	
	#print('Success! The new vector is', np.mean(components, axis=0))
	return np.mean(components, axis=0)

def add_noise(node, space_size, radius_noise=.2, angle_noise=.2):
	r, x_angle, y_angle = get_radial_vec(node) #math.sqrt(node[0]**2 + node[1]**2)
	#x_angle = math.acos(node[0] / r)
	#y_angle = math.asin(node[1] / r)
	return (
		r * (1 + np.random.normal() * radius_noise) * math.cos(x_angle * (1 + np.random.normal() * angle_noise)), 
		r * (1 + np.random.normal() * radius_noise) * math.sin(y_angle * (1 + np.random.normal() * angle_noise))
	)

def get_radial_vec(vector):
	r = np.sqrt(np.sum(np.array(vector)**2))
	x_angle, y_angle = math.acos(vector[0] / r), math.asin(vector[1] / r)
	return r, x_angle, y_angle

def drop_unseen_vectors(vectors, coords, max_angle=40, max_range=30, verbose=False):
	filtered = copy.deepcopy(vectors)
	orientations = vh.get_orientations(coords)
	
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

def remove_inconsistent_vectors(vectors, num_deduction_components, min_diffs_ratio, paths):
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
		if v not in intermediaries and is_deducible(v[0], v[1], temp, num_deduction_components):
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

def is_deducible(i1, i2, vectors, num_deduction_components):
	t = 0
	for k in range(len(vectors)):
		if k != i1 and k != i2:
			if not vh.is_missing(vectors[i1][k]) and not vh.is_missing(vectors[k][i2]):
				t += 1
	return t >= num_deduction_components

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

def get_consistent_vectors2(vectors, paths, num_deduction_components):
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
	for_deduction = []
	for ((i, k, j), diff) in diffs:
		if diff == 0:
			#print('Curr diff:', diff, i, k, j)
			temp[i][k] = vectors[i][k].copy()
			temp[i][j] = vectors[i][j].copy()
			temp[j][k] = vectors[j][k].copy()
		else:
			deduced = deduce_vectors(temp, paths, num_deduction_components, enforce_deduction=True)
			#print("Current chromosome:", temp)
			#print("Deduced chromosome:", deduced)
			if deduced is None:
				for_deduction += [(i, j), (j, k), (i, k)]
				temp[i][k] = vectors[i][k].copy()
				temp[i][j] = vectors[i][j].copy()
				temp[j][k] = vectors[j][k].copy()
			else:
				#print('BREAKING')
				break 

	print('Added vectors for deducability:', ', '.join(sorted(["M" + str(x[0]) + '-' + str(x[1]) for x in for_deduction])))
	return temp

def get_consistent_vectors(vectors, pair_paths, num_deduction_components, num_vectors=4, max_shapes=30, id_pairs=None):
	paths = vh.generate_paths_of_length(len(vectors), num_vectors, id_pairs=id_pairs)
	temp = np.zeros((len(vectors), len(vectors), 2))
	temp[:, :] = np.nan
	shapes = dict()
	
	pairs = list(paths.keys())
	random.shuffle(pairs)

	k = 0
	for pair in pairs:
		random.shuffle(paths[pair])
		for path in paths[pair]:
			if vh.is_valid(path, vectors):
				generated = np.array([0, 0])
				for i in range(1, len(path)):
					generated = generated + vectors[path[i - 1]][path[i]]
				
				diff = np.sum((vectors[pair[0]][pair[1]] - generated)**2)

				shapes[path] = np.sqrt(diff) 
				k += 1

			if k >= max_shapes:
				break
			
	for i in range(len(temp)):
		temp[i][i] = [0, 0]

	shapes = list(sorted(shapes.items(), key=lambda item: item[1]))
	
	for (path, diff) in shapes:
		if diff == 0:
			for i in range(1, len(path)):
				temp[path[i - 1]][path[i]] = vectors[path[i - 1]][path[i]].copy()
				#consistent.add((path[i - 1], path[i]))
		else:
			deduced = deduce_vectors(temp, pair_paths, num_deduction_components, enforce_deduction=True)
			if deduced is None:
				for i in range(1, len(path)):
					temp[path[i - 1]][path[i]] = vectors[path[i - 1]][path[i]].copy()
			else:
				break 

	return temp

def get_supergenes(vectors, pair_paths, num_deduction_components, num_vectors=4, id_pairs=None, max_num=20):
	paths = vh.generate_paths_of_length(len(vectors), num_vectors, id_pairs=id_pairs)
	shapes = dict()
	k = 0
	for pair in paths:
		for path in paths[pair]:
			generated = np.array([0, 0])
			for i in range(1, len(path)):
				generated = generated + vectors[path[i - 1]][path[i]]
			
			diff = np.sum((vectors[pair[0]][pair[1]] - generated)**2)
			shapes[path] = np.sqrt(diff) 

			if diff == 0:
				k += 1

			if k >= max_num:
				break
			
	shapes = list(sorted(shapes.items(), key=lambda item: item[1]))
	
	supergenes = set()
	for (path, diff) in shapes:
		if diff == 0:
			for i in range(1, len(path)):
				supergenes.add((path[i - 1], path[i]))

	return supergenes

def inject_noise(original, noise_ratio, space_size, coords, angle_noise, radius_noise):
	pairs = []
	for i in range(len(coords)):
		for j in range(len(coords)):
			if i != j and not vh.is_missing(original[i][j]):
				pairs.append((i, j))

	vectors = copy.deepcopy(original)
	pairs = random.sample(pairs, int(noise_ratio * (len(vh.matrix2indices(original)))))
	for (i, j) in pairs:
		noisy_vector = list(add_noise(vectors[i][j], space_size, angle_noise=angle_noise, radius_noise=radius_noise))
		noisy_vector[0] = min(space_size - coords[i][0], noisy_vector[0])
		noisy_vector[1] = min(space_size - coords[i][1], noisy_vector[1])
		noisy_vector[0] = max(-coords[i][0], noisy_vector[0])
		noisy_vector[1] = max(-coords[i][1], noisy_vector[1])
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
