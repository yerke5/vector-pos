from itertools import permutations
import numpy as np

MAX_K = 2

def format_indices(indices):
	return "{" + ", ".join(['M' + str(i + 1) + str(j + 1) for (i, j) in indices]) + '}'

def beautify_matrix(m):
	output = '[\n'
	for i in range(len(m)):
		s = "["
		for j in range(len(m)):
			s += str([round(x, 1) for x in m[i][j]]) + " "
		output += '  ' + s[:-1] + ']\n'
	output += ']\n'
	return output

def is_valid(path, measured):
	for i in range(1, len(path)):
		if is_missing(measured[path[i - 1]][path[i]]):
			return False 
	return True

def matrix2indices(matrix):
	v = []
	for i in range(len(matrix)):
		for j in range(len(matrix)):
			if i != j and not is_missing(matrix[i][j]):
				v.append((i, j))
	return v

def get_missing(matrix):
	v = []
	for i in range(len(matrix)):
		for j in range(len(matrix)):
			if i != j and is_missing(matrix[i][j]):
				v.append((i, j))
	return v

def is_missing(vector):
	return np.isnan(np.sum(vector))

def paths2dict(temp):
	paths = {}
	for path in temp:
		if (path[0], path[-1]) not in paths:
			paths[(path[0], path[-1])] = [path]
		else:
			paths[(path[0], path[-1])].append(path)
	return paths

def get_magnitude(vector):
	return np.sqrt(np.sum(vector**2))

def generate_paths(num_points, id_pairs=None, max_k=MAX_K, full=False):
	node_ids = [i for i in range(num_points)]
	pairs = list(permutations(node_ids, 2)) if not id_pairs else id_pairs
	paths = pairs.copy()
	num_perms = 1
	for i, pair in enumerate(pairs):
		full_middle_path = set(node_ids) - set(pair)
		start, end = pair

		if full:
			iters = min(max_k, len(full_middle_path))
		else:
			iters = 1
		
		for length in range(iters):
			middle_perms = list(permutations(full_middle_path, length + 1))
			for middle_perm in middle_perms:
				paths.append((start, *middle_perm, end))
				if i == 0:
					num_perms += 1

	return paths


