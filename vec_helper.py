from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt

MAX_K = 2

def format_indices(indices):
	return "{" + ", ".join(['M' + str(i + 1) + '-' + str(j + 1) for (i, j) in indices]) + '}'

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

def draw_vectors(space_size, coords, vectors, title, plot_orientations=False):
	origin_x = []
	origin_y = []
	vector_x = []
	vector_y = []

	if plot_orientations:
		orientations = np.zeros((len(coords), 2))
	
	for i in range(len(vectors)):
		if plot_orientations:
			orientations[i] = np.sum(np.concatenate([np.array(true_nodes)[:i, :], np.array(true_nodes)[i + 1:, :]]), axis=0) / (len(true_nodes) - 1)
		
		for j in range(len(vectors)):
			if i != j:
				origin_x.append(coords[i][0])
				origin_y.append(coords[i][1])
				vector_x.append(vectors[i][j][0])
				vector_y.append(vectors[i][j][1])

	plt.title(title)
	plt.scatter(origin_x, origin_y)

	for i, txt in enumerate([i for i in range(len(vectors))]):
   		plt.annotate("M" + str(txt), (origin_x[i], origin_y[i]))

	plt.quiver(origin_x, origin_y, vector_x, vector_y, angles='xy', scale_units='xy', scale=1)

	if plot_orientations:
		plt.quiver(origin_x, origin_y, orientations[:, 0], orientations[:, 1], angles='xy', scale_units='xy', scale=1, c="red")

	plt.axis('equal')  #<-- set the axes to the same scale
	plt.xlim(0, space_size)
	plt.ylim(0, space_size)
	plt.grid(b=True, which='major') #<-- plot grid lines
	plt.show()

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def scatter_locations(space_size, coords, vectors, title):
	origin_x = []
	origin_y = []
	vector_x = []
	vector_y = []
	cmap = get_cmap(len(vectors))

	for i in range(len(vectors)):
		plt.scatter(coords[i][0], coords[i][1], marker="x", c=cmap(i), label="Node " + str(i + 1), s=80)
		for j in range(len(vectors)):
			if i != j:
				plt.scatter(coords[i][0] + vectors[i][j][0], coords[i][1] + vectors[i][j][1], marker="o", c=cmap(j), s=20)

	plt.title(title)
	plt.legend()
	plt.axis('equal')  #<-- set the axes to the same scale
	#plt.xlim([-maxes[0],maxes[0]]) #<-- set the x axis limits
	#plt.ylim([-maxes[1],maxes[1]])
	plt.xlim(-space_size, space_size)
	plt.ylim(-space_size, space_size)
	plt.grid(b=True, which='major') #<-- plot grid lines
	plt.show()
