from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
import math
import copy

MAX_K = 2

def vec_matrix2edge_index(matrix):
	for i in range(len(matrix)):
		for j in range(len(matrix)):
			if i != j and not vh.is_missing(matrix[i][j]):
				pass

def replace_missing_vectors(vectors, value):
	updated = copy.deepcopy(vectors)
	for i in range(len(vectors)):
		for j in range(len(vectors)):
			if i == j:
				updated[i][j] = [0, 0]
			elif is_missing(vectors[i][j]):
				updated[i][j] = [value, value]
	return updated

def format_indices(indices, sort=True):
	return "{" + ", ".join([format_vector_name((i + 1, j + 1)) for (i, j) in sorted(indices)]) + '}'

def format_vector_name(index):
	return "-".join(["M" + str(i) for i in index])

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

def get_orientations(coords):
	orientations = np.zeros((len(coords), 2))

	for i in range(len(orientations)):
		orientations[i] = np.sum(np.concatenate([np.array(coords)[:i, :], np.array(coords)[i + 1:, :]]), axis=0) / (len(coords) - 1) - coords[i]

	return orientations

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

def draw_vectors(space_size, coords, vectors, title, plot_orientations=False, max_angle=40, show=False):
	origin_x = []
	origin_y = []
	vector_x = []
	vector_y = []

	if plot_orientations:
		orientations = np.zeros((len(coords), 2))
		orientations_left = np.zeros((len(coords), 2))
		orientations_right = np.zeros((len(coords), 2))
	
	for i in range(len(vectors)):
		if plot_orientations:
			orientations[i] = np.sum(np.concatenate([np.array(coords)[:i, :], np.array(coords)[i + 1:, :]]), axis=0) / (len(coords) - 1) - coords[i]
			orientations_left[i] = add_angle(orientations[i], -max_angle)
			orientations_right[i] = add_angle(orientations[i], max_angle)
		
		for j in range(len(vectors)):
			if i != j:
				origin_x.append(coords[i][0])
				origin_y.append(coords[i][1])
				vector_x.append(vectors[i][j][0])
				vector_y.append(vectors[i][j][1])
	
	plt.title(title)
	plt.scatter(origin_x, origin_y)

	plt.quiver(origin_x, origin_y, vector_x, vector_y, angles='xy', scale_units='xy', scale=1)

	if plot_orientations:
		plt.quiver(np.array(coords)[:, 0], np.array(coords)[:, 1], orientations[:, 0], orientations[:, 1], angles='xy', scale_units='xy', scale=1, color="red")
		plt.quiver(np.array(coords)[:, 0], np.array(coords)[:, 1], orientations_left[:, 0], orientations_left[:, 1], angles='xy', scale_units='xy', scale=1, color="green")
		plt.quiver(np.array(coords)[:, 0], np.array(coords)[:, 1], orientations_right[:, 0], orientations_right[:, 1], angles='xy', scale_units='xy', scale=1, color="blue")

	plt.axis('equal')  #<-- set the axes to the same scale
	plt.xlim(0, space_size)
	plt.ylim(0, space_size)
	plt.grid(b=True, which='major') #<-- plot grid lines
	if show:
		plt.show()

def add_angle(node, angle):
	r = math.sqrt(node[0]**2 + node[1]**2)
	x_angle = math.acos(node[0] / r)
	y_angle = math.asin(node[1] / r)
	return (
		r * math.cos(x_angle + angle * math.pi / 180), 
		r * math.sin(y_angle )#+ angle * math.pi / 180)# + angle)
	)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def generate_paths_of_length(num_points, length, id_pairs=None):
	node_ids = [i for i in range(num_points)]
	pairs = list(permutations(node_ids, 2)) if not id_pairs else id_pairs
	paths = {pair: [] for pair in pairs}

	for i, pair in enumerate(pairs):
		full_middle_path = set(node_ids) - set(pair)
		start, end = pair
		middle_perms = list(permutations(full_middle_path, length - 2))
		for middle_perm in middle_perms:
			paths[pair].append((start, *middle_perm, end))
			#yield (pair, (start, *middle_perm, end))

	return paths

def path_to_pairs(path):
	if path[0] == path[-1]:
		raise Exception("Wrong path")
	#print(path)
	pairs = []
	for i in range(1, len(path)):
		pairs.append((path[i - 1], path[i]))
	#print(pairs)
	return pairs

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

def plot_errors(df, noise):
	cols = ["Measured", "Generated", "GA"]#, "Generated GA"]
	#plt.style.use('seaborn')
	ax = df.plot(x="Noisy Vector Ratio", y=cols, kind="bar")
	ax.set_title(f"GA vs. Other Methods (Noise Level = {noise}%)")
	ax.set_ylabel("Positioning error")
	plt.show()
