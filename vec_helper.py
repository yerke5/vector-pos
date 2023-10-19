from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import random 
import collections 

MAX_K = 2

def vector_count(num_nodes, num_anchors):
	total_nodes = num_nodes + num_anchors
	return total_nodes * (total_nodes - 1) 

def log(s):
    print(f"[INFO] {s}")

def nPr(n, r):
    return int(math.factorial(n) / math.factorial(n - r))

def calculate_error(m1, m2):
	error = 0
	n = 0
	for i in range(len(m1)):
		for j in range(len(m1)):
			if i != j and not is_missing(m1[i][j]) and not is_missing(m2[i][j]):
				n += 1 
				error += np.sqrt(np.sum((m1[i][j] - m2[i][j])**2))

	if n == 0:
		return float("inf")
	return error / n

def format_indices(indices, sort=True):
	return "{" + ", ".join(['M' + str(i + 1) + '-' + str(j + 1) for (i, j) in sorted(indices)]) + '}'

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

def paths2len_dict(paths):
	#print("Paths:", paths)
	path_lens = {}
	min_len = float('inf')
	max_len = -1

	for path in paths:
		if len(path) > 2:
			if len(path) not in path_lens:
				path_lens[len(path)] = []
				min_len = min(min_len, len(path))
				max_len = max(max_len, len(path))
			path_lens[len(path)].append(path)
	return path_lens, min_len, max_len

def get_orientations(coords, space_size, random=False):
	orientations = np.zeros((len(coords), 2))

	if not random:
		for i in range(len(orientations)):
			orientations[i] = np.sum(np.concatenate([np.array(coords)[:i, :], np.array(coords)[i + 1:, :]]), axis=0) / (len(coords) - 1) - coords[i]
	else:
		orientations = (np.random.rand(len(coords), 2) * 2 - 1) * (space_size / 3)
	return orientations

def get_magnitude(vector):
	return np.sqrt(np.sum(vector**2))

def generate_paths(num_points, min_degree=1, max_degree=1):
	#print("Num points:", num_points, "min:", min_degree, "max:", max_degree)
	max_degree = min(max_degree + 1, num_points - 2)
	min_degree = max(min_degree, 1)
		
	node_ids = [i for i in range(num_points)]
	pairs = list(permutations(node_ids, 2))
	paths = pairs.copy()
	for i, pair in enumerate(pairs):
		full_middle_path = set(node_ids) - set(pair)
		start, end = pair

		for length in range(min_degree, max_degree):
			#print("Exploring length", length)
			middle_perms = list(permutations(full_middle_path, length))
			for middle_perm in middle_perms:
				paths.append((start, *middle_perm, end))
				#yield (start, *middle_perm, end)
				
	return paths

def draw_vectors(space_size, coords, vectors, title, max_angle=None, orientations=None, show=False, colors=None, boundaries=True, terminal_labels=True, figtitle=None, tl=None):
	plt.figure()
	origin_x = np.repeat(coords[:, 0], len(coords)).reshape(-1,)
	origin_y = np.repeat(coords[:, 1], len(coords)).reshape(-1,)
	vector_x = vectors[:, :, 0].reshape(-1,)
	vector_y = vectors[:, :, 1].reshape(-1,)

	if orientations is not None:
		orientations_left = np.zeros((len(coords), 2))
		orientations_right = np.zeros((len(coords), 2))
	
		for i in range(len(vectors)):
			if max_angle is None:
				raise Exception("If orientations are drawn, a max angle must be specified")
			orientations_left[i] = find_vector_at_angle(orientations[i], -max_angle)
			orientations_right[i] = find_vector_at_angle(orientations[i], max_angle)
	
	plt.title(title)
	coords = np.array(coords)
	
	if not colors:	
		plt.scatter(coords[:, 0], coords[:, 1])
	else:
		plt.scatter(coords[:, 0], coords[:, 1], c=colors)

	if terminal_labels:
		if tl is None:
			for i in range(len(coords)):
				plt.annotate(f"M{i + 1}", (coords[i, 0], coords[i, 1]))
		else:
			for i in range(len(coords)):
				plt.annotate(tl[i], (coords[i, 0], coords[i, 1]))

	plt.quiver(origin_x, origin_y, vector_x, vector_y, angles='xy', scale_units='xy', scale=1)
	
	if orientations is not None:
		plt.quiver(np.array(coords)[:, 0], np.array(coords)[:, 1], orientations[:, 0], orientations[:, 1], angles='xy', scale_units='xy', scale=1, color="red")
		if boundaries:
			plt.quiver(np.array(coords)[:, 0], np.array(coords)[:, 1], orientations_left[:, 0], orientations_left[:, 1], angles='xy', scale_units='xy', scale=1, color="green")
			plt.quiver(np.array(coords)[:, 0], np.array(coords)[:, 1], orientations_right[:, 0], orientations_right[:, 1], angles='xy', scale_units='xy', scale=1, color="blue")

	plt.axis('equal')  #<-- set the axes to the same scale
	plt.grid() #<-- plot grid lines

	if show:
		plt.show()
	
	if figtitle is not None:
		plt.savefig(figtitle)
	plt.close('all')

def add_angle(node, angle):
	r = math.sqrt(node[0]**2 + node[1]**2)
	x_angle = math.acos(node[0] / r)
	y_angle = math.asin(node[1] / r)
	return (
		r * math.cos(x_angle + angle * math.pi / 180), 
		r * math.sin(y_angle )#+ angle * math.pi / 180)# + angle)
	)

def find_vector_at_angle(vector, angle):
	angle = angle * np.pi / 180
	#return np.cos(angle * np.pi / 180) / vector * np.linalg.norm(vector)
	return np.dot(np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]), vector)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def generate_paths_of_length(num_points, length):
	node_ids = [i for i in range(num_points)]
	pairs = list(permutations(node_ids, 2))
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

def get_dims(space_size, max_range):
    x = min(space_size * np.sqrt(2), max_range)
    return -x, x

def trim_noisy_vectors(vectors, space_size, max_range):
	trimmed = copy.deepcopy(vectors)
	global_min, global_max = get_dims(space_size, max_range)
	trimmed[trimmed < global_min] = global_min
	trimmed[trimmed > global_max] = global_max
	return trimmed

def generate_random_nodes(num_nodes, size, num_anchors=0):
	return np.array([generate_random_node(size) for _ in range(num_nodes + num_anchors)]) 

def generate_random_node(size):
	return [random.random() * size, random.random() * size]

def coords2vectors(coords):
	vectors = np.zeros((len(coords), len(coords), 2))

	for i in range(len(coords)):
		for j in range(len(coords)):
			vectors[i][j] = coords[j] - coords[i] 

	return vectors 

def vecs2coords(vectors, space_size, pivot=0):
	# pivot at node 0
	coords = np.zeros((len(vectors), 2))
	for i in range(1, len(vectors)):
		if not is_missing(vectors[pivot][i]):
			coords[i] = vectors[pivot][i]
		else:
			for j in range(len(vectors)):
				if j != i and j != pivot and not is_missing(vectors[pivot][j]) and not is_missing(vectors[j][i]):
					coords[i] = vectors[pivot][j] + vectors[j][i]
					break
	gm = np.min(coords)
	if gm < 0:
		coords = coords - gm
	
	if len(coords[coords > space_size]) > 0:
		coords[coords > space_size] = space_size
	coords[coords < 0] = 0
	return coords

def get_empty_vec_matrix(num_nodes):
	temp = np.zeros((num_nodes, num_nodes, 2))
	temp[:, :] = np.nan
	
	for i in range(len(temp)):
		temp[i][i] = [0, 0]

	return temp

def get_origin_vectors(vectors, anchor_coords):
	origins = np.zeros((len(vectors), 2))
	origins[:] = np.nan
	origins[-len(anchor_coords):] = anchor_coords
	nums = collections.defaultdict(int)

	for j in range(len(vectors) - len(anchor_coords)):
		for a in range(len(anchor_coords)):
			aix = -(len(anchor_coords) - a)
			if not is_missing(vectors[aix][j]):
				if np.isnan(np.sum(origins[j])):
					origins[j] = np.array([0, 0])
				origins[j] += anchor_coords[aix] + vectors[aix][j]
				nums[j] += 1
	
	for i in range(len(origins) - len(anchor_coords)):
		origins[i] /= nums[i]
	return origins
				