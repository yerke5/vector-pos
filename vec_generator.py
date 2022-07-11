import math 
import copy
import random 
import numpy as np
import vec_manipulator as vm
import vec_helper as vh

def get_generated_vectors(measured, verbose=False):
	generated = np.zeros((len(measured), len(measured), 2))
	paths = vh.generate_paths(len(measured))

	#print('Paths:', paths)
	num_perms = dict()
	strpaths = dict()

	#print('Number of paths:', perms)
	for path in paths:
		if not vh.is_missing(measured[path[0]][path[-1]]):
			
			#print(f"Before {path[0] + 1} - {path[-1] + 1}: ", measured[path[0]][path[-1]])
			gen = get_generated_vector(path, measured) 
			if gen is not None:
				generated[path[0]][path[-1]] += gen
				if not (path[0], path[-1]) in num_perms:
					num_perms[(path[0], path[-1])] = 1
					strpaths[(path[0], path[-1])] = [''.join([str(p + 1) for p in path])]
				else:
					num_perms[(path[0], path[-1])] += 1
					strpaths[(path[0], path[-1])].append(''.join([str(p + 1) for p in path]))
		else:
			if (path[0], path[-1]) not in num_perms:
				generated[path[0]][path[-1]] = vm.deduce_vector(path[0], path[-1], measured, verbose=verbose)
				num_perms[(path[0], path[-1])] = 1
		#generated[path[0]][path[-1]] /= (perms + 1)
		#print(f"After {path[0] + 1} - {path[-1] + 1}: ", generated[path[0]][path[-1]])
		#print('-' * 50)

	for i in range(len(generated)):
		for j in range(len(generated)):
			if i != j:
				if verbose and (i, j) in strpaths:
					print("[NOT DEDUCED] M" + str(i + 1) + str(j + 1) + "(g): sum(" + "; ".join(strpaths[(i, j)]) + ") / " + str(num_perms[(i, j)]))
				generated[i][j] = generated[i][j] / num_perms[(i, j)]#(num_perms)#(perms + 1)
	return generated

def get_generated_vector(path, measured):
	new_vector = np.array([0.0, 0.0])

	for i in range(1, len(path)): 
		if vh.is_missing(measured[path[i - 1]][path[i]]):
			return None 
		#print(f"{path[i-1] + 1} - {path[i] + 1}: {measured[path[i - 1]][path[i]]}")
		new_vector += measured[path[i - 1]][path[i]]
	return new_vector

def generate_random_node(size):
	return [random.random() * size, random.random() * size]

def coords2vectors(coords, angle_noise=None, radius_noise=None):
	vectors = np.zeros((len(coords), len(coords), 2))
	noise = radius_noise is not None and angle_noise is not None
	for i in range(len(coords)):
		for j in range(len(coords)):
			vectors[i][j] = coords[j] - coords[i] 
			if noise and i != j:
				vectors[i][j] = vm.add_noise(vectors[i][j], angle_noise=angle_noise, radius_noise=radius_noise)
	return vectors 

