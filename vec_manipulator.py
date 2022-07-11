import copy
import numpy as np
import math
import vec_helper as vh
import vec_generator as vg

def deduce_vectors(measured, verbose=False):
	#print('Incomplete measured:\n', measured)
	deduced = copy.deepcopy(measured)
	for i in range(len(measured)):
		for j in range(len(measured)):
			if i != j and vh.is_missing(measured[i][j]):
				#print('Deducing', (i, j))
				deduced[i][j] = deduce_vector(i, j, deduced, verbose=verbose)

				if vh.is_missing(deduced[i][j]):
					return None
	return deduced 

def deduce_vector(i, j, measured, space_size=None, verbose=True, component_threshold=2):
	paths = vh.generate_paths(len(measured), id_pairs=[(i, j)])
	
	# remove pairs
	paths = [x for x in paths if len(x) > 2]

	#print('Filtered paths:', paths)
	if len(paths) < 2: # used to be 2
		return None  

	components = []
	cids = []
	for path in paths:
		#print('Trying path', path)
		new_vector = np.array([0.0, 0.0])
		cid = []
		broke = False
		for k in range(1, len(path)): 
			if not vh.is_missing(measured[path[k - 1]][path[k]]):
				#print('Found', path[k - 1], '-', path[k])
				new_vector += measured[path[k - 1]][path[k]]
				cid.append("M" + str(path[k - 1] + 1) + str(path[k] + 1))
				#j += 1
			else:
				broke = True 

		if not broke:#j == len(path) - 1:
			#print('Success! The new vector is', new_vector)
			components.append(new_vector.copy())
			cids.append(", ".join(cid))
			#print('Components:', components)
			#print('CIDs:', cids)

	#print('Components:', components)
	if len(components) < component_threshold: # used to be 2 
		return None  

	if verbose:
		print("[DEDUCED] M" + str(i + 1) + str(j + 1) + '(g):', 'sum(' + '; '.join(cids) + ') / ' + str(len(cids)))
	
	#print('Success! The new vector is', np.mean(components, axis=0))
	return np.mean(components, axis=0)

def add_noise(node, radius_noise=.2, angle_noise=.2):
	r = math.sqrt(node[0]**2 + node[1]**2)
	x_angle = math.acos(node[0] / r)
	y_angle = math.asin(node[1] / r)
	return (
		r * (1 + np.random.normal() * radius_noise) * math.cos(x_angle * (1 + np.random.normal() * angle_noise)), 
		r * (1 + np.random.normal() * radius_noise) * math.sin(y_angle * (1 + np.random.normal() * angle_noise))
	)

def drop_unseen_vectors(measured, true_nodes, space_size, max_angle=40, max_range=30, verbose=False):
	filtered = copy.deepcopy(measured)
	orientations = np.zeros((len(true_nodes), 2))

	# orientations[0] = [min(random.random() * space_size, space_size - true_nodes[0][0]), min(random.random() * space_size, space_size - true_nodes[0][1])]

	# for i in range(1, len(true_nodes)):
	# 	orientations[i] = np.mean(true_nodes[:i + 1], axis=0) - true_nodes[i] 
	# 	orientations[i][0] = min(space_size, orientations[i][0])
	# 	orientations[i][1] = min(space_size, orientations[i][1])

	for i in range(len(orientations)):
		orientations[i] = np.sum(np.concatenate([np.array(true_nodes)[:i, :], np.array(true_nodes)[i + 1:, :]]), axis=0) / (len(true_nodes) - 1)

	if verbose:
		print('Orientations:', orientations)
	
	true_vectors = vg.coords2vectors(true_nodes)

	for i in range(len(measured)):
		for j in range(len(measured)):
			if i != j:
				if np.sqrt(np.sum(measured[i][j]**2)) > max_range:
					filtered[i][j] = np.nan 
				else:
					curr_angle = math.acos(np.dot(orientations[i], true_vectors[i][j]) / (vh.get_magnitude(orientations[i]) * vh.get_magnitude(true_vectors[i][j]))) * 180 / math.pi
					if verbose:
						print(i, '->', j, ':', curr_angle)
					if curr_angle > max_angle:
						filtered[i][j] = np.nan 

	return filtered

