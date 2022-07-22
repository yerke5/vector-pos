import math 
import copy
import random 
import numpy as np
import vec_manipulator as vm
import vec_helper as vh

def get_generated_vectors(measured, paths, num_deduction_components, max_generation_paths=None, verbose=False):
	generated = np.zeros((len(measured), len(measured), 2))
	
	#print('Paths:', paths)
	num_perms = dict()
	strpaths = dict()

	#print('Number of paths:', perms)
	for pair in paths:
		if verbose and not vh.is_missing(measured[pair[0]][pair[1]]):
			print("[NOT DEDUCED] M" + str(pair[0] + 1) + str(pair[1] + 1) + "(g): ", end="")
		num_paths = 0
		if not vh.is_missing(measured[pair[0]][pair[1]]):
			for path in paths[pair]:
					gen = get_generated_vector(path, measured, verbose=verbose) 
					
					if gen is not None:
						generated[pair[0]][pair[-1]] += gen
						if not pair in num_perms:
							num_perms[pair] = 1

							if verbose:
								strpaths[pair] = [''.join([str(p + 1) for p in path])]
						else:
							num_perms[pair] += 1

							if verbose:
								strpaths[pair].append(''.join([str(p + 1) for p in path]))
		else:
			deduced = vm.deduce_vector(pair[0], pair[1], measured, paths[pair], num_deduction_components, verbose=verbose)
			if deduced is not None:
				generated[pair[0]][pair[1]] = deduced
			else:
				generated[pair[0]][pair[1]] = np.nan
	
			num_perms[pair] = 1
			#generated[path[0]][path[-1]] /= (perms + 1)
			#print(f"After {path[0] + 1} - {path[-1] + 1}: ", generated[path[0]][path[-1]])
			#print('-' * 50)

		if not vh.is_missing(measured[pair[0]][pair[1]]):
			generated[pair[0]][pair[1]] /= num_perms[pair]
			if verbose:
				print("/", num_perms[pair], '=', generated[pair[0]][pair[1]])

	for i in range(len(generated)):
		for j in range(len(generated)):
			if i != j:
				if verbose and (i, j) in strpaths:
					print("[NOT DEDUCED] M" + str(i + 1) + str(j + 1) + "(g): sum(" + "; ".join(strpaths[(i, j)]) + ") / " + str(num_perms[(i, j)]))
					#pass
				#generated[i][j] = generated[i][j] / num_perms[(i, j)]#(num_perms)#(perms + 1)
	return generated

def get_generated_vector(path, measured, verbose=False):
	new_vector = np.array([0.0, 0.0])
	vecstr = []
	for i in range(1, len(path)): 
		if vh.is_missing(measured[path[i - 1]][path[i]]):
			return None 
		#print(f"{path[i-1] + 1} - {path[i] + 1}: {measured[path[i - 1]][path[i]]}")
		new_vector += measured[path[i - 1]][path[i]]

		if verbose:
			vecstr.append(f"{measured[path[i - 1]][path[i]]}")
	
	if verbose:
		print("(" + " + ".join(vecstr) + ")" + " + ", end = "")

	return new_vector

def generate_random_node(size):
	return [random.random() * size, random.random() * size]

def coords2vectors(coords, space_size, angle_noise=.3, radius_noise=.3, noise_ratio=0, verbose=True):
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
				#print('Before:', coords[i], vectors[i][j])
				#print(space_size - coords[i])
				noisy_vector = list(vm.add_noise(vectors[i][j], space_size, angle_noise=angle_noise, radius_noise=radius_noise))
				noisy_vector[0] = min(space_size - coords[i][0], noisy_vector[0])
				noisy_vector[1] = min(space_size - coords[i][1], noisy_vector[1])
				noisy_vector[0] = max(-coords[i][0], noisy_vector[0])
				noisy_vector[1] = max(-coords[i][1], noisy_vector[1])
				vectors[i][j] = noisy_vector
				#print('After:', coords[i], vectors[i][j])

	if verbose and noise:
		print('Noisy vectors:', ", ".join(sorted(["M" + str(x[0] + 1) + '-' + str(x[1] + 1) for x in pairs])))
	return vectors 

