import copy
import random
import numpy as np
import vec_manipulator as vm 
import vec_helper as vh
import vec_generator as vg
import operator
import math
from numba import jit

DEFAULT_MUTATION_RATE = 0.2
DEFAULT_CROSSOVER_RATE = 0.8

def generate_population2(vectors, deletion_prob, size, min_diffs_ratio, paths, num_deduction_components, enforce_deduction=False):
	population = []
	ids = set()

	k = 0
	while len(population) < size and k < 10:
		temp = copy.deepcopy(vectors)
		for i in range(len(vectors)):
			for j in range(len(vectors)):
				if i != j and random.random() <= deletion_prob:
					temp[i][j] = np.nan

		to_proceed = not enforce_deduction or (vm.deduce_vectors(temp, paths, verbose=False, enforce_deduction=enforce_deduction) is not None)

		#if deduced is not None:
		# assume deduction is not important, meaning some vectors can be left ungenerated
		
		if to_proceed:
			curr_ids = vh.format_indices(vh.matrix2indices(temp))
			#curr_ids = set(vh.matrix2indices(temp))
			if curr_ids not in ids:
				if calculate_di(vg.get_generated_vectors(temp, paths, num_deduction_components), temp, paths=paths, min_diffs_ratio=min_diffs_ratio) is not None:
					population.append(temp)
					ids.add(curr_ids)
					k = 0

		k += 1

	return np.array(population)

def generate_population(vectors, deletion_prob, size, min_diffs_ratio, paths, num_deduction_components, di_type, enforce_deduction=False):
	population = []
	all_pairs = set(paths.keys())
	#print(all_pairs)
	pairs = set(paths.keys())
	to_keep = set()
	ids = set()

	most_consistent = vm.remove_inconsistent_vectors(vectors, num_deduction_components, min_diffs_ratio, paths)
	#print('Num most consistent vecs:', len(vh.matrix2indices(most_consistent)))
	#print('DI?', calculate_di(vg.get_generated_vectors(most_consistent, paths, num_deduction_components, enforce_deduction), most_consistent, paths=paths, min_diffs_ratio=min_diffs_ratio) is not None)
	mcdi = calculate_di(di_type, most_consistent, paths=paths, min_diffs_ratio=min_diffs_ratio) 
	print("Most consistent DI:", mcdi)

	assert mcdi is not None

	population.append(most_consistent)

	#consistent_vecs = vm.get_consistent_vectors(vectors)
	#print('Non-consistent vecs:', ", ".join(["M" + str(x[0] + 1) + str(x[1] + 1) for x in set(paths.keys()) - set(vh.matrix2indices(consistent_vecs))]))

	consistent_vecs2 = vm.get_consistent_vectors2(vectors, paths, num_deduction_components)
	print('Vectors deleted after the second heuristic:', ", ".join(sorted(["M" + str(x[0] + 1) + '-' + str(x[1] + 1) for x in set(paths.keys()) - set(vh.matrix2indices(consistent_vecs2))])))
	mcdi2 = calculate_di(di_type, consistent_vecs2, paths=paths, min_diffs_ratio=min_diffs_ratio) 
	print("Most consistent 2 DI:", mcdi2)

	assert mcdi2 is not None 
	population.append(consistent_vecs2)

	nodes = set([i for i in range(len(vectors))])
	while all_pairs:
		pivot = all_pairs.pop()
		#print(to_keep)
		#print(all_pairs)
		#print('Pivot:', pivot)
		candidates_to = set()
		candidates_from = set()

		for i in range(len(vectors)):
			if i not in pivot:
				candidates_to.add((pivot[0], i))
				candidates_from.add((i, pivot[1]))

		for i in range(max(1, int(min_diffs_ratio * (len(vectors) - 1))) + 1):
			chosen = None 
			chosen_to = candidates_to.intersection(to_keep)
			chosen_from = candidates_from.intersection(to_keep)
			#print("Candidates to:", chosen_to)
			#print("Candidates from:", chosen_from)
			if not chosen_to:
				if not chosen_from:
					if candidates_to:
						chosen = candidates_to.pop()[1]
					elif candidates_from:
						chosen = candidates_from.pop()[0]
					else:
						raise Exception('Not enough vectors for DI generation')
				else:
					chosen = chosen_from.pop()[0]
			else:
				if chosen_from:
					for ct in chosen_to:
						for cf in chosen_from:
							if ct[1] == cf[0]:
								chosen = ct[1]
								break
						if chosen is not None:
							break
				
				if chosen is None:
					if candidates_from:
						chosen = candidates_from.pop()[0]
					elif candidates_to:
						chosen = candidates_to.pop()[1]
					else:
						raise Exception('Not enough vectors for DI generation')
			#print('Chose', (pivot[0], chosen, pivot[1]))
			to_keep.add((pivot[0], chosen))
			to_keep.add((chosen, pivot[1]))
			#all_pairs -= set([(pivot[0], chosen), (chosen, pivot[1])])
			
			candidates_to -= set([(pivot[0], chosen)])
			candidates_from -= set([(chosen, pivot[1])])
	
	temp = copy.deepcopy(vectors)
	for p in pairs - to_keep:
		temp[p[0]][p[1]] = np.nan

	for i in range(len(vectors)):
		for j in range(len(vectors)):
			if vh.is_missing(temp[i][j]) and not vm.is_deducible(i, j, temp, num_deduction_components):
				comps = 0
				for k in range(len(vectors)):
					if (i, k) in to_keep:
						to_keep.add((k, j))
						temp[k][j] = vectors[k][j].copy()
					elif (k, j) in to_keep:
						to_keep.add((i, k))
						temp[i][k] = vectors[i][k].copy()
					else:
						to_keep.add((i, k))
						to_keep.add((k, j))
						temp[i][k] = vectors[i][k].copy()
						temp[k][j] = vectors[k][j].copy()
					
					comps += 1
					if comps >= num_deduction_components:
						break

	#to_keep = to_keep.union(vh.matrix2indices(consistent_vecs))

	print('Number of vectors to keep:', len(to_keep), 'out of', len(vectors) ** 2 - len(vectors))	
	
	#print(vh.beautify_matrix(vm.deduce_vectors(temp, paths, num_deduction_components, enforce_deduction=enforce_deduction)))
	#print('DI?', calculate_di(vg.get_generated_vectors(temp, paths, num_deduction_components), temp, paths=paths, min_diffs_ratio=min_diffs_ratio) is not None)

	optional = pairs - to_keep
	k = 0
	while len(population) < size and k < 100:
		new_set = to_keep.union(random_subset(optional, deletion_prob=deletion_prob))
		#print(len(new_set))
		new_id = vh.format_indices(sorted(list(new_set), key=lambda item: (item[0], item[1])))
		if new_id not in ids:
			temp = copy.deepcopy(vectors)
			for i in range(len(temp)):
				for j in range(len(temp)):
					if i != j and (i, j) not in new_set:
						temp[i][j] = np.nan
			
			to_proceed = not enforce_deduction or (vm.deduce_vectors(temp, paths, verbose=False, enforce_deduction=enforce_deduction, num_deduction_components=num_deduction_components) is not None)

			# assume deduction is not important, meaning some vectors can be left ungenerated
			if to_proceed:
				#print('DEDUCTION SUCCESS')
				di = calculate_di(di_type, temp, paths=paths, min_diffs_ratio=min_diffs_ratio) 
				if di is not None: # vg.get_generated_vectors(temp, paths, num_deduction_components)
					#print('DI SUCCESS')
					population.append(temp)
					ids.add(new_id)
					k = 0
		k += 1

	return np.array(population)

def random_subset(s, deletion_prob):
	return set(filter(lambda x: random.random() < deletion_prob, s))

def tournament_selection(population, population_fitness, num_elites, k=2):
	selection = []

	# preserve the elite
	for i in range(num_elites):
		try:
			selection.append(population_fitness[i][0])
		except:
			print('Population fitness:', population_fitness)
			raise Exception('Not enough chromosomes for selection')

	for _ in range(len(population) - num_elites):
		b = random.randint(0, len(population_fitness) - 1)
		for i in np.random.randint(0, len(population_fitness), k - 1):
			if population_fitness[i][1] > population_fitness[b][1]:
				b = i 
		selection.append(population_fitness[b][0])
	
	return [population[i] for i in selection]

def get_population_fitness(population, min_diffs_ratio, paths, num_deduction_components, di_type, verbose=False):
	population_fitness = dict()
	for i, individual in enumerate(population):
		#deduced = deduce_vectors(individual)
		if verbose:
			print('Chromosome:', vh.format_indices(vh.matrix2indices(individual)), f'(Missing: {vh.format_indices(vh.get_missing(individual))})')
		di = calculate_di(
			di_type,
			individual, 
			paths=paths,
			verbose=verbose, 
			min_diffs_ratio=min_diffs_ratio
		) 

		if di is None:
			#print(vh.format_indices(vh.matrix2indices(individual)))
			#print(vh.beautify_matrix(individual))
			raise Exception('Cannot calculate DI for one chromosome. Something went wrong during reproduction or initial population generation')

		if di == 0:
			population_fitness[i] = float('inf')
		else:
			population_fitness[i] = 1 / di
	
	# for k, v in population_fitness.items():
	# 	if v == 0:
	# 		population_fitness[k] = float('inf')
	# 	else:
	# 		population_fitness[k] = 1 / v
	return sorted(population_fitness.items(), key=operator.itemgetter(1), reverse=True)

def generate_offspring(
	selection, 
	num_elites, 
	min_diffs_ratio,
	enforce_deduction,
	paths,
	space_size,
	measured,
	num_deduction_components,
	di_type,
	mutation_rate=DEFAULT_MUTATION_RATE, 
	crossover_rate=DEFAULT_CROSSOVER_RATE,
	variations=None,
	mutation_percent_change=.3,
	verbose=False,
	mutation_type=4,
	inner_workings=False,
	faster=False
):
	if len(selection) < 2:
		return selection
	offspring = []

	# preserve the elite
	for i in range(num_elites):
		offspring.append(selection[i])

	k = 0
	while len(offspring) < len(selection) and k < 100:
		i1 = i2 = -1
		while i1 == i2:
			i1 = random.randint(0, len(selection) - 1)
			i2 = random.randint(0, len(selection) - 1)
		
		# crossover
		if inner_workings:
			print('[INFO] Attempting crossover...')
		child = crossover(selection[i1], selection[i2], crossover_rate, paths, num_deduction_components, variations=variations, verbose=verbose)

		to_proceed = not enforce_deduction or (vm.deduce_vectors(child, paths, num_deduction_components, verbose=False, enforce_deduction=enforce_deduction) is not None)
		
		if to_proceed:
			# mutation
			if inner_workings:
				print('[INFO] Attempting mutation...')
			#child = mutation3(child, mutation_rate, mutation_percent_change, min_diffs_ratio, paths, space_size)
			if mutation_type == 4:
				child = mutation4(child, mutation_rate, mutation_percent_change, min_diffs_ratio, paths, space_size)
			elif mutation_type == 3:
				child = mutation3(child, mutation_rate, mutation_percent_change, min_diffs_ratio, paths, space_size)
			elif mutation_type == 5:
				child = mutation5(measured, child, mutation_rate, mutation_percent_change, min_diffs_ratio, paths, space_size)
			
			di = calculate_di(
				di_type, 
				child, 
				paths=paths, 
				min_diffs_ratio=min_diffs_ratio, 
				faster=faster
			)

			if di is not None:
				offspring.append(child)
			
				if inner_workings:
					print('[INFO] One reproduction instance successful')
				
		k += 1

	if len(offspring) > len(selection):
		offspring = offspring[:len(selection)]
	
	return offspring

def crossover(p1, p2, crossover_rate, paths, num_deduction_components, verbose=False, variations=None):
	if random.random() < crossover_rate:
		v1 = vh.matrix2indices(p1)
		v2 = vh.matrix2indices(p2)

		#np.random.shuffle(v1)
		#np.random.shuffle(v2)
		i1 = random.randint(0, len(v1) - 1)
		i2 = random.randint(0, len(v2) - 1)

		
		indexes = set(v1[:i1]).union(set(v2[i2:]))

		child = np.zeros((len(p1), len(p1), 2))
		child[:] = np.nan

		for i in range(len(child)):
			child[i][i] = [0, 0]

		for (i, j) in indexes:
			if not vh.is_missing(p1[i][j]):
				child[i][j] = p1[i][j].copy()
			else:
				child[i][j] = p2[i][j].copy()
		
		if verbose:
			print('CROSSOVER:')
			print('Parent 1:', vh.format_indices(v1))
			print('Parent 2:', vh.format_indices(v2))
			print('Cutting before index', i1, 'at parent 1')
			print('Cutting after index', i2, 'at parent 2')
			print('--> Child:', vh.format_indices(vh.matrix2indices(child)))

		deduced = vm.deduce_vectors(child, paths, num_deduction_components)
		if deduced is not None and variations:
			choice = random.sample(variations, 1)[0]
			if choice == "deduced":
				return deduced
			if choice == "generated-deduced":
				return vg.get_generated_vectors(deduced, paths, num_deduction_components, enforce_deduction)
			if choice == "generated":
				return vg.get_generated_vectors(child, paths, num_deduction_components, enforce_deduction)
		return child 
	if random.random() < 0.5:
		return copy.deepcopy(p1)
	return copy.deepcopy(p2)

def mutation(individual, mutation_rate, percent_change, min_diffs_ratio, paths, verbose=False):
	temp = copy.deepcopy(individual)
	for i in range(len(temp)):
		for j in range(len(temp)):
			if i != j and not vh.is_missing(temp[i][j]) and random.random() <= mutation_rate:
				prev_di = calculate_di(vg.get_generated_vectors(temp, paths=paths), temp, paths=paths, min_diffs_ratio=min_diffs_ratio)
				temp[i][j][0] = (1 + percent_change) * individual[i][j][0]
				di = calculate_di(vg.get_generated_vectors(temp, paths=paths), temp, paths=paths, min_diffs_ratio=min_diffs_ratio)
				if di > prev_di:
					temp[i][j][0] = (1 - percent_change) * individual[i][j][0]
					di = calculate_di(vg.get_generated_vectors(temp, paths=paths), temp, paths=paths, min_diffs_ratio=min_diffs_ratio)
					if di > prev_di:
						temp[i][j][0] = individual[i][j][0]
					else:
						prev_di = di
				else:
					prev_di = di

				temp[i][j][1] = (1 + percent_change) * individual[i][j][1]
				di = calculate_di(vg.get_generated_vectors(temp, paths=paths), temp, paths=paths, min_diffs_ratio=min_diffs_ratio)
				if di > prev_di:
					temp[i][j][1] = (1 - percent_change) * individual[i][j][1]
					di = calculate_di(vg.get_generated_vectors(temp, paths=paths), temp, paths=paths, min_diffs_ratio=min_diffs_ratio)
					if di > prev_di:
						temp[i][j][1] = individual[i][j][1]
	return temp

def mutation2(individual, mutation_rate, percent_change, min_diffs_ratio, paths, verbose=False):
	temp = copy.deepcopy(individual)
	for i in range(len(temp)):
		for j in range(len(temp)):
			if i != j and not vh.is_missing(temp[i][j]) and random.random() <= mutation_rate:
				if random.random() <= 0.5:
					temp[i][j] *= (1 + percent_change) 
				else:
					temp[i][j] *= (1 - percent_change) 
				
	return temp

def mutation3(individual, mutation_rate, percent_change, min_diffs_ratio, paths, space_size, verbose=False):
	temp = copy.deepcopy(individual)
	for i in range(len(temp)):
		for j in range(len(temp)):
			if i != j and not vh.is_missing(temp[i][j]) and random.random() <= mutation_rate:
				t = random.random()
				if t <= 1/3:
					temp[i][j] = vm.add_noise(temp[i][j], space_size, radius_noise=0, angle_noise=-percent_change)
				elif t > 2/3:
					temp[i][j] = vm.add_noise(temp[i][j], space_size, radius_noise=-percent_change, angle_noise=-percent_change)
				else:
					temp[i][j] = vm.add_noise(temp[i][j], space_size, radius_noise=-percent_change, angle_noise=0)
	return temp

def mutation4(individual, mutation_rate, percent_change, min_diffs_ratio, paths, space_size, verbose=False):
	temp = copy.deepcopy(individual)
	prev_di = calculate_di(vg.get_generated_vectors(temp, paths=paths), temp, paths=paths, min_diffs_ratio=min_diffs_ratio)
	for i in range(len(temp)):
		for j in range(len(temp)):
			if i != j and not vh.is_missing(temp[i][j]) and random.random() <= mutation_rate:
				#print('Before:', vh.beautify_matrix(temp))
				temp[i][j] = vm.add_noise(individual[i][j], space_size, radius_noise=0, angle_noise=-percent_change)
				#print('After:', vh.beautify_matrix(temp))
				di = calculate_di(vg.get_generated_vectors(temp, paths=paths), temp, paths=paths, min_diffs_ratio=min_diffs_ratio)
				if di > prev_di:
					temp[i][j] = vm.add_noise(individual[i][j], space_size, radius_noise=-percent_change, angle_noise=-percent_change)
					di = calculate_di(vg.get_generated_vectors(temp, paths=paths), temp, paths=paths, min_diffs_ratio=min_diffs_ratio)
					if di > prev_di:
						temp[i][j] = vm.add_noise(individual[i][j], space_size, radius_noise=-percent_change, angle_noise=0)
						di = calculate_di(vg.get_generated_vectors(temp, paths=paths), temp, paths=paths, min_diffs_ratio=min_diffs_ratio)
						if di > prev_di:
							temp[i][j] = individual[i][j]
	return temp

def mutation5(measured, individual, mutation_rate, percent_change, min_diffs_ratio, paths, verbose=False):
	temp = copy.deepcopy(individual)
	for i in range(len(temp)):
		for j in range(len(temp)):
			if i != j and vh.is_missing(temp[i][j]) and random.random() <= mutation_rate:
				temp[i][j] = measured[i][j].copy()
	return temp

def calculate_di2(generated, measured, min_diffs_ratio, paths, faster=False, verbose=False):
	di = 0
	generated = np.array(generated)
	measured = np.array(measured)
	num_vectors = len(vh.matrix2indices(measured))
	if verbose:
		print('Calculating DI')
	
	min_required_components = 2 #max(2, int(min_diffs_ratio * (len(measured) - 1)))
	for i in range(len(generated)):
		for j in range(len(generated)):
			c1 = ""
			curr = 0
			if i != j and not vh.is_missing(measured[i][j]):
				curr += np.sum((measured[i][j] - generated[i][j])**2)
				num_perms = 1
				
				if verbose:
					print("M" + str(i + 1) + str(j + 1) + ": " + f"sqrt(({measured[i][j]} - {generated[i][j]})**2", end="")
					c1 += "M" + str(i + 1) + str(j + 1) + ": " + "sqrt(((M" + str(i + 1) + str(j + 1) + "(m) - " + "M" + str(i + 1) + str(j + 1) + "(g))**2 "
				
				for path in paths[(i, j)]:
					if len(path) > 2 and vh.is_valid(path, measured):
						if verbose:
							print(f" + ({generated[i][j]} - (", end="")
							c1 += " + (M" + str(i + 1) + str(j + 1) + "(g) - " + "M" + str(i + 1) + str(j + 1) + f"(g{','.join([str(p + 1) for p in path])}))**2 "
						curr += np.sum((generated[i][j] - vg.get_generated_vector(path, measured, verbose=verbose))**2)
						
						if verbose:
							print("))**2", end="")
							
						num_perms += 1

						if faster and num_perms >= min_required_components:
							break

				if num_perms < min_required_components:
					#print('NOT ENOUGH PERMS:', num_perms[(i,j)])
					return None

				curr /= num_perms
				di += math.sqrt(curr)

				if verbose:
					print(f') / {num_perms})')
					c1 += f') / {num_perms})'
					
			if c1 != "":
				print(c1)
				
	return di / num_vectors #(len(generated)**2 - len(generated))

def calculate_di(di_type, measured, min_diffs_ratio, paths, enforce_deduction=True, num_deduction_components=2, faster=False, verbose=False):
	if di_type == "generated":
		deduced = vm.deduce_vectors(measured, paths, num_deduction_components, verbose=False, enforce_deduction=True)
		generated = vg.get_generated_vectors(deduced, paths, num_deduction_components)
		measured = copy.deepcopy(generated)#deduced)
		num_vectors = len(measured) ** 2 - len(measured) #
	else:
		measured = vm.deduce_vectors(measured, paths, num_deduction_components, verbose=False, enforce_deduction=True)
		generated = vg.get_generated_vectors(measured, paths, num_deduction_components)
		num_vectors = len(vh.matrix2indices(measured))
	
	di = 0
	
	if verbose:
		print('Calculating DI')
	
	min_required_components = max(2, int(min_diffs_ratio * (len(measured) - 1))) # set min_diffs_ratio to 0 to disable constraints on the number of required components
	
	for i in range(len(generated)):
		for j in range(len(generated)):
			c1 = ""
			curr = 0
			if i != j:
				if (di_type == "generated" and vh.is_missing(measured[i][j])):
					raise Exception('Some vectors are missing for DI calculation')
				curr += np.sum((measured[i][j] - generated[i][j])**2)
				num_perms = 1
				
				if verbose:
					print("M" + str(i + 1) + str(j + 1) + ": " + f"sqrt(({measured[i][j]} - {generated[i][j]})**2", end="")
					c1 += "M" + str(i + 1) + str(j + 1) + ": " + "sqrt(((M" + str(i + 1) + str(j + 1) + "(m) - " + "M" + str(i + 1) + str(j + 1) + "(g))**2 "
				
				for path in paths[(i, j)]:
					if len(path) > 2 and ((di_type == "measured" and vh.is_valid(path, measured)) or di_type == "generated"):
						if verbose:
							print(f" + ({generated[i][j]} - (", end="")
							c1 += " + (M" + str(i + 1) + str(j + 1) + "(g) - " + "M" + str(i + 1) + str(j + 1) + f"(g{','.join([str(p + 1) for p in path])}))**2 "

						intermediary = vg.get_generated_vector(path, measured, verbose=verbose)
						curr += np.sum((generated[i][j] - intermediary)**2)
						
						if verbose:
							print("))**2", end="")
							
						num_perms += 1

						if faster and num_perms >= min_required_components:
							break

				if num_perms < min_required_components:
					return None

				curr /= num_perms
				di += math.sqrt(curr)

				if verbose:
					print(f') / {num_perms})')
					c1 += f') / {num_perms})'
					
			if c1 != "":
				print(c1)
				
	return di / num_vectors 

#@jit(nopython=True)
def calculate_di_measured(generated, measured, min_diffs_ratio, paths, faster=False, verbose=False):
	di = 0
	generated = np.array(generated)
	measured = np.array(measured)
	#temp = vh.generate_paths(len(generated))
	#paths = vh.paths2dict(temp)
	num_vectors = len(vh.matrix2indices(measured))
	if verbose:
		print('Calculating DI')
	#deduced = deduce_vectors(measured)
	#if deduced is None:
		#raise Exception('Error in calculating DI')
	min_required_components = max(2, int(min_diffs_ratio * (len(measured) - 1)))
	for i in range(len(generated)):
		for j in range(len(generated)):
			c1, c2 = "", ""
			curr = 0
			if i != j and not vh.is_missing(measured[i][j]):
				curr += np.sum((measured[i][j] - generated[i][j])**2)
				num_perms = 1
				if verbose:
					#print("M" + str(i + 1) + str(j + 1) + ": " + "sqrt(((M" + str(i + 1) + str(j + 1) + "(m) - " + "M" + str(i + 1) + str(j + 1) + "(g))**2", end='')
					print("M" + str(i + 1) + str(j + 1) + ": " + f"sqrt(({measured[i][j]} - {generated[i][j]})**2", end="")

					c1 += "M" + str(i + 1) + str(j + 1) + ": " + "sqrt(((M" + str(i + 1) + str(j + 1) + "(m) - " + "M" + str(i + 1) + str(j + 1) + "(g))**2 "
					#c2 += "M" + str(i + 1) + str(j + 1) + ": " + f"sqrt((({measured[i][j]} - {generated[i][j]})**2 "
				for path in paths[(i, j)]:
					if len(path) > 2 and vh.is_valid(path, measured):
						if verbose:
							#print(" + (M" + str(i + 1) + str(j + 1) + "(g) - " + "M" + str(i + 1) + str(j + 1) + f"(g{','.join([str(p + 1) for p in path])}))**2", end='')
							print(f" + ({generated[i][j]} - (", end="")
							c1 += " + (M" + str(i + 1) + str(j + 1) + "(g) - " + "M" + str(i + 1) + str(j + 1) + f"(g{','.join([str(p + 1) for p in path])}))**2 "
							#c2 += f" + ({generated[i][j]} - " + f"{generated[i][j]}" + f"(g{','.join([str(p + 1) for p in path])}))**2 "
						curr += np.sum((generated[i][j] - vg.get_generated_vector(path, measured, verbose=verbose))**2)
						
						if verbose:
							print("))**2", end="")
							
						num_perms += 1

						if faster and num_perms >= min_required_components:
							break

				if num_perms < min_required_components:
					#print('NOT ENOUGH PERMS:', num_perms[(i,j)])
					return None

				curr /= num_perms
				di += math.sqrt(curr)

				if verbose:
					print(f') / {num_perms})')
					c1 += f') / {num_perms})'
					#c2 += f') / {num_perms[(i, j)]})'
			if c1 != "":
				print(c1)
				#print(c2)
		#print(c2)

	return di / num_vectors #(len(generated)**2 - len(generated))
