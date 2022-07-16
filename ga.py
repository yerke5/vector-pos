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

def generate_population(vectors, deletion_prob, size, min_diffs_ratio, paths, enforce_deduction=False):
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
				if calculate_di(vg.get_generated_vectors(temp, paths=paths), temp, paths=paths, min_diffs_ratio=min_diffs_ratio) is not None:
					population.append(temp)
					ids.add(curr_ids)
					k = 0

		k += 1

	#print(ids)
	return np.array(population)

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

def get_population_fitness(population, min_diffs_ratio, paths, verbose=False):
	population_fitness = dict()
	for i, individual in enumerate(population):
		#deduced = deduce_vectors(individual)
		if verbose:
			print('Chromosome:', vh.format_indices(vh.matrix2indices(individual)), f'(Missing: {vh.format_indices(vh.get_missing(individual))})')
		di = calculate_di(
			vg.get_generated_vectors(individual, paths=paths, verbose=verbose), 
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
	
	#population_fitness = {i: calculate_di(get_generated_vectors(population[i]), population[i]) for i in range(len(population))}

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
	mutation_rate=DEFAULT_MUTATION_RATE, 
	crossover_rate=DEFAULT_CROSSOVER_RATE,
	mutation_percent_change=.3,
	verbose=False,
	mutation_type=4
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
		print('[INFO] Attempting crossover...')
		child = crossover(selection[i1], selection[i2], crossover_rate, paths=paths, verbose=verbose)

		to_proceed = not enforce_deduction or (vm.deduce_vectors(child, paths=paths, verbose=False, enforce_deduction=enforce_deduction) is not None)
		
		if to_proceed:
			di = calculate_di(vg.get_generated_vectors(child, paths=paths), child, paths=paths, min_diffs_ratio=min_diffs_ratio)
			if di is not None:
				# mutation
				print('[INFO] Attempting mutation...')
				#child = mutation3(child, mutation_rate, mutation_percent_change, min_diffs_ratio, paths, space_size)
				if mutation_type == 4:
					child = mutation4(child, mutation_rate, mutation_percent_change, min_diffs_ratio, paths, di, space_size)
				elif mutation_type == 3:
					child = mutation3(child, mutation_rate, mutation_percent_change, min_diffs_ratio, paths, space_size)
				print('[INFO] One reproduction instance successful')
				offspring.append(child)
			
		k += 1

	if len(offspring) > len(selection):
		offspring = offspring[:len(selection)]
	
	return offspring

def crossover(p1, p2, crossover_rate, paths, verbose=False):
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

		deduced = vm.deduce_vectors(child, paths=paths)
		if deduced is not None:
			variations = {"raw"}#"generated", "deduced"}#, "generated-deduced"}
			# choice = random.sample(variations, 1)[0]
			# if choice == "deduced":
			# 	return deduced
			# if choice == "generated-deduced":
			# 	return vg.get_generated_vectors(deduced)
			# if choice == "generated":
			# 	return vg.get_generated_vectors(child)
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

def mutation4(individual, mutation_rate, percent_change, min_diffs_ratio, paths, prev_di, space_size, verbose=False):
	temp = copy.deepcopy(individual)
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

#@jit(nopython=True)
def calculate_di(generated, measured, min_diffs_ratio, paths, verbose=False):
	di = 0
	generated = np.array(generated)
	measured = np.array(measured)
	#temp = vh.generate_paths(len(generated))
	num_perms = dict()
	#paths = vh.paths2dict(temp)
	num_vectors = len(vh.matrix2indices(measured))
	if verbose:
		print('Calculating DI')
	#deduced = deduce_vectors(measured)
	#if deduced is None:
		#raise Exception('Error in calculating DI')
	for i in range(len(generated)):
		for j in range(len(generated)):
			c1, c2 = "", ""
			curr = 0
			if i != j and not vh.is_missing(measured[i][j]):
				curr += np.sum((measured[i][j] - generated[i][j])**2)
				num_perms[(i, j)] = 1
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
							
						num_perms[(i, j)] += 1

				if num_perms[(i,j)] < max(2, int(min_diffs_ratio * (len(measured) - 1))):
					return None

				curr /= num_perms[(i, j)]
				di += math.sqrt(curr)

				if verbose:
					print(f') / {num_perms[(i, j)]})')
					c1 += f') / {num_perms[(i, j)]})'
					#c2 += f') / {num_perms[(i, j)]})'
			if c1 != "":
				print(c1)
				#print(c2)
		#print(c2)

	return di / num_vectors #(len(generated)**2 - len(generated))
