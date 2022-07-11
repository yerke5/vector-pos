import copy
import random
import numpy as np
import vec_manipulator as vm 
import vec_helper as vh
import vec_generator as vg
import operator
import math

DEFAULT_MUTATION_RATE = 0.01
DEFAULT_CROSSOVER_RATE = 0.8

def generate_population(vectors, deletion_prob=.3, size=10):
	population = []
	ids = set()

	k = 0
	while len(population) < size and k < 100:
		temp = copy.deepcopy(vectors)
		for i in range(len(vectors)):
			for j in range(len(vectors)):
				if i != j and random.random() <= deletion_prob:
					temp[i][j] = np.nan

		deduced = vm.deduce_vectors(temp, verbose=False)
		if deduced is not None:
			curr_ids = vh.format_indices(vh.matrix2indices(temp))
			if curr_ids not in ids:
				population.append(temp)
				ids.add(curr_ids)

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

def get_population_fitness(population, verbose=False):
	population_fitness = dict()
	for i, individual in enumerate(population):
		#deduced = deduce_vectors(individual)
		if verbose:
			print('Chromosome:', vh.format_indices(vh.matrix2indices(individual)), f'(Missing: {vh.format_indices(vh.get_missing(individual))})')
		di = calculate_di(vg.get_generated_vectors(individual, verbose=verbose), individual) 
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
	mutation_rate=DEFAULT_MUTATION_RATE, 
	crossover_rate=DEFAULT_CROSSOVER_RATE,
	verbose=False
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
		child = crossover(selection[i1], selection[i2], crossover_rate, verbose=verbose)

		# mutation

		deduced = vm.deduce_vectors(child, verbose=False)
		if deduced is not None:
			offspring.append(child)
			
		k += 1

	if len(offspring) > len(selection):
		offspring = offspring[:len(selection)]
	
	return offspring

def crossover(p1, p2, crossover_rate, verbose=False):
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

		deduced = vm.deduce_vectors(child)
		if deduced is not None:
			variations = {"raw"}#"generated", "deduced"}#, "generated-deduced"}
			choice = random.sample(variations, 1)[0]
			if choice == "deduced":
				return deduced
			if choice == "generated-deduced":
				return vg.get_generated_vectors(deduced)
			if choice == "generated":
				return vg.get_generated_vectors(child)
		return child 
	if random.random() < 0.5:
		return copy.deepcopy(p1)
	return copy.deepcopy(p2)

	

def calculate_di(generated, measured, verbose=False):
	di = 0
	generated = np.array(generated)
	measured = np.array(measured)
	temp = vh.generate_paths(len(generated))
	num_perms = dict()
	paths = vh.paths2dict(temp)
	num_vectors = len(vh.matrix2indices(measured))
	if verbose:
		print('Calculating DI')
	#deduced = deduce_vectors(measured)
	#if deduced is None:
		#raise Exception('Error in calculating DI')
	for i in range(len(generated)):
		for j in range(len(generated)):
			curr = 0
			if i != j and not vh.is_missing(measured[i][j]):
				curr += np.sum((measured[i][j] - generated[i][j])**2)
				num_perms[(i, j)] = 1
				if verbose:
					print("M" + str(i + 1) + str(j + 1) + ": " + "sqrt(((M" + str(i + 1) + str(j + 1) + "(m) - " + "M" + str(i + 1) + str(j + 1) + "(g))**2", end='')
				for path in paths[(i, j)]:
					if len(path) > 2 and vh.is_valid(path, measured):
						if verbose:
							print(" + (M" + str(i + 1) + str(j + 1) + "(g) - " + "M" + str(i + 1) + str(j + 1) + f"(g{','.join([str(p + 1) for p in path])}))**2", end='')
						curr += np.sum((generated[i][j] - vg.get_generated_vector(path, measured))**2)
						
						num_perms[(i, j)] += 1
				curr /= num_perms[(i, j)]
				di += math.sqrt(curr)

				if verbose:
					print(f') / {num_perms[(i, j)]})')

	return di / num_vectors #(len(generated)**2 - len(generated))
