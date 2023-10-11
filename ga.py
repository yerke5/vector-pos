import copy
import random
import numpy as np
import vec_manipulator as vm 
import vec_helper as vh
import vec_generator as vg
import consistency_checker as vc
import operator
import math
import random

DEFAULT_MUTATION_RATE = 0.2
DEFAULT_CROSSOVER_RATE = 0.8
MAX_ITERS_CONSISTENCY_CHECK = 10

class GA:
	def __init__(
		self, 
		measured,
		population_size, 
		iters,
		vecgen,
		conscheck,
		elite_proportion=.1,
		crossover_rate=.8,
		mutation_rate=.1, 
		crossover_type=2, 
		mutation_type=6, 
		deletion_prob=.3,
		max_init_consistency_shapes=30,
		max_supergene_shapes=10,
		max_di_components=5,
		min_di_components=1,
		missing_vectors_weight=.2,
		verbose=False,
		init_consistent_shapes_ratio = .3,
		init_consistency_shape_delta=.05
	):
		self.population_size = population_size
		self.vector_generator = vecgen
		self.consistency_checker = conscheck
		self.crossover_rate = crossover_rate
		self.crossover_type = crossover_type 
		self.mutation_rate = mutation_rate
		self.mutation_type = mutation_type 
		self.verbose = verbose
		self.max_init_consistency_shapes = max_init_consistency_shapes
		self.max_supergene_shapes = max_supergene_shapes
		self.num_elites = int(elite_proportion * population_size)
		self.measured = copy.deepcopy(measured)
		self.deletion_prob = deletion_prob
		self.iters = iters
		self.max_di_components = max_di_components
		self.min_di_components = min_di_components
		self.missing_vectors_weight = missing_vectors_weight
		self.init_consistent_shapes_ratio = init_consistent_shapes_ratio
		self.init_consistency_shape_delta = init_consistency_shape_delta
		
	def generate_population(self):
		vectors = self.measured
		population = [copy.deepcopy(self.measured)]
		all_pairs = set(self.vector_generator.params.generation_paths.keys())
		available_pairs = all_pairs - set(vh.get_missing(vectors))
		population_fitness = {0: self.calculate_di(vectors)}

		if self.vector_generator.params.consistency_paths is not None:
			k = 0
			while len(population) < self.population_size * self.init_consistent_shapes_ratio and k < 100:
				consistent_vectors = self.consistency_checker.get_consistency_shapes(vectors, max_shapes=self.max_init_consistency_shapes, delta=self.init_consistency_shape_delta)
				
				# add random missing vectors
				rvs = self.random_vectors(available_pairs)
				for rv in rvs:
					if vh.is_missing(consistent_vectors[rv[0]][rv[1]]):
						consistent_vectors[rv[0]][rv[1]] = self.measured[rv[0]][rv[1]].copy()

				di = self.calculate_di(consistent_vectors)
				if di is not None: 
					population.append(consistent_vectors)
					population_fitness[len(population) - 1] = 1 / di if di != 0 else float('inf')
					k = 0
				k += 1
				
		print("[INFO] Finished searching for consistent shapes; current population size:", len(population))

		k = 0
		while len(population) < self.population_size and k < 100:
			deleted_pairs = self.random_subset(available_pairs)
			ind = copy.deepcopy(vectors)
			for pair in deleted_pairs:
				ind[pair[0]][pair[1]] = np.nan
			population.append(ind)
			population_fitness[len(population) - 1] = self.calculate_di(ind)
			k += 1

		print("[INFO] Finished searching for random chromosomes; current population size:", len(population))
		return np.array(population), sorted(population_fitness.items(), key=operator.itemgetter(1), reverse=True)

	def random_subset(self, s):
		return set(filter(lambda x: random.random() < self.deletion_prob, s))

	def random_vectors(self, s):
		out = set()
		for el in s:                                                                                                                    
			# random coin flip
			if random.randint(0, 1) == 0:
				out.add(el)
		return out

	def tournament_selection(self, population, population_fitness, k=2):
		selection = []

		# preserve the elite
		for i in range(self.num_elites):
			try:
				selection.append(population_fitness[i][0])
			except:
				print('Population fitness:', population_fitness)
				raise Exception('Not enough chromosomes for selection')

		for _ in range(len(population) - self.num_elites):
			b = random.randint(0, len(population_fitness) - 1)
			for i in np.random.randint(0, len(population_fitness), k - 1):
				if population_fitness[i][1] > population_fitness[b][1]:
					b = i 
			selection.append(population_fitness[b][0])
		
		return [population[i] for i in selection]

	def generate_offspring(
		self,
		selection, 
		inner_workings=False
	):
		population_fitness = dict()
		if len(selection) < 2:
			return selection
		offspring = []

		# preserve the elite
		for i in range(self.num_elites):
			di = self.calculate_di(selection[i])
			population_fitness[i] = 1 / di if di != 0 else float("inf")
			offspring.append(selection[i])

		k = 0
		while len(offspring) < self.population_size and k < 100:
			i1 = i2 = -1
			while i1 == i2:
				i1 = random.randint(0, len(selection) - 1)
				i2 = random.randint(0, len(selection) - 1)
			
			# crossover
			if inner_workings:
				print('[INFO] Attempting crossover...')

			if self.crossover_type == 2:
				#if not supergenes:
					#raise Exception("Missing supergenes")
				if self.vector_generator.params.consistency_paths is None:
					raise Exception("Consistency paths missing from input for crossover of specified type")
				supergenes1 = self.consistency_checker.get_supergenes(selection[i1], max_shapes=self.max_supergene_shapes)
				supergenes2 = self.consistency_checker.get_supergenes(selection[i2], max_shapes=self.max_supergene_shapes)
				children = self.crossover2(selection[i1], selection[i2], supergenes1, supergenes2)

			for child in children:
				if len(offspring) < len(selection):
					to_proceed = not self.vector_generator.params.enforce_inference or (self.vector_generator.get_inferrable_vectors(child) is not None)
					
					if to_proceed:
						# mutation
						if inner_workings:
							print('[INFO] Attempting mutation...')
						
						if self.mutation_type == 6:
							child = self.mutation6(child)

						di = self.calculate_di(child)

						if di is not None:
							#print("Adding child!", 1 / di if di != 0 else float('inf'))
							offspring.append(child)
							population_fitness[len(offspring) - 1] = 1 / di if di != 0 else float('inf')

							if inner_workings:
								print('[INFO] One reproduction instance successful')
					
			k += 1

		# if len(offspring) > len(selection):
		# 	offspring = offspring[:len(selection)]
		# 	population_fitness = population_fitness[:len(selection)]
		
		return offspring, sorted(population_fitness.items(), key=operator.itemgetter(1), reverse=True)

	def crossover2(self, p1, p2, perfect_genes1, perfect_genes2):
		if random.random() < self.crossover_rate:
			i1 = set(vh.matrix2indices(p1))
			i2 = set(vh.matrix2indices(p2))

			if not i1 or not i2:
				raise Exception("Found empty chromosomes; something went wrong")
			
			v1 = list(i1 - set(perfect_genes1))
			v2 = list(i2 - set(perfect_genes2))

			if not v1 and not v2:
				return [p1, p2]

			if not v1:
				indexes1 = perfect_genes2
				indexes2 = perfect_genes1.union(v2)
			elif not v2:
				indexes2 = perfect_genes1
				indexes1 = perfect_genes2.union(v1)
			else:
				i1 = random.randint(0, len(v1) - 1)
				i2 = random.randint(0, len(v2) - 1)
				indexes1 = set(v1[:i1]).union(set(v2[i2:])).union(perfect_genes1)
				indexes2 = set(v2[:i1]).union(set(v1[i2:])).union(perfect_genes2)

			child1 = np.zeros((len(p1), len(p1), 2))
			child2 = np.zeros((len(p1), len(p1), 2))
			child1[:] = np.nan
			child2[:] = np.nan

			for i in range(len(child1)):
				child1[i][i] = [0, 0]
				child2[i][i] = [0, 0]

			for (i, j) in indexes1:
				if not vh.is_missing(p1[i][j]):
					child1[i][j] = p1[i][j].copy()
				else:
					child1[i][j] = p2[i][j].copy()
			
			for (i, j) in indexes2:
				if not vh.is_missing(p2[i][j]):
					child2[i][j] = p2[i][j].copy()
				else:
					child2[i][j] = p1[i][j].copy()

			if self.verbose:
				print('CROSSOVER:')
				print('Parent 1:', vh.format_indices(v1))
				print('Parent 2:', vh.format_indices(v2))
				print('Cutting before index', i1, 'at parent 1')
				print('Cutting after index', i2, 'at parent 2')
				print('--> Child 1:', vh.format_indices(vh.matrix2indices(child1)))
				print('--> Child 2:', vh.format_indices(vh.matrix2indices(child2)))

			return [child1, child2]
		return [p1, p2]
	
	def crossover(self, p1, p2, perfect_genes1, perfect_genes2, delta=1e-6):
		if random.random() < self.crossover_rate:
			i1 = set(vh.matrix2indices(p1))
			i2 = set(vh.matrix2indices(p2))

			if not i1 or not i2:
				raise Exception("Found empty chromosomes; something went wrong")
			
			v1 = list(i1 - set(perfect_genes1))
			v2 = list(i2 - set(perfect_genes2))

			if not v1 and not v2:
				return [p1, p2]

			gp2 = self.vector_generator.get_generated_vectors(p2)
			for (pi, pj) in perfect_genes1:
				for i in range(len(p1)):
					if np.sum(((p1[pi][pj] + gp2[pj][i]) - p1[pi][i]))**2 <= delta:
						pass
			#return [child1, child2]
		return [p1, p2]

	def mutation6(self, individual):
		if self.vector_generator.params.consistency_paths is None:
			raise Exception("Consistency paths not passed for mutation of specified type")
		temp = copy.deepcopy(individual)

		if random.random() <= self.mutation_rate:
			#if random.random() < .7:
				supergenes = self.consistency_checker.get_supergenes(individual)
				
				# select gene for deletion
				missing = set(vh.get_missing(individual))
				ds = (set(vh.matrix2indices(individual)) - set(supergenes) - missing)
				to_delete = ds.pop() if ds else None

				if to_delete:
					temp[to_delete[0]][to_delete[1]] = np.nan 
				# select gene for addition
				
				if ds:
					to_add = ds.pop()
					if to_add:
						temp[to_add[0]][to_add[1]] = self.measured[to_add[0]][to_add[1]].copy()
			# else:
			# 	for i in range(len(individual)):
			# 		for j in range(len(individual)):
			# 			if random.random() <= self.mutation_rate:
			# 				individual[i][j] += (-1)**random.randint(0, 1) * random.random() * .05

		return temp

	# this one includes the number of missing vectors in the cost
	def calculate_di(self, vectors):
		#measured = vg.get_inferrable_vectors(measured, paths, self.vector_generator.params.max_infer_components, verbose=False, self.vector_generator.params.enforce_inference=False)

		if self.vector_generator.params.enforce_inference and vectors is None:
			return None 

		generated = self.vector_generator.get_generated_vectors(vectors)
		num_vectors = len(vh.matrix2indices(vectors))
		if num_vectors == 0:
			return 100000000
		
		di = 0
		
		if self.verbose:
			print('Calculating DI')
		
		for i in range(len(generated)):
			for j in range(len(generated)):
				c1 = ""
				curr = 0
				if i != j:
					if vh.is_missing(vectors[i][j]):
						continue 
					curr += np.sum((vectors[i][j] - generated[i][j])**2)
					num_perms = 1
					
					if self.verbose:
						print("M" + str(i + 1) + str(j + 1) + ": " + f"sqrt(({vectors[i][j]} - {generated[i][j]})**2", end="")
						c1 += "M" + str(i + 1) + str(j + 1) + ": " + "sqrt(((M" + str(i + 1) + str(j + 1) + "(m) - " + "M" + str(i + 1) + str(j + 1) + "(g))**2 "
					
					curr_paths = self.vector_generator.params.generation_paths[(i, j)]#[:min_required_components]
					random.shuffle(curr_paths)

					for path in curr_paths:
						if len(path) > 2:# and ((di_type == "measured" and vh.is_valid(path, measured)) or di_type == "generated"):
							if self.verbose:
								print(f" + ({generated[i][j]} - (", end="")
								c1 += " + (M" + str(i + 1) + str(j + 1) + "(g) - " + "M" + str(i + 1) + str(j + 1) + f"(g{','.join([str(p + 1) for p in path])}))**2 "

							intermediary = self.vector_generator.get_generated_vector(path, vectors)
							if intermediary is None:
								continue 
							curr += np.sum((generated[i][j] - intermediary)**2)
							
							if self.verbose:
								print("))**2", end="")
								
							num_perms += 1

							if num_perms >= self.max_di_components:
								break

					# if num_perms < self.min_di_components:
					# 	return None

					curr /= num_perms
					di += math.sqrt(curr)

					if self.verbose:
						print(f') / {num_perms})')
						c1 += f') / {num_perms})'
				
				if c1 != "":
					print(c1)
		
		return (di / num_vectors) / self.vector_generator.params.space_size * (1 - self.missing_vectors_weight) + (len(vh.get_missing(generated)) / (len(vectors) * (len(vectors) - 1)) * self.missing_vectors_weight if self.missing_vectors_weight > 0 else 0)

	def run( 
		self,
		true_vectors=None,
		verbose=False
	):
		if verbose:
			print('[INFO] Generating population...')
		
		population, population_fitness = self.generate_population()
		
		if len(population) == 0:
			raise Exception('Unable to generate an initial population')
		
		if verbose:
			print("[INFO] Initial population (first 5):")
			for i in population[:5]:
				print(vh.format_indices(vh.matrix2indices(i)))
		
		best_individual = None 
		dis = []
		best_errors = []

		for i in range(self.iters):
			print(f'[INFO] Iteration {i + 1}:')
			print('[INFO] Calculating population fitness')
			
			if best_individual is None or (population_fitness[0][1] > 0 and 1 / population_fitness[0][1] < (1 / best_fitness if best_fitness > 0 else float('inf'))):
				res = self.vector_generator.infer_vectors(population[population_fitness[0][0]]) #self.vector_generator.get_inferrable_vectors(population[population_fitness[0][0]], coverage=None)

				if res is None:
					if not self.vector_generator.params.enforce_inference:
						print('[WARNING] Since deduction was not enforced, some vectors will be missing')
					else:
						raise Exception('Violation of enforcing deduction detected, which means that something went wrong')
				else:
					best_individual, mrpl = res
					best_fitness = population_fitness[0][1]

					if true_vectors is not None:
						best_error = vh.calculate_error(best_individual, true_vectors)

			if true_vectors is not None:
				best_errors.append(best_error)
			
			dis.append(1 / best_fitness)

			if true_vectors is not None:
				print("[INFO] Best individual error:", best_error)
			
			print("[INFO] Number of missing vectors:", len(vh.get_missing(best_individual)))
			print("[INFO] Best individual DI:", 1/best_fitness if best_fitness != float('inf') else 0)
			
			print('[INFO] Performing selection...')
			selection = self.tournament_selection(population, population_fitness)

			print('[INFO] Generating offspring...')
			population, population_fitness = self.generate_offspring(selection)
			#print("Fitness:", population_fitness)

			if len(population) == 0:
				raise Exception('Unable to generate offspring')

		nnv = self.vector_generator.get_negative_vectors(best_individual)

		return best_individual, best_fitness
