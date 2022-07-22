import random 
import math
import numpy as np
import matplotlib.pyplot as plt
import copy 
import operator
import ga
import vec_generator as vg 
import vec_helper as vh
import vec_manipulator as vm
import sys

TEST = False

def main(num_tests, plot=False):
	#errors = dict()
	percentages = {"generated2actual": [], "generated2measured": [], 'measured2actual': []}
	for i in range(num_tests):
		t = run_test_case(verbose=False or TEST)#num_tests == 1)
		if i == 0:
			errors = t
		else:
			for key in errors:
				errors[key] += t[key]

		change = (t["ga from measured; raw ga output to measured"] - t['generated to measured']) / t['generated to measured']
		percentages["generated2measured"].append(change)

		change = (t["ga from measured; raw ga output to actual"] - t['generated to actual']) / t['generated to actual']
		percentages["generated2actual"].append(change)

		change = (t["ga from measured; raw ga output to measured"] - t['measured to actual']) / t['measured to actual']
		percentages["measured2actual"].append(change)

	print('*' * 50)
	for key in errors:
		errors[key] /= num_tests
		print("average", key, "error:", errors[key], end='')
		if key != "generated to measured" and ("to measured" in key):
			print(f" ({round(100 * (errors[key] - errors['generated to measured']) / errors['generated to measured'], 2)}% change compared to gen to measured)")

		elif key != "generated to actual" and key != "measured to actual" and "to actual" in key:
			print(f" ({round(100 * (errors[key] - errors['generated to actual']) / errors['generated to actual'], 2)}% change compared to gen to actual)")
		else:
			print()


	for key in percentages:
		print(f'average change with respect to {key}:', np.mean(percentages[key]))
		#print(key, percentages[key])
	
	if plot:
		plt.hist(percentages["generated2measured"], density=True)
		plt.show()
		plt.hist(percentages["generated2actual"], density=True)
		plt.show()

def run_test_case(
	di_type="measured",
	verbose=False,
	population_size = 20,
	iters = 10,
	angle_noise = .3,
	radius_noise = .3,
	deletion_prob = .3,
	crossover_rate = .8,
	mutation_rate=0.01,
	space_size = 30,
	num_nodes = 5,
	elite_proportion = .1,
	test = TEST,
	max_angle = 60,
	max_range = 30,
	noise_ratio = .5,
	min_diffs_ratio=.1,
	enforce_deduction=True,
	mutation_percent_change=.7,
	mutation_type=5,
	num_deduction_components=2,
	variations=None,
	faster=False,
	draw_vectors=False
):
	print('=' * 50)
	print("PARAMETERS")
	print('Number of nodes:', num_nodes)
	print('Number of iterations:', iters)
	print('Crossover rate:', crossover_rate)
	print('Mutation rate:', mutation_rate)
	print('Space size:', space_size)
	print('Population size:', population_size)
	print('Angle noise:', angle_noise)
	print('Radius noise:', radius_noise)
	print('Noisy vectors ratio:', noise_ratio)
	print('Minimum diffs ratio for DI calculation:', min_diffs_ratio)
	print('Enforce deduction:', enforce_deduction)
	print('=' * 50)
	errors = {}
	true_nodes = []

	for i in range(num_nodes):
		node = vg.generate_random_node(space_size)
		true_nodes.append(node)
		
	if test:
		true_nodes = [
			[1, 4], 
			[6, 8],
			[10, 5],
			[5, 1]
		]

	if verbose:
		print('Coordinates:', [[round(x[0], 2), round(x[1], 2)] for x in true_nodes])
	true_nodes = np.array(true_nodes)
	#print(generate_paths(len(true_nodes)))
	#measured_nodes = np.array(measured_nodes)
	#plt.scatter(true_nodes[:, 0], true_nodes[:, 1], color="b", label="true")
	#plt.scatter(measured_nodes[:, 0], measured_nodes[:, 1], color="r", label="measured")
	#plt.legend()
	#plt.show()

	true_vectors = vg.coords2vectors(true_nodes, space_size)

	if not test:
		measured = vg.coords2vectors(true_nodes, space_size, angle_noise=angle_noise, radius_noise=radius_noise, noise_ratio=noise_ratio)
	else:
		measured = np.array([
			[[0, 0], [5 ,4], [9, 1], [4, -3]],
			[[-5, -4], [0, 0], [4, -3], [-1, -7]],
			[[-9, -1], [-4, 3], [0, 0], [-4, -3]], # M34 = [-5, -4]
			[[-4, 3], [-2, 8], [5, 4], [0, 0]] # M42 = [1,7]
		], dtype=float)


	paths = vh.paths2dict(vh.generate_paths(len(measured)))

	# print('Error before:', calculate_gen_error(measured, true_vectors))
	# print(vh.beautify_matrix(measured))
	# measured = vm.remove_inconsistent_vectors(measured, num_deduction_components)
	# print(vh.beautify_matrix(measured))
	# print('Error after:', calculate_gen_error(vm.deduce_vectors(measured, paths=paths, enforce_deduction=enforce_deduction), true_vectors))

	filtered = vm.drop_unseen_vectors(measured, true_nodes, space_size, max_angle=max_angle, max_range=max_range, verbose=verbose)
	generated = vg.get_generated_vectors(measured, paths, num_deduction_components)

	errors["measured to actual"] = calculate_gen_error(measured, true_vectors)
	errors["generated to measured"] = calculate_gen_error(generated, measured)
	errors["generated to actual"] = calculate_gen_error(generated, true_vectors)
	
	if verbose:
		# results
		print('Generated true vectors:', vh.beautify_matrix(vg.get_generated_vectors(true_vectors, paths, num_deduction_components)))
		print('Real vectors:', vh.beautify_matrix(true_vectors))
		print('Noisy vectors:', vh.beautify_matrix(measured))
		print('Generated vectors:', vh.beautify_matrix(generated))
	
	best_individual, best_fitness = run_ga(
		population_size, 
		measured, 
		true_vectors, 
		di_type=di_type,
		num_deduction_components=num_deduction_components,
		space_size=space_size,
		paths=paths,
		elite_proportion=elite_proportion, 
		deletion_prob=deletion_prob, 
		iters=iters,
		crossover_rate=crossover_rate,
		mutation_rate=mutation_rate,
		verbose=verbose,
		min_diffs_ratio=min_diffs_ratio,
		enforce_deduction=enforce_deduction,
		mutation_percent_change=mutation_percent_change,
		mutation_type=mutation_type,
		variations=variations,
		faster=faster
	)

	generated_best = vg.get_generated_vectors(best_individual, paths, num_deduction_components)
	errors["ga from measured; raw ga output to measured"] = calculate_gen_error(measured, best_individual)
	errors["ga from measured; raw ga output to actual"] = calculate_gen_error(true_vectors, best_individual)
	errors["ga from measured; generated ga output to measured"] = calculate_gen_error(measured, generated_best)
	errors["ga from measured; generated ga output to actual"] = calculate_gen_error(true_vectors, generated_best)

	if verbose:
		# results
		#print('Generated true vectors:', vh.beautify_matrix(vg.get_generated_vectors(true_vectors, paths, num_deduction_components, enforce_deduction)))
		print('Real vectors:', vh.beautify_matrix(true_vectors))
		print('Noisy vectors:', vh.beautify_matrix(measured))
		print('Generated vectors:', vh.beautify_matrix(generated))
	
		# filtered
		#print('Filtered vectors:', vh.beautify_matrix(filtered))
		#print('Deducable?', vm.deduce_vectors(filtered, paths=paths, enforce_deduction=enforce_deduction) is not None)

	if draw_vectors:
		vh.draw_vectors(space_size, true_nodes, true_vectors, "True Vectors")
		vh.draw_vectors(space_size, true_nodes, measured, "Noisy Vectors")
		#vh.draw_vectors(space_size, true_nodes, generated, "Generated Vectors")
		vh.draw_vectors(space_size, true_nodes, best_individual, "GA Vectors")
		#vh.draw_vectors(space_size, true_nodes, generated_best, "Generated GA Vectors")

		# vh.scatter_locations(space_size, true_nodes, measured, "Noisy Vectors")
		# vh.scatter_locations(space_size, true_nodes, generated, "Generated Vectors")
		# vh.scatter_locations(space_size, true_nodes, best_individual, "GA Vectors")
		# vh.scatter_locations(space_size, true_nodes, generated_best, "Generated GA Vectors")
	
	init_di = ga.calculate_di(di_type, measured, paths=paths, min_diffs_ratio=min_diffs_ratio) 

	print('*' * 50)
	print('RESULTS')
	print('Measured to actual error:', errors["measured to actual"])
	#print('Generated to measured error:', errors["generated to measured"])
	print('Generated to actual error:', errors["generated to actual"])
	print("Initial DI:", init_di)
	print('Raw GA best DI:', 1/best_fitness if best_fitness != float('inf') else 0)
	#print('Raw GA best fitness:', best_fitness)
	#print('Raw GA output:', vh.beautify_matrix(best_individual))
	#print('Measured to raw GA output error:', errors["ga from measured; raw ga output to measured"])
	print('Raw GA output to actual error:', errors["ga from measured; raw ga output to actual"])
	#print('Generated GA output to measured error:', errors["ga from measured; generated ga output to measured"])
	print('Generated GA output to actual error:', errors["ga from measured; generated ga output to actual"])
	print("*" * 50)
	
	# best_individual, best_fitness = run_ga(
	# 	population_size, 
	# 	generated, 
	# 	true_vectors, 
	# 	elite_proportion=elite_proportion, 
	# 	deletion_prob=deletion_prob, 
	# 	iters=iters,
	# 	crossover_rate=crossover_rate
	# )

	# generated_best = vg.get_generated_vectors(best_individual)
	# errors["ga from generated; raw ga output to measured"] = calculate_gen_error(measured, best_individual)
	# errors["ga from generated; raw ga output to actual"] = calculate_gen_error(true_vectors, best_individual)
	# errors["ga from generated; generated ga output to measured"] = calculate_gen_error(measured, generated_best)
	# errors["ga from generated; generated ga output to actual"] = calculate_gen_error(true_vectors, generated_best)

	# if verbose:
	# 	print('Input = generated')
	# 	print('Raw GA best DI:', 1/best_fitness if best_fitness != float('inf') else 0)
	# 	print('Raw GA output:', vh.beautify_matrix(best_individual))
	# 	print('Measured to raw GA output error:', errors["ga from generated; raw ga output to measured"])
	# 	print('Raw GA output to actual error:', errors["ga from generated; raw ga output to actual"])
	# 	print('Generated GA output to measured error:', errors["ga from generated; generated ga output to measured"])
	# 	print('Generated GA output to actual error:', errors["ga from generated; generated ga output to actual"])

	return errors

def calculate_gen_error(m1, m2):
	return np.sum(np.sqrt(np.sum((np.array(m1) - np.array(m2))**2, axis=2))) / (len(m1) ** 2 - len(m1))

def run_ga(
	population_size, 
	measured, 
	true_vectors, 
	paths, 
	space_size, 
	num_deduction_components, 
	di_type,
	deletion_prob=.3, 
	iters=10, 
	elite_proportion=.1, 
	mutation_rate=ga.DEFAULT_MUTATION_RATE, 
	crossover_rate=ga.DEFAULT_CROSSOVER_RATE, 
	verbose=False, 
	min_diffs_ratio=.3, 
	enforce_deduction=False, 
	mutation_percent_change=.2, 
	mutation_type=4,
	variations=None,
	faster=False
):
	print('[INFO] Generating population...')
	population = ga.generate_population(
		measured, 
		di_type=di_type,
		paths=paths, 
		deletion_prob=deletion_prob, 
		size=population_size, 
		min_diffs_ratio=min_diffs_ratio, 
		enforce_deduction=enforce_deduction, 
		num_deduction_components=num_deduction_components
	)

	print("Most consistent child error:", calculate_gen_error(vm.deduce_vectors(population[0], paths, num_deduction_components), true_vectors))

	if TEST:
		population = [copy.deepcopy(measured) for i in range(1)]
		for k in range(len(measured)):
			for n in range(len(measured)):
				if k != n and (k, n) in [(3, 1), (2, 3)]:#[(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0), (2, 3), (3, 0), (3, 1)]:
					population[0][k][n] = np.nan

	if len(population) == 0:
		raise Exception('Unable to generate an initial population')
	
	if verbose:
		print("[INFO] Initial population (first 5):")
		for i in population[:5]:
			print(vh.format_indices(vh.matrix2indices(i)))
	
	num_elites = int(population_size * elite_proportion)
	best_individual = None 
	dis = []
	for i in range(iters):
		to_print = (i == 0 or (i + 1) % 5 == 0) and verbose
		if True:#to_print:
			print(f'[INFO] Iteration {i + 1}:')

		print('[INFO] Calculating population fitness')
		population_fitness = ga.get_population_fitness(population, min_diffs_ratio, paths, num_deduction_components, di_type, verbose=verbose)

		if best_individual is None or (population_fitness[0][1] > 0 and 1 / population_fitness[0][1] < (1 / best_fitness if best_fitness > 0 else float('inf'))):
			best_individual = copy.deepcopy(vm.deduce_vectors(population[population_fitness[0][0]], paths, num_deduction_components, verbose=False, enforce_deduction=enforce_deduction))

			if best_individual is None:
				if not enforce_deduction:
					print('[WARNING] Since deduction was not enforced, some vectors will be missing')
				else:
					raise Exception('Violation of enforcing deduction detected, which means that something went wrong')
			
			best_fitness = population_fitness[0][1]
		dis.append(best_fitness)
		
		print('[INFO] Performing selection...')
		selection = ga.tournament_selection(population, population_fitness, num_elites)

		print('[INFO] Generating offspring...')
		population = ga.generate_offspring(
			selection, 
			num_elites,
			di_type=di_type,
			num_deduction_components=num_deduction_components,
			measured=measured,
			paths=paths,
			space_size=space_size,
			crossover_rate=crossover_rate,
			mutation_rate=mutation_rate,
			verbose=to_print,
			min_diffs_ratio=min_diffs_ratio,
			enforce_deduction=enforce_deduction,
			mutation_percent_change=mutation_percent_change,
			mutation_type=mutation_type,
			variations=variations,
			faster=faster
		)

		print('New population:')
		for i in population[:5]:
			print(vh.format_indices(vh.matrix2indices(i)))

		if len(population) == 0:
			raise Exception('Unable to generate offspring')

	#plt.plot(dis)
	#plt.show()
	return best_individual, best_fitness

if __name__ == "__main__":
	#num_tests = 1
	#main(num_tests)

	# run_test_case(
	# 	di_type="measured",
	# 	verbose=False,
	# 	population_size = 40,
	# 	iters = 50,
	# 	angle_noise = .5,
	# 	radius_noise = .5,
	# 	deletion_prob = .3,
	# 	crossover_rate = .8,
	# 	mutation_rate=0.4,
	# 	space_size = 10,
	# 	num_nodes = 5,
	# 	elite_proportion = .1,
	# 	test = TEST,
	# 	variations=None,#{"raw", "generated", "deduced", "generated-deduced"},
	# 	max_angle = 60,
	# 	max_range = 30,
	# 	noise_ratio = .25,
	# 	min_diffs_ratio=0,#5/10,
	# 	enforce_deduction=True,
	# 	mutation_percent_change=.7,
	# 	mutation_type=5,
	# 	num_deduction_components=2,
	# 	faster=False,
	# 	draw_vectors=True
	# )

	# sys.exit(1)

	num_nodess = [10, 20, 30, 50]
	noise_ratios = [.25, .5, .85]

	for i, noise_ratio in enumerate(noise_ratios):
	
		generated_errors = []
		ga_errors = []
		generated_ga_errors = []
		measured_errors = []

		plt.subplot(2, 2, i+1)
		for num_nodes in num_nodess:
			errors = run_test_case(
				verbose=False,
				population_size = 10,
				iters = 5,
				angle_noise = .1,
				radius_noise = .1,
				deletion_prob = .3,
				crossover_rate = .8,
				mutation_rate=0.01,
				space_size = 20,
				num_nodes = num_nodes,
				elite_proportion = .1,
				test = TEST,
				variations=None,#{"raw", "generated", "deduced", "generated-deduced"},
				max_angle = 60,
				max_range = 30,
				noise_ratio = noise_ratio,
				min_diffs_ratio=5/num_nodes,
				enforce_deduction=True,
				mutation_percent_change=.7,
				mutation_type=5,
				num_deduction_components=2,
				faster=True
			)

			generated_errors.append(errors["generated to actual"])
			measured_errors.append(errors["measured to actual"])
			generated_ga_errors.append(errors["ga from measured; generated ga output to actual"])
			ga_errors.append(errors["ga from measured; raw ga output to actual"])

		plt.title(f'Noisy vectors ratio = {noise_ratio}')
		plt.plot(num_nodess, generated_errors, label='Generated', color="#444444", linestyle="dashed", marker="*")
		plt.plot(num_nodess, generated_ga_errors, label='Generated GA', color="green", marker="x")
		plt.plot(num_nodess, ga_errors, label='GA', color="blue", marker="s")
		plt.plot(num_nodess, measured_errors, label='Measured', color="black", linestyle="dotted", marker="o")
		plt.xlabel('Number of nodes')
		plt.ylabel('Positioning error')
		plt.legend()
		plt.grid()

	plt.show()


