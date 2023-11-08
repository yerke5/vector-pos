import enum
import json
from pylab import plot, show, savefig, xlim, figure, \
                ylim, legend, boxplot, setp, axes
from importlib.resources import path
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
import pandas as pd
import seaborn as sns
import collections
from params import Params
import matplotlib
import matplotlib.patches as mpatches
from mycolorpy import colorlist as mcp
import consistency_checker as vc 

POS_ERROR_LIMITS = [0, 1.2]
DETECTION_RATE_LIMITS = [0, 100]
GRAPH_DIR = "../../diagrams/simulations/algs/graphs"

def get_noise_params():
	return 0, .2, 0, .2, None, None, None, None
	
def dict2df(d):
	cmp = None
	colors = collections.defaultdict(list) 
	dfs = {}
	for max_gen_degree in d:
		dfs[f"Max K = {max_gen_degree}"] = []
		N = len(d[max_gen_degree])
		
		if not cmp:
			cmp = mcp.gen_color(cmap="cool", n=N) #["#2C7BB6", '#D7191C', '#9cbd1a']
			cmp = {x: c for x, c in zip(d[max_gen_degree].keys(), cmp)}
		
		for num_nodes in d[max_gen_degree]:
			ks = d[max_gen_degree][num_nodes].items()
			iters = len(list(ks)[0][1])

			for it in range(iters):
				dfs[f"Max K = {max_gen_degree}"].append([0] * (len(ks) + 1))
				dfs[f"Max K = {max_gen_degree}"][-1][-1] = num_nodes

				for j, k in enumerate(d[max_gen_degree][num_nodes]):
					# print(it, k)
					# print(d[max_gen_degree][num_nodes][k], len(d[max_gen_degree][num_nodes][k]))
					dfs[f"Max K = {max_gen_degree}"][-1][j] = d[max_gen_degree][num_nodes][k][it] # f"Max K = {max_gen_degree}"
	
	
		dfs[f"Max K = {max_gen_degree}"] = pd.DataFrame(dfs[f"Max K = {max_gen_degree}"], columns=[f"K = {i}" for i in range(max_gen_degree + 1)] + ["N"])#.sort_values(by=["Max K", "K", "N"])
	
	return dfs 
	
def box_plot(path_lens):
	dfs = dict2df(path_lens)
	for cat, df in dfs.items():
		df_long = pd.melt(df, "N", var_name="K", value_name="% of generated vectors")
		PROPS = {
			'boxprops':{'edgecolor':'none'},
			'medianprops':{'color':'black'},
			'whiskerprops':{'color':'black', 'linestyle': '--'},
			'capprops':{'color':'black'}
		}
		sns.boxplot(x="K", hue="N", y="% of generated vectors", data=df_long, linewidth=0.5, palette="cool", **PROPS)
		#plt.title(f"Distribution of Generated Vectors for {cat}")
		plt.ylim(0, 100)
		plt.title("")
		plt.tight_layout()
		plt.show()

def bar_plot(data, x_labels, title, xa_label, ya_label, get_axis=False, separate_legend=False, cols=None, ylimits=None):
	df = pd.DataFrame(data=data, index=x_labels)
	if cols:
		df = df[cols]
	
	ax = df.plot.bar(rot=15, title=title, colormap="cool", edgecolor="white")
	ax.set(xlabel=xa_label, ylabel=ya_label)

	if separate_legend:
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

		# Put a legend to the right of the current axis
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	if ylimits is not None:
		ax.set_ylim(bottom=ylimits[0], top=ylimits[1])
	plt.tight_layout()
	if not get_axis:
		plt.show()
	else:
		return ax

def algs_barplot(d, title, ylabel):
	df = pd.DataFrame(d)
	df.set_index(['R', 'N'], inplace=True)
	#df.plot.bar(rot=15, colormap="cool")

	# Create figure with a subplot for each factory zone with a relative width
	# proportionate to the number of factories
	zones = df.index.levels[0]
	nplots = zones.size
	plots_width_ratios = [df.xs(zone).index.size for zone in zones]
	fig, axes = plt.subplots(nrows=1, ncols=nplots, sharey=True, figsize=(10, 4), gridspec_kw = dict(width_ratios=plots_width_ratios, wspace=0))

	if nplots == 1:
		axes = [axes]
	# Loop through array of axes to create grouped bar chart for each factory zone
	alpha = 0.3 # used for grid lines, bottom spine and separation lines between zones
	first = True 
	for zone, ax in zip(zones, axes):
		# Create bar chart with grid lines and no spines except bottom one
		df.xs(zone).plot.bar(ax=ax, legend=None, zorder=2, colormap="cool", edgecolor="white")
		#ax.grid(axis='y', zorder=1, color='black', alpha=alpha)
		for spine in ['top', 'left', 'right']:
			#ax.spines[spine].set_visible(False)
			ax.spines[spine].set_alpha(alpha)
		ax.spines['bottom'].set_alpha(alpha)
		
		# Set and place x labels for factory zones
		ax.set_xlabel(zone)
		ax.xaxis.set_label_coords(x=0.5, y=-0.2)
		
		# Format major tick labels for factory names: note that because this figure is
		# only about 10 inches wide, I choose to rewrite the long names on two lines.
		ticklabels = [name for name in df.xs(zone).index]
		ax.set_xticklabels(ticklabels, rotation=0, ha='center')
		ax.tick_params(axis='both', length=0, pad=7)
		
		# Set and format minor tick marks for separation lines between zones: note
		# that except for the first subplot, only the right tick mark is drawn to avoid
		# duplicate overlapping lines so that when an alpha different from 1 is chosen
		# (like in this example) all the lines look the same
		if ax.is_first_col():
			ax.set_xticks([*ax.get_xlim()], minor=True)
		else:
			ax.set_xticks([ax.get_xlim()[1]], minor=True)
		ax.tick_params(which='minor', length=100, width=0.8, color=[0, 0, 0, alpha])
		ax.set_ylabel(ylabel)

		if first:
			ax.legend(loc="upper left", fontsize=15)
			first = False
	# Add legend using the labels and handles from the last subplot
	#fig.legend(*ax.get_legend_handles_labels(), loc="upper left")

	fig.suptitle(title, size=20)
	fig.tight_layout()

	#plt.legend(loc="upper left")
	plt.show()

# fixed P and different K
def cumulative_path_lengths(iters, num_nodess, max_gen_degrees, params): 
	space_size = params["space_size"]
	angle_noise = params["angle_noise"]
	radius_noise = params["radius_noise"]
	max_angle = params["max_angle"]
	max_range = params["max_range"]
	noise_ratio = params["noise_ratio"]
	consider_orientation = params["consider_orientation"]
	random_orientations = params["random_orientations"] 
	num_anchors = params["num_anchors"]
	sequential = params["sequential"] 
	noise_trim = params["noise_trim"]
	
	measured_rates = []
	inferred_rates = []
	measured_errors = []
	inferred_errors = []

	path_lens = collections.defaultdict(dict)
	rates = {}
	pos_errors = {}

	for num_nodes in num_nodess:
		for max_gen_degree in max_gen_degrees: 
			path_lens[max_gen_degree][num_nodes] = {}
			for k in range(0, max_gen_degree + 1):
				path_lens[max_gen_degree][num_nodes][k + 2] = [0] * iters
			
	for n, num_nodes in enumerate(num_nodess):
		print("Number of nodes:", num_nodes)
		
		for it in range(iters):
			print(f"Iteration {it}")
			# measured_rate = measured_error = 0
			# inferred_rate = inferred_error = 0

			dpl = collections.defaultdict(list)
			true_nodes = vh.generate_random_nodes(num_nodes, space_size, num_anchors=num_anchors)
			anchor_nodes = true_nodes[-num_anchors:]
			true_vectors = vh.coords2vectors(true_nodes)
			
			# filter out vectors based on the ground truth
			orientations = vh.get_orientations(true_nodes, space_size, random=random_orientations) if consider_orientation else None 
			filtered = vm.drop_unseen_vectors(true_vectors, orientations, max_angle=max_angle, max_range=max_range, verbose=False)
			#print("Detection rate after filtering unreachable nodes:", measured_rate)

			# inject noise into filtered vectors
			measured = vm.inject_noise(filtered, noise_ratio, space_size, true_nodes, angle_noise, radius_noise, num_anchors=num_anchors)
			
			if not f"Measured vectors" in pos_errors:
				pos_errors[f"Measured vectors"] = [0] * len(num_nodess)
			
			pos_errors[f"Measured vectors"][n] += vh.calculate_error(true_vectors, measured) / iters
			print("Measured error:", pos_errors[f"Measured vectors"][n])
			
			if not f"Measured vectors" in rates:
				rates[f"Measured vectors"] = [0] * len(num_nodess)
			
			rates[f"Measured vectors"][n] += (1 - len(vh.get_missing(filtered)) / vh.vector_count(num_nodes, num_anchors)) / iters * 100

			for max_gen_degree in max_gen_degrees:
				params = Params(
					num_nodes,
					space_size,
					num_anchors=num_anchors,
					max_infer_components=1, 
					max_gen_components=1, 
					min_gen_degree=1,
					max_gen_degree=max_gen_degree,
					min_consistency_degree=1,
					max_consistency_degree=2,
					noise_trim=False,
					max_range=max_range,
					max_angle=max_angle,
					enforce_inference=False
				)
				#print("Generation paths:", params.generation_paths)
				vecgen = vg.VectorGenerator(params=params, verbose=False, sequential=sequential)

				inferred, inference_path_lengths = vecgen.infer_vectors(measured, generation_mode=True) #COME BAACK!!!!! used to be infer_vectors
				
				#print(inference_path_lengths)
				
				for pl in inference_path_lengths:
					if pl == 4:
						print("K = 2!!!!", inference_path_lengths[pl], "out of", (num_nodes * (num_nodes - 1)))
					#dpl[pl].append(inference_path_lengths[pl] * 100 / (num_nodes * (num_nodes - 1))) # record the ratio of vectors at this K 
					path_lens[max_gen_degree][num_nodes][pl][it] = inference_path_lengths[pl] * 100 / (num_nodes * (num_nodes - 1))

				if not f"Generated vectors (Max K = {max_gen_degree})" in pos_errors:
					pos_errors[f"Generated vectors (Max K = {max_gen_degree})"] = [0] * len(num_nodess)

				pos_errors[f"Generated vectors (Max K = {max_gen_degree})"][n] += vh.calculate_error(true_vectors, inferred) / iters
				
				if not f"Generated vectors (Max K = {max_gen_degree})" in rates:
					rates[f"Generated vectors (Max K = {max_gen_degree})"] = [0] * len(num_nodess)
				
				rates[f"Generated vectors (Max K = {max_gen_degree})"][n] += (1 - len(vh.get_missing(inferred)) / vh.vector_count(num_nodes, num_anchors)) / iters * 100

	plt.tight_layout()
	box_plot(path_lens)
	bar_plot(pos_errors, num_nodess, "Max K vs Positioning Error", "Number of nodes", "Positioning error (m)", separate_legend=True)#, ylimits=POS_ERROR_LIMITS)
	bar_plot(rates, num_nodess, "Max K vs Detection Rate", "Number of nodes", "Detection rate (%)", separate_legend=True, ylimits=DETECTION_RATE_LIMITS)

# different P
def num_gen_components(iters, num_nodess, Ps, params, k_as_ratios=True):
	space_size = params["space_size"]
	angle_noise = params["angle_noise"]
	radius_noise = params["radius_noise"]
	max_angle = params["max_angle"]
	max_range = params["max_range"]
	noise_ratio = params["noise_ratio"]
	consider_orientation = params["consider_orientation"]
	random_orientations = params["random_orientations"] 
	num_anchors = params["num_anchors"]
	sequential = params["sequential"] 
	noise_trim = params["noise_trim"]
	ges = {"Measured vectors": [0] * len(num_nodess)}

	for n, num_nodes in enumerate(num_nodess):
		print("Number of nodes:", num_nodes)
		
		# measured_rate = 0
		# gen_rates = np.zeros(len(Ps),)
		measured_error = 0
		gen_errors = np.zeros(len(Ps),)
		
		for it in range(iters):
			print("Iteration", it + 1)
			true_nodes = vh.generate_random_nodes(num_nodes, space_size, num_anchors=num_anchors)
			true_vectors = vh.coords2vectors(true_nodes)

			# filter out vectors based on the ground truth
			orientations = vh.get_orientations(true_nodes, space_size, random=random_orientations) if consider_orientation else None 
			filtered = vm.drop_unseen_vectors(true_vectors, orientations, max_angle=max_angle, max_range=max_range, verbose=False)
			#fr = (1 - len(vh.get_missing(filtered)) / vh.vector_count(num_nodes, num_anchors)) / iters
			
			# inject noise into filtered vectors
			measured = vm.inject_noise(filtered, noise_ratio, space_size, true_nodes, angle_noise, radius_noise, num_anchors=num_anchors)
			me = vh.calculate_error(true_vectors, measured)

			for j, k in enumerate(Ps):
				print("Number of generation components:", k)
				if k == -1:
					if f"Max P" not in ges:
						ges["Max P"] = [0] * len(num_nodess)
				elif k > num_nodes - 1:
					continue
				elif f"P = {k}" not in ges:
					ges[f"P = {k if not k_as_ratios else k * 100}"] = [0] * len(num_nodess)
				

				params = Params(
					num_nodes,
					space_size,
					max_infer_components=(k if not k_as_ratios else max(1, int(k * (num_nodes - 1)))) if k != -1 else float("inf"), 
					max_gen_components=(k if not k_as_ratios else max(1, int(k * (num_nodes - 1)))) if k != -1 else float("inf"), 
					min_gen_degree=1, 
					max_gen_degree=1, 
					min_consistency_degree=1, 
					max_consistency_degree=1,
					noise_trim=noise_trim,
					max_range=max_range,
					max_angle=max_angle,
					enforce_inference=False
				)
				vecgen = vg.VectorGenerator(params=params, verbose=False, sequential=sequential)

				generated = vecgen.get_generated_vectors(measured)
				#gen_rates[j] += (1 - len(vh.get_missing(generated)) / vh.vector_count(num_nodes, num_anchors)) * 100 / iters
				gen_errors[j] += vh.calculate_error(true_vectors, generated) / iters

				#pos_errors[f"P = {k}"].append(gen_errors.copy()
				
			#measured_rate += fr * 100 / iters
			measured_error += me / iters
			
		ges["Measured vectors"][n] = measured_error
		for j, k in enumerate(Ps):
			if k > num_nodes - 1:
				continue
			ges[f"P = {k if not k_as_ratios else k * 100}" if k != -1 else "Max P"][n] = gen_errors[j]
		
	bar_plot(ges, num_nodess, f"P vs Positioning Error", "Number of nodes", "Positioning error (m)", separate_legend=True, cols=["Measured vectors"] + ["Max P"] + [f"P = {k}" for k in [1, 2, 5, 10, 20]])#, ylimits=POS_ERROR_LIMITS)

# positioning algorithms (HA + GA)
def cmp_algs(iters, num_nodess, noise_ratios, params_, algs, draw_vectors=False, savefigs=False):
	space_size = params_["space_size"]
	angle_noise = params_["angle_noise"]
	radius_noise = params_["radius_noise"]
	max_angle = params_["max_angle"]
	max_range = params_["max_range"]
	consider_orientation = params_["consider_orientation"]
	random_orientations = params_["random_orientations"] 
	num_anchors = params_["num_anchors"]
	sequential = params_["sequential"] 
	noise_trim = params_["noise_trim"]
	max_gen_degree = params_["max_gen_degree"]
	min_gen_degree = params_["min_gen_degree"]
	min_consistency_degree = params_["min_consistency_degree"]
	max_consistency_degree = params_["max_consistency_degree"]
	di_p = params_["DI_P"]
	gen_p = params_["GEN_P"]
	population_size = params_["population_size"]
	elite_proportion = params_["elite_proportion"]
	crossover_rate = params_["crossover_rate"]
	mutation_rate = params_["mutation_rate"]
	missing_vectors_weight = params_["missing_vectors_weight"]
	ga_iters = params_["ga_iters"]
	init_consistent_shapes_ratio = params_["init_consistent_shapes_ratio"]
	max_init_consistency_shapes = params_["max_init_consistency_shapes"]
	mirror_consistent_vectors = params_["mirror_consistent_vectors"]
	include_baseline_vector = params_["include_baseline_vector"]
	try_negative_vectors = params_["try_negative_vectors"]
	mean_radius_noise = 0
	mean_angle_noise = 0
	ha_delta = params_["ha_delta"]
	supergene_delta = params_["supergene_delta"]
	low_dist_noise_mean, low_dist_noise_std, low_ra_noise_mean, low_ra_noise_std = params_["low_dist_noise_mean"], params_["low_dist_noise_std"], params_["low_ra_noise_mean"], params_["low_ra_noise_std"]
	
	if "mean_radius_noise" in params_:
		mean_radius_noise = params_["mean_radius_noise"]
	if "mean_angle_noise" in params_:
		mean_angle_noise = params_["mean_angle_noise"]

	pos_errors = []
	rates = []

	for noise_ratio in noise_ratios:
		for n, num_nodes in enumerate(num_nodess):
			params = Params(
				num_nodes,
				space_size,
				max_infer_components=gen_p, 
				max_gen_components=gen_p, 
				min_gen_degree=min_gen_degree, 
				max_gen_degree=max_gen_degree, 
				min_consistency_degree=min_consistency_degree, 
				max_consistency_degree=max_consistency_degree,
				noise_trim=False,
				max_range=max_range,
				max_angle=max_angle,
				enforce_inference=False
			)
			vecgen = vg.VectorGenerator(params, verbose=False, sequential=sequential)
			conscheck = vc.ConsistencyChecker(
				params, 
				vector_generator=vecgen, 
				verbose=False, 
				mirror_consistent_vectors=mirror_consistent_vectors, 
				include_baseline_vector=include_baseline_vector, 
				try_negative_vectors=try_negative_vectors,
				delta = ha_delta,
				supergene_delta = supergene_delta
			)
			
			pos_error = {"N": f"{num_nodes}", "R": f"R = {noise_ratio * 100}%", "Measured vectors": 0, "Generated vectors": 0, "GA": 0, "HA": 0}
			rate = {"N": f"{num_nodes}", "R": f"R = {noise_ratio * 100}%", "Measured vectors": 0, "Generated vectors": 0, "GA": 0, "HA": 0}

			for it in range(iters):
				true_nodes = vh.generate_random_nodes(num_nodes, space_size, num_anchors=num_anchors)
				anchor_nodes = true_nodes[-num_anchors:]
				true_vectors = vh.coords2vectors(true_nodes)
				
				# filter out vectors based on the ground truth
				if params_["consider_orientation"]:
					orientations = vh.get_orientations(true_nodes, space_size, random=random_orientations) if consider_orientation else None 
					filtered = vm.drop_unseen_vectors(true_vectors, orientations, max_angle=max_angle, max_range=max_range, verbose=False)
				else:
					filtered = true_vectors
				
				# inject noise into filtered vectors
				measured = vm.inject_noise(filtered, noise_ratio, space_size, true_nodes, angle_noise, radius_noise, num_anchors=num_anchors, mean_angle_noise=mean_angle_noise, mean_radius_noise=mean_radius_noise, mean_angle_noise_low=low_ra_noise_mean, mean_radius_noise_low=low_dist_noise_mean, angle_noise_low=low_ra_noise_std, radius_noise_low=low_dist_noise_std)
				curr_measured_error = vh.calculate_error(true_vectors, measured)
				curr_measured_rate = (1 - len(vh.get_missing(measured)) / vh.vector_count(num_nodes, num_anchors)) * 100
				
				pos_error["Measured vectors"] += curr_measured_error / iters
				rate[f"Measured vectors"] += curr_measured_rate / iters 

				generated = vecgen.get_generated_vectors(measured)#vecgen.infer_vectors(measured, generation_mode=True) #vecgen.get_generated_vectors(measured) 
				curr_gen_error = vh.calculate_error(true_vectors, generated)
				curr_gen_rate = (1 - len(vh.get_missing(generated)) / vh.vector_count(num_nodes, num_anchors)) * 100
				
				pos_error[f"Generated vectors"] += curr_gen_error / iters
				rate[f"Generated vectors"] += curr_gen_rate / iters 
				
				if draw_vectors or savefigs:
					vh.draw_vectors(space_size, true_nodes, true_vectors, "Actual", show=draw_vectors, figtitle=f"{GRAPH_DIR}/actual-{noise_ratio}-{num_nodes}-{it+1}.png" if savefigs else None)
					vh.draw_vectors(space_size, true_nodes, measured, "Measured", show=draw_vectors, figtitle=f"{GRAPH_DIR}/measured-{noise_ratio}-{num_nodes}-{it+1}.png" if savefigs else None)
					vh.draw_vectors(space_size, true_nodes, generated, "Generated", show=draw_vectors, figtitle=f"{GRAPH_DIR}/generated-{noise_ratio}-{num_nodes}-{it+1}.png" if savefigs else None)

				print(">>> Noise ratio:", noise_ratio, "number of nodes:", num_nodes, "iteration:", it + 1)
						
				for alg in algs:
					if alg == "GA":
						genalg = ga.GA(
							measured=measured, 
							max_di_components=di_p, 
							population_size=population_size, 
							vecgen=vecgen, 
							conscheck=conscheck,
							elite_proportion=elite_proportion,
							crossover_rate=crossover_rate, 
							mutation_rate=mutation_rate, 
							iters=ga_iters,
							missing_vectors_weight=missing_vectors_weight,
							verbose=False,
							init_consistent_shapes_ratio=init_consistent_shapes_ratio,
							min_di_components=2,
							max_init_consistency_shapes=num_nodes//2,
							max_supergene_shapes=num_nodes//2
						)
						
						res, best_fitness = genalg.run(
							true_vectors=true_vectors, 
							verbose=False
						)

					elif alg == "HA":
						res, nmv = conscheck.build_consistent_vectors(measured)
					
					if draw_vectors or savefigs:
						vh.draw_vectors(space_size, true_nodes, res, alg, show=draw_vectors, figtitle=f"{GRAPH_DIR}/{alg}-{noise_ratio}-{num_nodes}-{it + 1}.png" if savefigs else None)
					
					curr_alg_error = vh.calculate_error(true_vectors, res)
					curr_alg_rate = (1 - len(vh.get_missing(res)) / (num_nodes * (num_nodes - 1))) * 100
					pos_error[alg] += curr_alg_error / iters 
					rate[alg] += curr_alg_rate / iters
					print(f"    >>> {alg} error:", curr_alg_error)
					print(f"    >>> {alg} detection rate:", curr_alg_rate)
				print("    >>> Measured error:", curr_measured_error)
				print("    >>> Measured detection rate:", curr_measured_rate)
				print("    >>> Generated error:", curr_gen_error)
				print("    >>> Generated detection rate:", curr_gen_rate)

			pos_errors.append(pos_error)
			rates.append(rate)

	algs_barplot(pos_errors, "Positioning Error of Different Algorithms", "Positioning error (m)")
	algs_barplot(rates, "Detection Rate of Different Algorithms", "Detection rate (%)")

	# df = pd.DataFrame(rates)
	# df.index = pd.MultiIndex.from_arrays(df[["N", "R"]].values.T)
	# df.drop(["N", "R"], axis=1, inplace=True)
	# df.plot.bar(rot=15, colormap="cool")
	# plt.title("Detection Rate of Different Algorithms")
	# plt.ylabel("Detection rate (%)")
	# plt.xlabel("")
	# plt.show()

ra_noise_mean, ra_noise_std, dist_noise_mean, dist_noise_std, low_dist_noise_mean, low_dist_noise_std, low_ra_noise_mean, low_ra_noise_std = get_noise_params(default=True)

params = {
	"space_size": 30,
	"angle_noise": ra_noise_std,#.2,
	"radius_noise": dist_noise_std,#.2,
	"max_angle": 50,
	"max_range": 40,
	"noise_ratio": 1,
	"consider_orientation": True,
	"random_orientations": True,
	"num_anchors": 0,
	"sequential": False,
	"noise_trim": False,
	"min_gen_degree": 1,
	"max_gen_degree": 1,
	"min_consistency_degree": 1,
	"max_consistency_degree": 2,
	"DI_P": float("inf"),
	"GEN_P": float("inf"),
	"ga_iters": 20,
	"population_size": 30,
	"elite_proportion": .1,
	"crossover_rate": .8,
	"mutation_rate": .01,
	"missing_vectors_weight": .3,
	"init_consistent_shapes_ratio": .2,
	"max_init_consistency_shapes": 10,
	"mirror_consistent_vectors": False,
	"include_baseline_vector": False,
	"try_negative_vectors": True,
	"mean_angle_noise": ra_noise_mean,
	"mean_dist_noise": dist_noise_mean,
	"ha_delta": 1e-6,
	"supergene_delta": 1e-6,
	"low_dist_noise_mean": low_dist_noise_mean, 
	"low_dist_noise_std": low_dist_noise_std,
	"low_ra_noise_mean": low_ra_noise_mean,
	"low_ra_noise_std": low_ra_noise_std
}
print("-" * 50)
for param in params:
	print(param, params[param])

print("-" * 50)
Ps = [1, 2, 5, 10, 20, -1]
iters = 10
max_gen_degrees = [1, 2]
path_lens = [3, 4, 5]
noise_ratios = [.3, .5, .7, .9]
algs = ["HA", "GA"]
matplotlib.rcParams.update({'font.size': 20})

# uncomment one of the lines below to run the simulations
cumulative_path_lengths(iters=iters, num_nodess=[5, 10, 30, 50], max_gen_degrees=max_gen_degrees, params=params)

#num_gen_components(iters=iters, num_nodess=[5, 10, 30, 50], Ps=Ps, params=params, k_as_ratios=False)

#cmp_algs(iters=3, num_nodess=[10, 20, 30], noise_ratios=noise_ratios, params_=params, algs=algs, savefigs=True)#, draw_vectors=True)

