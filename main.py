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

def boxplot2(path_lens):
	df, colors, cmp = dict2df(path_lens)
	sns.boxplot(
		data=df[['K', 'N']].melt('fruit'),
		x='fruit',
		y='value',
		hue='variable',
		width=0.5,
		order=['apple', 'mango', 'orange'],
	)

def dict2df2(d):
	arr = []
	cmp = None
	colors = collections.defaultdict(list) 
	#print("DICT:", json.dumps(d, sort_keys=True, indent=4))

	for max_gen_degree in d:

		N = len(d[max_gen_degree])
		if not cmp:
			cmp = mcp.gen_color(cmap="cool", n=N) #["#2C7BB6", '#D7191C', '#9cbd1a']
			cmp = {x: c for x, c in zip(d[max_gen_degree].keys(), cmp)}
		
		i = 0
		for num_nodes in d[max_gen_degree]:
			for k in d[max_gen_degree][num_nodes]:
				for val in d[max_gen_degree][num_nodes][k]:
					arr.append([f"Max K = {max_gen_degree}", k - 2, num_nodes, val]) # f"Max K = {max_gen_degree}"
				#colors[f"Max K = {max_gen_degree}"].append(cmp[num_nodes])
			i += 1
	
	df = pd.DataFrame(arr, columns=["Max K", "K", "N", "%"]).sort_values(by=["Max K", "K", "N"])
	x = df.groupby(["Max K", "K", "N"]).mean()
	#idx = np.concatenate([x.index.get_level_values(1), x.index.get_level_values(2)], axis=1)
	maxks = x.index.get_level_values(0)
	ns = x.index.get_level_values(2)
	
	for maxk in maxks:
		for n in ns:
			colors[maxk].append(cmp[n])

	return df, colors, cmp

def dict2df(d):
	cmp = None
	colors = collections.defaultdict(list) 
	#print("DICT:", json.dumps(d, sort_keys=True, indent=4))
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
	x = df.groupby(["Max K", "K", "N"]).mean()
	#idx = np.concatenate([x.index.get_level_values(1), x.index.get_level_values(2)], axis=1)
	maxks = x.index.get_level_values(0)
	ns = x.index.get_level_values(2)
	
	for maxk in maxks:
		for n in ns:
			colors[maxk].append(cmp[n])

	return df, colors, cmp

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def box_plot2(path_lens):
	df, colors, cmp = dict2df2(path_lens)
	
	# df_long = pd.melt(df[df["Max K"] == 1], "N", var_name="K", value_name="%")
	# sns.boxplot(x="K", hue="N", y="%", data=df_long)
	# plt.plot()
	# return

	fig, ax1 = plt.subplots() #patch_artist=True, 
	bp_dict = df.groupby('Max K').boxplot(ax=ax1, by=['K', "N"], return_type="both", patch_artist=True, layout=(1, len(path_lens)), figsize=(6,8), whiskerprops = dict(linestyle='--', linewidth=.5))

	patches = []
	for n, c in cmp.items():
		patches.append(mpatches.Patch(color=c, label=f'N = {n}'))

	for col, bp in bp_dict.items():
		for b in bp.values:
			# b.ax.set_xticks([])
			# b.ax.xaxis.set_label_text('foo')
			# b.ax.xaxis.label.set_visible(False)	
			b.ax.yaxis.set_label_text("% of generated vectors out of all vectors")
			
			for patch in b.lines["whiskers"]:
				patch.set_linestyle("dashed")
				patch.set_linewidth(1)
				# patch.set_color(colors[col][i])
				# i += 2
			
			for patch, color in zip(b.lines["medians"], colors[col]):
				patch.set_color("black")

			for patch, color in zip(b.lines['boxes'], colors[col]):#for patch, color in zip(bp.values[0].lines['boxes'], colors):
				patch.set_facecolor(color)
				patch.set_edgecolor(color)
				#patch.set_linestyle("dashed")
			
			b.ax.grid(b=True, which='both', axis='both', linestyle='-')
			b.ax.legend(handles=patches)

	ax1.grid(b=True, which='both', axis='both', linestyle='-')
	ax1.set_title("Distribution of Generated Vectors by K")
	ax1.set_ylabel("% of generated vectors")
	fig.suptitle("Distribution of Generated Vectors by K")
	plt.show()

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
	#print(df)
	
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

def fixed_path_lengths(iters, num_nodess, path_lens, params): 
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

	rates = {}
	pos_errors = {}

	for n, num_nodes in enumerate(num_nodess):
		print("Number of nodes:", num_nodes)
		
		for it in range(iters):
			print(f"Iteration {it}")
			measured_rate = measured_error = 0
			inferred_rate = inferred_error = 0

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
			
			if not f"Measured vectors" in rates:
				rates[f"Measured vectors"] = [0] * len(num_nodess)
			
			rates[f"Measured vectors"][n] += (1 - len(vh.get_missing(filtered)) / vh.vector_count(num_nodes, num_anchors)) / iters * 100

			for path_len in path_lens:
				if path_len >= num_nodes - 1:
					continue
				params = Params(
					num_nodes,
					space_size,
					num_anchors=num_anchors,
					max_infer_components=num_nodes-1, 
					max_gen_components=num_nodes-1, 
					min_gen_degree=path_len-2,
					max_gen_degree=path_len-2,
					min_consistency_degree=1,
					max_consistency_degree=2,
					noise_trim=False,
					max_range=max_range,
					max_angle=max_angle,
					enforce_inference=False
				)
				#print("Generation paths:", params.generation_paths)
				vecgen = vg.VectorGenerator(params=params, verbose=False, sequential=sequential)

				inferred, inference_path_lengths = vecgen.infer_vectors(measured, generation_mode=True) 
				
				if not f"Generated vectors (K = {path_len - 2})" in pos_errors:
					pos_errors[f"Generated vectors (K = {path_len - 2})"] = [0] * len(num_nodess)

				pos_errors[f"Generated vectors (K = {path_len - 2})"][n] += vh.calculate_error(true_vectors, inferred) / iters
				
				if not f"Generated vectors (K = {path_len - 2})" in rates:
					rates[f"Generated vectors (K = {path_len - 2})"] = [0] * len(num_nodess)
				
				rates[f"Generated vectors (K = {path_len - 2})"][n] += (1 - len(vh.get_missing(inferred)) / vh.vector_count(num_nodes, num_anchors)) / iters * 100

			measured_rates.append(measured_rate * 100)
			inferred_rates.append(inferred_rate * 100)
			measured_errors.append(measured_error)
			inferred_errors.append(inferred_error)

	plt.tight_layout()
	
	bar_plot(pos_errors, num_nodess, "Fixed K vs Positioning Error", "Number of nodes", "Positioning error (m)", separate_legend=True)#, ylimits=POS_ERROR_LIMITS)
	bar_plot(rates, num_nodess, "Fixed K vs Detection Rate", "Number of nodes", "Detection rate (%)", separate_legend=True, ylimits=DETECTION_RATE_LIMITS)

def fixed_path_lengths2(iters, num_nodes, path_lens, params):
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

	for path_len in path_lens:
		#print("Path length:", path_len)
		measured_rate = inferred_rate = 0
		measured_error = inferred_error = 0 
		params = Params(
			num_nodes,
			space_size,
			max_infer_components=num_nodes - 1, 
			max_gen_components=num_nodes - 1, 
			min_gen_degree=path_len - 2,
			max_gen_degree=path_len - 2,
			min_consistency_degree=1,
			max_consistency_degree=2,
			noise_trim=False,
			max_range=max_range,
			max_angle=max_angle,
			enforce_inference=False
		)
		#print("Generation paths:", params.generation_paths)
		vecgen = vg.VectorGenerator(params=params, verbose=False, sequential=sequential)

		for it in range(iters):
			print(f"Iteration {it}")
			true_nodes = vh.generate_random_nodes(num_nodes, space_size, num_anchors=num_anchors)
			true_vectors = vh.coords2vectors(true_nodes)

			# filter out vectors based on the ground truth
			orientations = vh.get_orientations(true_nodes, space_size, random=random_orientations) if consider_orientation else None 
			filtered = vm.drop_unseen_vectors(true_vectors, orientations, max_angle=max_angle, max_range=max_range, verbose=False)
			measured_rate += (1 - len(vh.get_missing(filtered)) / vh.vector_count(num_nodes, num_anchors)) / iters
			#print("Detection rate after filtering unreachable nodes:", measured_rate)
			
			# inject noise into filtered vectors
			measured = vm.inject_noise(filtered, noise_ratio, space_size, true_nodes, angle_noise, radius_noise, num_anchors=num_anchors)
			
			inferred, inference_path_lengths = vecgen.infer_vectors(measured, generation_mode=True)
			inferred_rate += (1 - len(vh.get_missing(inferred)) / vh.vector_count(num_nodes, num_anchors)) / iters
	
			measured_error += vh.calculate_error(true_vectors, measured) / iters
			inferred_error += vh.calculate_error(true_vectors, inferred) / iters
			
		measured_rates.append(measured_rate * 100)
		inferred_rates.append(inferred_rate * 100)
		measured_errors.append(measured_error)
		inferred_errors.append(inferred_error)
	
	rates = {
		"Measured vectors": measured_rates,
		"Generated vectors": inferred_rates
	}

	pos_errors = {
		"Measured vectors": measured_errors,
		"Generated vectors": inferred_errors
	}
	
	# bar_plot(num_nodess, rates, "Number of nodes", "Detection rates")
	# bar_plot(num_nodess, pos_errors, "Number of nodes", "Positioning errors")
	bar_plot(pos_errors, path_lens, "Path lengths vs Positioning Error", "Path length (fixed)", "Positioning error (m)", ylimits=POS_ERROR_LIMITS)
	bar_plot(rates, path_lens, "Path lengths vs Detection Rate", "Path length (fixed)", "Detection rate (%)", ylimits=DETECTION_RATE_LIMITS)

# this function explores detection rate (how many vectors are available) when orientations of terminals are pointing in random directions
# it also explores how positioning error grows when generated vectors are used to generated other missing vectors sequentially
# current results: error grows if sequential inference is used
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
			measured_rate = measured_error = 0
			inferred_rate = inferred_error = 0

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

				inferred, inference_path_lengths = vecgen.infer_vectors(measured, generation_mode=True)
				
				#print(inference_path_lengths)
				
				for pl in inference_path_lengths:
					#dpl[pl].append(inference_path_lengths[pl] * 100 / (num_nodes * (num_nodes - 1))) # record the ratio of vectors at this K 
					path_lens[max_gen_degree][num_nodes][pl][it] = inference_path_lengths[pl] * 100 / (num_nodes * (num_nodes - 1))

				if not f"Generated vectors (Max K = {max_gen_degree})" in pos_errors:
					pos_errors[f"Generated vectors (Max K = {max_gen_degree})"] = [0] * len(num_nodess)

				pos_errors[f"Generated vectors (Max K = {max_gen_degree})"][n] += vh.calculate_error(true_vectors, inferred) / iters
				
				if not f"Generated vectors (Max K = {max_gen_degree})" in rates:
					rates[f"Generated vectors (Max K = {max_gen_degree})"] = [0] * len(num_nodess)
				
				rates[f"Generated vectors (Max K = {max_gen_degree})"][n] += (1 - len(vh.get_missing(inferred)) / vh.vector_count(num_nodes, num_anchors)) / iters * 100

				# -----------------
				# if it == 1 and num_nodes == num_nodess[1]:
				# 	ov = vh.get_origin_vectors(true_vectors, anchor_nodes)
				# 	plt.scatter(true_nodes[:, 0], true_nodes[:, 1], label="True nodes")
				# 	plt.scatter(ov[:, 0], ov[:, 1], label="Origin vectors")
				# 	plt.legend()
				# 	plt.show()
					
				# 	ov = vh.get_origin_vectors(measured, anchor_nodes)
				# 	plt.scatter(true_nodes[:, 0], true_nodes[:, 1], label="True nodes")
				# 	plt.scatter(ov[:, 0], ov[:, 1], label="Origin vectors")
				# 	plt.legend()
				# 	plt.show()
				# ------------------

				#print("Detection rate after deducing missing vectors:", inferred_rate)
				
				# if it == 1 and num_nodes == num_nodess[0] and draw_vectors:
				# 	vh.draw_vectors(space_size, true_nodes, true_vectors, "Unfiltered Actual", show=True, max_angle=max_angle, orientations=orientations)
				# 	vh.draw_vectors(space_size, true_nodes, filtered, "Filtered Actual", show=True, max_angle=max_angle, orientations=orientations)
				# 	vh.draw_vectors(space_size, true_nodes, measured, "Filtered Measured", show=True, max_angle=max_angle, orientations=orientations)
				# 	vh.draw_vectors(space_size, true_nodes, inferred, "Deduced Measured", show=True, max_angle=max_angle, orientations=orientations)
				# 	vh.draw_vectors(space_size, true_nodes, true_vectors, "Actual", show=True, max_angle=max_angle, orientations=orientations)
				
				# for pl in inference_path_lengths:
				# 	if pl not in dpl:
				# 		dpl[pl] = (inference_path_lengths[pl], 1)
				# 	else:
				# 		dpl[pl] = (dpl[pl][0] + inference_path_lengths[pl], dpl[pl][1] + 1)
				
			measured_rates.append(measured_rate * 100)
			inferred_rates.append(inferred_rate * 100)
			measured_errors.append(measured_error)
			inferred_errors.append(inferred_error)
			
			#pl = sorted(list(dpl.keys()))

			# plt.bar(pl, [dpl[p] for p in pl], label=f"Path lengths for inference (num_nodes = {num_nodes})")
			# plt.xlabel("Path length")
			# plt.ylabel("Number of vectors")
			# plt.legend()
			# plt.show()	

			# path_lengths = {
			# 	"Ratio of vectors generated": [dpl[p] for p in pl]
			# }
			
			# bar_plot(pl, path_lengths, "Path length", "%% of vectors", title=f"Number of nodes - {num_nodes})")

			# box plot 
			
		# plt.plot(num_nodess, measured_rates, label="filtered rate")
		# plt.plot(num_nodess, inferred_rates, label="inferred rate")
		# plt.legend()
		# plt.show()

		# plt.plot(num_nodess, measured_errors, label="measured error")
		# plt.plot(num_nodess, inferred_errors, label="inferred error")
		# plt.legend()
		# plt.show()

	# -------------------
	# plt.title(f"% of Vectors vs Path Lengths for Vector Generation")
	# plt.boxplot([dpl[p] for p in pl], labels=pl)
	# plt.xlabel("Path length")
	# plt.ylabel("% of vectors")
	# plt.show()

	plt.tight_layout()
	box_plot(path_lens)
	bar_plot(pos_errors, num_nodess, "Max K vs Positioning Error", "Number of nodes", "Positioning error (m)", separate_legend=True)#, ylimits=POS_ERROR_LIMITS)
	bar_plot(rates, num_nodess, "Max K vs Detection Rate", "Number of nodes", "Detection rate (%)", separate_legend=True, ylimits=DETECTION_RATE_LIMITS)

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
	#rates = {}
	pos_errors = {} #{"Measured errors": []}
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
			
		#rates[f"measured - {num_nodes}"] = measured_rates.copy()
		#rates[f"N = {num_nodes}"] = gen_rates.copy()
		#pos_errors[f"measured - {num_nodes}"] = measured_error.copy()
		ges["Measured vectors"][n] = measured_error
		for j, k in enumerate(Ps):
			if k > num_nodes - 1:
				continue
			ges[f"P = {k if not k_as_ratios else k * 100}" if k != -1 else "Max P"][n] = gen_errors[j]
		
		#pos_errors["Measured errors"].append(measured_error)
		
	# rates = {
	# 	"Measured vectors": measured_rates,
	# 	"generated vectors": gen_rates
	# }

	# pos_errors = {
	# 	"measured error": measured_errors,
	# 	"generated error": gen_errors
	# }

	# bar_plot(rates, [k * 100 for k in Ps], f"Ratio of Vector Generation Components vs Detection Rate for N = {num_nodes}", "Ratio of Vector Generation Components (%)", "Detection rate (%)")
	# bar_plot(pos_errors, [k * 100 for k in Ps], f"Ratio of Vector Generation Components vs Positioning Error for N = {num_nodes}", "Ratio of Vector Generation Components (%)", "Positioning error (m)")
	
	#ax = bar_plot(rates, [k * 100 for k in Ps], f"Ratio of Vector Generation Components vs Detection Rate", "Ratio of Vector Generation Components (%)", "Detection rate (%)")
	#ax = bar_plot(pos_errors, [k * 100 for k in Ps], f"Ratio of Intermediate Generated Vectors vs Positioning Error", "Ratio of Intermediate Generated Vectors (%)", "Positioning error (m)")
	bar_plot(ges, num_nodess, f"P vs Positioning Error", "Number of nodes", "Positioning error (m)", separate_legend=True, cols=["Measured vectors"] + ["Max P"] + [f"P = {k}" for k in [1, 2, 5, 10, 20]])#, ylimits=POS_ERROR_LIMITS)

def num_anchors():
	pass

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
	plt.tight_layout()

	#plt.legend(loc="upper left")
	plt.show()

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
			conscheck = vc.ConsistencyChecker(params, vector_generator=vecgen, verbose=False)
			
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
				
				# BAD PRACTICE
				#filtered = vecgen.get_generated_vectors(filtered)
				
				# inject noise into filtered vectors
				measured = vm.inject_noise(filtered, noise_ratio, space_size, true_nodes, angle_noise, radius_noise, num_anchors=num_anchors)
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
							max_init_consistency_shapes=max_init_consistency_shapes
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

params = {
	"space_size": 40,
	"angle_noise": .2,
	"radius_noise": .2,
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
	"min_consistency_degree": 2,
	"max_consistency_degree": 2,
	"DI_P": float("inf"),
	"GEN_P": 5,#float("inf"),
	"ga_iters": 20,
	"population_size": 30,
	"elite_proportion": .1,
	"crossover_rate": .8,
	"mutation_rate": .01,
	"missing_vectors_weight": .2,
	"init_consistent_shapes_ratio": .2,
	"max_init_consistency_shapes": 30
}

Ps = [1, 2, 5, 10, 20, -1]#[.1, .25, .5, .75, 1]
iters = 10
max_gen_degrees = [1, 2]
path_lens = [3, 4, 5]
noise_ratios = [.3, .5, .7, .9] #[.3, .4, .5, .6, .7]
algs = ["GA", "HA"]
matplotlib.rcParams.update({'font.size': 20})

#cumulative_path_lengths(iters=iters, num_nodess=[5, 10, 30, 50], max_gen_degrees=max_gen_degrees, params=params)
#fixed_path_lengths(iters=iters, num_nodess=[10, 20, 30], path_lens=path_lens, params=params)
#num_gen_components(iters=iters, num_nodess=[5, 10, 30, 50], Ps=Ps, params=params, k_as_ratios=False)

cmp_algs(iters=3, num_nodess=[10, 20, 30], noise_ratios=noise_ratios, params_=params, algs=algs, savefigs=True)#, draw_vectors=True)
