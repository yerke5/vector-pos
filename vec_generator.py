import math 
import copy
import random
import numpy as np
import vec_manipulator as vm
import vec_helper as vh
import collections

class VectorGenerator:
	def __init__(self, params, verbose=False, sequential=False):
		self.params = params 
		self.verbose = verbose
		self.sequential = sequential

	def get_generated_vectors(self, measured):
		generated = np.zeros((len(measured), len(measured), 2))
		for i in range(len(generated)):
			generated[i][i] = np.nan

		num_perms = dict()
		strpaths = dict()

		#print('Number of generation_paths:', perms)
		for pair in self.params.generation_paths:
			random.shuffle(self.params.generation_paths[pair])
			
			if self.verbose and not vh.is_missing(measured[pair[0]][pair[1]]):
				print("[NOT DEDUCED] M" + str(pair[0] + 1) + str(pair[1] + 1) + "(g): ", end="")

			if not vh.is_missing(measured[pair[0]][pair[1]]):
				# # start out with the measured vector
				# generated[pair[0]][pair[-1]] = copy.deepcopy(measured[pair[0]][pair[1]])
				# if not pair in num_perms:
				# 	num_perms[pair] = 1

				# 	if self.verbose:
				# 		strpaths[pair] = [''.join([str(p + 1) for p in path])]
				
				# add intermediate generated vectors
				for path in self.params.generation_paths[pair]:
					# if len(path) < 3:
					# 	continue 
					gen = self.get_generated_vector(path, measured) 
					
					if gen is not None:
						generated[pair[0]][pair[-1]] += gen
						
						if not pair in num_perms:
							num_perms[pair] = 0
						num_perms[pair] += 1

						if num_perms[pair] >= self.params.max_gen_components:
							break

						if self.verbose:
									strpaths[pair].append(''.join([str(p + 1) for p in path]))
			else:
				res = self.infer_vector(measured, self.params.generation_paths[pair])
				if res is not None:
					inferred, mrpl = res
					generated[pair[0]][pair[1]] = inferred.copy()
				else:
					#print("Couldn't infer vector!", pair)
					generated[pair[0]][pair[1]] = np.nan
		
				num_perms[pair] = 1
				#generated[path[0]][path[-1]] /= (perms + 1)
				#print(f"After {path[0] + 1} - {path[-1] + 1}: ", generated[path[0]][path[-1]])
				#print('-' * 50)

			if not vh.is_missing(measured[pair[0]][pair[1]]):
				generated[pair[0]][pair[1]] /= num_perms[pair]
				if self.verbose:
					print("/", num_perms[pair], '=', generated[pair[0]][pair[1]])

		# infer missing vectors anyway
		#print("Number of missing vectors in generated version:", len(vh.get_missing(generated)))
		#generated = vg.get_inferrable_vectors(generated, generation_paths, max_infer_components, self.params.enforce_inference=self.params.enforce_inference)
		if self.verbose:
			for i in range(len(generated)):
				for j in range(len(generated)):
					if i != j:
						if (i, j) in strpaths:
							print("[NOT DEDUCED] M" + str(i + 1) + str(j + 1) + "(g): sum(" + "; ".join(strpaths[(i, j)]) + ") / " + str(num_perms[(i, j)]))
							#pass
						#generated[i][j] = generated[i][j] / num_perms[(i, j)]#(num_perms)#(perms + 1)
		return generated

	def get_generated_vector(self, path, measured):
		new_vector = np.array([0.0, 0.0])
		vecstr = []
		for i in range(1, len(path)): 
			try:
				if vh.is_missing(measured[path[i - 1]][path[i]]):
					return None 
			except:
				raise Exception("For some reason path is empty:", path, "index:", i, path[i - 1], path[i], len(measured), len(measured[0]), measured[1])
			
			try:
				new_vector += measured[path[i - 1]][path[i]]
			except:
				print("ERROR:", new_vector, measured[path[i-1]][path[i]])

			if self.verbose:
				vecstr.append(f"{measured[path[i - 1]][path[i]]}")
		
		if self.verbose:
			print("(" + " + ".join(vecstr) + ")" + " + ", end = "")

		return new_vector

	def infer_vectors(self, measured, generation_mode=False):
		#print('Incomplete measured:\n', measured)
		inferred = copy.deepcopy(measured)
		num_inferred_vectors = float('inf')
		inference_path_lengths = collections.defaultdict(int)
		# keep deducing until no vectors could be inferred

		while num_inferred_vectors > 0:
			num_inferred_vectors = 0
			for i in range(len(inferred)):
				for j in range(len(inferred)):
					if i != j and vh.is_missing(inferred[i][j]): # change ro inferred
						degree0 = False
						try:
							res = self.infer_vector(measured if not self.sequential else inferred, self.params.generation_paths[(i, j)], generation_mode=generation_mode) # change ro inferred
						except:
							print(i, j, len(measured))
						if res is not None:
							inferred[i][j], rmpl = res
							
							if rmpl != -1:
								inference_path_lengths[rmpl] += 1
							else:
								degree0 = True 
						else: 
							degree0 = True 
						
						if degree0:
							# in the worst case, set it to -inferred[j][i]
							if not vh.is_missing(measured[j][i] if not self.sequential else inferred[j][i]): # change ro inferred
								if (not generation_mode) or vh.is_missing(inferred[i][j]):
									inferred[i][j] = (-measured[j][i] if not self.sequential else -inferred[j][i])

								inferred[i][j] = (inferred[i][j] + (-measured[j][i] if not self.sequential else -inferred[j][i]))/2 # change ro inferred
								inference_path_lengths[2] += 1
							elif self.params.enforce_inference:
								return None
							else:
								num_inferred_vectors -= 1
						num_inferred_vectors += 1
			
			if not self.sequential:
				break
						
		return inferred, inference_path_lengths

	def get_inferrable_vectors(self, vectors, coverage=None):
		inferred = copy.deepcopy(vectors)
		num_missing_vectors = 0
		num_inferred_vectors = float('inf')
		# keep deducing until no vectors could be inferred

		while num_inferred_vectors > 0:
			num_inferred_vectors = 0
			for i in range(len(vectors)):
				for j in range(len(vectors)):
					if i != j and vh.is_missing(inferred[i][j]):
						was_inferred = False
						res = self.infer_vector(inferred, self.params.generation_paths[(i, j)])
						if res is None:
							num_missing_vectors += 1
						else:
							inferred[i][j], mrpl = res
							was_inferred = True 

						if not was_inferred and not vh.is_missing(inferred[j][i]):
							inferred[i][j] = -1 * inferred[j][i]
							was_inferred = True 
						
						if was_inferred:
							if coverage is not None:
								coverage[i][j] = True
							num_inferred_vectors += 1
		
		return inferred, num_missing_vectors

	def infer_vector(self, measured, paths, generation_mode=False):
		assert self.params.max_infer_components >= 1
		# categorise paths by length
		path_lens, min_ipl, max_ipl = vh.paths2len_dict(paths)
		#print('Min ipl:', min_ipl, "max ipl:", max_ipl)
		#print(path_lens)
		#print(paths)
		ndc = 0
		dv = np.zeros(2,).astype(np.float32)
		reached_max_ipl = -1
		
		for inference_path_length in range(min_ipl, max_ipl + 1):
			
			# if inference_path_length > 3:
			# 	print(f"Elevating inference level to {inference_path_length}!")
			for path in path_lens[inference_path_length]:
				new_vector = np.array([0.0, 0.0])
				broke = False
				for k in range(1, len(path)): 
					if not vh.is_missing(measured[path[k - 1]][path[k]]):
						new_vector += measured[path[k - 1]][path[k]]
					else:
						broke = True 
						break

				if not broke:
					dv += new_vector
					ndc += 1
				
			if ndc >= self.params.max_infer_components:
				if reached_max_ipl == -1:
					reached_max_ipl = inference_path_length
				break 
			reached_max_ipl = max(inference_path_length, reached_max_ipl)

		if ndc == 0: # used to be 2 
			return None  

		if generation_mode:
			x = list(paths)[0]
			if not vh.is_missing(measured[x[0]][x[-1]]):
				print(f"Adding measured vector to {x[0]}-{x[-1]}")
				dv += measured[x[0]][x[-1]]
				ndc += 1

		return dv / ndc, reached_max_ipl

	def is_inferable(self, i1, i2, vectors):
		t = 0
		for k in range(len(vectors)):
			if k != i1 and k != i2:
				if not vh.is_missing(vectors[i1][k]) and not vh.is_missing(vectors[k][i2]):
					t += 1
		return t >= self.params.max_infer_components

	def get_negative_vectors(self, vectors):
		n = 0
		for i in range(len(vectors)):
			for j in range(len(vectors)):
				if vh.is_missing(vectors[i][j]) and not vh.is_missing(vectors[j][i]):
					vectors[i][j] = -1 * vectors[j][i]
					n += 1
		return n