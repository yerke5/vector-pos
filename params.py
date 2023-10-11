import vec_helper as vh
import numpy as np

class Params:
	def __init__(
		self, 
		num_nodes,
		space_size,
		max_infer_components=float("inf"), 
		max_gen_components=float("inf"), 
		min_gen_degree=1, 
		max_gen_degree=1, 
		min_consistency_degree=2, 
		max_consistency_degree=2,
		noise_trim=False,
		max_range=40,
		max_angle=50,
		enforce_inference=False,
		num_anchors=0
	):
		self.space_size = space_size
		self.max_infer_components = max_infer_components
		self.max_gen_components = max_gen_components
		self.min_gen_degree = min_gen_degree
		self.max_gen_degree = max_gen_degree
		self.min_consistency_degree = min_consistency_degree
		self.max_consistency_degree = max_consistency_degree
		self.enforce_inference = enforce_inference
		self.noise_trim = noise_trim
		self.generation_paths = vh.paths2dict(vh.generate_paths(num_nodes + num_anchors, min_degree=min_gen_degree, max_degree=max_gen_degree))
		self.consistency_paths = None 
		self.max_range = max_range
		self.max_angle = max_angle
		self.num_anchors = num_anchors
		if max_consistency_degree > 0:
			self.consistency_paths = vh.paths2dict(vh.generate_paths(num_nodes + num_anchors, min_degree=min_consistency_degree, max_degree=max_consistency_degree))
