import copy
import random
import numpy as np
import vec_manipulator as vm 
import vec_helper as vh
import vec_generator as vg
import operator
import math
import random

class ConsistencyChecker:
    def __init__(self, params, delta=1e-6, vector_generator=None, verbose=False):
        self.params = params 
        if not vector_generator:
            vector_generator = vg.VectorGenerator(params)
        else:
            self.vector_generator = vector_generator
        self.delta = 1e-6
        
        self.verbose = verbose

    def get_consistent_vectors(self, vectors, max_shapes=30):
        temp = np.zeros((len(vectors), len(vectors), 2))
        temp[:, :] = np.nan
        shapes = dict()
        
        pairs = list(self.params.consistency_paths.keys())
        random.shuffle(pairs)

        k = 0
        for pair in pairs:
            random.shuffle(self.params.consistency_paths[pair])
            for path in self.params.consistency_paths[pair]:
                if vh.is_valid(path, vectors):
                    generated = np.array([0, 0])
                    for i in range(1, len(path)):
                        generated = generated + vectors[path[i - 1]][path[i]]
                    
                    diff = np.sum((vectors[pair[0]][pair[1]] - generated)**2)

                    shapes[path] = np.sqrt(diff) 
                    k += 1

                if k >= max_shapes:
                    break
                
        for i in range(len(temp)):
            temp[i][i] = [0, 0]

        shapes = list(sorted(shapes.items(), key=lambda item: item[1]))
        
        for (path, diff) in shapes:
            if diff == 0:
                for i in range(1, len(path)):
                    temp[path[i - 1]][path[i]] = vectors[path[i - 1]][path[i]].copy()
                    #consistent.add((path[i - 1], path[i]))
            else:
                #inferred = infer_vectors(temp, pair_paths, max_infer_components, enforce_inference=True)
                temp, num_missing_vectors = self.vector_generator.get_inferrable_vectors(temp, coverage=None)
                if num_missing_vectors > 0:
                    for i in range(1, len(path)):
                        if vh.is_missing(temp[path[i - 1]][path[i]]):
                            temp[path[i - 1]][path[i]] = vectors[path[i - 1]][path[i]].copy()
                else:
                    break 

        return temp

    def get_consistency_shapes(self, vectors, max_shapes=30, delta=.3):
        temp = np.zeros((len(vectors), len(vectors), 2))
        temp[:, :] = np.nan
        shapes = dict()
        
        pairs = list(self.params.consistency_paths.keys())
        random.shuffle(pairs)

        k = 0
        for pair in pairs:
            random.shuffle(self.params.consistency_paths[pair])
            for path in self.params.consistency_paths[pair]:
                curr_path = ()
                generated = np.array([0, 0])
                for i in range(1, len(path)):
                    if not vh.is_missing(vectors[path[i - 1]][path[i]]):
                        generated = generated + vectors[path[i - 1]][path[i]]
                        curr_path = curr_path + ((path[i - 1], path[i]),)
                    elif not vh.is_missing(vectors[path[i]][path[i - 1]]):
                        generated = generated - vectors[path[i]][path[i - 1]]
                        curr_path = curr_path + ((path[i], path[i - 1]),)
                    else:
                        continue
                
                diff = np.sum((vectors[pair[0]][pair[1]] - generated)**2)
                if diff == 0:
                    for (k1, k2) in curr_path:
                        temp[k1][k2] = vectors[k1][k2].copy()
                        if vh.is_missing(temp[k2][k1]) and not vh.is_missing(vectors[k2][k1]):
                            temp[k2][k1] = vectors[k2][k1].copy()
                elif diff <= delta:
                    shapes[curr_path] = np.sqrt(diff) 
                
                k += 1

                if k >= max_shapes:
                    break
                
        for i in range(len(temp)):
            temp[i][i] = [0, 0]
        
        temp, _ = self.vector_generator.get_inferrable_vectors(temp)
                 
        shapes = list(sorted(shapes.items(), key=lambda item: item[1]))
        
        for (path, diff) in shapes:
            for (k1, k2) in path:
                if vh.is_missing(temp[k1][k2]):
                    temp[k1][k2] = vectors[k1][k2].copy()
                    temp, _ = self.vector_generator.get_inferrable_vectors(temp)
                    
        return temp

    def get_supergenes(self, vectors, max_shapes=20):
        shapes = dict()
        supergenes = set()
        k = 0
        for pair in self.params.consistency_paths:
            for path in self.params.consistency_paths[pair]:
                generated = np.array([0, 0])
                negated = set()
                for i in range(1, len(path)):
                    if not vh.is_missing(vectors[path[i - 1]][path[i]]):
                        generated = generated + vectors[path[i - 1]][path[i]]
                    elif not vh.is_missing(vectors[path[i]][path[i - 1]]):
                        generated = generated - vectors[path[i]][path[i - 1]]
                        negated.add((path[i - 1], path[i]))
                    else:
                        continue 
                
                diff = np.sum((vectors[pair[0]][pair[1]] - generated)**2)
                
                if diff == 0:
                    for i in range(1, len(path)):
                        if (path[i - 1], path[i]) in negated:
                            supergenes.add((path[i], path[i - 1]))
                        else:
                            supergenes.add((path[i - 1], path[i]))
                    k += 1

                if k >= max_shapes:
                    break
                
        return supergenes

    def init_coverage_matrix(self, vectors):
        covered = np.zeros((len(vectors), len(vectors))).astype(bool)
        np.fill_diagonal(covered, True)
        return covered

    def get_num_covered_vectors(self, coverage):
        return np.count_nonzero(coverage == True)

    def get_uncovered_vectors(self, coverage):
        uncovered = set()
        for i in range(len(coverage)):
            for j in range(len(coverage)):
                if not coverage[i][j]:
                    uncovered.add((i + 1, j + 1))
        return uncovered

    def build_consistent_vectors(self, vectors, max_num_consistency_shapes=float("inf")):
        consistent_vectors = vh.get_empty_vec_matrix(len(vectors))
        shapes = dict()
        pairs = list(self.params.generation_paths.keys())
        coverage = self.init_coverage_matrix(vectors)
        
        if self.verbose:
            print("Initial coverage:", self.get_num_covered_vectors(coverage))

        covered = False   
        for (start_vertex, end_vertex) in pairs:
            if not coverage[start_vertex][end_vertex]:
                if self.verbose:
                    vh.log(f"Performing dfs on M{start_vertex + 1}-{end_vertex + 1}")
                self.consistency_dfs(start_vertex, end_vertex, vectors, consistent_vectors, coverage, shapes, 0, max_num_consistency_shapes=max_num_consistency_shapes)
                
            if self.all_vectors_covered(coverage):
                covered = True 
                break 
            
        consistent_vectors, num_missing_vectors = self.vector_generator.get_inferrable_vectors(consistent_vectors, coverage=coverage)
        nmv = num_missing_vectors

        # copy most consistent vectors from measured vectors
        shapes = list(sorted(shapes.items(), key=lambda item: item[1]))
        deduction_complete = False
        for path, _ in shapes:
            for i in range(len(path)):
                if vh.is_missing(consistent_vectors[path[i - 1]][path[i]]):
                    consistent_vectors[path[i - 1]][path[i]] = vectors[path[i - 1]][path[i]].copy()
                    consistent_vectors, num_missing_vectors = self.vector_generator.get_inferrable_vectors(consistent_vectors, coverage=coverage)
        
                    if num_missing_vectors == 0:
                        deduction_complete = True 
                        break 
            if deduction_complete:
                break 
        
        nnv = self.vector_generator.get_negative_vectors(consistent_vectors)
        # copy other missing vectors
        for i in range(len(vectors)):
            for j in range(len(vectors)):
                if i != j and vh.is_missing(consistent_vectors[i][j]) and not vh.is_missing(vectors[i][j]):
                    consistent_vectors[i][j] = vectors[i][j].copy()

        if self.params.noise_trim:
            consistent_vectors = vh.trim_noisy_vectors(consistent_vectors, self.params.space_size, self.params.max_range)
        #print("Result:", vh.beautify_matrix(consistent_vectors))
        return consistent_vectors, nmv - nnv

    # this function modifies the vector matrix in place to preserve consistent shapes
    def consistency_dfs(self, a, z, vectors, consistent_vectors, coverage, shapes, curr_num_shapes, max_num_consistency_shapes=float("inf")):
        if (curr_num_shapes >= max_num_consistency_shapes) or self.all_vectors_covered(coverage) or vh.is_missing(vectors[a][z]):
            return 

        for path in self.params.consistency_paths[(a, z)]:
            if path not in shapes:# and vh.is_valid(path, vectors):
                generated = np.array([0, 0])
                negated = set()
                for i in range(1, len(path)):
                    if not vh.is_missing(vectors[path[i - 1]][path[i]]):
                        generated = generated + vectors[path[i - 1]][path[i]]
                    elif not vh.is_missing(vectors[path[i]][path[i - 1]]):
                        generated = generated - vectors[path[i]][path[i - 1]]
                        negated.add((path[i - 1], path[i]))
                    else:
                        continue
                
                diff = np.sqrt(np.sum((vectors[a][z] - generated)**2))

                if diff <= self.delta:
                    coverage[a][z] = True 
                    uncovered = set()
                    for i in range(1, len(path)):
                        k1, k2 = path[i-1], path[i]
                        if (path[i - 1], path[i]) in negated:
                            k2, k1 = k1, k2
                        
                        if not coverage[k1][k2]:
                            uncovered.add((k1, k2))
                            coverage[k1][k2] = True 
                            if self.verbose:
                                print(f"Copying M{k1 + 1}-{k2 + 1} to consistent vectors")
                            consistent_vectors[k1][k2] = vectors[k1][k2].copy()
                    
                    if len(uncovered) > 0:
                        inferred, num_missing_vectors = self.vector_generator.get_inferrable_vectors(consistent_vectors, coverage=coverage) 
                        consistent_vectors[:, :, :] = copy.deepcopy(inferred)
                        if num_missing_vectors == 0:
                            coverage[:, :] = True
                            return 

                    for (x, y) in uncovered:
                        #print("Starting to explore", x, "-", y)
                        self.consistency_dfs(x, y, vectors, consistent_vectors, coverage, shapes, curr_num_shapes + 1, max_num_consistency_shapes=max_num_consistency_shapes)
                else:
                    if path not in shapes:
                        shapes[path] = diff 
                    else:
                        shapes[path] = min(diff, shapes[path])

    def all_vectors_covered(self, covered):
        return np.all(covered) == True 
