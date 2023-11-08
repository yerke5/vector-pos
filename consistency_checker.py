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
    def __init__(
            self, 
            params, 
            delta=1e-6, 
            supergene_delta=1e-6, 
            vector_generator=None, 
            verbose=False,
            try_negative_vectors=False, 
            include_baseline_vector=True, 
            mirror_consistent_vectors=True
        ):
        self.params = params 
        if not vector_generator:
            vector_generator = vg.VectorGenerator(params)
        else:
            self.vector_generator = vector_generator
        self.delta = delta
        self.supergene_delta = supergene_delta
        self.verbose = verbose
        self.try_negative_vectors = try_negative_vectors
        self.include_baseline_vector = include_baseline_vector
        self.mirror_consistent_vectors = mirror_consistent_vectors

    def get_consistent_vectors(self, vectors, max_shapes):
        temp = np.zeros((len(vectors), len(vectors), 2))
        temp[:, :] = np.nan
        shapes = dict()
        
        pairs = list(self.params.consistency_paths.keys())
        random.shuffle(pairs)

        k = 0
        for pair in pairs:
            #random.shuffle(self.params.consistency_paths[pair])
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

    def get_consistency_shapes(self, vectors, max_shapes):
        temp = np.zeros((len(vectors), len(vectors), 2))
        temp[:, :] = np.nan
        shapes = dict()
        
        pairs = list(self.params.consistency_paths.keys())
        random.shuffle(pairs)

        k = 0
        for pair in pairs:
            if vh.is_missing(vectors[pair[0]][pair[1]]):
                continue
            for path in self.params.consistency_paths[pair]:
                if len(path) < 3:
                    continue
                curr_path = ()
                generated = np.array([0, 0])
                broke = False
                for i in range(1, len(path)):
                    if not vh.is_missing(vectors[path[i - 1]][path[i]]):
                        generated = generated + vectors[path[i - 1]][path[i]]
                        curr_path = curr_path + ((path[i - 1], path[i]),)
                    elif self.try_negative_vectors and not vh.is_missing(vectors[path[i]][path[i - 1]]):
                        generated = generated - vectors[path[i]][path[i - 1]]
                        curr_path = curr_path + ((path[i], path[i - 1]),)
                    else:
                        broke = True 
                
                if broke:
                    continue 
                
                diff = np.sum((vectors[pair[0]][pair[1]] - generated)**2)
                if diff == 0:
                    k += 1

                    # path vectors
                    for (k1, k2) in curr_path:
                        if vh.is_missing(temp[k1][k2]):
                            #print("Copying", k1+1, "-", k2 + 1, "to consistent vectors")
                            temp[k1][k2] = vectors[k1][k2].copy()
                            if self.mirror_consistent_vectors and vh.is_missing(temp[k2][k1]):
                                #print("Copying the negative version of", k1+1, "-", k2 + 1, "to consistent vectors")
                                temp[k2][k1] = -vectors[k1][k2].copy()
                    
                    # baseline vector
                    if self.include_baseline_vector:
                        if vh.is_missing(temp[pair[0]][pair[1]]):
                            #print("Copying", pair[0]+1, "-", pair[1]+ 1, "to consistent vectors")
                            temp[pair[0]][pair[1]] = vectors[pair[0]][pair[1]].copy()
                        if self.mirror_consistent_vectors and vh.is_missing(vectors[pair[1]][pair[0]]):
                            temp[pair[0]][pair[1]] = -temp[pair[0]][pair[1]]
                    
                elif diff <= self.supergene_delta:
                    shapes[curr_path, pair] = np.sqrt(diff) 
                    k += 1

                if k >= max_shapes:
                    break
                
        for i in range(len(temp)):
            temp[i][i] = [0, 0]
        
        temp, nmv = self.vector_generator.get_inferrable_vectors(temp)

        if nmv > 0:         
            shapes = list(sorted(shapes.items(), key=lambda item: item[1]))

            deduction_complete = False
            for (path, pair), diff in shapes:
                # path vectors
                for (k1, k2) in path:
                    if vh.is_missing(temp[k1][k2]):
                        #print("Copying", k1+1, "-", k2+1, "as backup")
                        temp[k1][k2] = vectors[k1][k2].copy()
                        
                    if self.mirror_consistent_vectors and vh.is_missing(temp[k2][k1]):
                        temp[k2][k1] = -temp[k1][k2]

                    temp, nmv = self.vector_generator.get_inferrable_vectors(temp)
                    if nmv == 0:
                        deduction_complete = True 
                        break 
                if deduction_complete:
                    break 
                
                # baseline vector 
                if self.include_baseline_vector:
                    if vh.is_missing(temp[pair[0]][pair[1]]):
                        #print("Copying", pair[0]+1, "-", pair[1]+1, "as backup")
                        temp[pair[0]][pair[1]] = vectors[pair[0]][pair[1]].copy()
                    if self.mirror_consistent_vectors and vh.is_missing(vectors[pair[1]][pair[0]]):
                        temp[pair[0]][pair[1]] = -temp[pair[0]][pair[1]]
                    temp, nmv = self.vector_generator.get_inferrable_vectors(temp)
                    if nmv == 0:
                        break 

        return temp

    def get_supergenes(self, vectors, max_shapes):
        temp = np.zeros((len(vectors), len(vectors), 2))
        temp[:, :] = np.nan
        supergenes = set()
        
        pairs = list(self.params.consistency_paths.keys())
        random.shuffle(pairs)

        k = 0
        for pair in pairs:
            if vh.is_missing(vectors[pair[0]][pair[1]]):
                continue
            for path in self.params.consistency_paths[pair]:
                if len(path) < 3:
                    continue
                curr_path = ()
                generated = np.array([0, 0])
                broke = False
                for i in range(1, len(path)):
                    if not vh.is_missing(vectors[path[i - 1]][path[i]]):
                        generated = generated + vectors[path[i - 1]][path[i]]
                        curr_path = curr_path + ((path[i - 1], path[i]),)
                    elif self.try_negative_vectors and not vh.is_missing(vectors[path[i]][path[i - 1]]):
                        generated = generated - vectors[path[i]][path[i - 1]]
                        curr_path = curr_path + ((path[i], path[i - 1]),)
                    else:
                        broke = True 
                
                if broke:
                    continue 
                
                diff = np.sum((vectors[pair[0]][pair[1]] - generated)**2)
                if diff <= self.supergene_delta:
                    k += 1
                    if self.include_baseline_vector:
                        supergenes.add(pair)
                    for (k1, k2) in curr_path:
                        supergenes.add((k1, k2))
                        if self.mirror_consistent_vectors:
                            supergenes.add((k2, k1))
                        
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

    def build_consistent_vectors(self, vectors, max_num_consistency_shapes=float("inf"), include_baseline_vector=False):
        consistent_vectors = vh.get_empty_vec_matrix(len(vectors))
        shapes = dict()
        pairs = list(self.params.generation_paths.keys())
        coverage = self.init_coverage_matrix(vectors)
        
        if self.verbose:
            print("Initial coverage:", self.get_num_covered_vectors(coverage))

        covered = False  
        nnv = nmv = 0 
        for (start_vertex, end_vertex) in pairs:
            if not coverage[start_vertex][end_vertex]:
                if self.verbose:
                    vh.log(f"Performing dfs on M{start_vertex + 1}-{end_vertex + 1}")
                self.consistency_dfs(start_vertex, end_vertex, vectors, consistent_vectors, coverage, shapes, 0, max_num_consistency_shapes=max_num_consistency_shapes)
                
            if self.all_vectors_covered(coverage):
                covered = True 
                break 
        
        if True:#not covered:    
            consistent_vectors, nmv = self.vector_generator.get_inferrable_vectors(consistent_vectors, coverage=coverage)
            
            if nmv > 0:
                # copy most consistent vectors from measured vectors
                shapes = list(sorted(shapes.items(), key=lambda item: item[1]))
                deduction_complete = False
                for (path, pair), _ in shapes:
                    if include_baseline_vector and vh.is_missing(consistent_vectors[pair[0]][pair[1]]) and not vh.is_missing(vectors[pair[0]][pair[1]]):
                        consistent_vectors[pair[0]][pair[1]] = vectors[pair[0]][pair[1]].copy()
                    for (k1, k2) in path:
                        if vh.is_missing(consistent_vectors[k1][k2]):
                            consistent_vectors[k1][k2] = vectors[k1][k2].copy()
                            consistent_vectors, nmv = self.vector_generator.get_inferrable_vectors(consistent_vectors, coverage=coverage)
                            #print(f"Copying {k1+1}-{k2+1} from consistent shapes")
                            if nmv == 0:
                                deduction_complete = True 
                                break 
                    if deduction_complete:
                        break 
                
                if nmv > 0:
                    nnv = self.vector_generator.get_negative_vectors(consistent_vectors)
                
                # copy other missing vectors
                if nmv - nnv > 0:
                    for i in range(len(vectors)):
                        for j in range(len(vectors)):
                            if i != j and vh.is_missing(consistent_vectors[i][j]) and not vh.is_missing(vectors[i][j]):
                                #print(f"Copying {k1+1}-{k2+1} from measured")
                                consistent_vectors[i][j] = vectors[i][j].copy()
                    #consistent_vectors, nmv = self.vector_generator.get_inferrable_vectors(consistent_vectors, coverage=coverage)
                    consistent_vectors = self.vector_generator.get_generated_vectors(consistent_vectors)
        if self.params.noise_trim:
            consistent_vectors = vh.trim_noisy_vectors(consistent_vectors, self.params.space_size, self.params.max_range)
        
        return consistent_vectors, nmv - nnv

    # this function modifies the vector matrix in place to preserve consistent shapes
    def consistency_dfs(self, a, z, vectors, consistent_vectors, coverage, shapes, curr_num_shapes, max_num_consistency_shapes=float("inf")):
        if (curr_num_shapes >= max_num_consistency_shapes) or self.all_vectors_covered(coverage) or vh.is_missing(vectors[a][z]):
            return 
        
        for path in self.params.consistency_paths[(a, z)]:
            if len(path) < 3:
                continue 
            if path not in shapes:# and vh.is_valid(path, vectors):
                generated = np.array([0, 0])
                #negated = set()
                curr_path = ()
                broke = False 
                for i in range(1, len(path)):
                    if not vh.is_missing(vectors[path[i - 1]][path[i]]):
                        generated = generated + vectors[path[i - 1]][path[i]]
                        curr_path = curr_path + ((path[i-1], path[i]),)
                    elif self.try_negative_vectors and not vh.is_missing(vectors[path[i]][path[i - 1]]):
                        generated = generated - vectors[path[i]][path[i - 1]]
                        #negated.add((path[i - 1], path[i]))
                        curr_path = curr_path + ((path[i], path[i-1]),)
                    else:
                        broke = True 
                        break
                if broke:
                    continue 
                
                diff = np.sqrt(np.sum((vectors[a][z] - generated)**2))
  
                if diff <= 1e-6:
                    # baseline vector
                    coverage[a][z] = True 
                    if self.include_baseline_vector:
                        if vh.is_missing(consistent_vectors[a][z]):
                            consistent_vectors[a][z] = vectors[a][z].copy()
                    
                        if self.mirror_consistent_vectors and vh.is_missing(consistent_vectors[z][a]):
                            coverage[z][a] = True 
                            consistent_vectors[z][a] = -consistent_vectors[a][z]

                    # path vectors
                    uncovered = set()
                    for (k1, k2) in curr_path:#for i in range(1, len(path)):
                        # k1, k2 = path[i-1], path[i]
                        # if self.try_negative_vectors and (path[i - 1], path[i]) in negated:
                        #     k2, k1 = k1, k2
                        
                        if not coverage[k1][k2]:
                            uncovered.add((k1, k2))
                            coverage[k1][k2] = True 
                        
                        if vh.is_missing(consistent_vectors[k1][k2]):
                            if self.verbose:
                                print(f"Copying M{k1 + 1}-{k2 + 1} to consistent vectors")
                            consistent_vectors[k1][k2] = vectors[k1][k2].copy()
                            if self.mirror_consistent_vectors and vh.is_missing(consistent_vectors[k2][k1]):
                                consistent_vectors[k2][k1] = -consistent_vectors[k1][k2]
                                coverage[k2][k1] = True 
                    
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
                        shapes[curr_path, (a, z)] = diff 
                    else:
                        shapes[curr_path, (a, z)] = min(diff, shapes[path, (a, z)])

    def all_vectors_covered(self, covered):
        return np.all(covered) == True 
