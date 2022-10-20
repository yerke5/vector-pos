import vec_generator as vg
import vec_helper as vh
import vec_manipulator as vm
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

def calculate_error(m1, m2):
	error = 0
	n = 0
	for i in range(len(m1)):
		for j in range(len(m1)):
			if i != j and not vh.is_missing(m1[i][j]) and not vh.is_missing(m2[i][j]):
				n += 1 
				error += np.sqrt(np.sum((m1[i][j] - m2[i][j])**2))

	#print("Calculated error for", n, "vectors")
	return error / n

def extract_vec_idxs(df):
	idxs = []
	for _, r in df.iterrows():
		idxs.append([])
		for col in range(num_nodes):
			if r[str(col)] == 1:
				idxs[-1].append(col)
				if len(idxs[-1]) == 2:
					if r["Direction"] == -1:
						idxs[-1] = idxs[-1][::-1]
					break

	assert np.array(idxs).shape == (df.shape[0], 2)
	return idxs

def df_to_vec_matrix(df, num_nodes):
	vector_matrix = np.zeros((num_nodes, num_nodes, 2))
	vector_matrix[:, :] = np.nan

	for k in range(df.shape[0]):
		i, j = df.loc[k, "Index"] # map(int, X_test.loc[k, "Index"][1:].split("-"))
		vector_matrix[i][j] = np.array([df.loc[k, "VectorX"], df.loc[k, "VectorY"]])

	return vector_matrix

def flatten_vec_matrix(matrix):
	record = []
	for i in range(len(matrix)):
		for j in range(len(matrix)):
			record += list(matrix[i][j]) if not vh.is_missing(matrix[i][j]) and i != j else [0, 0]
	return record

def unflatten_record(record):
	num_nodes = int(np.sqrt(len(record) // 2))
	matrix = np.zeros((num_nodes, num_nodes, 2))
	matrix[:, :] = np.nan

	for i in range(num_nodes):
		for j in range(num_nodes):
			end_idx = 2 * (i * (num_nodes) + j)
			if record[end_idx] != 0 and record[end_idx + 1] != 0:
				matrix[i][j] = [record[end_idx], record[end_idx + 1]]

	return matrix

def generate_dataset(
	num_samples, 
	num_nodes, 
	space_size, 
	num_deduction_components=1, 
	noise_ratio=.5, 
	angle_noise=.1, 
	radius_noise=.1, 
	max_angle=40, 
	max_range=30, 
	verbose=False, 
	enforce_deduction=False, 
	flatten=True, 
	normalise=True, 
	custom_missing_value=0
):
	noise = 0
	X, y = [], []
	paths = vh.paths2dict(vh.generate_paths(num_nodes))

	for k in range(num_samples):
		# generate ground truth vectors
		true_nodes = np.array([vg.generate_random_node(space_size) for i in range(num_nodes)])
		true_vectors = vg.coords2vectors(true_nodes)

		# filter unseen vectors
		filtered = vm.drop_unseen_vectors(true_vectors, true_nodes, max_angle=max_angle, max_range=max_range, verbose=verbose)
		#print("Detection rate after filtering unreachable nodes:", 1 - len(vh.get_missing(filtered)) / (num_nodes ** 2 - num_nodes))

		# deduce missing vectors
		deduced = vm.deduce_vectors(filtered, paths, num_deduction_components, enforce_deduction=enforce_deduction)
		#print("Detection rate after inferring missing vectors:", 1 - len(vh.get_missing(deduced)) / (num_nodes ** 2 - num_nodes))

		# inject noise into filtered vectors
		measured = vm.inject_noise(deduced, noise_ratio, space_size, true_nodes, angle_noise, radius_noise)

		#measured = vh.replace_missing_vectors(measured, 0)
		#generated = vg.get_generated_vectors(measured, paths, num_deduction_components)
		noise += calculate_error(measured, true_vectors)
		#gen_error += calculate_error(generated, true_vectors)

		measured = np.where(np.isnan(measured), custom_missing_value, measured)
		X.append(measured.reshape(-1,) if flatten else measured)
		y.append(true_vectors.reshape(-1,) if flatten else true_vectors)

		if k > 0 and k % (num_samples // 5) == 0:
			print(f"Generated {k / num_samples * 100}% of the dataset")
	
	X, y = np.array(X), np.array(y)
	x_train_noisy, x_test_noisy, x_train, x_test = train_test_split(X, y, test_size=0.33, random_state=42)

	res = {}
	# preprocessing (only fit on the training dataset but scale both input sets)
	if normalise:
		noisy_scaler, true_scaler = StandardScaler(), StandardScaler()
		x_train_noisy = noisy_scaler.fit_transform(x_train_noisy)
		x_train = true_scaler.fit_transform(x_train)
		x_test_noisy = noisy_scaler.transform(x_test_noisy)
		x_test = true_scaler.transform(x_test)
		res = {
			"noisy_scaler": noisy_scaler,
			"true_scaler": true_scaler
		}

	res = {**res, **{
		"x_train_noisy": x_train_noisy, 
		"x_test_noisy": x_test_noisy, 
		"x_train": x_train, 
		"x_test": x_test
	}}

	res["average noise"] = noise / num_samples

	return res

if __name__ == "__main__":
	num_nodes = 10
	max_angle = 40
	max_range = 30
	space_size = 50
	verbose = False 
	enforce_deduction = False #True 
	noise_ratio = .7
	angle_noise = radius_noise = .1
	num_deduction_components = 1
	LARGE_VALUE = 0#1e6
	normalise = True 
	include_missing_vectors = False
	paths = vh.paths2dict(vh.generate_paths(num_nodes))
	num_samples = 1000

	res = generate_dataset(num_samples, num_nodes, space_size, num_deduction_components=num_deduction_components, noise_ratio=noise_ratio, angle_noise=angle_noise, radius_noise=radius_noise, max_angle=max_angle, max_range=max_range, verbose=False, enforce_deduction=False, flatten=True, normalise=normalise, custom_missing_value=0)

	X_train, X_test, y_train, y_test = res["x_train_noisy"], res["x_test_noisy"], res["x_train"], res["x_test"]

	# calculate error
	print("Average noise:", res["average noise"])
	#print("Average generated error:", gen_error / num_runs)

	# training
	model = MLPRegressor(hidden_layer_sizes=(32,32))#SVR(C=1.0, epsilon=0.2)
	model.fit(X_train, y_train)

	# generate predictions
	y_pred = model.predict(X_test)

	if normalise:
		true_scaler, noisy_scaler = res["true_scaler"], res["noisy_scaler"]
		# convert everything back to scale
		y_pred = true_scaler.inverse_transform(y_pred)
		X_test = noisy_scaler.inverse_transform(X_test)
		X_train = noisy_scaler.inverse_transform(X_train)
		y_train = true_scaler.inverse_transform(y_train)
		y_test = true_scaler.inverse_transform(y_test)

	error = 0
	for i in range(y_pred.shape[0]):
		predicted = unflatten_record(y_pred[i])
		actual = unflatten_record(y_test[i])
		error += calculate_error(predicted, actual)

	print("Average vector error:", error / y_pred.shape[0])

	# plot the first predicted set of vectors
	predicted = unflatten_record(y_pred[0])
	actual = unflatten_record(y_test[0])

	true_coords = [[0, 0]] + [actual[0][i] for i in range(1, num_nodes)]
	vh.draw_vectors(space_size, true_coords, predicted, "Predicted", show=True)
	vh.draw_vectors(space_size, true_coords, actual, "Actual", show=True)