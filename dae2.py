import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from nn_flatten import generate_dataset, calculate_error
from sklearn.preprocessing import StandardScaler
import os
import vec_generator as vg
import vec_helper as vh
import vec_manipulator as vm
import random

num_samples = 10000
num_nodes = 10
space_size = 20
normalise = True
flatten = True 
noise_ratio = .5
angle_noise = radius_noise = .1

signature = f"{num_nodes}_{num_samples}_{space_size}_{angle_noise}_{noise_ratio}"
if os.path.isfile(f"x_train_noisy_" + signature + ".txt"):
    x_train_noisy = np.loadtxt(f"x_train_noisy_" + signature + ".txt")
    x_train = np.loadtxt(f"x_train_" + signature + ".txt")
    x_test_noisy = np.loadtxt(f"x_test_noisy_" + signature + ".txt")
    x_test = np.loadtxt(f"x_test_" + signature + ".txt")
else:
    res = generate_dataset(num_samples, num_nodes, space_size, flatten=flatten, normalise=False, angle_noise=angle_noise, radius_noise=radius_noise, noise_ratio=noise_ratio)
    print("Finished generating training dataset")
    x_train_noisy, x_test_noisy, x_train, x_test = res["x_train_noisy"], res["x_test_noisy"], res["x_train"], res["x_test"]
    np.savetxt("x_train_noisy_" + signature + ".txt", x_train_noisy)
    np.savetxt("x_train_" + signature + ".txt", x_train)
    np.savetxt("x_test_noisy_" + signature + ".txt", x_test_noisy)
    np.savetxt("x_test_" + signature + ".txt", x_test)

# normalise
if normalise:
    noisy_scaler, true_scaler = StandardScaler(), StandardScaler()
    x_train_noisy = noisy_scaler.fit_transform(x_train_noisy)
    x_train = true_scaler.fit_transform(x_train)
    x_test_noisy = noisy_scaler.transform(x_test_noisy)
    x_test = true_scaler.transform(x_test)

input_dim = x_train.shape[1]
# create model
model = Sequential()
#model.add(Dense(512, input_dim=input_dim, activation='relu'))
#model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
#model.add(Dense(512, input_dim=input_dim, activation='relu'))
#model.add(Dense(256, activation='relu'))
model.add(Dense(input_dim, activation='relu'))

# compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# train the model
model.fit(x_train_noisy, x_train, validation_data=(x_test_noisy, x_test), epochs=30, batch_size=64)

# draw a random set of vectors
x_pred = model.predict(x_test_noisy)

# undo normalisation
if normalise:
    x_test_noisy = noisy_scaler.inverse_transform(x_test_noisy)
    x_pred = true_scaler.inverse_transform(x_pred)
    x_test = true_scaler.inverse_transform(x_test)

x_test_noisy = x_test_noisy.reshape(-1, num_nodes, num_nodes, 2)
x_pred = x_pred.reshape(-1, num_nodes, num_nodes, 2)
x_test = x_test.reshape(-1, num_nodes, num_nodes, 2)

random_idx = random.randint(0, x_pred.shape[0])
noisy = x_test_noisy[random_idx]
predicted = x_pred[random_idx]
actual = x_test[random_idx]

true_coords = [[0, 0]] + [actual[0][i] for i in range(1, num_nodes)]
vh.draw_vectors(space_size, true_coords, noisy, "Measured", show=True)
vh.draw_vectors(space_size, true_coords, predicted, "Predicted", show=True)
vh.draw_vectors(space_size, true_coords, actual, "Actual", show=True)

print("Noise:", calculate_error(noisy, actual))
print("Predicted vector error:", calculate_error(predicted, actual))