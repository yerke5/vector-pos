import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv, GATv2Conv
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import vec_helper as vh 
import vec_generator as vg 
import consistency_checker as vc 
import vec_manipulator as vm
from params import Params
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.nn import Linear, ReLU, Sequential
import os
from sklearn.preprocessing import StandardScaler
import copy 
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd 
import time 

matplotlib.rcParams.update({'font.size': 20})

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

def generate_params(num_nodess):
    vecgens = {}
    for num_nodes in num_nodess:
        params = Params(
            num_nodes,
            40,
            num_anchors=0,
            max_infer_components=num_nodes-1,
            max_gen_components=num_nodes-1,
            min_gen_degree=1,
            max_gen_degree=1,
            max_consistency_degree=0,
            noise_trim=False,
            max_range=40,
            max_angle=50,
            enforce_inference=False
        )

        vecgens[num_nodes] = vg.VectorGenerator(params=params, verbose=False, sequential=False)

    return vecgens

def cmp_algs(space_size, noisy_coords, true_coords, predicted_coords, edge_attr, edge_index, average_features, input_generated=True, vecgens=None, draw=False, polar=False):
    noisy_coords = noisy_coords.cpu().detach().numpy()
    edge_index = edge_index.cpu().detach().numpy()

    if not average_features:
        edge_attr = np.mean(edge_attr.cpu().detach().numpy().reshape(len(edge_attr), len(edge_attr), -1, 2), axis=-2)
    else:
        edge_attr = edge_attr.cpu().detach().numpy()

    gnn_vectors = vh.coords2vectors(predicted_coords)
    true_vectors = vh.coords2vectors(true_coords)

    # generated
    num_nodes = len(true_coords)

    if input_generated:
        generated = vh.sparse2vecs(edge_attr, edge_index, num_nodes)
        if polar:
            generated = vh.polar2cartesian(generated)
        
        if average_features:
            measured = vh.coords2vectors(noisy_coords)
        else:
            measured = vh.coords2vectors(np.mean(noisy_coords.reshape(noisy_coords.shape[0], -1, 2), axis=1))

    else:
        measured = vh.sparse2vecs(edge_attr, edge_index, num_nodes)
        if polar:
            measured = vh.polar2cartesian(measured)
        
        if vecgens is None:
            params = Params(
                num_nodes,
                space_size,
                num_anchors=0,
                max_infer_components=num_nodes-1,
                max_gen_components=num_nodes-1,
                min_gen_degree=1,
                max_gen_degree=1,
                max_consistency_degree=0,
                noise_trim=False,
                max_range=40,
                max_angle=50,
                enforce_inference=False
            )
            vecgen = vg.VectorGenerator(params=params, verbose=False, sequential=False)
        else:
            vecgen = vecgens[num_nodes]
        
        st = time.time()
        generated = vecgen.get_generated_vectors(measured)
        generated_rt = time.time() - st

    errors = {
        "Measured vectors": vh.calculate_error(measured, true_vectors),
        #"HA": vh.calculate_error(ha_res, true_vectors),
        "Generated vectors": vh.calculate_error(generated, true_vectors),
        "CollabGNN": vh.calculate_error(gnn_vectors, true_vectors)
    }

    rates = {
        "Measured vectors": (1 - len(vh.get_missing(measured)) / (num_nodes * (num_nodes - 1))) * 100,
        #"HA": vh.calculate_error(ha_res, true_vectors),
        "Generated vectors": (1 - len(vh.get_missing(generated)) / (num_nodes * (num_nodes - 1))) * 100,
        "CollabGNN": (1 - len(vh.get_missing(gnn_vectors)) / (num_nodes * (num_nodes - 1))) * 100
    }

    if draw:
        vh.draw_vectors(None, true_coords, gnn_vectors, "GNN vectors", max_angle=50, show=True)
        vh.draw_vectors(None, true_coords, measured, "Measured vectors", max_angle=50, show=True)
        vh.draw_vectors(None, true_coords, generated, "Generated vectors", max_angle=50, show=True)

    return errors, rates, num_nodes, generated_rt

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def generate_dataset(
    train_size,
    num_nodess=None,
    generate_vectors=True,
    constant_node_features=False,
    average_timestamps=True,
    min_num_nodes=50,
    max_num_nodes=100,
    min_space_size=10,
    max_space_size=50,
    min_noise_ratio=1,
    max_noise_ratio=1,
    num_anchors=0,
    min_angle_noise=.05,
    max_angle_noise=.1,
    save_data_dir=None,
    num_timestamps=5,
    random_orientations=True,
    polar=False,
    debug=False
):
    max_range = 40
    max_angle = 50
    dataset = []
    xs, ys, edge_indexes, edge_attrs = [], [], [], []

    if num_nodess is not None and generate_vectors:
        vecgens = generate_params(num_nodess)

    for i in range(train_size):
        if i % 10 == 0:
            print("Iteration", i)

        if save_data_dir and os.path.isfile(f"{save_data_dir}/noisy-coords-{i}.npy"):
            noisy_coords = np.load(f'{save_data_dir}/noisy-coords-{i}.npy', allow_pickle=True)
            true_coords = np.load(f'{save_data_dir}/true-coords-{i}.npy', allow_pickle=True)
            edge_index = np.load(f'{save_data_dir}/edge-index-{i}.npy', allow_pickle=True)
            edge_attr = np.load(f'{save_data_dir}/edge-attr-{i}.npy', allow_pickle=True)
        else:
            num_nodes = int(np.random.random() * (max_num_nodes - min_num_nodes) + min_num_nodes) if num_nodess is None else num_nodess[i % len(num_nodess)]
            space_size = int(np.random.random() * (max_space_size - min_space_size) + min_space_size) #np.random.randint(min_space_size, max_space_size)
            noise_ratio = np.random.random() * (max_noise_ratio - min_noise_ratio) + min_noise_ratio
            angle_noise = radius_noise = np.random.random() * (max_angle_noise - min_angle_noise) + min_angle_noise #np.random.uniform(min_angle_noise, max_angle_noise)

            true_coords = np.array(vh.generate_random_nodes(num_nodes, space_size, num_anchors=num_anchors))
            true_vectors = vh.coords2vectors(true_coords)

            true_coords = vh.vecs2coords(true_vectors, space_size) # always get coordinates from node 0's perspective

            orientations = vh.get_orientations(true_coords, space_size, random=random_orientations)
            filtered = vm.drop_unseen_vectors(true_vectors, orientations, max_angle=max_angle, max_range=max_range, verbose=False)

            # inject noise into filtered vectors
            measured_vectors = []
            noisy_coordss = []
            for _ in range(num_timestamps):
                measured = vm.inject_noise(filtered, noise_ratio, space_size, true_coords, angle_noise, radius_noise, num_anchors=num_anchors)

                #noisy_coords = np.array([vm.add_noise(node, space_size, radius_noise, angle_noise) for node in true_coords])
                noisy_coords = vh.vecs2coords(measured, space_size) #/ space_size
                measured_vectors.append(measured)
                noisy_coordss.append(copy.deepcopy(noisy_coords))

            measured = np.concatenate(measured_vectors, axis=-1)
            if average_timestamps:
                measured = np.mean(measured.reshape(measured.shape[0], measured.shape[1], -1, 2), axis=-2)
                if polar:
                    measured = vh.cartesian2polar(measured)

                if num_nodess is not None and generate_vectors:
                    if polar:
                        measured = vh.polar2cartesian(measured)
                    measured = vecgens[num_nodes].get_generated_vectors(measured)
                    if polar:
                        measured = vh.cartesian2polar(measured)

            if constant_node_features:
                noisy_coords = np.array([1] * num_nodes)
            else:
                noisy_coords = np.concatenate(noisy_coordss, axis=-1)
                if average_timestamps:
                    noisy_coords = vh.vecs2coords(vh.polar2cartesian(measured) if polar else measured, space_size) #np.mean(noisy_coords.reshape(noisy_coords.shape[0], -1, 2), axis=1)
                    
            edge_index, edge_attr = vh.vecs2sparse(measured)
            
            if save_data_dir:
                if not os.path.isdir(save_data_dir):
                    os.mkdir(save_data_dir)

                np.save(f'{save_data_dir}/noisy-coords-{i}.npy', noisy_coords)
                np.save(f'{save_data_dir}/true-coords-{i}.npy', true_coords)
                np.save(f'{save_data_dir}/edge-index-{i}.npy', edge_index)
                np.save(f'{save_data_dir}/edge-attr-{i}.npy', edge_attr)

        xs.append(copy.deepcopy(noisy_coords))
        ys.append(copy.deepcopy(true_coords))
        edge_indexes.append(copy.deepcopy(edge_index))
        edge_attrs.append(copy.deepcopy(edge_attr))

    xs, ys, edge_indexes, edge_attrs = np.array(xs), np.array(ys), (edge_indexes), (edge_attrs)

    for x, y, edge_attr, edge_index in zip(xs, ys, edge_attrs, edge_indexes):
        x = torch.tensor(x.reshape(-1, 1) if constant_node_features else x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        graph_data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        dataset.append(graph_data)

    return dataset

class DynamicGCN(nn.Module):
    def __init__(self, num_node_feats, num_edge_feats, hidden_units, out_channels, nn1h=2, nnih=8, layer_type="nnconv", num_hidden_layers=1):
        super(DynamicGCN, self).__init__()
        self.layers = nn.ModuleList()
        if layer_type == "nnconv":
            if num_hidden_layers > 0:
                self.initial_layer = NNConv(num_node_feats, hidden_units, nn=nn.Sequential(
                    nn.Linear(num_edge_feats, nn1h),
                    nn.ReLU(),
                    # nn.Linear(nn1h, nn1h),
                    # nn.ReLU(),
                    nn.Linear(nn1h, num_node_feats * hidden_units)
                ), aggr="mean")
            else:
                self.initial_layer = NNConv(num_node_feats, out_channels, nn=nn.Sequential(
                    nn.Linear(num_edge_feats, nn1h),
                    nn.ReLU(),
                    # nn.Linear(nn1h, nn1h),
                    # nn.ReLU(),
                    nn.Linear(nn1h, num_node_feats * out_channels)
                ), aggr="mean")
            
        elif layer_type == "gatv2conv":
            if num_hidden_layers > 0:
                self.initial_layer = GATv2Conv(num_node_feats, hidden_units, edge_dim=num_edge_feats, add_self_loops=False, concat=False)
            else:
                self.initial_layer = GATv2Conv(num_node_feats, out_channels, edge_dim=num_edge_feats, add_self_loops=False, concat=False)

        for i in range(num_hidden_layers - 1):
            if layer_type == "nnconv":
                self.layers.append(
                    NNConv(hidden_units, hidden_units, nn=nn.Sequential(
                        nn.Linear(num_edge_feats, nnih),
                        nn.ReLU(),
                        nn.Linear(nnih, hidden_units * hidden_units)
                    ), aggr="mean")
                )
            elif layer_type == "gatv2conv":
                self.layers.append(
                    GATv2Conv(hidden_units, hidden_units, edge_dim=num_edge_feats, add_self_loops=False, concat=False)
                )
        
        if num_hidden_layers > 0:
            if layer_type == "nnconv":
                self.final_layer = NNConv(hidden_units, out_channels, nn=nn.Sequential(
                    nn.Linear(num_edge_feats, nnih),
                    nn.ReLU(),
                    # nn.Linear(nnih, nnih),
                    # nn.ReLU(),
                    nn.Linear(nnih, hidden_units * out_channels)
                ), aggr="mean")
        
            elif layer_type == "gatv2conv":
                self.final_layer = GATv2Conv(hidden_units, out_channels, edge_dim=num_edge_feats, add_self_loops=False, concat=False)
        else:
            self.final_layer = None

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.initial_layer(x, edge_index, edge_attr)
        if self.final_layer is not None:
            x = F.relu(x)

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x, edge_index, edge_attr))
        
        if self.final_layer is not None:
            x = self.final_layer(x, edge_index, edge_attr)
        return x

save_data_dir = "training-graphs"
num_epochs = 100
in_features = 1
hidden_units = 256
out_features = 1
train_size = 1000
num_timestamps = 1
min_num_nodes = 50
max_num_nodes = 100
average_timestamps = True
constant_node_features = False
input_generated = False
train_batch_size = test_batch_size = 1
random_orientations = True
space_size = 30
min_angle_noise = max_angle_noise = .2
polar = False
debug = False
num_nodess = [50, 60, 70]

if not os.path.isdir(save_data_dir):
    os.makedirs(save_data_dir)

initial_dataset = generate_dataset(
    train_size,
    generate_vectors=input_generated,
    constant_node_features=constant_node_features,
    average_timestamps=average_timestamps,
    num_nodess=num_nodess,
    min_noise_ratio=1,
    max_noise_ratio=1,
    min_num_nodes=min_num_nodes,
    max_num_nodes=max_num_nodes,
    num_anchors=0,
    min_angle_noise=min_angle_noise, #.05
    max_angle_noise=max_angle_noise, #.05
    num_timestamps=num_timestamps,
    save_data_dir=save_data_dir,
    min_space_size=space_size,
    max_space_size=space_size,
    random_orientations=random_orientations,
    polar=polar,
    debug=debug
)

dataset = train_val_dataset(initial_dataset, val_split=.1)
print("Size of dataset:", len(dataset["train"]))
train_dataloader = DataLoader(dataset["train"], batch_size=train_batch_size, shuffle=False)
val_dataloader = DataLoader(dataset["val"], batch_size=test_batch_size, shuffle=False)

model = DynamicGCN(
    num_node_feats=1 if constant_node_features else (2 if average_timestamps else 2 * num_timestamps),
    num_edge_feats=2 if average_timestamps else 2 * num_timestamps,
    hidden_units=hidden_units,
    out_channels=2,
    nn1h=8,
    nnih=8,
    layer_type="nnconv",
    num_hidden_layers=1
)# 2*num_timestamps

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if os.path.isfile("model.pt"):
    model = torch.load("model.pt")
else:
    model.train()  # Set the model to training mode
    
    for epoch in range(num_epochs):
        for data in train_dataloader:  # Iterate over batches of data
            optimizer.zero_grad()  # Zero the gradients
            output = model(data)  # Forward pass
            loss = criterion(output, data.y)  # Compute the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

        # Validation
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            validation_loss = 0.0
            example = True
            for val_data in val_dataloader:  # Iterate over validation data
                val_output = model(val_data)

                val_loss = criterion(val_output, val_data.y)
                validation_loss += val_loss.item()

        # Calculate and print average validation loss
        average_validation_loss = validation_loss / len(val_dataloader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Validation Loss: {average_validation_loss}')

    if not os.path.isfile("model.pt"):
        torch.save(model, "model.pt")
# Testing
model.eval()
test_loss = 0.0
k = 0
example = True
all_errors = {}
all_rates = {}
all_rts = {}

with torch.no_grad():
    st = time.time()
    for test_data in val_dataloader:  # Iterate over test data
        st = time.time()
        test_output = model(test_data)
        gnn_rt = time.time() - st
        test_loss += criterion(test_output, test_data.y)
        if k % 2 == 0:
            print("Iteration", k, "/", len(val_dataloader))
        k+=1
        
        errors, rates, num_nodes, generated_rt = cmp_algs(
            space_size,
            test_data.x,
            test_data.y,
            test_output,
            test_data.edge_attr,
            test_data.edge_index,
            average_timestamps,
            input_generated=input_generated,
            draw=False,#k<10,
            polar=polar,
            debug=debug
        )
        if k < 10:
            print(errors)
        
        for key in errors:
            if key not in all_errors:
                all_errors[key] = {}
                all_rates[key] = {}
                if key != "Measured vectors":
                    all_rts[key] = {}
            if num_nodes not in all_errors[key]:
                all_errors[key][num_nodes] = []
                all_rates[key][num_nodes] = []
                if key != "Measured vectors":
                    all_rts[key][num_nodes] = []
            all_errors[key][num_nodes].append(errors[key])
            all_rates[key][num_nodes].append(rates[key])
        
        all_rts["Generated vectors"][num_nodes].append(generated_rt)
        all_rts["CollabGNN"][num_nodes].append(gnn_rt)
    
for key in all_errors:
    for n in all_errors[key]:
        all_errors[key][n] = np.mean(all_errors[key][n])
        all_rates[key][n] = np.mean(all_rates[key][n])
        if key != "Measured vectors":
            all_rts[key][n] = np.mean(all_rts[key][n])

keys = list(all_errors.keys())

bar_plot(all_errors, num_nodess, "", "Number of nodes", "Positioning error (m)", separate_legend=True)#, ylimits=POS_ERROR_LIMITS)
bar_plot(all_rates, num_nodess, "", "Number of nodes", "Detection rate (%)", separate_legend=True)

print("Running times:", all_rts)