#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import time
from typing import List
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from epde.integrate import OdeintAdapter
from epde.integrate import BOPElement

# Define paths for data and results storage
# Load the exchange_rate_nips.сsv coefficient data

# Analyze patterns in coefficients
most_common_structure = df.groupby(df.columns.tolist()).size().reset_index(name='count')
most_common_structure = most_common_structure.sort_values(by='count', ascending=False)
print(most_common_structure.head(1))

# Identify the most common zero-coefficients pattern
zero_patterns = (df == 0).astype(int).astype(str).agg(''.join, axis=1)
pattern_counts = zero_patterns.value_counts()
most_common_pattern = pattern_counts.idxmax()
count_of_most_common_pattern = pattern_counts.max()
print(f"Most common zero pattern: {most_common_pattern}")
print(f"Occurrences: {count_of_most_common_pattern}")

# Normalize and filter the dataset
scaler = StandardScaler()
normalized_df = scaler.fit_transform(df.values)
df_normalized = pd.DataFrame(normalized_df, columns=df.columns, index=df.index)
df_filtered = df_normalized.loc[df.sum(axis=1) != -1]
df_filtered = df_filtered.loc[:, (df_filtered != 0).any(axis=0)]
print(df_filtered.head())

# Generate Monte Carlo samples for the dataset
num_samples = 300
monte_carlo_samples = pd.DataFrame(index=df.index, columns=df.columns)
for column in df.columns:
    mean_value = df[column].mean()
    std_dev = df[column].std()
    min_value = df[column].min()
    max_value = df[column].max()
    for i in range(num_samples):
        current_mean = df[column].iloc[i % df.shape[0]]
        current_std_dev = std_dev * 0.01
        noise = np.random.normal(loc=0, scale=current_std_dev)
        sample = current_mean + noise
        sample = np.clip(sample, min_value, max_value)
        monte_carlo_samples.loc[i, column] = sample

monte_carlo_samples = monte_carlo_samples.dropna()

# Calculate pairwise distances and select similar rows
distances = pairwise_distances(df, monte_carlo_samples)
threshold = 2.0
similar_rows = [monte_carlo_samples.iloc[j] for j in range(distances.shape[1]) if distances[0][j] < threshold]
new_coeff = pd.DataFrame(similar_rows)
new_coeff.columns = monte_carlo_samples.columns


def construct_general_equation_dict(new_coeff):
    equations = []
    for _, row in new_coeff.iterrows():
        # Eequations based on terms observed in the dataset
        equation_dict = {
            'u{power: 1.0}': {'coeff': row.get('u{power: 1.0}', 0.0), 
                               'term': [[None]], 
                               'pow': [1.0], 
                               'var': [0]},
            'du/dx0{power: 1.0}': {'coeff': row.get('du/dx0{power: 1.0}', 0.0), 
                                    'term': [[0]], 
                                    'pow': [1.0], 
                                    'var': [0]},
            'd^2u/dx0^2{power: 1.0}': {'coeff': row.get('d^2u/dx0^2{power: 1.0}', 0.0), 
                                        'term': [[1]], 
                                        'pow': [1.0], 
                                        'var': [0]},
            'u{power: 2.0}': {'coeff': row.get('u{power: 2.0}', 0.0), 
                              'term': [[None]], 
                              'pow': [2.0], 
                              'var': [0]},
            'x_0{power: 2.0} * cos{power: 1.0}': {'coeff': row.get('x_0{power: 2.0, dim: 0.0} * cos{power: 1.0}', 0.0), 
                                                  'term': [[2]], 
                                                  'pow': [2.0, 1.0], 
                                                  'var': [0]},
            'x_0{power: 2.0} * sin{power: 1.0}': {'coeff': row.get('x_0{power: 2.0, dim: 0.0} * sin{power: 1.0}', 0.0), 
                                                  'term': [[3]], 
                                                  'pow': [2.0, 1.0], 
                                                  'var': [0]},
            'C': {'coeff': row.get('C', 0.0), 
                  'term': [[None]], 
                  'pow': [0], 
                  'var': [0]},
        }
        
        # Append the dictionary for this equation
        equations.append(equation_dict)
    
    return equations

eqs = [Equation() for i in range(10)]
    for eq_idx, eq in enumerate(eqs):
        eq.add(equations[eq_idx])

def build_ann() -> torch.nn.Sequential:
    """Creates a feedforward neural network with 3 hidden layers using Tanh activation."""
    return torch.nn.Sequential(
        torch.nn.Linear(2, 100),  # Input layer (2 features) -> first hidden layer (100 neurons)
        torch.nn.Tanh(),         # Activation (Tanh)
        torch.nn.Linear(100, 100),  # First hidden layer -> second hidden layer
        torch.nn.Tanh(),         # Activation (Tanh)
        torch.nn.Linear(100, 100),  # Second hidden layer -> third hidden layer
        torch.nn.Tanh(),         # Activation (Tanh)
        torch.nn.Linear(100, 1)  # Third hidden layer -> output layer (1 neuron)
    )

   #Build one ANN per equation
   anns = [build_ann() for _ in eqs]
    c_cache = cache.Cache(cache_verbose=False, model_randomize_parameter=1e-6)
    cb_es = early_stopping.EarlyStopping(eps=1e-5,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=5,
                                         randomize_parameter=1e-10,
                                         info_string_every=500
                                         )
    cb_plots = plot.Plots(save_every=None, print_every=None)
    # Optimizer for model training
    optimizer = Optimizer('Adam', {'lr': 1e-3})

print(f'eqs are {eqs}')
start = time.time()
for eq_idx, equation in enumerate(eqs):
    model = Model(anns[eq_idx], domain, equation, boundaries)  # batch_size = 390
    # print('batch size', model.batch_size)
    model.compile('NN', lambda_operator=1, lambda_bound=100)
    model.train(optimizer, 3000, save_model=False, callbacks=[cb_es, c_cache, cb_plots])
    end = time.time()
    print('Time taken 10= ', end - start)

    solutions = []
    for net_idx, net in enumerate(anns):
        anns[net_idx] = net.to(device=device_type())
        solutions.append(anns[net_idx](domain.build('NN')))  # .detach().numpy().reshape(-1))
        solutions_tensor = torch.stack(solutions, dim=0)  # Tensor containing all solutions
print(f"Solutions tensor shape: {solutions_tensor.shape}")
average_solution_tensor = solutions_tensor.mean(dim=0)
average_solution = average_solution_tensor.detach().numpy().reshape(-1)  # Reshape to 1D for saving
#Save solutions to results storage
pt_directory = r''
os.makedirs(pt_directory, exist_ok=True)
solution_file_name = f"several_solutions_{len(solutions)}_shape_{solutions_tensor.shape}.pt"
torch.save(solutions_tensor, os.path.join(pt_directory, pt_file_name))