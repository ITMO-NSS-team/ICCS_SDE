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
PATH = 'C:/Users/Ksenia/NSS/ODE_projects/SDE-DS/hunter_prey'
TENSOR_DIR = 'C:\\Users\\Ksenia\\NSS\\ODE_projects\\SDE-DS\\lotka-volterra\\pt'
NPY_DIR = 'C:\\Users\\Ksenia\\NSS\\ODE_projects\\SDE-DS\\lotka-volterra\\np'
os.makedirs(TENSOR_DIR, exist_ok=True)
os.makedirs(NPY_DIR, exist_ok=True)

# Load the Lotka-Volterra coefficient data
df = pd.read_csv(f'{PATH}/output_main_lotka_volterra_equations.csv', 
                 index_col='Unnamed: 0', sep='\t', encoding='utf-8')

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
num_samples = 84
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


# Construct ODEs from the coefficient DataFrame
def construct_equation_dict(new_coeff):
    equations = []
    for _, row in new_coeff.iterrows():
        hare_dict = {
            'u{power: 1.0}_u': {'coeff': row['u{power: 1.0}_u'], 'term': [[None]], 'pow': [1], 'var': [0]},
            'v{power: 1.0}_u': {'coeff': row['v{power: 1.0}_u'], 'term': [[None]], 'pow': [1], 'var': [1]},
            'v{power: 1.0} * u{power: 1.0}_u': {'coeff': row['v{power: 1.0} * u{power: 1.0}_u'],
                                                'term': [[None], [None]], 'pow': [1, 1], 'var': [1, 0]},
            'du/dx0': {'coeff': row['du/dx0{power: 1.0}_u'], 'term': [[0]], 'pow': [1], 'var': [0]},
            'C_u': {'coeff': row['C_u'], 'term': [[None]], 'pow': [0], 'var': [0]},
        }
        lynx_dict = {
            'v{power: 1.0}_v': {'coeff': row['v{power: 1.0}_v'], 'term': [[None]], 'pow': [1], 'var': [1]},
            'u{power: 1.0}_v': {'coeff': row['u{power: 1.0}_v'], 'term': [[None]], 'pow': [1], 'var': [0]},
            'dv/dx0': {'coeff': row['dv/dx0{power: 1.0}_v'], 'term': [[0]], 'pow': [1], 'var': [1]},
            'C_v': {'coeff': row['C_v'], 'term': [[None]], 'pow': [0], 'var': [0]},
            'v{power: 1.0} * u{power: 1.0}_v': {'coeff': row['v{power: 1.0} * u{power: 1.0}_v'],
                                                'term': [[None], [None]], 'pow': [1, 1], 'var': [1, 0]},
        }
        equations.append((hare_dict, lynx_dict))
    return equations


equations = construct_equation_dict(new_coeff)

# Time grid and initial conditions
t = np.linspace(0, 20, 201)
data_initial = np.load(f'{PATH}/data_synth.npy')
x0, y0 = data_initial.T[0][0], data_initial.T[1][0]


# Helper function to define boundary conditions
def get_ode_bop(key, var, term, grid_loc, value):
    bop = BOPElement(axis=0, key=key, term=term, power=1, var=var)
    bop.set_grid(torch.tensor([[grid_loc]]).float())
    bop.values = torch.tensor([[value]]).float()
    return bop


bop_u = get_ode_bop('u', 0, [None], t[0], x0)
bop_v = get_ode_bop('v', 1, [None], t[0], y0)

# Solve ODE system for all equations using Radau method
adapter = OdeintAdapter('Radau')
solutions = []
for eq in equations:
    result = adapter.solve_epde_system(system=[('u', eq[0]), ('v', eq[1])],
                                       grids=[torch.from_numpy(t)],
                                       boundary_conditions=[bop_u, bop_v],
                                       vars_to_describe=['u', 'v'])
    solutions.append(result)

# Save family of solutions
family_solutions = np.array([sol[1] for sol in solutions])
family_tensor = torch.tensor(family_solutions)
torch.save(family_tensor, os.path.join(TENSOR_DIR, f'lv_{len(solutions)}_family_solutions_shape_{family_tensor.shape}.pt'))

# Calculate and save the average solution
average_solution = np.mean(family_solutions, axis=0)
np.save(os.path.join(NPY_DIR, f'lv_{len(solutions)}_solutions_shape_{average_solution.shape}.npy'), average_solution)
average_tensor = torch.tensor(average_solution)
torch.save(average_tensor, os.path.join(TENSOR_DIR, f'lv_{len(solutions)}_solutions_shape_{average_tensor.shape}.pt'))

print(f'Average solution saved as tensor in: {os.path.join(TENSOR_DIR, f"lv_{len(solutions)}_solutions_shape_{average_tensor.shape}.pt")}')

