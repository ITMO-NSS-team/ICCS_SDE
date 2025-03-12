#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import copy
import math
import os
import pickle
import random
import numpy as np
import pandas as pd
import epde
import epde.interface.interface as epde_alg
from gluonts.dataset.repository.datasets import get_dataset

# Load dataset
dataset_name = "exchange_rate_nips"
dataset = get_dataset(dataset_name, regenerate=False)

# Process the training data into DataFrames
train_data = list(dataset.train)
data_frames = []

for data in train_data:
    periods = pd.date_range(start=data['start'].start_time, periods=len(data['target']), freq='B')
    df = pd.DataFrame({
        'date': periods,
        'value': data['target'],
        'item_id': data['item_id']
    })
    data_frames.append(df)

# Combine all DataFrames into a single one
final_df = pd.concat(data_frames, ignore_index=True)
final_df.set_index('date', inplace=True)

# Print the combined DataFrame
print(final_df)

# Filter for item_id == 4 and extract its 'value' column
combined_df = pd.concat(data_frames, ignore_index=True)
combined_df = combined_df[combined_df['item_id'] == 4]
x = combined_df['value'].values

# Display filtered data properties
print('x are ', x)
print(type(x), x.size)

# EPDE initialization parameters
bnd = 1000
n_epochs = 100
popsize = 5
max_axis_idx = x.ndim - 1
t = np.arange(0, len(x))  # Time values

diff_mode = 'FD'

# Initialize EPDE search object
epde_search_obj = epde.EpdeSearch(use_solver=False, multiobjective_mode=True,
                                  boundary=bnd, dimensionality=max_axis_idx,
                                  coordinate_tensors=[t, ])

# Set equation factors limits
factors_max_number = {'factors_num': [1, 2], 'probas': [0.6, 0.4]}

# Set differentiation mode
if diff_mode == 'ANN':
    epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
                                     preprocessor_kwargs={'epochs_max': 50000})
elif diff_mode == 'poly':
    epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                     preprocessor_kwargs={'use_smoothing': False, 
                                                          'sigma': 1,
                                                          'polynomial_window': 3, 
                                                          'poly_order': 3})
elif diff_mode == 'FD':
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD')
else:
    raise NotImplementedError('Incorrect differentiation mode selected.')

# Define tokens for EPDE
grid_tokens = epde.GridTokens(['x_0'], dimensionality=max_axis_idx, max_power=2)
trig_tokens = epde.TrigonometricTokens(freq=(0.95, 1.05), dimensionality=max_axis_idx)

# Set MOEA/DD parameters for EPDE
epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=n_epochs)

# Perform EPDE fitting
epde_search_obj.fit(data=[x], variable_names=['u'], max_deriv_order=(2,),
                    equation_terms_max_number=4, data_fun_pow=2,
                    additional_tokens=[trig_tokens, grid_tokens],
                    equation_factors_max_number=factors_max_number,
                    eq_sparsity_interval=(1e-12, 1e-10))

# Extract and display the resulting equations
res = epde_search_obj.equations(True)
print(res)

# Save `res` to a CSV file for further use
# res.to_csv('results.csv')

