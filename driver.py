#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 22:26:39 2024

@author: adamgreenberg
"""

import torch as tc
import utilities as utl
import time as tm
tc.manual_seed(1)

# define truth model
num_voxel = 200
num_mode = 30
truth_model = utl.generate_gaussian_model(num_voxel, num_mode)

# define observational parameters
angles = tc.linspace(0,tc.pi,10)
axis = tc.Tensor([0,0,1])
noise = 0.01

# simulate measurements - 4E5 total pixels
images = [utl.observe_model(utl.rotate_model(truth_model, axis, angle), noise) \
          for angle in angles]

# recover model via fit - 8E6 total free model parameters
tic = tm.time()
num_epoch = 75
cand_model = tc.zeros((num_voxel,num_voxel,num_voxel), requires_grad=True)
optimizer = tc.optim.Adam([cand_model], lr=4E-2)
for k in range(num_epoch):
    optimizer.zero_grad()
    loss = utl.loss_function(images, cand_model, axis, angles)
    loss.backward()
    optimizer.step()
    print(f"Iteration: {k:3.0f}, Loss: {loss.item():8.1f}")
print(f"Total time: {tm.time()-tic:.1f} s")

# compare truth model with recovered model
random_axis = tc.rand((3,))
import matplotlib.pyplot as plt
for angle in tc.linspace(0.05, tc.pi+0.05, 30):
    args = [random_axis, angle]
    img       = utl.observe_model(utl.rotate_model(cand_model,  *args))
    img_truth = utl.observe_model(utl.rotate_model(truth_model, *args))
    
    plt.figure(1), plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(img.detach().numpy())
    plt.subplot(1,2,2)
    plt.imshow(img_truth.detach().numpy())
    plt.pause(0.1)

