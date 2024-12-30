#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 22:26:39 2024

@author: adamgreenberg
"""
import sys
sys.path.append('./src/')

import torch as tc
import utilities as utl
import time as tm
import plotter as plt
tc.manual_seed(1)

# figure out if GPU is available
if tc.cuda.is_available():
    print("I found a GPU.")
    device = "cuda:0"
else:
    print("I did not find a GPU.")
    device = "cpu"


# define truth model
num_voxel = 200
num_mode = 30
truth_model = utl.generate_gaussian_model(num_voxel, num_mode)
truth_model.to(device)

# define observational parameters
angles = tc.linspace(0,tc.pi,10)
axis = tc.Tensor([0,0,1])
axis.to(device)
noise = 0.05

# simulate measurements - 4E5 total pixels
images = [utl.observe_model(utl.rotate_model(truth_model, axis, angle), noise) \
          for angle in angles]


# recover model via fit - 8E6 total free model parameters
tic = tm.time()
num_epoch = 75
cand_model = tc.zeros((num_voxel,num_voxel,num_voxel), requires_grad=True)
cand_model.to(device)

optimizer = tc.optim.Adam([cand_model], lr=4E-2)
for k in range(num_epoch):
    optimizer.zero_grad()
    loss = utl.loss_function(images, cand_model, axis, angles)
    loss.backward()
    optimizer.step()
    print(f"Iteration: {k:3.0f}, Loss: {loss.item():8.1f}")
print(f"Total time: {tm.time()-tic:.1f} s")

outfile = 'outputs/comparison.png'
plt.show_comparison(images, cand_model, axis, angles, outfile)

