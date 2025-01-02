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

# define truth model
num_voxel = 200
num_mode = 30
truth_model = utl.generate_gaussian_model(num_voxel, num_mode)

# define observational parameters
angle_rate = tc.pi
times = tc.linspace(0,1,10)

axis = tc.Tensor([1,0,0])
noise = 0.05

# simulate measurements - 4E5 total pixels
images = [utl.observe_model(utl.rotate_model(truth_model, axis, angle), noise) \
          for angle in angle_rate*times]

# recover model via fit - 8E6 total free model parameters
tic = tm.time()
num_epoch = 50
cand_model = tc.zeros((num_voxel,num_voxel,num_voxel), requires_grad=True)
cand_angle_rate = tc.ones((1,), requires_grad=True)

# tc.no_grad context manager necessary to keep cand_angle_rate as a valid leaf 
# node
with tc.no_grad(): cand_angle_rate *= 2.5

# define parameter groups to enable to different learning rates for fit-for 
# variables
params = [{'params': cand_model,      'lr': 5E-2}, 
          {'params': cand_angle_rate, 'lr': 5E-3}]
optimizer = tc.optim.Adam(params)

# run optimization loop
for k in range(num_epoch):
    optimizer.zero_grad()
    loss = utl.loss_function(times, images, cand_model, cand_angle_rate, axis)
    loss.backward(retain_graph=True)
    optimizer.step()
    print(f"Iteration: {k:3.0f}, Loss: {loss.item():8.1f}")

cand_model.detach_().round_()
print(f"Total time: {tm.time()-tic:.1f} s")

outfile = 'outputs/comparison.png'
plt.show_comparison(images, cand_model, axis, times*cand_angle_rate, outfile)