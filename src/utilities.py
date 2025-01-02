#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 09:35:04 2024

@author: adamgreenberg
"""

import torch as tc
from torch.nn import functional as fn

def generate_gaussian_model(num_voxel, num_mode, 
                            thresh=5E-2, std_max=0.2, mean_max=0.5):
    # generate voxel model (i.e., a 3D occupancy grid) as a gaussian mixture
    
    space = tc.zeros((num_voxel, num_voxel, num_voxel))
    indexing = tc.linspace(-1, 1, num_voxel)
    xgrid, ygrid, zgrid = tc.meshgrid((indexing, indexing, indexing))

    mode_func =  lambda mu, std: tc.exp(-0.5 * ((xgrid-mu[0])**2
                                              + (ygrid-mu[1])**2
                                              + (zgrid-mu[2])**2)/std**2)

    for _ in range(num_mode):
        mu = mean_max * (2*tc.rand((3,))-1)
        std = std_max * (2*tc.rand((1,))-1)
        space += mode_func(mu, std)

    return (space>thresh).to(dtype=tc.float)

def rotate_model(model, axis, angle):
    
    # initialize helper variables for affine transformation matrix definition
    ax = axis/tc.linalg.norm(axis, ord=2)
    cos,sin = tc.cos(angle), tc.sin(angle)
    ncos = 1-cos
    
    # initialize arguments for grid functions
    grid_args = {'align_corners':False}
    grid_shape = (1,1)+tuple(model.shape)
    
    # via https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    # (this weird construction is necessary to maintain autograd graph)
    zero = tc.zeros(1)[0]
    R1 = tc.stack((ax[0]**2*ncos+cos, ax[0]*ax[1]*ncos-ax[2]*sin, ax[0]*ax[2]*ncos+ax[1]*sin,zero))
    R2 = tc.stack((ax[0]*ax[1]*ncos+ax[2]*sin, ax[1]**2*ncos+cos, ax[1]*ax[2]*ncos-ax[0]*sin,zero))
    R3 = tc.stack((ax[0]*ax[2]*ncos-ax[1]*sin, ax[1]*ax[2]*ncos+ax[0]*sin, ax[2]**2*ncos+cos,zero))
    R = tc.stack((R1,R2,R3))
    
    # use torch's interpolating (differentiable) matrix affine transformation
    grid = fn.affine_grid(R[None,:,:], grid_shape, **grid_args) 
    model_rot = fn.grid_sample(model[None,None,:,:,:], grid, **grid_args)
    
    return model_rot[0,0,:,:,:]
    
def observe_model(model, noise = 0):
    
    image = model.sum(dim=0)/model.shape[0]
    image += tc.normal( tc.zeros(image.shape), noise)
    return image

def loss_function(times, images, model, angle_rate, axis):
    assert(len(times)==len(images))
            
    # initialize loss with continuous penalty function for non-physical 
    # occupancy values (basically just a reciprocal super-gaussian)
    center,width = 0.5,0.5
    loss = tc.sum( tc.exp(((model-center)/width/1.5)**8)-1 )/1E3
    
    # loss is sum-squared residuals between synthetic and measured images
    for image,angle in zip(images, angle_rate*times):
        image_sim = observe_model(rotate_model(model, axis, angle))
        loss += tc.sum((image-image_sim)**2)
    
    return loss