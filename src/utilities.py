#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 09:35:04 2024

@author: adamgreenberg
"""

import torch as tc

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
    # apply a continuous rotation to a voxel model
    
    num_voxel = model.shape[0]    
    ax = axis/tc.linalg.norm(axis, ord=2)
    cos = tc.cos(angle)
    ncos = 1-cos
    sin = tc.sin(angle)
    
    # via https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    R = tc.Tensor([[ax[0]**2*ncos+cos, ax[0]*ax[1]*ncos-ax[2]*sin, ax[0]*ax[2]*ncos+ax[1]*sin], \
                   [ax[0]*ax[1]*ncos+ax[2]*sin, ax[1]**2*ncos+cos, ax[1]*ax[2]*ncos-ax[0]*sin], \
                   [ax[0]*ax[2]*ncos-ax[1]*sin, ax[1]*ax[2]*ncos+ax[0]*sin, ax[2]**2*ncos+cos]])
    
    # define array coordinate system
    indexing = tc.arange(num_voxel)
    xgrid, ygrid, zgrid = tc.meshgrid((indexing, indexing, indexing))
    grids = (xgrid.flatten(), ygrid.flatten(), zgrid.flatten())
    coords = tc.stack(grids).to(dtype=tc.int64)
    
    # apply rotation to coordinate system
    coords_rot = (tc.matmul(R, coords-num_voxel/2)+num_voxel/2).round().to(dtype=tc.int64)
    
    # discard coordinates outside of array bounds
    mask_lo = coords_rot>=0
    mask_hi = coords_rot<num_voxel
    mask = tc.all(tc.logical_and(mask_lo, mask_hi), dim=0)
    coords = coords[:,mask].T
    coords_rot = coords_rot[:,mask].T
    
    # apply rotated coordinate system
    model_rot = tc.zeros(model.shape).to(dtype=model.dtype)
    model_rot[coords[:,0],coords[:,1],coords[:,2]] \
        = model[coords_rot[:,0],coords_rot[:,1],coords_rot[:,2]]
        
    return model_rot

def rotate_model_fast(model, coords, coords_rot):
    # apply a continuous rotation to a voxel model with precomputed coords tensors
    # apply rotated coordinate system
    model_rot = tc.zeros(model.shape).to(dtype=model.dtype)
    model_rot[coords[:,0],coords[:,1],coords[:,2]] \
        = model[coords_rot[:,0],coords_rot[:,1],coords_rot[:,2]]

    return model_rot

def compute_coords_tensors(num_voxel, axis, angle):
    # precompute coords tensors in order to apply a continuous rotation to a voxel model

    ax = axis/tc.linalg.norm(axis, ord=2)
    cos = tc.cos(angle)
    ncos = 1-cos
    sin = tc.sin(angle)

    # via https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    R = tc.Tensor([[ax[0]**2*ncos+cos, ax[0]*ax[1]*ncos-ax[2]*sin, ax[0]*ax[2]*ncos+ax[1]*sin], \
                   [ax[0]*ax[1]*ncos+ax[2]*sin, ax[1]**2*ncos+cos, ax[1]*ax[2]*ncos-ax[0]*sin], \
                   [ax[0]*ax[2]*ncos-ax[1]*sin, ax[1]*ax[2]*ncos+ax[0]*sin, ax[2]**2*ncos+cos]])

    # define array coordinate system
    indexing = tc.arange(num_voxel)
    xgrid, ygrid, zgrid = tc.meshgrid((indexing, indexing, indexing))
    grids = (xgrid.flatten(), ygrid.flatten(), zgrid.flatten())
    coords = tc.stack(grids).to(dtype=tc.int64)

    # apply rotation to coordinate system
    coords_rot = (tc.matmul(R, coords-num_voxel/2)+num_voxel/2).round().to(dtype=tc.int64)

    # discard coordinates outside of array bounds
    mask_lo = coords_rot>=0
    mask_hi = coords_rot<num_voxel
    mask = tc.all(tc.logical_and(mask_lo, mask_hi), dim=0)
    coords = coords[:,mask].T
    coords_rot = coords_rot[:,mask].T

    return coords, coords_rot

def observe_model(model, noise = 0):
    image = model.sum(dim=0)/model.shape[0]
    image += tc.normal( tc.zeros(image.shape), noise)
    return image

def loss_function(images, model, axis, angles):
    assert(len(angles)==len(images))
        
    # initialize loss with continuous penalty function for non-physical
    # occupancy values (basically just a reciprocal super-gaussian)
    center,width = 0.5,0.5
    loss = tc.sum( tc.exp(((model-center)/width/1.5)**4)-1 )/1E3
    
    # loss is sum-squared residuals between synthetic and measured images
    for angle,image in zip(angles, images):
        image_sim = observe_model(rotate_model(model, axis, angle))
        loss += tc.sum((image-image_sim)**2)
    
    return loss

def loss_function_fast(images, model, coords_list, coords_rot_list):
    assert(len(coords_list)==len(images))
    assert(len(coords_rot_list)==len(images))

    # initialize loss with continuous penalty function for non-physical
    # occupancy values (basically just a reciprocal super-gaussian)
    center,width = 0.5,0.5
    loss = tc.sum( tc.exp(((model-center)/width/1.5)**4)-1 )/1E3

    # loss is sum-squared residuals between synthetic and measured images
    for coords,coords_rot,image in zip(coords_list, coords_rot_list, images):
        image_sim = observe_model(rotate_model_fast(model, coords, coords_rot))
        loss += tc.sum((image-image_sim)**2)

    return loss


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt

    model = generate_gaussian_model(200, 30)
    angle_rate = tc.Tensor([tc.pi/100])
    
    for k in range(100):

        model2 = rotate_model(model, tc.Tensor([0,0,1]), k*angle_rate)
        image = observe_model(model2, 0.05)
    
        plt.figure(1)
        plt.clf()
        plt.imshow(image)
        plt.pause(0.05)
