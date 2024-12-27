#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 09:53:22 2024

@author: adamgreenberg
"""

import matplotlib.pyplot as plt
import utilities as utl

def show_comparison(images, model, axis, angles, outfile = None, num_image_max=4):
    assert(len(angles)==len(images))
        
    num_image = min(num_image_max, len(images))
    
    fig, ax = plt.subplots(num_image, 2, figsize=(2*2, 2*num_image))

    for k,(angle,image) in enumerate(zip(angles, images)):
        
        image_sim = utl.observe_model(utl.rotate_model(model, axis, angle))
        
        ax[k,0].pcolormesh(image)
        ax[k,0].get_xaxis().set_ticks([])
        ax[k,0].get_yaxis().set_ticks([])
        ax[k,0].set_aspect('equal')
        
        ax[k,1].pcolormesh(image_sim.detach().numpy())
        ax[k,1].get_xaxis().set_ticks([])
        ax[k,1].get_yaxis().set_ticks([])
        ax[k,1].set_aspect('equal')
        
        if k==0: ax[k,0].set_title('Measured'), ax[k,1].set_title('Synthetic')
        if k==num_image-1: break

    fig.tight_layout(pad=1.5, w_pad=0, h_pad=0)
    
    if outfile is not None: plt.savefig(outfile)
