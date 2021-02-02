# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 17:14:43 2020

@author: Shukla
"""

# This file is the set of functions used for this thesis
# The functions are used in the main file.
#%% Importing Libraries
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import pandas as pd
import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.metrics import precision_score, recall_score

CentralFolder =os.getcwd()
sys.path.append(CentralFolder)
import HelperFunctions as HelpFunc

global _DEVICE
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#%% Functions and classes Defination


## Random onehot softmax labels
def one_hot_softmax(true_confidence,mode ='soft_one_hot',samples=500,classes=10):
    """
    Inputs:
        true_confidence :- It is the probablity value for the given classes in the dataset. \n
        mode            :- "soft_one_hot" if requred samples with distributed probablity values.\n
                            Ex. [0.1,0.1,0.8] \n
                           "one_hot" if required samples in one hot probablity. \n
                            Ex. [0.,0.,0.8] \n
        samples         :- Number of examples to produce per class. \n
        classes         :- Number of classes in the Dataset. \n
    
    Returns:
        A nested list of tensors containing samples from each classes. \n
        Ex. [[class1],[class2],[class3]]

    """
   
    false_confidence =1-true_confidence
    
    true_range  = [(true_confidence,true_confidence) for i in range(classes)]
    false_range = [(0,false_confidence) for i in range(classes)]
    
    samp = []
    for i in range(classes):
        (f_min, f_max) = false_range[i]
        (t_min,t_max) = true_range[i]
        
        out     = torch.FloatTensor(samples,classes).uniform_(0.00,0.00)
        corr    = torch.FloatTensor(samples,1).uniform_(t_min,t_max)
        one_hot = out.index_copy(1,torch.tensor([i]),corr)
        
        if mode == 'soft_one_hot':
            rem     = (torch.ones((samples,1)) - corr)/(classes-1) 
            app_list = [j for j in range(classes) if j !=i ]
            for j in app_list:
                one_hot   = one_hot.index_copy(1,torch.tensor(j), rem)
        elif mode == 'one_hot':
            one_hot =one_hot        
        samp.append(one_hot)
         
    return samp

## Function to plot RGB images 
def imshow(img,dataset):
    """
    Unnormalize the image and plot it in figure
    """
    img = img/2 +0.5
    if dataset =='mnist':
        plt.imshow(img.squeeze().cpu().detach(), cmap="gray")
    elif dataset == 'cifar10':
        torchimg =img.permute( 1, 2, 0)
        plt.imshow(torchimg.squeeze().cpu().detach())

def PlotViz(fixed_data, recon_image, img_title_list,FigureDict,PlotTitle, dataset):
    """
    Inputs:
        fixed_data  : Real image to reconstruct. \n
        recon_image : Generator reconstructions. \n
        img_title_list : Titles for both images. \n
        FigureDict  : Dictionary object to store figures. 'None' if not available.\n
        PlotTitle   : Title for Figure. 'None' if not available.\n
        dataset     : The current working Dataset. \n
        
    Returns:
        Matplotlib figure stored in working directory.
        
    """
    fig_ae  = plt.figure(figsize=(10,10),dpi=300)
    if fig_ae is not None:
        fig_ae.suptitle(PlotTitle)
        
    image_list = []
    for i in range(64):
        if i %2 ==0:
            image_list.append(fixed_data[i//2])
        else:
            image_list.append(recon_image[i//2])
    
    for i in range(64):
        ax=plt.subplot(8,8,i+1)
        imshow(image_list[i],dataset)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.xticks([])
        plt.yticks([])
        ax.set_title(img_title_list[0] if i%2==0 else img_title_list[1])
        

    if FigureDict is not None:
        FigureDict.StoreFig(fig=fig_ae, name=PlotTitle,saving=True)
    plt.show()