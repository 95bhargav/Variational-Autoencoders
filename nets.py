# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:45:49 2021

@author: Shukla
"""


import torch 
import torch.nn as nn


class autoencoder_new(nn.Module):
    def __init__(self, Out, feature_size =40, label_size=10, image_size= 32):
        
        super(autoencoder_new,self).__init__()
        
        self.label_size =label_size
        self.feature_size = feature_size
        self.image_size = image_size
        self.Out    = Out
        
        self.feat_scaler= nn.Sequential(nn.Linear(self.feature_size, (self.Out*4)*2*2),
                                        nn.LeakyReLU(0.2, inplace= True))
        
        self.class_labels = nn.Sequential(nn.Linear(self.label_size,4), nn.Sigmoid())
        
        if self.image_size ==32:
            self.encoder = nn.Sequential(
                # 3x32x32
                nn.Conv2d(3, self.Out, 4, 2, 1, bias = False),
                nn.BatchNorm2d(self.Out),
                nn.LeakyReLU(0.2, inplace=True),)
        
        elif self.image_size == 64:
            self.encoder = nn.Sequential(
                # 3x32x32
                nn.Conv2d(3, self.Out, 4, 2, 1, bias = False),
                nn.BatchNorm2d(self.Out),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.Out, self.Out, 4, 2, 1, bias = False),
                nn.BatchNorm2d(self.Out),
                nn.LeakyReLU(0.2, inplace=True),)
        
        self.encoder.add_module("Body", nn.Sequential(
            
            # self.Outx16x16
            nn.Conv2d(self.Out, self.Out*2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.Out*2),
            nn.LeakyReLU(0.2, inplace=True),
            # self.Out*2x8x8
            nn.Conv2d(self.Out*2, self.Out*4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.Out*4),
            nn.LeakyReLU(0.2, inplace=True),
            # self.Out*4x4x4
            nn.Conv2d(self.Out*4, self.Out*4, 4, 2, 1, bias = False),
            nn.Sigmoid()))
            # self.Out*4x2x2
            
            
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d((self.Out*4)+1, self.Out*4, 4 ,2, 1, bias= False),
                            nn.BatchNorm2d(self.Out*4),
                            nn.LeakyReLU(0.2,inplace=True),
            # self.Out*4 x4x4
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d( self.Out*4, self.Out*4, 3, 1, 1, bias= False),
                            nn.BatchNorm2d(self.Out*4),
                            nn.LeakyReLU(0.2, inplace=True), 
            # self.Out*4x8x8
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d( self.Out*4, self.Out*2, 3, 1, 1, bias= False),
                            nn.BatchNorm2d(self.Out*2),
                            nn.LeakyReLU(0.2, inplace=True),
            # self.Out*2x16x16  
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.Out*2 ,self.Out, 3, 1, 1, bias= False),
                            nn.BatchNorm2d(self.Out),
                            nn.LeakyReLU(0.2, inplace=True),)
            # self.Outx32x32 
            
        if self.image_size == 32:
            self.decoder.add_module("final",nn.Sequential(
            nn.Conv2d(self.Out,3, 3, 1, 1, bias= False),
                            nn.Tanh()))
            # 3x32x32
        elif self.image_size == 64:
            self.decoder.add_module("final",nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.Out, self.Out, 3, 1, 1, bias= False),
            nn.BatchNorm2d(self.Out),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(self.Out, 3, 3, 1, 1, bias= False),
                            nn.Tanh()))
                                    
                                    
    def forward(self, inputs, labels, mode = "normal", train_mode = "ae"):
        
        if train_mode == "ae":
            feat = self.encoder(inputs)
        elif train_mode == "gan":
            feat = self.feat_scaler(inputs).reshape(-1,self.Out*4, 2,2)   
            
        if mode == "normal":
            labels = nn.Softmax(1)(labels)
        elif mode == "one_hot":
            labels = labels
            
        labs = self.class_labels(labels).reshape(-1, 1, feat.shape[2],feat.shape[3])
        concat = torch.cat((feat,labs),1)
                       
        return self.decoder(concat)

