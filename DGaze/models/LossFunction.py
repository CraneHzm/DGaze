# Copyright (c) Hu Zhiming 2019/7/15 jimmyhu@pku.edu.cn All Rights Reserved.

import torch
import torch.nn as nn

# define my own loss functions


# define the huber loss.    
class HuberLoss(nn.Module):
    def __init__(self, beta= 1.0, size_average=True):
        super().__init__()
        self.beta = beta
        self.size_average = size_average
    def forward(self,input,target):
        """
        very similar to the smooth_l1_loss from pytorch, but with
        the extra beta parameter
        """
        n = torch.abs(input - target)
        cond = n < self.beta
        loss = torch.where(cond, 0.5 * n ** 2, self.beta*(n - 0.5 * self.beta))
        if self.size_average:
            return loss.mean()
        return loss.sum() 
    
    
# define the custom loss.    
class CustomLoss(nn.Module):
    def __init__(self, beta= 1.0, size_average=True):
        super().__init__()
        self.beta = beta
        self.size_average = size_average
    def forward(self,input,target):
        
        n = torch.abs(input - target)
        cond = n < self.beta
        loss = torch.where(cond, 0*n, n - self.beta)
        if self.size_average:
            return loss.mean()
        return loss.sum()     