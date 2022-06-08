import torch
import torch.nn as nn
#from torchvision import models
import torch.nn.init as init
import torch.nn.functional as F
from math import floor


# DGaze Model.
class DGaze(nn.Module):
    def __init__(self, seqLength, seqFeatureNum, saliencyWidth, saliencyNum, n_output, dropoutRate):
        super(DGaze, self).__init__()
        
        # the input params
        self.seqLength = seqLength
        self.seqFeatureNum = seqFeatureNum
        self.seqSize = self.seqLength * self.seqFeatureNum
        self.saliencyWidth = saliencyWidth
        self.saliencyNum = saliencyNum
        self.saliencySize = self.saliencyWidth * self.saliencyWidth * self.saliencyNum
        
        
        # the model params
        seqCNN1D_outChannels = 128
        seqCNN1D_poolingRate = 2
        seqCNN1D_kernelSize = 2
        self.seqCNN1D_outputSize = floor((self.seqLength - seqCNN1D_kernelSize + 1)/seqCNN1D_poolingRate)* seqCNN1D_outChannels
        #print(self.seqCNN1D_outputSize)
        saliencyFC_outputSize = 64
        prdFC_inputSize = saliencyFC_outputSize + self.seqCNN1D_outputSize
        prdFC_linearSize1 = 128
        prdFC_linearSize2 = 128
        
        
        # the headobject sequence encoder layer
        self.SeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.seqFeatureNum, out_channels=seqCNN1D_outChannels,kernel_size=seqCNN1D_kernelSize),
            nn.BatchNorm1d(seqCNN1D_outChannels),
            nn.ReLU(),
            nn.MaxPool1d(seqCNN1D_poolingRate),
            nn.Dropout(p = dropoutRate),
             )
        
        # the saliency encoder layer 
        self.SaliencyFC = nn.Sequential(
            nn.Linear(self.saliencySize, saliencyFC_outputSize),
            nn.BatchNorm1d(saliencyFC_outputSize),
            nn.ReLU(),
            nn.Dropout(p = dropoutRate),
              )
       
        # the prediction fc layer for DGaze.
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = dropoutRate),
            nn.Linear(prdFC_linearSize2, n_output)
             )
               
        # the prediction fc layer for DGaze without using the saliency features.
        self.PrdFC2 = nn.Sequential(
            nn.Linear(self.seqCNN1D_outputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = dropoutRate),
            nn.Linear(prdFC_linearSize2, n_output)
             )   
        
        # the prediction fc layer for DGaze without using the head sequence.
        self.headFeatureNum = 2
        self.SeqCNN1D3 = nn.Sequential(
            nn.Conv1d(in_channels=self.seqFeatureNum - self.headFeatureNum, out_channels=seqCNN1D_outChannels,kernel_size=seqCNN1D_kernelSize),
            nn.BatchNorm1d(seqCNN1D_outChannels),
            nn.ReLU(),
            nn.MaxPool1d(seqCNN1D_poolingRate),
            nn.Dropout(p = dropoutRate),
             )        
       
        # the prediction fc layer for DGaze without using the object sequence.
        self.objectFeatureNum = self.seqFeatureNum - self.headFeatureNum
        self.SeqCNN1D4 = nn.Sequential(
            nn.Conv1d(in_channels=self.seqFeatureNum - self.objectFeatureNum, out_channels=seqCNN1D_outChannels,kernel_size=seqCNN1D_kernelSize),
            nn.BatchNorm1d(seqCNN1D_outChannels),
            nn.ReLU(),
            nn.MaxPool1d(seqCNN1D_poolingRate),
            nn.Dropout(p = dropoutRate),
             )  
             
            
    # DGaze Model.
    def forward1(self, x):
        headObjectSeq = x[:, 0:self.seqSize]       
        saliencyFeatures = x[:, self.seqSize:]
        
        headObjectSeq = headObjectSeq.reshape(-1, self.seqLength, self.seqFeatureNum)
        headObjectSeq = headObjectSeq.permute(0,2,1)
        seqOut = self.SeqCNN1D(headObjectSeq)
        seqOut = seqOut.reshape(-1, self.seqCNN1D_outputSize)
        saliencyOut = self.SaliencyFC(saliencyFeatures)
        prdInput = torch.cat((seqOut, saliencyOut), 1)
        out = self.PrdFC(prdInput)
        return out
       
        
    # DGaze without the saliency features.
    def forward2(self, x):
        headObjectSeq = x[:, 0:self.seqSize]       
        
        headObjectSeq = headObjectSeq.reshape(-1, self.seqLength, self.seqFeatureNum)
        headObjectSeq = headObjectSeq.permute(0,2,1)
        seqOut = self.SeqCNN1D(headObjectSeq)
        seqOut = seqOut.reshape(-1, self.seqCNN1D_outputSize)
        out = self.PrdFC2(seqOut)
        return out
    
    
    # DGaze without the head sequence.
    def forward3(self, x):
        headObjectSeq = x[:, 0:self.seqSize]       
        saliencyFeatures = x[:, self.seqSize:]
        
        headObjectSeq = headObjectSeq.reshape(-1, self.seqLength, self.seqFeatureNum)
        headObjectSeq = headObjectSeq.permute(0,2,1)
        objectSeq = headObjectSeq[:, self.headFeatureNum:,:]
        
        seqOut = self.SeqCNN1D3(objectSeq)
        seqOut = seqOut.reshape(-1, self.seqCNN1D_outputSize)
        saliencyOut = self.SaliencyFC(saliencyFeatures)
        prdInput = torch.cat((seqOut, saliencyOut), 1)
        out = self.PrdFC(prdInput)
        return out
        
        
    # DGaze without the object sequence.
    def forward4(self, x):
        headObjectSeq = x[:, 0:self.seqSize]       
        saliencyFeatures = x[:, self.seqSize:]
        
        headObjectSeq = headObjectSeq.reshape(-1, self.seqLength, self.seqFeatureNum)
        headObjectSeq = headObjectSeq.permute(0,2,1)
        headSeq = headObjectSeq[:, 0:self.headFeatureNum,:]
        
        seqOut = self.SeqCNN1D4(headSeq)
        seqOut = seqOut.reshape(-1, self.seqCNN1D_outputSize)
        saliencyOut = self.SaliencyFC(saliencyFeatures)
        prdInput = torch.cat((seqOut, saliencyOut), 1)
        out = self.PrdFC(prdInput)
        return out 
        
        
    def forward(self, x):
        out = self.forward1(x)
        return out  
    
    
# DGaze Model using only head and object data as input features.
class DGaze_HeadObject(nn.Module):
    def __init__(self, seqLength, seqFeatureNum, n_output, dropoutRate):
        super(DGaze_HeadObject, self).__init__()
        
        # the input params
        self.seqLength = seqLength
        self.seqFeatureNum = seqFeatureNum
        self.seqSize = self.seqLength * self.seqFeatureNum        
        
        
        # the model params
        seqCNN1D_outChannels = 128
        seqCNN1D_poolingRate = 2
        seqCNN1D_kernelSize = 2
        self.seqCNN1D_outputSize = floor((self.seqLength - seqCNN1D_kernelSize + 1)/seqCNN1D_poolingRate)* seqCNN1D_outChannels
        #print(self.seqCNN1D_outputSize)        
        prdFC_inputSize = self.seqCNN1D_outputSize
        prdFC_linearSize1 = 128
        prdFC_linearSize2 = 128
        
        
        # the headobject sequence encoder layer
        self.SeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.seqFeatureNum, out_channels=seqCNN1D_outChannels,kernel_size=seqCNN1D_kernelSize),
            nn.BatchNorm1d(seqCNN1D_outChannels),
            nn.ReLU(),
            nn.MaxPool1d(seqCNN1D_poolingRate),
            nn.Dropout(p = dropoutRate),
             )
        
       
        # the prediction fc layer for DGaze.
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = dropoutRate),
            nn.Linear(prdFC_linearSize2, n_output)
             )
                
    def forward1(self, x):
        headObjectSeq = x[:, 0:self.seqSize]               
        
        headObjectSeq = headObjectSeq.reshape(-1, self.seqLength, self.seqFeatureNum)
        headObjectSeq = headObjectSeq.permute(0,2,1)
        seqOut = self.SeqCNN1D(headObjectSeq)
        seqOut = seqOut.reshape(-1, self.seqCNN1D_outputSize)               
        prdInput = seqOut
        out = self.PrdFC(prdInput)
        return out
    
    def forward(self, x):
        out = self.forward1(x)
        return out  

    
    
# DGaze Model for SGaze Dataset.
class DGaze_SGazeDataset(nn.Module):
    def __init__(self, seqLength, seqFeatureNum, saliencyWidth, saliencyNum, n_output, dropoutRate):
        super(DGaze_SGazeDataset, self).__init__()
        
        # the input params
        self.seqLength = seqLength
        self.seqFeatureNum = seqFeatureNum
        self.seqSize = self.seqLength * self.seqFeatureNum
        self.saliencyWidth = saliencyWidth
        self.saliencyNum = saliencyNum
        self.saliencySize = self.saliencyWidth * self.saliencyWidth * self.saliencyNum
        
        
        # the model params
        seqCNN1D_outChannels = 128
        seqCNN1D_poolingRate = 2
        seqCNN1D_kernelSize = 2
        self.seqCNN1D_outputSize = floor((self.seqLength - seqCNN1D_kernelSize + 1)/seqCNN1D_poolingRate)* seqCNN1D_outChannels
        #print(self.seqCNN1D_outputSize)
        saliencyFC_outputSize = 64
        prdFC_inputSize = saliencyFC_outputSize + self.seqCNN1D_outputSize
        prdFC_linearSize1 = 128
        prdFC_linearSize2 = 128
        
        
        # the head sequence encoder layer
        self.SeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.seqFeatureNum, out_channels=seqCNN1D_outChannels,kernel_size=seqCNN1D_kernelSize),
            nn.BatchNorm1d(seqCNN1D_outChannels),
            nn.ReLU(),
            nn.MaxPool1d(seqCNN1D_poolingRate),
            nn.Dropout(p = dropoutRate),
             )
        
        # the saliency encoder layer 
        self.SaliencyFC = nn.Sequential(
            nn.Linear(self.saliencySize, saliencyFC_outputSize),
            nn.BatchNorm1d(saliencyFC_outputSize),
            nn.ReLU(),
            nn.Dropout(p = dropoutRate),
              )
       
        # the prediction fc layer for DGaze.
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = dropoutRate),
            nn.Linear(prdFC_linearSize2, n_output)
             )
              
    # DGaze net.
    def forward1(self, x):
        headSeq = x[:, 0:self.seqSize]       
        saliencyFeatures = x[:, self.seqSize:]
        
        headSeq = headSeq.reshape(-1, self.seqLength, self.seqFeatureNum)
        headSeq = headSeq.permute(0,2,1)
        seqOut = self.SeqCNN1D(headSeq)
        seqOut = seqOut.reshape(-1, self.seqCNN1D_outputSize)
        saliencyOut = self.SaliencyFC(saliencyFeatures)
        prdInput = torch.cat((seqOut, saliencyOut), 1)
        out = self.PrdFC(prdInput)
        return out
        
    def forward(self, x):
        out = self.forward1(x)
        return out      
    
    
# DGaze_ET Model.
class DGaze_ET(nn.Module):
    def __init__(self, seqLength, seqFeatureNum, saliencyWidth, saliencyNum, n_output, dropoutRate):
        super(DGaze_ET, self).__init__()
        
        # the input params
        self.seqLength = seqLength
        self.seqFeatureNum = seqFeatureNum
        self.seqSize = self.seqLength * self.seqFeatureNum
        self.saliencyWidth = saliencyWidth
        self.saliencyNum = saliencyNum
        self.saliencySize = self.saliencyWidth * self.saliencyWidth * self.saliencyNum
        
        
        # the model params
        seqCNN1D_outChannels = 128
        seqCNN1D_poolingRate = 2
        seqCNN1D_kernelSize = 2
        self.seqCNN1D_outputSize = floor((self.seqLength - seqCNN1D_kernelSize + 1)/seqCNN1D_poolingRate)* seqCNN1D_outChannels
        #print(self.seqCNN1D_outputSize)
        saliencyFC_outputSize = 64
        prdFC_inputSize = saliencyFC_outputSize + self.seqCNN1D_outputSize
        prdFC_linearSize1 = 128
        prdFC_linearSize2 = 128
        
        
        # the head sequence encoder layer
        self.SeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.seqFeatureNum, out_channels=seqCNN1D_outChannels,kernel_size=seqCNN1D_kernelSize),
            nn.BatchNorm1d(seqCNN1D_outChannels),
            nn.ReLU(),
            nn.MaxPool1d(seqCNN1D_poolingRate),
            nn.Dropout(p = dropoutRate),
             )
        
        # the saliency encoder layer 
        self.SaliencyFC = nn.Sequential(
            nn.Linear(self.saliencySize, saliencyFC_outputSize),
            nn.BatchNorm1d(saliencyFC_outputSize),
            nn.ReLU(),
            nn.Dropout(p = dropoutRate),
              )
       
        # the prediction fc layer for DGaze.
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = dropoutRate),
            nn.Linear(prdFC_linearSize2, n_output)
             )
              
    # DGaze net.
    def forward1(self, x):
        gazeHeadObjectSeq = x[:, 0:self.seqSize]       
        saliencyFeatures = x[:, self.seqSize:]
        
        gazeHeadObjectSeq = gazeHeadObjectSeq.reshape(-1, self.seqLength, self.seqFeatureNum)
        gazeHeadObjectSeq = gazeHeadObjectSeq.permute(0,2,1)
        seqOut = self.SeqCNN1D(gazeHeadObjectSeq)
        seqOut = seqOut.reshape(-1, self.seqCNN1D_outputSize)
        saliencyOut = self.SaliencyFC(saliencyFeatures)
        prdInput = torch.cat((seqOut, saliencyOut), 1)
        out = self.PrdFC(prdInput)
        return out
        
    def forward(self, x):
        out = self.forward1(x)
        return out          
    
    
# DGaze_ET Model using only gaze, head and object data as input features
class DGaze_ET_GazeHeadObject(nn.Module):
    def __init__(self, seqLength, seqFeatureNum, n_output, dropoutRate):
        super(DGaze_ET_GazeHeadObject, self).__init__()
        
        # the input params
        self.seqLength = seqLength
        self.seqFeatureNum = seqFeatureNum
        self.seqSize = self.seqLength * self.seqFeatureNum        
        
        
        # the model params
        seqCNN1D_outChannels = 128
        seqCNN1D_poolingRate = 2
        seqCNN1D_kernelSize = 2
        self.seqCNN1D_outputSize = floor((self.seqLength - seqCNN1D_kernelSize + 1)/seqCNN1D_poolingRate)* seqCNN1D_outChannels
        #print(self.seqCNN1D_outputSize)        
        prdFC_inputSize = self.seqCNN1D_outputSize
        prdFC_linearSize1 = 128
        prdFC_linearSize2 = 128
        
        
        # the headobject sequence encoder layer
        self.SeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.seqFeatureNum, out_channels=seqCNN1D_outChannels,kernel_size=seqCNN1D_kernelSize),
            nn.BatchNorm1d(seqCNN1D_outChannels),
            nn.ReLU(),
            nn.MaxPool1d(seqCNN1D_poolingRate),
            nn.Dropout(p = dropoutRate),
             )
        
       
        # the prediction fc layer for DGaze.
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = dropoutRate),
            nn.Linear(prdFC_linearSize2, n_output)
             )
                
    def forward1(self, x):
        headObjectSeq = x[:, 0:self.seqSize]               
        
        headObjectSeq = headObjectSeq.reshape(-1, self.seqLength, self.seqFeatureNum)
        headObjectSeq = headObjectSeq.permute(0,2,1)
        seqOut = self.SeqCNN1D(headObjectSeq)
        seqOut = seqOut.reshape(-1, self.seqCNN1D_outputSize)               
        prdInput = seqOut
        out = self.PrdFC(prdInput)
        return out
    
    def forward(self, x):
        out = self.forward1(x)
        return out  
   