# Copyright (c) Hu Zhiming 2019/7/15 jimmyhu@pku.edu.cn All Rights Reserved.


# deep neural network for gaze prediction.
import sys
sys.path.append('../')
from utils import CalAngularDist, LoadTrainingData, LoadTestData, RemakeDir, MakeDir
from utils.Misc import adjust_learning_rate, AverageMeter
from tensorboardX import SummaryWriter
from models import HuberLoss, CustomLoss
from models import weight_init
from models.DGazeModels import *
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import time
import datetime
import argparse
import os


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# set the random seed to ensure reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)



def main(args):
    
    # Create the model.
    print('\n==> Creating the model...')
    input_size = args.featureNum
    # regress 2 values
    n_output = 2
    model = DGaze_ET(args.seqLength, args.seqFeatureNum, args.saliencyWidth, args.saliencyNum, n_output, args.dropout_rate)
    #model = DGaze_ET_GazeHeadObject(args.seqLength, args.seqFeatureNum, n_output, args.dropout_rate)
    model.apply(weight_init)
    model = torch.nn.DataParallel(model)
    if args.loss == 'L1':
        criterion = nn.L1Loss()
        print('\n==> Loss Function: L1')
    if args.loss == 'MSE':
        criterion = nn.MSELoss()
        print('\n==> Loss Function: MSE')
    if args.loss == 'SmoothL1':
        criterion = nn.SmoothL1Loss()
        print('\n==> Loss Function: SmoothL1')
    if args.loss == 'Huber':
        criterion = HuberLoss(args.loss_beta)
        print('\n==> Loss Function: Huber')
    if args.loss == 'Custom':
        criterion = CustomLoss(args.loss_beta)
        print('\n==> Loss Function: Custom')
    
    # train the model.
    if args.trainFlag == 1:
        # load the training data.
        train_loader = LoadTrainingData(args.datasetDir, args.batch_size)
        # if train the model from scratch.
        if not args.resume:
            # optimizer and loss.
            lr = args.lr
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)  
            #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            # training start epoch
            start_epoch = 0
            # remake checkpoint directory
            RemakeDir(args.checkpoint)
            # remake the summary directory
            RemakeDir(args.summaryDir)
            current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            MakeDir(args.summaryDir + 'Train/')
            summaryDir = args.summaryDir + 'Train/' + current_time
            MakeDir(summaryDir) 
            summaryWriter = SummaryWriter(summaryDir)
        # resume training by loading existing checkpoint.
        else:
            if os.path.isfile(args.resume):
                print("\n==> Loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint['model_state_dict'])
                lr = checkpoint['lr']
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)  
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                loss = checkpoint['loss']
                print('Latest Epoch: {}, Latest Loss: {:.4f}, LR: {:.16f}'.format(start_epoch, loss, lr))
                current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                summaryDir = args.summaryDir + 'Train/' + current_time
                MakeDir(summaryDir) 
                summaryWriter = SummaryWriter(summaryDir)
            else:
                print("\n==> No checkpoint found at '{}'".format(args.resume))    
                
        # training.
        localtime = time.asctime(time.localtime(time.time()))
        print('\nTraining starts at ' + localtime)
        # the number of training steps in an epoch.
        stepNum = len(train_loader)
        num_epochs = args.epochs
        #print('stepNum: {}'.format(stepNum))
        #print('loss_frequency: {}'.format(args.loss_frequency))
        #print('num: {}'.format(int(stepNum/args.loss_frequency)))
        startTime = datetime.datetime.now()
        for epoch in range(start_epoch, num_epochs):
            lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
            print('\nEpoch: {} | LR: {:.16f}'.format(epoch + 1, lr))
            epoch_losses = AverageMeter()
            for i, (features, labels) in enumerate(train_loader):  
                # Move tensors to the configured device
                features = features.reshape(-1, input_size).to(device)
                labels = labels.reshape(-1, n_output).to(device)

                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                epoch_losses.update(loss.item(), features.size(0))
                #print(features.size(0))
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm(model.parameters(), 30)
                optimizer.step()

                # record the global_step and the corresponding loss
                global_step = i+1+epoch*stepNum
                summaryWriter.add_scalar('Step Loss', loss.item(), global_step)

                # output the loss
                if (i+1) % int(stepNum/args.loss_frequency) == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, stepNum, loss.item()))
            
            # record the average loss at each epoch.
            summaryWriter.add_scalar('Epoch Loss',epoch_losses.avg, epoch+1)
            endTime = datetime.datetime.now()
            totalTrainingTime = (endTime - startTime).seconds/60
            print('\nEpoch [{}/{}], Total Training Time: {:.2f} min'.format(epoch+1, num_epochs, totalTrainingTime))    
            #print('Loss1: {:.4f}, Loss2: {:.4f}'.format(loss.item(), epoch_losses.avg))

            # save the checkpoint
            if (epoch +1) % args.interval == 0:
                save_path = os.path.join(args.checkpoint, "checkpoint_epoch_{}.tar".format(str(epoch+1).zfill(3)))
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'lr': lr,
                 }, save_path)

        
        summaryWriter.close()
        localtime = time.asctime(time.localtime(time.time()))
        print('\nTraining ends at ' + localtime)
        
    # test all the existing models.
    # load the existing models to test.
    if os.path.isdir(args.checkpoint):
        filelist = os.listdir(args.checkpoint)
        checkpoints = []
        checkpointNum = 0
        for name in filelist:
            # checkpoints are stored as tar files.
            if os.path.splitext(name)[-1][1:] == 'tar':
                checkpoints.append(name)
                checkpointNum +=1
        # test the checkpoints.
        if checkpointNum:
            print('\nCheckpoint Number : {}'.format(checkpointNum))
            checkpoints.sort()
            summaryDir = args.summaryDir + 'Test/'
            RemakeDir(summaryDir)
            current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            summaryDir = args.summaryDir + 'Test/' + current_time
            MakeDir(summaryDir)
            summaryWriter = SummaryWriter(summaryDir)
            # load the test data.
            test_loader = LoadTestData(args.datasetDir, args.batch_size)
            # load the test labels.
            testY = np.load(args.datasetDir + 'testY.npy')
            # save the predictions.
            if args.save:
                prdDir = args.predictionDir
                RemakeDir(prdDir)
            localtime = time.asctime(time.localtime(time.time()))
            print('\nTest starts at ' + localtime)
            for name in checkpoints:
                print("\n==> Test checkpoint : {}".format(name))
                checkpoint = torch.load(args.checkpoint + name)
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint['epoch']
                # the model's predictions.
                prdY = []
                # evaluate mode
                model.eval()
                epoch_losses = AverageMeter()
                for i, (features, labels) in enumerate(test_loader): 
                    # Move tensors to the configured device
                    features = features.reshape(-1, input_size).to(device)
                    labels = labels.reshape(-1, n_output).to(device)
                    
                    # Forward pass
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    epoch_losses.update(loss.item(), features.size(0))
                   
                    # save the outputs.
                    outputs_npy = outputs.data.cpu().detach().numpy()  
                    if(len(prdY) >0):
                        prdY = np.concatenate((prdY, outputs_npy))
                    else:
                        prdY = outputs_npy
                              
                # save the evaluation results.
                summaryWriter.add_scalar('Epoch Loss',epoch_losses.avg, epoch)
                # Calculate the prediction error (angular distance) between groundtruth and our predictions.
                testSize = testY.shape[0]
                prdError = np.zeros(testSize)
                for i in range(testSize):
                    prdError[i] = CalAngularDist(testY[i, 0:2], prdY[i, 0:2])
                meanPrdError = prdError.mean()
                summaryWriter.add_scalar('Prediction Error',meanPrdError, epoch)
                # standard error of the mean
                SEM = prdError.std()/np.sqrt(testSize)
                print('Epoch: {}, Prediction Mean Error: {:.2f}, SEM: {:.2f}'.format(epoch, meanPrdError, SEM))
                
                # save the predictions.
                if args.save:
                    prdDir = args.predictionDir + 'predictions_epoch_{}/'.format(str(epoch).zfill(3))
                    MakeDir(prdDir)
                    predictions = np.zeros(shape = (testSize, 4))
                    predictions[:, 0:2] = testY
                    predictions[:, 2:4] = prdY
                    np.savetxt(prdDir + 'predictions.txt', predictions)
                  
            localtime = time.asctime(time.localtime(time.time()))
            print('\nTest ends at ' + localtime)   
            summaryWriter.close()            
        else:
            print('\n==> No valid checkpoints in directory {}'.format(args.checkpoint))
    else:
        print('\n==> Invalid checkpoint directory: {}'.format(args.checkpoint))
   
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'DGaze: Predict Gaze Position using Deep Learning model.')
    # the number of input features.
    parser.add_argument('-f', '--featureNum', default=1802, type=int,
                        help='the number of input features (default: 1802)')    
    # the parameters for the net.
    parser.add_argument('--seqLength', default=50, type=int,
                        help=' the length of the head sequence (default: 50)')        
    parser.add_argument('--seqFeatureNum', default=13, type=int,
                        help=' the number of features in the head sequence (default: 13)')        
    parser.add_argument('--saliencyWidth', default=24, type=int,
                        help='the width/height of the saliency map (default: 24)')    
    parser.add_argument('--saliencyNum', default=2, type=int,
                        help='the number of the input saliency maps(default: 2)')                     
    # the dropout rate of the model.
    parser.add_argument('--dropout_rate', default=0.5, type=float,
                        help='the dropout rate of the model (default: 0.5)')       
    # the directory that saves the dataset.
    parser.add_argument('-d', '--datasetDir', default = '../../DGazeDataset/dataset/DGaze_ET/', type = str, 
                        help = 'the directory that saves the dataset (default: ../../DGazeDataset/dataset/DGaze_ET/)')
    # trainFlag = 1 means train new models; trainFlag = 0 means test existing models.
    parser.add_argument('-t', '--trainFlag', default = 1, type = int, help = 'set the flag to train the model (default: 1)')
    # path to save checkpoint
    parser.add_argument('-c', '--checkpoint', default = '../checkpoint/DGaze_ET/', type = str, 
                        help = 'path to save checkpoint (default: ../checkpoint/DGaze_ET/)')
    # path to save the summary
    parser.add_argument('-s', '--summaryDir', default = '../summary/DGaze_ET/', type = str, 
                        help = 'path to save the summary (default: ../summary/DGaze_ET/)')
    # resume training from the latest checkpoint.
    parser.add_argument('-r', '--resume', default='', type=str,
                        help='path to the existing checkpoint (default: none)')
    # save the prediction results or not.
    parser.add_argument('--save', default = 0, type = int, help = 'save the prediction results (1) or not (0) (default: 0)')
    # the directory that saves the prediction results.
    parser.add_argument('-p', '--predictionDir', default = '../predictions/DGaze_ET/', type = str, 
                        help = 'the directory that saves the prediction results (default: ../predictions/DGaze_ET/)')
    # the number of total epochs to run
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='number of total epochs to run (default: 30)')
    # the batch size
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help='the batch size (default: 64)')
    # the interval that we save the checkpoint
    parser.add_argument('-i', '--interval', default=5, type=int,
                        help='the interval that we save the checkpoint (default: 5)')
    # the initial learning rate.
    parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float,
                        help='initial learning rate (default: 1e-2)')
    parser.add_argument('--weight_decay', '--wd', default=0.0, type=float,
                        help='weight decay (default: 0.0)')
    # Decrease learning rate at these epochs.
    parser.add_argument('--schedule', type=int, nargs='+', default=[5, 10, 15, 20, 25, 30],
                        help='Decrease learning rate at these epochs (default: [5, 10, 15, 20, 25, 30])')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule (default: 0.1)')
    # the loss function.
    parser.add_argument('--loss', default="L1", type=str,
                        help='Different loss to train the network: L1 | MSE | SmoothL1 | Huber | Custom (default: L1)')
    parser.add_argument('--loss_beta', type=float, default=1.0,
                        help='The beta parameter for Huber Loss and Custom Loss (default: 1.0)')
    # the frequency that we output the loss in an epoch.
    parser.add_argument('--loss_frequency', default=5, type=int,
                        help='the frequency that we output the loss in an epoch (default: 5)')
    main(parser.parse_args())
    