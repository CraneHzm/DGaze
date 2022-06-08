# Copyright (c) Hu Zhiming 2022/06/07 jimmyhu@pku.edu.cn All Rights Reserved.

# run a pre-trained DGaze_ET model for a single input data.


from models.DGazeModels import *
from utils import AngularCoord2ScreenCoord
import torch
import numpy as np
import zmq


# model parameters
seqLength = 50
seqFeatureNum = 13
n_output = 2
dropout_rate = 0.5

inputSize = seqLength*seqFeatureNum

# path to a pre-trained model of predicting gaze in the future 100 ms.
modelPath = './checkpoint/DGaze_100_GazeHeadObject/checkpoint_epoch_030.tar'


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the server
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")


def main():
	# Create the model
	print('\n==> Creating the model...')
	# We only utilize gaze, head and object data as input features because saliency features are difficult to obtain in real time. The experimental results in our paper validate that using only gaze, head and object data can achieve good results.
	model = DGaze_ET_GazeHeadObject(seqLength, seqFeatureNum, n_output, dropout_rate)
	model = torch.nn.DataParallel(model)
	if device == torch.device('cuda'):
		checkpoint = torch.load(modelPath)
		print('\nDevice: GPU')
	else:
		checkpoint = torch.load(modelPath, map_location=lambda storage, loc: storage)
		print('\nDevice: CPU')
	model.load_state_dict(checkpoint['model_state_dict'])                                          
	# evaluate mode
	model.eval()
	
	while True:
		#  Wait for next request from client
		message = socket.recv()		
		data = message.decode('utf-8').split(',')
		timeStamp = data[0]
		print("Time Stamp: {}".format(timeStamp))
		features = np.zeros((1, inputSize),  dtype=np.float32)
		for i in range(inputSize):			
			features[0, i] = float(data[i+1])
				
		singleInput = torch.tensor(features, dtype=torch.float32, device=device)			
		# Forward pass
		outputs = model(singleInput)
		outputs_npy = outputs.data.cpu().detach().numpy()[0]  			
		# The model outputs angular coordinates. Convert it to screen coordinates for better usage in Unity.
		# Angular coordinates: (0 deg, 0 deg) at screen center
		# Screen coordinates: (0, 0) at Bottom-left, (1, 1) at Top-right
		gaze = AngularCoord2ScreenCoord(outputs_npy)
		print("On-Screen Gaze Position: {}".format(gaze))				
		gaze = str(gaze).encode('utf-8')
		socket.send(gaze)
		
if __name__ == '__main__':
    main()