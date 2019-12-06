# DGaze: CNN-Based Gaze Prediction in Dynamic Scenes
Project homepage: https://cranehzm.github.io/DGaze.


This repository contains the source code of our model and many pre-trained models.
The pre-trained models are saved in "checkpoint" and there are 22 models in total.
The models include: DGaze for relatime gaze prediction; DGaze for future 100 ms, 200 ms, ..., 1000 ms prediction; DGaze_ET for future 100 ms, 200 ms, ..., 1000 ms prediction; and DGaze for realtime prediction on SGaze dataset.


## Abstract
```
We conduct novel analyses of users' gaze behaviors in dynamic virtual scenes and, based on our analyses, we present a novel CNN-based model called DGaze for gaze prediction in HMD-based applications. 
We first collect 43 users' eye tracking data in 5 dynamic scenes under free-viewing conditions. 
Next, we perform statistical analysis of our data and observe that dynamic object positions, head rotation velocities, and salient regions are correlated with users' gaze positions. 
Based on our analysis, we present a CNN-based model (DGaze) that combines object position sequence, head velocity sequence, and saliency features to predict users' gaze positions. 
Our model can be applied to predict not only realtime gaze positions but also gaze positions in the near future and can achieve better performance than prior method. 
In terms of realtime prediction, DGaze achieves a 22.0% improvement over prior method in dynamic scenes and obtains an improvement of 9.5% in static scenes, based on using the angular distance as the evaluation metric. 
We also propose a variant of our model called DGaze_ET to predict future gaze positions with higher precision by combining accurate past gaze data from an eye tracker.
We further analyze our CNN architecture and verify the effectiveness of each component in our model. 
We also apply our model to gaze-contingent rendering and a game, and also present the evaluation results from a user study.
```	

# Environments:
Ubuntu: 18.04;
python 3.6+;
pytorch 1.1.0+;
tensorboardX 1.8+;
CUDA 9.0+;


# Usage:
Step 1: Download the dataset from our project homepage: https://cranehzm.github.io/DGaze.

Step 2: Run the scripts "run_DGaze.sh"、"run_DGaze_ET.sh" in "scripts/" to retrain or test DGaze model and DGaze_ET model.
		Run "run_DGaze_SGazeDataset.sh" to retrain or test DGaze model on SGaze dataset.

