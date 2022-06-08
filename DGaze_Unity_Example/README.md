## Solution Explanation

'DGaze_Unity_Example' contains an example of running DGaze model in Unity.  


"DGaze_Server.py" loads a pre-trained DGaze model (predict real-time gaze position) and then waits for input data from Unity client to run the DGaze model. Note that we only utilize head and object data as input features because saliency features are difficult to obtain in real time. The experimental results in our paper validate that using only head and object data can achieve good results.  


"DGaze_ET_Server.py" loads a pre-trained DGaze_ET model (predict future gaze position) and then waits for input data from Unity client to run the DGaze_ET model. Note that we only utilize gaze, head and object data as input features because saliency features are difficult to obtain in real time. The experimental results in our paper validate that using only gaze, head and object data can achieve good results.  


"Unity_Client_DGaze.unity" collects head and object data and sends the data to the python server, i.e. "DGaze_Server.py".  


"Unity_Client_DGaze_ET.unity" collects gaze, head and object data and sends the data to the python server, i.e. "DGaze_ET_Server.py".  


"Unity_Client/Assets/Plugins/" contains the required netmq plugins.  


Unity Scripts:  
"CalculateHeadVelocity.cs": calculates the velocity of a head camera.    
"DataRecorder.cs": collects head and object data.   
"DataRecorder_DGaze_ET.cs": collects gaze, head and object data.   
"Client.cs": sends the collected head and object data to a python server.   
"Client_DGaze_ET.cs": sends the collected gaze, head and object data to a python server.   
"TrackObjects.cs": track the positions of the dynamic objects in the scene.   


Using this example, you can do a lot of interesting things, e.g.  
1. Apply our pre-trained model to your Unity scene.  
2. Collect your own data to retrain our model or train your own model.  
3. Communicate between a Unity client and a Python server to do whatever you like.:)  


## Requirements:
Unity 2019.4.13+  
python 3.6+  
pytorch 1.1.0+  
pyzmq  
netmq  


## Usage:
Run DGaze Model:  
Step 1: Run "DGaze_Server/DGaze_Server.py".  
Step 2: Use Unity to open "Unity_Client" and run "Unity_Client/Assets/Unity_Client_DGaze.unity".  


Run DGaze_ET Model:  
Step 1: Run "DGaze_Server/DGaze_ET_Server.py".  
Step 2: Use Unity to open "Unity_Client" and run "Unity_Client/Assets/Unity_Client_DGaze_ET.unity".  

