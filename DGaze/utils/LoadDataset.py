# the class to load our data
import torch
import torch.utils.data as data
import numpy as np


# create our dataset.
class MyDataset(data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        feature, target = self.features[index], self.labels[index]
        return feature, target
    
    def __len__(self):
        return len(self.features)

    
# load the training data.    
def LoadTrainingData(datasetDir, batch_size):
    print("\nLoading the training data...")
    trainingX = torch.from_numpy(np.load(datasetDir + 'trainingX.npy')).float()
    print('\nTraining Data Size: {}'.format(list(trainingX.size())))
    trainingY = torch.from_numpy(np.load(datasetDir + 'trainingY.npy')).float()
    # resize the image.
    #saliencyMap = trainingY[:, 65:].reshape(32, 32, 2)
    
    
    train_dataset = MyDataset(trainingX, trainingY)
    # Data loader
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
    
    
    
    
# load the test data
def LoadTestData(datasetDir, batch_size):
    print("\nLoading the test data...")
    testX = torch.from_numpy(np.load(datasetDir + 'testX.npy')).float()
    print('\nTest Data Size: {}'.format(list(testX.size())))
    testY = torch.from_numpy(np.load(datasetDir + 'testY.npy')).float()
    test_dataset = MyDataset(testX, testY)
    # Data loader
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader    