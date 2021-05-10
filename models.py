## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        #self.fc1_drop = nn.Dropout(p=0.4) #46
        
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        #self.fc2_drop = nn.Dropout(p=0.4) #21
        
        #self.conv3 = nn.Conv2d(64, 128, 5) 
        #self.pool3 = nn.MaxPool2d(2, 2)
        #self.fc3_drop = nn.Dropout(p=0.3) #26

        #self.conv4 = nn.Conv2d(128, 224, 3) 
        #self.pool4 = nn.MaxPool2d(2, 2)
        #self.fc4_drop = nn.Dropout(p=0.3) #12
        
        self.fc5 = nn.Linear(64*21*21, 1000)
        #self.fc5_drop = nn.Dropout(p=0.4)
        
        self.fc6 = nn.Linear(1000, 500)
        
        
        self.fc7 = nn.Linear(500, 136)
        self.fc6_drop = nn.Dropout(p=0.4)
        #self.softmax = nn.Softmax(dim=1)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #x = self.pool3(F.relu(self.conv3(x)))
        #x = self.fc4_drop(self.pool4(F.relu(self.conv4(x))))
        
        x = x.view(-1, 64*21*21)
        
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        
        x = self.fc7(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
