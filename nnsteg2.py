import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from math import log

'''
****************************************************************
Jacob Cleveland
4/12/21
ECEN 4920
University of Nebraska, Lincoln
Department of Electrical and Computer Engineering

Artifical Neural Network steganography implementation module.
****************************************************************
'''

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 794
hidden_size = 500
stage1_output = 784
num_classes = 10
num_epochs = 1
batch_size = 1
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

stegModel = NeuralNet(input_size, hidden_size, stage1_output).to(device)
decModel = NeuralNet(stage1_output, hidden_size, num_classes).to(device)

criterion = nn.MSELoss()
stegOptim = torch.optim.Adam(stegModel.parameters(), lr=learning_rate)  
decOptim = torch.optim.Adam(decModel.parameters(), lr=learning_rate)
message = [-75,-115,-45,-45,-15,75,-15,25,-45,-55]
origMess = np.copy(message)
hello = []
alt = []

#Combine message with picture input
#since the model takes 794 floats as input
for i in range(0,1):
    buf = message
    altbuf = []
    for j in range(0,784):
        buf.append(1.*i+j)
        altbuf.append(1+1.*i+j)
    hello.append(buf)
    alt.append(altbuf)

#create torch tensor of input vector
hello = torch.tensor(hello)
alt = torch.tensor(alt)
alt.requires_grad = True
#create torch sensor of original message
origMess = torch.tensor(origMess).float()
origMess.requires_grad = True
#train 100 times
for i in range(0,100):
    stegImage = stegModel(hello)
    decMess = decModel(stegImage)
    decMess = decMess[0]
    decLoss = criterion(origMess,decMess)
    decOptim.zero_grad()
    decLoss.backward(retain_graph = True)
    decOptim.step()
    stegLoss = criterion(alt,stegImage)
    if(i%10 == 0):
        #print progress update
        print((i,stegLoss.item(),decLoss.item()))
    stegOptim.zero_grad()
    stegLoss.backward()
    stegOptim.step()
#save a copy of the original image
alt = np.copy(alt.detach())
alt = np.reshape(alt, (28,28))
altImage = Image.fromarray(alt)
altImage = altImage.convert('RGB')
altImage.save("alt.png")
#save a copy of the embedded image
stegImage = np.copy(stegImage.detach())
stegImage = np.reshape(stegImage, (28,28))
stegImage = Image.fromarray(stegImage)
stegImage = stegImage.convert('RGB')
stegImage.save("stegImage.png")
#print out decoded and original messages
print(np.copy(decMess.detach()))    
print(np.copy(origMess.detach()))  
    
# Save the model checkpoint
torch.save(stegModel.state_dict(), 'stegModel.ckpt')
torch.save(decModel.state_dict(), 'decModel.ckpt')
