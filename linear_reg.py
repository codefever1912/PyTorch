import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

weight = 0.5
bias = 0.3

x = torch.arange(0,1,0.02)
y = weight*x+bias

train_len = round(0.8*len(x))
train_data = x[:train_len]
train_labels = y[:train_len]
test_data = x[train_len:]
test_labels = y[train_len:]

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1,requires_grad=True))
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.weights*x + self.bias

def plot_pred(train_data=train_data,train_labels=train_labels,test_data=test_data,test_labels=test_labels,predictions=None): # predictions will be a tensor containing the predictions made by the model on test data
    plt.figure(figsize=(10,6))
    plt.scatter(train_data,train_labels,c='g',s=5,label="Training Data")
    plt.scatter(test_data,test_labels,c='b',s=5,label="Test Data")
    if predictions is not None:
       plt.scatter(test_data,predictions,c='r',s=5,label="Predictions")
    plt.legend()
    plt.title("Linear Regression Model")
    plt.show()