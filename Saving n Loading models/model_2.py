import torch
import torch.nn as nn
from pathlib import Path

class LinearRegModel(nn.Module): 
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1,requires_grad=True))

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.weights*x + self.bias
    

weight  = 0.3
bias = 0.9

x = torch.arange(0,1,0.01)
y = weight*x + bias

train_len = round(0.8*len(x))

train_data = x[:train_len]
train_labels = y[:train_len]
test_data = x[train_len:]
test_labels = y[train_len:]

torch.manual_seed(42)
    
train_model = LinearRegModel()
loss_func = nn.MSELoss()
optimiser = torch.optim.SGD(params=train_model.parameters(),lr=0.01)

train_loss = {};test_loss = {}

#training loop
epochs = 900
for epoch in range(epochs):
    train_model.train()
    predictions = train_model(train_data)
    trn_loss = loss_func(predictions,train_labels)
    optimiser.zero_grad()
    trn_loss.backward()
    optimiser.step()
    print(train_model.state_dict())

    train_model.eval()
    if epoch%20==0:
        with torch.inference_mode():
            preds = train_model(test_data)
        tst_loss = loss_func(preds,test_labels)
        test_loss[epoch] = tst_loss.detach().numpy()
        train_loss[epoch] = trn_loss.detach().numpy()

model_path = Path.cwd()
model_name = "model_2.pt"
model_location = model_path/model_name

torch.save(obj=train_model.state_dict(),f=model_location)