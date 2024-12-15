import torch
from linear_reg import LinearRegression

trained_model = LinearRegression()
trained_model.load_state_dict(torch.load(f="model_2.pt"))

print(trained_model.state_dict())