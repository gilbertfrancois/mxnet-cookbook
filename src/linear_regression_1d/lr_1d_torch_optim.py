#    Copyright 2021 Gilbert Francois Duivesteijn
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch

# %%
# Helper function 
def log_info(net, loss):
    [w] = net.parameters()
    print(f"Epoch: {epoch}, Weight: {w[0].item():.3f}, Loss: {loss.item():.3f}, dW: {w.grad.item():.3f}")

# %% 
# Configuration
EPOCHS = 50
LR = 0.005

# %% 
# Target function to train:
# f: y = xW = 2x
#
# Data x and label y
x = torch.tensor([[-2], [-1], [0], [1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float32)
y = 2*x

n_samples, n_features = x.shape
in_features = n_features
out_features = n_features

# %%
# Network with 1 trainable parameter
class LinearRegression(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(LinearRegression, self).__init__()
        # Linear => xW + b, where b=0
        self.layer = torch.nn.Linear(in_features, out_features, bias=False)
        # Init weight
        self.layer.weight.data.fill_(0.001)

    def forward(self, x):
        return self.layer(x)

net = LinearRegression(in_features, out_features)

# %% 
# Loss function: MSE
loss_fn = torch.nn.MSELoss()

# %%
# Optimizer, Stochastic Gradient Decent
optimizer = torch.optim.SGD(net.parameters(), lr=LR)

# %%
# Training loop
for epoch in range(EPOCHS):
    # Compute f(x) = Wx
    y_pred = net(x)
    # Compute loss
    loss = loss_fn(y_pred, y) 
    # Compute dL/dW 
    loss.backward()
    # Show intermediate values to screen
    log_info(net, loss)
    # Update weights 
    optimizer.step()
    # Reset all gradients for the next iteration
    optimizer.zero_grad()

# %%
# Test the model: f(5) = 2*5 = 10
with torch.no_grad():
    x_test = torch.tensor([[5]], dtype=torch.float32)
    print(f"f(5) = {net(x_test)}")

