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
def log_info(W, loss):
    print(f"Epoch: {epoch}, Weight: {W.item():.3f}, Loss: {loss:.3f}, dW: {W.grad.item():.3f}")

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

# %%
# Trainable parameter
W = torch.tensor([[0.001]], dtype=torch.float32, requires_grad=True)

# %% 
# Network as a forward function
def forward(x):
    return torch.matmul(x, W)

# %% 
# Loss function: MSE
def loss_fn(y_pred, y):
    return torch.mean(torch.square(y_pred - y))

# %%
# Training loop
for epoch in range(EPOCHS):
    # Compute f(x) = Wx
    y_pred = forward(x)
    # Compute loss
    loss = loss_fn(y_pred, y) 
    # Compute dL/dW 
    loss.backward()
    # Show intermediate values to screen
    log_info(W, loss)
    # Update weights 
    with torch.no_grad():
        W -= LR * W.grad
        # Reset dW for the next iteration
        W.grad.zero_()

# %%
# Test the model: f(5) = 2*5 = 10
x_test = torch.tensor([[5]], dtype=torch.float32)
with torch.no_grad():
    print(f"f(5) = {forward(x_test)}")
