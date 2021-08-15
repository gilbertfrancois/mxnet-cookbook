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

import numpy as np

# %%
# Helper function 
def log_info(W, W_grad, loss):
    print(f"Epoch: {epoch}, Weight: {W[0][0]:.3f}, Loss: {loss:.3f}, dW: {W_grad:.3f}")

# %% 
# Configuration
EPOCHS = 50
LR = 0.005

# %% 
# Target function to train:
# f: y = xW = 2x
#
# Data x and label y
x = np.array([[-2], [-1], [0], [1], [2], [3], [4], [5], [6], [7], [8]], dtype=np.float32)
y = (2*x).astype(np.float32)

# %%
# Trainable parameter
W = np.array([[0.001]], dtype=np.float32)

# %% 
# Network as a forward function
def forward(x):
    return np.matmul(x, W)

# %% 
# Loss function: MSE
def loss_fn(y_pred, y):
    return np.mean(np.square(y_pred - y))

# %% 
# Compute gradient function
#     L = 1/N ∑(xW - y)^2
# ∂L/∂W = 2/N ∑(xW - y)·x
def backward(x, y, y_pred):
    return np.mean(2*x*(y_pred - y))
    
# %%
# Training loop
for epoch in range(EPOCHS):
    # Compute f(x) = Wx
    y_pred = forward(x)
    # Compute loss
    loss = loss_fn(y_pred, y) 
    # Compute dL/dW 
    W_grad = backward(x, y, y_pred)
    # Show intermediate values to screen
    log_info(W, W_grad, loss)
    # Update weights 
    W -= LR * W_grad

# %%
# Test the model: f(5) = 2*5 = 10
x_test = np.array([[5]])
print(f"f(5) = {forward(x_test)}")

