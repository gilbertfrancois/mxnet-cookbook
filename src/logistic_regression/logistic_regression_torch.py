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
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# %%
# Helper function 
def log_info(net, loss):
    with torch.no_grad():
        [w, b] = net.parameters()
        print(f"Epoch: {epoch}, Weight: {w[0].mean():.3f}+-{w[0].std():.3f}, Loss: {loss.item():.3f}, " \
              f"dW: {w.grad.mean():.3f}, learning_rate: {LR:.3e}")

# %% 
# Configuration
EPOCHS = 500
LR = 0.01

# %%
# Data and label
data = load_breast_cancer()
X = data.data.astype(np.float32)
y = data.target.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, shuffle=True, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train).view(-1, 1)
y_test = torch.from_numpy(y_test).view(-1, 1)

n_samples = X.shape[0]
in_features = X.shape[1]
out_features = 1

# %%
# Network with 1 trainable parameter
class LogisticRegression(nn.Module):

    def __init__(self, in_features, out_features=1):
        super(LogisticRegression, self).__init__()
        # Linear => xW + b, where b=0
        self.layer = nn.Linear(in_features, out_features)
        self.layer.weight.data.fill_(0.001)
        
    def forward(self, x):
        x = self.layer(x)
        x = torch.sigmoid(x)
        return x


net = LogisticRegression(in_features, out_features)

# %%
# Loss function: Binary Cross Entropy
loss_fn = torch.nn.BCELoss()

# %%
# Optimizer, Stochastic Gradient Decent
optimizer = torch.optim.SGD(net.parameters(), lr=LR)

# %%
# Training loop
for epoch in range(EPOCHS):
    # Compute f(x) = Wx
    y_pred = net(X_train)
    # Compute loss
    loss = loss_fn(y_pred, y_train) 
    # Compute dL/dW 
    loss.backward()
    # Show intermediate values to screen
    if epoch % 10 == 0:
        log_info(net, loss)
    # Update weights 
    optimizer.step()
    # Reset all gradients for the next iteration
    optimizer.zero_grad()
    # change learning rate for every 50 steps
    if epoch % 50 == 0 and epoch > 0:
        LR /= 3.0

# %%
# Test the model
with torch.no_grad():
    y_pred = net(X_train)
    y_pred = torch.round(y_pred)
    acc_train = y_pred.eq(y_train).sum()/len(y_train)
    y_pred = net(X_test)
    y_pred = torch.round(y_pred)
    acc_test = y_pred.eq(y_test).sum()/len(y_test)
    print(f"acc_train: {acc_train:0.4f}")
    print(f"acc_test:  {acc_test:0.4f}")
