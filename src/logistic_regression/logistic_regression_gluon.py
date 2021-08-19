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

import mxnet as mx
import mxnet.gluon.nn as nn
from mxnet import nd
from mxnet import autograd
from mxnet import gluon
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# %%
# Helper function 
def log_info(net, loss):
    w = net.layer.weight.data().asnumpy()
    w_mean = np.mean(w)
    w_std = np.std(w)
    b_mean = nd.mean(net.layer.bias.data()).asscalar()
    w_grad = net.layer.weight.grad().asnumpy()
    w_grad_mean = np.mean(w_grad)
    w_grad_std = np.std(w_grad)
    _loss = nd.mean(loss).asscalar()
    print(f"Epoch: {epoch}, W: {w_mean:.3f}±{w_std:.3f}, b: {b_mean:.3f}, " \
          f"Loss: {_loss:.3f}, dW: {w_grad_mean:.3f}±{w_grad_std:.3f} learning_rate: {LR:.3e}")

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

X_train = nd.array(X_train, dtype="float32")
X_test = nd.array(X_test, dtype="float32")
y_train = nd.array(y_train, dtype="float32").reshape(-1, 1)
y_test = nd.array(y_test, dtype="float32").reshape(-1, 1)

n_samples = X.shape[0]
in_features = X.shape[1]
out_features = 1

# %%
# Network with 1 trainable layer.
class LogisticRegression(gluon.nn.Block):

    def __init__(self, in_features, out_features=1):
        super(LogisticRegression, self).__init__()
        # Linear => xW + b, where b=0
        self.layer = nn.Dense(in_units=in_features, units=out_features, use_bias=True, 
                weight_initializer=mx.init.Constant(0.01), bias_initializer=mx.init.Constant(0.0))
        
    def forward(self, x):
        x = self.layer(x)
        # Note that the loss function has the sigmoid operation for better numerical stability. When
        # doing inference, we need to add the sigmoid function to the model.
        if not autograd.is_training():
            x = nd.sigmoid(x)
        return x

net = LogisticRegression(in_features, out_features)
net.collect_params().initialize()

# %%
# Loss function: Binary Cross Entropy
loss_fn = gluon.loss.SigmoidBinaryCrossEntropyLoss()

# %%
# Optimizer, Stochastic Gradient Decent
optimizer = mx.optimizer.SGD(learning_rate=LR, wd=0.0, momentum=0.0)
trainer = gluon.Trainer(net.collect_params(), optimizer)

# %%
# Training loop
for epoch in range(EPOCHS):
    with autograd.record(train_mode=True):
        # Compute f(x) = Wx
        y_pred = net(X_train)
        # Compute loss
        loss = loss_fn(y_pred, y_train) 
    # Compute dL/dW 
    loss.backward()
    # Show intermediate values to screen
    if epoch % 10 == 0:
        log_info(net, loss)
    # Update weights, normalization of grads happens here, not in the loss function
    trainer.step(batch_size=len(X_train))
    # change learning rate for every 50 steps
    if epoch % 50 == 0 and epoch > 0:
        LR /= 3.0

# %%
# Test the model
y_pred = net(X_train)
y_pred = nd.round(y_pred).asnumpy()
acc_train = np.sum(np.equal(y_pred, y_train.asnumpy()))/len(y_train)
y_pred = net(X_test)
y_pred = nd.round(y_pred).asnumpy()
acc_test = np.sum(np.equal(y_pred, y_test.asnumpy()))/len(y_test)
print(f"acc_train: {acc_train:0.4f}")
print(f"acc_test:  {acc_test:0.4f}")
