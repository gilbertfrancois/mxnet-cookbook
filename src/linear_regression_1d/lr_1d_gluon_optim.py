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
from mxnet import nd
from mxnet import autograd
from mxnet import gluon

# %%
# Helper function 
def log_info(net, loss):
    w = net.layer.weight.data().asnumpy()[0][0]
    w_grad = net.layer.weight.grad().asnumpy()[0][0]
    _loss = nd.mean(loss).asscalar()
    print(f"Epoch: {epoch}, Weight: {w:.3f}, Loss: {_loss:.3f}, dW: {w_grad:.3f}")

# %% 
# Configuration
EPOCHS = 50
LR = 0.005 * 2  # Learning rate x 2 to get the same results as other implementations.

# %% 
# Target function to train:
# f: y = xW = 2x
#
# Data x and label y
x = nd.array([[-2], [-1], [0], [1], [2], [3], [4], [5], [6], [7], [8]])
y = 2*x

n_samples, n_features = x.shape
in_features = n_features
out_features = n_features

# %%
# Network with 1 trainable parameter
class LinearRegression(gluon.nn.HybridBlock):

    def __init__(self, in_features, out_features):
        super(LinearRegression, self).__init__()
        # Linear => xW + b, where b=0
        self.layer = gluon.nn.Dense(in_units=in_features, units=out_features, use_bias=False,
                weight_initializer=mx.init.Constant(0.001))

    def forward(self, x):
        return self.layer(x)

net = LinearRegression(in_features, out_features)
net.collect_params().initialize()

# %% 
# Be aware that the L2 and MSE implementations are not the same!
#  L2 Loss function:  nd.sum(nd.square(y_pred - y)) / 2
# MSE Loss function: nd.mean(nd.square(y_pred - y))
#
# NOTE: 
# 1) The L2Loss function does not compute the mean, but the sum of losses of all samples in the batch. 
#    The normalization of the gradient with respect to the batch size happens in the trainer.step(..) function.
# 2) The L2Loss is defined as 1/2 MSE. So to get the same results per iteration as from the other methods, you
#    need to double the learning rate.
loss_fn = gluon.loss.L2Loss()

# %%
# Optimizer, Stochastic Gradient Decent
optimizer = mx.optimizer.SGD(learning_rate=LR, wd=0.0, momentum=0.0)
trainer = gluon.Trainer(net.collect_params(), optimizer)

# %%
# Training loop
for epoch in range(EPOCHS):
    with autograd.record():
        # Compute f(x) = Wx
        y_pred = net(x)
        # Compute loss
        loss = loss_fn(y_pred, y) 
    # Compute dL/dW 
    loss.backward()
    # Show intermediate values to screen
    log_info(net, loss)
    # Update weights, normalization of grads happens here, not in the loss function
    trainer.step(batch_size=len(x))

# %%
# Test the model: f(5) = 2*5 = 10
x_test = nd.array([[5]])
print(f"f(5) = {net(x_test)}")

