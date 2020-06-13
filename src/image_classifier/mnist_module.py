#    Copyright 2019 Gilbert Francois Duivesteijn
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

import logging
import time

import mxnet as mx
import numpy as np
from mxnet import nd, gluon

# Set logging to see some output during training
logging.getLogger().setLevel(logging.DEBUG)

# Fix the seed
mx.random.seed(42)

# Set the compute context, GPU is available otherwise CPU
mx_ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

# %%
# -- Constants

EPOCHS = 3
BATCH_SIZE = 256


# %%
# -- Transformation helper functions (in Gluon we use Transformer)

def to_tensor(X):
    X = X.astype(np.float32) / 255.0
    X = nd.moveaxis(X, 3, 1)
    return X


def normalize(X, mean_, std_):
    assert std_ != 0
    return (X - mean_) / nd.power(std_, 2)


# %%
# -- Get some data to train on

# Get data
mnist_train = gluon.data.vision.datasets.FashionMNIST(train=True)
mnist_eval = gluon.data.vision.datasets.FashionMNIST(train=False)

X_train = mnist_train._data
y_train = mnist_train._label

X_eval = mnist_eval._data
y_eval = mnist_eval._label

# To Tensor
X_train = to_tensor(X_train)
X_eval = to_tensor(X_eval)

# Normalize
X_train = normalize(X_train, 0.286, 0.353)
X_eval = normalize(X_eval, 0.286, 0.353)

# %%
# -- Create a data iterator that feeds batches

train_iter = mx.io.NDArrayIter(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True,  label_name="softmax_label")
eval_iter  = mx.io.NDArrayIter(X_eval,  y_eval,  batch_size=BATCH_SIZE, shuffle=False, label_name="softmax_label")

# %%
# -- Define the model
#    Note that the loss function is implicitly given by the layer "SoftmaxOutput"

# placeholders for input and output data
data = mx.sym.Variable("data")
label = mx.sym.Variable("softmax_label")
# layer 1
net = mx.sym.Convolution(data, kernel=(5, 5), pad=(5 // 2, 5 // 2), num_filter=32)
net = mx.sym.Activation(net, act_type="relu")
net = mx.sym.BatchNorm(net, momentum=0.995, eps=0.001)
# layer 2
net = mx.sym.Convolution(net, kernel=(5, 5), pad=(5 // 2, 5 // 2), num_filter=32)
net = mx.sym.Activation(net, act_type="relu")
net = mx.sym.BatchNorm(net, momentum=0.995, eps=0.001)
net = mx.sym.Pooling(net, kernel=(2, 2), pool_type="max", stride=(2, 2))
# layer 3
net = mx.sym.Convolution(net, kernel=(3, 3), pad=(3 // 2, 3 // 2), num_filter=64)
net = mx.sym.Activation(net, act_type="relu")
net = mx.sym.BatchNorm(net, momentum=0.995, eps=0.001)
# layer 4
net = mx.sym.Convolution(net, kernel=(3, 3), pad=(3 // 2, 3 // 2), num_filter=64)
net = mx.sym.Activation(net, act_type="relu")
net = mx.sym.BatchNorm(net, momentum=0.995, eps=0.001)
net = mx.sym.Pooling(net, kernel=(2, 2), pool_type="max", stride=(2, 2))
# layer 5
net = mx.sym.Flatten(net)
net = mx.sym.FullyConnected(net, num_hidden=1024)
net = mx.sym.Activation(net, act_type="relu")
net = mx.sym.BatchNorm(net, momentum=0.995, eps=0.001)
# layer 6
net = mx.sym.Dropout(net, p=0.3)
net = mx.sym.FullyConnected(net, num_hidden=10)
net = mx.sym.SoftmaxOutput(net, label=label, name="softmax")

model = mx.mod.Module(symbol=net, data_names=['data'], label_names=['softmax_label'], context=mx_ctx)

# %%
# -- Print summary
mx.viz.print_summary(net, shape={'data': train_iter.provide_data[0].shape})

# %%
# -- Train
#    Note that the optimizer is given as parameter in the fit() function as an argument.

t0 = time.time()

model.fit(train_iter, eval_iter,
          num_epoch=EPOCHS,
          optimizer="adam",
          optimizer_params={"learning_rate": 0.001},
          eval_metric="acc",
          batch_end_callback=mx.callback.Speedometer(BATCH_SIZE)
          )

print("Elapsed time: {:0.2f} seconds".format(time.time() - t0))

# %%
# -- Save parameters only

model.save_params("fashion_mnist_module.params")


# %%
# -- Save model definition and parameters

model.save_checkpoint("fashion_mnist_module", EPOCHS)
