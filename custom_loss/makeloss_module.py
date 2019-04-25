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
from mxnet import nd
import numpy as np

logging.getLogger().setLevel(logging.DEBUG)

# %%
# -- Constants

NUM_INPUTS = 2
NUM_OUTPUTS = 1
NUM_EPOCHS = 10
BATCH_SIZE = 4


# %%
# -- Create some data to train on

def get_x(num_examples, num_inputs):
    return nd.random.normal(shape=(num_examples, num_inputs))


def get_y(X):
    y = 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2
    noise = 0.01 * nd.random.normal(shape=(len(X),))
    return y + noise


# %%
# -- Split train / eval data

X_train = get_x(9000, NUM_INPUTS)
y_train = get_y(X_train)

X_eval = get_x(1000, NUM_INPUTS)
y_eval = get_y(X_eval)

X_test = get_x(1000, NUM_INPUTS)

# %%
# -- Create a data loader that feeds batches

train_iter = mx.io.NDArrayIter(X_train, y_train, BATCH_SIZE, shuffle=True, label_name='lin_reg_label')
eval_iter = mx.io.NDArrayIter(X_eval, y_eval, BATCH_SIZE, shuffle=False, label_name='lin_reg_label')
test_iter = mx.io.NDArrayIter(X_test, None, BATCH_SIZE)


# %%
# -- Some custom loss function

def loss_fn(out, label):
    # the out of fc1 has shape [batch_size, 1], so we have to reshape it first to get the same shape as `label`.
    out = mx.sym.Reshape(out, shape=0)
    # Don't compute the gradient of the label.
    label = mx.sym.BlockGrad(label)
    loss = mx.sym.mean(mx.sym.abs(out - label))
    return loss


# %%
# -- Define the model
#    Note that the loss function is implicitly given by the layer "LinearRegressionOutput"

# placeholders for input and output data
data = mx.sym.Variable('data')
label = mx.sym.Variable('lin_reg_label')
# layer 1
fc1 = mx.sym.FullyConnected(data, num_hidden=1, name='fc1')

# Create output for the metric that blocks computation of the metric to indicate that it is the last in the chain.
fc1_output = mx.sym.BlockGrad(fc1, name='y')

# Custom loss function
lro = mx.sym.MakeLoss(loss_fn(fc1, label), name="lro")

# Add fc1 output to the list of outputs for the custom eval metric
sym = mx.sym.Group([lro, fc1_output])

model = mx.mod.Module(symbol=sym, data_names=['data'], label_names=['lin_reg_label'])


# %%
# -- IMPORTANT: Since the MakeLoss function returns the gradient of the loss, instead of the loss value, we need
#               to create a custom metric. The standard MSE compares the label with the output of the loss function,
#               which is in this case the gradent of (y_pred-label). It will look like the model is not converging,
#               but actually it is...
#
#               For illustration purposes, both outputs will be measured. Notice how metric_fc1 shows convergence,
#               metric_lro shows garbage.

def eval_metric(label, out):
    out = np.squeeze(out)
    return np.mean(np.square(label - out))


metric_fc1 = mx.metric.CustomMetric(eval_metric, name='mse_y', output_names=['y_output'], label_names=['lin_reg_label'])
metric_lro = mx.metric.CustomMetric(eval_metric, name='mse_lro', output_names=['lro_output'],
                                    label_names=['lin_reg_label'])

# %%
# -- Train
#    Note that the optimizer is given as parameter in the fit() function as an argument.

t0 = time.time()

model.fit(train_iter, eval_iter,
          optimizer='sgd',
          optimizer_params={'learning_rate': 0.001, 'momentum': 0.9},
          num_epoch=NUM_EPOCHS,
          eval_metric=[metric_fc1, metric_lro],
          force_init=True,
          batch_end_callback=mx.callback.Speedometer(BATCH_SIZE, 1000)
          )

print("Elapsed time: {:0.2f} seconds".format(time.time() - t0))

# %%
# -- Take a peak into the trained weights and bias

print(model.get_params())
