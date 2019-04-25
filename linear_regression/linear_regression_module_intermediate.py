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

import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import nd
from mpl_toolkits.mplot3d import Axes3D

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
# -- Define the model
#    Note that the loss function is implicitly given by the layer "LinearRegressionOutput"

# placeholders for input and output data
data = mx.sym.Variable('data')
label = mx.sym.Variable('lin_reg_label')
# layer 1
net = mx.sym.FullyConnected(data, num_hidden=1, name='fc1')
net = mx.sym.LinearRegressionOutput(net, label=label, name='lro')

model = mx.mod.Module(symbol=net, data_names=['data'], label_names=['lin_reg_label'])

# %%
# -- Train
#    Intermediate level training (alternative to high level fit() function).

log = []

model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label, for_training=True)
model.init_params(initializer=mx.init.Xavier(magnitude=2), force_init=True)
model.init_optimizer(optimizer="sgd", optimizer_params={'learning_rate': 0.001, 'momentum': 0.9}, )
metric = mx.metric.MSE()
t0 = time.time()

for epoch in range(NUM_EPOCHS):
    tic = time.time()
    train_iter.reset()
    metric.reset()
    for batch in train_iter:
        model.forward(batch, is_train=True)
        model.update_metric(metric, batch.label)
        model.backward()
        model.update()
        log.append(metric.get())
    toc = time.time()
    name, val = metric.get_name_value()[0]
    print("Epoch: {:05d}, Chrono: {:0.3f}, Train-{}: {:e}".format(epoch, (toc-tic), name, val))

print("Elapsed time: {:0.3f} seconds".format(time.time() - t0))


# %%
# -- Predict y for the test data and plot the result as a point cloud

y_pred = model.predict(test_iter)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0].asnumpy(), X_test[:, 1].asnumpy(), y_pred.asnumpy())
fig.show()

# %%
# -- Take a peak into the trained weights and bias

print(model.get_params())

# %%
# -- Plot loss

l2_loss = [item[1] for item in log]
plt.figure()
plt.plot(l2_loss)
plt.show()
