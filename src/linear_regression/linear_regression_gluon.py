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

import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet import gluon
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

# %%
# -- Constants

data_ctx = mx.cpu()
model_ctx = mx.cpu()

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

train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
eval_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X_eval, y_eval), batch_size=BATCH_SIZE, shuffle=False)
test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X_test), batch_size=BATCH_SIZE, shuffle=False)

# %%
# -- Define model

# layer 1
net = gluon.nn.Dense(1)

# %%
# -- Initialize parameters

net.collect_params().initialize(mx.init.Normal(sigma=1.0), ctx=model_ctx)

# %%
# -- Define loss function and optimizer

loss_fn = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001, 'momentum': 0.9})

# %%
# -- Train

t0 = time.time()

loss_sequence = []
for epoch in range(NUM_EPOCHS):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = loss_fn(output, label)
        loss.backward()
        trainer.step(BATCH_SIZE)
        cumulative_loss += nd.mean(loss).asscalar()
    print("Epoch {}, loss: {}".format(epoch, cumulative_loss / len(X_train)))
    loss_sequence.append(cumulative_loss)

print("Elapsed time: {:0.2f} seconds".format(time.time() - t0))

# %%
# -- Plot training loss on a log scale

plt.plot(np.log10(loss_sequence))
plt.savefig("lr_gluon_loss.png")
# plt.show()

# %%
# -- Predict y for the test data and plot the result as a point cloud

y_pred = net(X_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0].asnumpy(), X_test[:, 1].asnumpy(), y_pred.asnumpy())
plt.savefig("lr_gluon_test.png")
# fig.show()

# %%
# -- Take a peak into the trained weights and bias

params = net.collect_params()
print("Params type is: {}".format(type(params)))

for param in params.values():
    print(param.name, param.data())
