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
# Create a data loader that feeds batches

train_iter = mx.io.NDArrayIter(X_train, y_train, BATCH_SIZE, shuffle=True, label_name='lin_reg_label')
eval_iter = mx.io.NDArrayIter(X_eval, y_eval, BATCH_SIZE, shuffle=False, label_name='lin_reg_label')
test_iter = mx.io.NDArrayIter(X_test, None, BATCH_SIZE)

# %%
# -- Define the model
#    Note that the loss function is implicitly given by the layer "LinearRegressionOutput"

data = mx.sym.Variable('data')
label = mx.sym.Variable('lin_reg_label')

net = mx.sym.FullyConnected(data, num_hidden=1, name='fc1')
net = mx.sym.LinearRegressionOutput(net, label=label, name='lro')

model = mx.mod.Module(symbol=net, data_names=['data'], label_names=['lin_reg_label'])

# %%
# -- Train
#    Note that the optimizer is given as parameter in the fit() function as an argument.

t0 = time.time()

model.fit(train_iter, eval_iter,
          optimizer='sgd',
          optimizer_params={'learning_rate': 0.001, 'momentum': 0.9},
          num_epoch=NUM_EPOCHS,
          eval_metric='mse',
          force_init=True,
          batch_end_callback=mx.callback.Speedometer(len(X_train) // BATCH_SIZE))

print("Elapsed time: {:0.2f} seconds".format(time.time() - t0))

# %%
# -- Plot the predictions

y_pred = model.predict(test_iter)

#%%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0].asnumpy(), X_test[:, 1].asnumpy(), y_pred.asnumpy())
fig.show()

# %%
# -- Take a peak into the trained weights and bias

model.get_params()