import numpy as np
import matplotlib.pyplot as plt

import mxnet

print("Using mxnet version {}".format(mxnet.__version__))

import os
import mxnet as mx
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

import time

# %%
BATCH_SIZE = 256
EPOCHS = 20
NET_PARAMS_CHECKPOINT = '1549639509-net.params'

ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)


# %%

mnist_train = gluon.data.vision.datasets.MNIST(train=True)
mnist_valid = gluon.data.vision.datasets.MNIST(train=False)

transformer = transforms.Compose([
    transforms.ToTensor()
])

train_data = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=4)

valid_data = gluon.data.DataLoader(mnist_valid.transform_first(transformer),
                                   batch_size=BATCH_SIZE,
                                   num_workers=4
                                   )


#%%

net = nn.Sequential()
net.add(
    nn.Conv2D(channels=32, kernel_size=(5, 5), padding=(5 // 2, 5 // 2), activation='relu'),
    nn.Conv2D(channels=32, kernel_size=(5, 5), padding=(5 // 2, 5 // 2), activation='relu'),
    nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    nn.Conv2D(channels=64, kernel_size=(5, 5), padding=(5 // 2, 5 // 2), activation='relu'),
    nn.Conv2D(channels=64, kernel_size=(5, 5), padding=(5 // 2, 5 // 2), activation='relu'),
    nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    nn.Conv2D(channels=128, kernel_size=(3, 3), padding=(3 // 2, 3 // 2), activation='relu'),
    nn.Conv2D(channels=128, kernel_size=(3, 3), padding=(3 // 2, 3 // 2), activation='relu'),
    nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    nn.Flatten(),
    nn.Dense(2, activation='relu'),
    nn.Dense(10)
)
# net.initialize(mx.init.Xavier(), ctx=ctx)
if os.path.exists(NET_PARAMS_CHECKPOINT):
    net.load_parameters(NET_PARAMS_CHECKPOINT, ctx)
# print(net.summary(nd.random.randn(1, 1, 28, 28)))

# %%

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'Adam', {'learning_rate': 0.0001, 'epsilon': 1e-8})

# %%%

def acc(output, label, ctx):
    output = output.as_in_context(ctx)
    label = label.as_in_context(ctx)
    return (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()

# %%

for epoch in range(EPOCHS):
    train_loss = 0
    train_acc = 0
    valid_acc = 0
    t0 = time.time()

    for data, label in train_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        with autograd.record():
            output = net(data)
            loss = loss_fn(output, label)
        loss.backward()
        trainer.step(batch_size=BATCH_SIZE)
        train_loss += loss.mean().asscalar()
        train_acc += acc(output, label, ctx)

    for data, label in valid_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        valid_acc += acc(output, label, ctx)

    t1 = time.time() - t0
    print("epoch {}, loss {:0.3f}, acc {:0.3f}, val_acc {:0.3f}, time {:0.3f}".format(
        epoch, train_loss / len(train_data), train_acc / len(train_data), valid_acc / len(valid_data), t1))

# %%
timestamp = str(int(time.time()))
net.save_parameters('{}-net.params'.format(timestamp))
# %%
y_pred_list = []
label_list = []
for data, label in train_data:
    data = data.as_in_context(ctx)
    label = label.as_in_context(ctx)

    y_pred = net(data)[:11]
    y_pred_list.append(y_pred.asnumpy())
    label_list.append(label.asnumpy())

#%%