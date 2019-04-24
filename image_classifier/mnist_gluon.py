import logging
import time

import mxnet as mx
from mxnet import gluon, init, autograd
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import Dense
from mxnet.gluon.nn import Dropout
from mxnet.gluon.nn import Flatten
from mxnet.gluon.nn import MaxPool2D
from mxnet.gluon.nn import Sequential

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
# -- Get some data to train on

mnist_train = gluon.data.vision.datasets.FashionMNIST(train=True)
mnist_valid = gluon.data.vision.datasets.FashionMNIST(train=False)

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.286, 0.353)
])

# %%
# -- Create a data iterator that feeds batches

train_data = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=4)

eval_data = gluon.data.DataLoader(mnist_valid.transform_first(transformer),
                                  batch_size=BATCH_SIZE,
                                  num_workers=4
                                  )

# %%
# -- Define the model

net = Sequential()
with net.name_scope():
    net.add(
        # layer 1
        Conv2D(channels=32, kernel_size=(5, 5), padding=(5 // 2, 5 // 2), activation='relu'),
        BatchNorm(axis=1, momentum=0.995, epsilon=0.001),
        # layer 2
        Conv2D(channels=32, kernel_size=(5, 5), padding=(5 // 2, 5 // 2), activation='relu'),
        BatchNorm(axis=1, momentum=0.995, epsilon=0.001),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        # layer 3
        Conv2D(channels=64, kernel_size=(3, 3), padding=(3 // 2, 3 // 2), activation='relu'),
        BatchNorm(axis=1, momentum=0.995, epsilon=0.001),
        # layer 4
        Conv2D(channels=64, kernel_size=(3, 3), padding=(3 // 2, 3 // 2), activation='relu'),
        BatchNorm(axis=1, momentum=0.995, epsilon=0.001),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        # layer 5
        Flatten(),
        Dense(1024, activation='relu'),
        BatchNorm(axis=1, momentum=0.995, epsilon=0.001),
        # layer 6
        Dropout(0.3),
        Dense(10)
    )

# %%
# -- Initialize parameters

net.initialize(init=init.Xavier(), ctx=mx_ctx)

# %%
# -- Define loss function and optimizer

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'Adam', {'learning_rate': 0.001})


# %%%
# -- Custom metric function

def acc(output, label):
    return (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()


# %%
# -- Train

t0 = time.time()

for epoch in range(EPOCHS):
    train_loss = 0
    train_acc = 0
    eval_acc = 0

    for data, label in train_data:
        data = data.as_in_context(mx_ctx)
        label = label.as_in_context(mx_ctx)
        with autograd.record():
            output = net(data)
            loss = loss_fn(output, label)
        loss.backward()
        trainer.step(batch_size=BATCH_SIZE)
        train_loss += loss.mean().asscalar()
        train_acc += acc(output, label)

    for data, label in eval_data:
        data = data.as_in_context(mx_ctx)
        label = label.as_in_context(mx_ctx)
        eval_acc += acc(net(data), label)

    print("epoch {}, loss {:0.3f}, acc {:0.3f}, val_acc {:0.3f}".format(
        epoch, train_loss / len(train_data), train_acc / len(train_data), eval_acc / len(eval_data)))

print("Elapsed time {:02f} seconds".format(time.time() - t0))


# %%
# -- Save parameters

net.save_parameters('fashion_mnist_gluon.params')
