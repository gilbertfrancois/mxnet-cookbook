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
import mxnet.ndarray as nd
from mxnet import gluon, init, autograd
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import Dense
from mxnet.gluon.nn import Dropout
from mxnet.gluon.nn import Flatten
from mxnet.gluon.nn import MaxPool2D
from mxnet.gluon.nn import HybridSequential

# Set logging to see some output during training
logging.getLogger().setLevel(logging.DEBUG)

# Fix the seed
mx.random.seed(42)

# Set the compute context, GPU is available otherwise CPU
mx_ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

# %%
# -- Constants

EPOCHS = 50
BATCH_SIZE = 64

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
                                   num_workers=8)

eval_data = gluon.data.DataLoader(mnist_valid.transform_first(transformer),
                                  batch_size=BATCH_SIZE,
                                  num_workers=8
                                  )

# %%
# -- Define the model

net = HybridSequential()
with net.name_scope():
    net.add(
        # block 1
        Conv2D(channels=32, kernel_size=(5, 5), padding=(5 // 2, 5 // 2), activation='relu'),
        BatchNorm(axis=1, momentum=0.9, epsilon=1e-5),
        Conv2D(channels=32, kernel_size=(5, 5), padding=(5 // 2, 5 // 2), activation='relu'),
        BatchNorm(axis=1, momentum=0.9, epsilon=1e-5),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        BatchNorm(axis=1, momentum=0.9, epsilon=1e-5),
        Dropout(0.5),
        # block 2
        Conv2D(channels=64, kernel_size=(3, 3), padding=(3 // 2, 3 // 2), activation='relu'),
        BatchNorm(axis=1, momentum=0.9, epsilon=1e-5),
        Conv2D(channels=64, kernel_size=(3, 3), padding=(3 // 2, 3 // 2), activation='relu'),
        BatchNorm(axis=1, momentum=0.9, epsilon=1e-5),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        BatchNorm(axis=1, momentum=0.9, epsilon=1e-5),
        Dropout(0.5),
        # block 3
        Flatten(),
        Dense(128, activation='relu'),
        BatchNorm(axis=1, momentum=0.9, epsilon=1e-5),
        Dropout(0.3),
        Dense(10)
    )

# %%
# -- Initialize parameters

net.hybridize()
net.initialize(init=init.Xavier(), ctx=mx_ctx)
x = nd.random.randn(1, 1, 28, 28, ctx=mx_ctx)
_ = net(x)

# %%
# -- Define loss function and optimizer

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
lr_scheduler = mx.lr_scheduler.FactorScheduler(base_lr=0.001, factor=0.333, step=10*len(train_data))
trainer = gluon.Trainer(net.collect_params(), 'Adam', {'lr_scheduler': lr_scheduler})


# %%%
# -- Custom metric function

def acc(output, label):
    return (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()


# %%
# -- Train

t0 = time.time()
timestamp = str(int(t0))
param_file_prefix = f"{timestamp}_fashion_mnist"

prev_val_acc = -1

for epoch in range(EPOCHS):
    tic_epoch = time.time()
    train_loss = 0
    train_acc = 0
    eval_acc = 0
    train_emb = []
    train_labels = []
    eval_emb = []
    eval_labels = []

    for i, (data, label) in enumerate(train_data):
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

    train_loss /= len(train_data)
    train_acc /= len(train_data)
    eval_acc /= len(eval_data)

    toc_epoch = time.time()
    chrono_epoch = toc_epoch - tic_epoch
    img_per_sec = len(train_data) / chrono_epoch
    print(f"epoch {epoch}, loss {train_loss:0.3f}, acc {train_acc:0.3f}, val_acc {eval_acc:0.3f}, lr: {trainer.learning_rate}, img/sec: {img_per_sec:0.2f}")

    # Save parameters if they are better than the previous epoch
    if eval_acc > prev_val_acc:
        net.save_parameters(f"{param_file_prefix}-{epoch:04d}.params")
        prev_val_acc = eval_acc


print("Elapsed time {:02f} seconds".format(time.time() - t0))


# %%
# -- Save parameters

net.save_parameters(f"{param_file_prefix}-final.params")
