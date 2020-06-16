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
from mxboard import SummaryWriter
from mxnet import gluon, init, autograd
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import Dense
from mxnet.gluon.nn import Dropout
from mxnet.gluon.nn import Flatten
from mxnet.gluon.nn import MaxPool2D
from mxnet.gluon.nn import Activation
from mxnet.gluon.nn import HybridSequential

# Set logging to see some output during training
logging.getLogger().setLevel(logging.DEBUG)

# Hyperparameters

N_GPUS = 2
mx_ctx = [mx.gpu(i) for i in range(N_GPUS)] if N_GPUS > 0 else [mx.cpu()]
N_WORKERS = 4
EPOCHS = 50
PER_DEVICE_BATCH_SIZE = 64
BATCH_SIZE = PER_DEVICE_BATCH_SIZE * max(len(mx_ctx), 1)


# Fix the seed
mx.random.seed(42)

# Set the compute context, GPU is available otherwise CPU
# mx_ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

# %%
# -- Constants


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
                                   num_workers=N_WORKERS)

eval_data = gluon.data.DataLoader(mnist_valid.transform_first(transformer),
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=N_WORKERS)

# %%
# -- Define the model

net = HybridSequential()
with net.name_scope():
    net.add(
        # block 1
        Conv2D(channels=32, kernel_size=(5, 5), strides=(1, 1), padding=(2, 2)),
        Activation("relu"),
        BatchNorm(axis=1, momentum=0.9, epsilon=1e-5),
        Conv2D(channels=32, kernel_size=(5, 5), strides=(1, 1), padding=(2, 2)),
        Activation("relu"),
        BatchNorm(axis=1, momentum=0.9, epsilon=1e-5),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        BatchNorm(axis=1, momentum=0.9, epsilon=1e-5),
        Dropout(0.5),
        # block 2
        Conv2D(channels=64, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
        Activation("relu"),
        BatchNorm(axis=1, momentum=0.9, epsilon=1e-5),
        Conv2D(channels=128, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1)),
        Activation("relu"),
        BatchNorm(axis=1, momentum=0.9, epsilon=1e-5),
        # MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        # BatchNorm(axis=1, momentum=0.9, epsilon=1e-5),
        Dropout(0.5),
        # block 3
        Flatten(),
        Dense(128),
        Activation("relu"),
        BatchNorm(),
        Dropout(0.3),
        Dense(10)
    )

# %%
# -- Initialize parameters

net.initialize(init=init.Xavier(), ctx=mx_ctx)

# %%
# -- Print summary before hybridizing

x = nd.random.randn(1, 1, 28, 28, ctx=mx_ctx[0])
net.summary(x)

# %%
# -- Hybridize and run a forward pass once to generate a symbol which will be used later
#    for plotting the network.

net.hybridize()
net(x)


# %%
# -- Define loss function and optimizer

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
lr_scheduler = mx.lr_scheduler.FactorScheduler(base_lr=0.001, factor=0.333, step=10*len(train_data))
trainer = gluon.Trainer(net.collect_params(), 'Adam', {'lr_scheduler': lr_scheduler})

train_metric = mx.metric.Accuracy()
eval_metric = mx.metric.Accuracy()

# %%%
# -- Custom metric function

def acc(output, label):
    return (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()


# %%
# -- Train

t0 = time.time()
timestamp = str(int(t0))
param_file_prefix = f"{timestamp}_fashion_mnist"

with SummaryWriter(logdir=f"../../log/{timestamp}_mnist_gluon", flush_secs=5) as sw:
    sw.add_graph(net)

    prev_val_acc = -1

    for epoch in range(EPOCHS):
        train_metric.reset()
        eval_metric.reset()
        train_loss = 0
        eval_loss = 0
        train_acc = 0
        eval_acc = 0

        # Forward pass and update weights
        tic_epoch = time.time()
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=mx_ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=mx_ctx, batch_axis=0, even_split=False)
            with autograd.record():
                output_list = [net(X) for X in data]
                loss_list = [loss_fn(yhat, y) for yhat, y in zip(output_list, label)]
            for loss in loss_list:
                loss.backward()
            trainer.step(batch_size=BATCH_SIZE)
            train_loss += sum([loss.mean().asscalar() for loss in loss_list]) / len(loss_list)
            train_metric.update(label, output_list)
        toc_epoch = time.time()
        chrono_epoch = toc_epoch - tic_epoch

        # Evaluate
        for i, batch in enumerate(eval_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=mx_ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=mx_ctx, batch_axis=0, even_split=False)
            output_list = [net(X) for X in data]
            loss_list = [loss_fn(yhat, y) for yhat, y in zip(output_list, label)]
            eval_loss += sum([loss.mean().asscalar() for loss in loss_list]) / len(loss_list)
            eval_metric.update(label, output_list)

        train_loss /= len(train_data)
        eval_loss /= len(eval_data)
        _, train_acc = train_metric.get()
        _, eval_acc = eval_metric.get()

        img_per_sec = len(mnist_train) / chrono_epoch
        print(f"{len(mnist_train)}, epoch {epoch}, train_loss {train_loss:0.3f}, acc {train_acc:0.3f}, val_loss: {eval_loss:0.3f}, val_acc {eval_acc:0.3f}, lr: {trainer.learning_rate}, img/sec: {img_per_sec:0.2f}, chrono: {chrono_epoch:0.2f}")

        # Add some values to mxboard
        sw.add_scalar("Accuracy", ("train_acc", train_acc), global_step=epoch)
        sw.add_scalar("Accuracy", ("val_acc", eval_acc), global_step=epoch)
        for key, param in net.collect_params(".*weight").items():
            sw.add_histogram(f"grad_{param.name}", param.grad(ctx=mx_ctx[0]), global_step=epoch)

        # Save parameters if they are better than the previous epoch
        if eval_acc > prev_val_acc:
            net.save_parameters(f"{param_file_prefix}-{epoch:04d}.params")
            prev_val_acc = eval_acc


print("=== Total training time: {:02f} seconds".format(time.time() - t0))

# %%
# -- Save parameters

net.save_parameters(f"{param_file_prefix}-final.params")
