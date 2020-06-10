#    Copyright 2020 Gilbert Francois Duivesteijn
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

import os
import time
import datetime

import mxnet as mx
import numpy as np
from gluoncv import model_zoo
from mxnet import autograd
from mxnet import gluon
from mxnet import init, nd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

# %%
# -- Hyperparameters

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
classes = 23
epochs = 5
lr = 0.001
per_device_batch_size = 64
momentum = 0.9
wd = 0.0001

lr_factor = 0.75
lr_steps = [10, 20, 30, np.inf]

num_gpus = 0
num_workers = 8
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
batch_size = per_device_batch_size * max(num_gpus, 1)
pretrained_model_name = "ResNet50_v2"
model_name = f"minc_{pretrained_model_name.lower()}_{timestamp}"
print(f"--- Model name: {model_name}")

# %%
# -- Data augmentation

jitter_param = 0.4
lightning_param = 0.1

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param, saturation=jitter_param),
    transforms.RandomLighting(lightning_param),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# %%
# -- Data loaders

# data_folder = os.path.join(os.path.dirname(__file__), "..", "..", "..", "minc-2500")
data_folder = "/Users/gilbert/Development/git/mxnet-cookbook/data/minc-2500"
train_data_folder = os.path.join(data_folder, "train")
val_data_folder = os.path.join(data_folder, "val")
test_data_folder = os.path.join(data_folder, "test")

train_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(train_data_folder).transform_first(transform_train),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

val_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(val_data_folder).transform_first(transform_test),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

test_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(test_data_folder).transform_first(transform_test),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

# %%
# -- Model and Trainer

x = nd.random.randn(1, 3, 224, 224)
net = model_zoo.get_model(pretrained_model_name, pretrained=True)

# %%
#
with net.name_scope():
    net.output = nn.Dense(classes)
net.output.initialize(init.Xavier(), ctx=ctx)
net.collect_params().reset_ctx(ctx)
# net.summary(x)
net.hybridize()

# %%
# --

trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": lr, "momentum": momentum, "wd": wd})
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
metric = mx.metric.Accuracy()

# %%
# -- Test function

def test(net_, val_data_, ctx_):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data_):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx_, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx_, batch_axis=0, even_split=False)
        outputs = [net_(X) for X in data]
        metric.update(label, outputs)
    return metric.get()


# %%
# -- Training Loop

lr_counter = 0
epoch = 0
num_batch = len(train_data)
prev_val_acc = -1

for epoch in range(epochs):
    # Adapt learning rate if applicable
    if epoch == lr_steps[lr_counter]:
        trainer.set_learning_rate(trainer.learning_rate * lr_factor)
        lr_counter += 1
    # Init training loop
    tic = time.time()
    train_loss = 0
    metric.reset()
    # Start training loop
    print(f"*** Starting training loop...")
    print(f"--- Using: {ctx}")
    for i, batch in enumerate(train_data):
        # Splits an NDArray into len(ctx_list) slices along batch_axis and loads each slice to one context in ctx_list.
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        with autograd.record():
            output_list = [net(X) for X in data]
            loss_list = [loss_fn(yhat, y) for yhat, y in zip(output_list, label)]
        # Compute gradients
        for loss in loss_list:
            loss.backward()
        # Update weights
        trainer.step(batch_size)
        # Compute batch loss and add to total training loss
        batch_loss = sum([loss.mean().asscalar() for loss in loss_list]) / len(loss_list)
        train_loss += batch_loss
        metric.update(label, output_list)
        print(f"[Epoch {epoch}], batch: {i:04d}, batch_loss: {batch_loss:0.5f}")
    # Get metrics
    train_loss /= num_batch
    _, train_acc = metric.get()
    _, val_acc = test(net, val_data, ctx)
    # Save best model, based on val_acc
    if val_acc > prev_val_acc:
        net.export(model_name, epoch)
    # Save every N epochs to be able to restart
    if epoch % 10 == 0:
        net.export(f"checkpoint_{model_name}", epoch)

    toc = time.time()
    chrono = toc - tic
    print(f"[Epoch {epoch}], train_acc: {train_acc:0.3f}, val_acc: {val_acc:0.3f}, chrono: {chrono}")

# Save last epoch checkpoint
net.export(f"checkpoint_{model_name}", epoch)
