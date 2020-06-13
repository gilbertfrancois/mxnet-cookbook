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

import datetime
import os
import pickle as pkl
import time

import matplotlib.pyplot as plt
import mxnet as mx
from gluoncv import model_zoo
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

# %%
# -- Hyperparameters

pretrained_model_name = "ResNet50_v2"
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
classes = 23
transfer_epochs = 50
per_device_batch_size = 32
sgd_momentum = 0.9
sgd_wd = 0.0001
lr_base = 0.0001
lr_factor = 0.333
lr_epoch_steps = 10
num_gpus = 2
num_workers = 16
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
batch_size = per_device_batch_size * max(num_gpus, 1)
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

# data_folder = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "minc-2500")
data_folder = "/home/gilbert/Development/git/mxnet-cookbook/data/minc-2500"
# data_folder = "/Users/gilbert/Development/git/mxnet-cookbook/data/minc-2500"
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
# -- Pretrained model

x = nd.random.randn(1, 3, 224, 224)
pretrained_net = model_zoo.get_model(pretrained_model_name, pretrained=True)

# %%
# -- Freeze feature layers and add new tail

if pretrained_model_name == "VGG16":
    pretrained_features = pretrained_net.features[:31]
else:
    pretrained_features = pretrained_net.features
# Freeze pretrained layers
for param in pretrained_features.collect_params().values():
    param.grad_req = 'null'
# Create new layers for the tail
if pretrained_model_name == "VGG16":
    new_tail = nn.HybridSequential()
    new_tail.add(
        nn.Dense(4096),
        nn.Dropout(0.5),
        nn.Activation("relu"),
        nn.Dense(4096),
        nn.Dropout(0.5),
        nn.Dense(classes)
    )
    new_tail.initialize(init=mx.init.Xavier(), ctx=ctx)
    # Create new net for transfer learning
    net = nn.HybridSequential()
    with net.name_scope():
        for layer in pretrained_features:
            net.add(layer)
        for layer in new_tail:
            net.add(layer)
else:
    net = pretrained_net
    net.output = nn.Dense(classes)
    net.output.initialize(init=mx.init.Xavier(), ctx=ctx)


# DEBUG
for index, param in enumerate(net.collect_params().values()):
    print(f"{param.name:>40}: {param.grad_req}")

# %%
#

net.collect_params().reset_ctx(ctx)

# %%
x = x.as_in_context(ctx[0])
net.summary(x)
net.hybridize()

# %%
# --

# scheduler = mx.lr_scheduler.FactorScheduler(base_lr=lr_base, factor=lr_factor, step=lr_epoch_steps * len(train_data))
# trainer = gluon.Trainer(net.collect_params(), "sgd",
#                          {"lr_scheduler": scheduler, "momentum": sgd_momentum, "wd": sgd_wd})
# trainer = gluon.Trainer(net.collect_params(), "adam", {"lr_scheduler": scheduler})
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
metric = mx.metric.Accuracy()


# %%
# -- Test function

def test(net_, val_data_, ctx_):
    _metric = mx.metric.Accuracy()
    val_loss = -1
    for i, _batch in enumerate(val_data_):
        _data = gluon.utils.split_and_load(_batch[0], ctx_list=ctx_, batch_axis=0, even_split=False)
        _label = gluon.utils.split_and_load(_batch[1], ctx_list=ctx_, batch_axis=0, even_split=False)
        output_list = [net_(X) for X in _data]
        loss_list = [loss_fn(yhat, y) for yhat, y in zip(output_list, _label)]
        val_loss = sum([loss.mean().asscalar() for loss in loss_list]) / len(loss_list)
        _metric.update(_label, output_list)
    _, val_acc = _metric.get()
    return val_acc, val_loss


# %%
# -- Plot history

def plot_history(data, model_name):
    plt.figure()
    plt.plot(data["train_acc"], label="train_acc")
    plt.plot(data["val_acc"], label="val_acc")
    plt.savefig(f"{model_name}.png")
    plt.close()

# %%
# -- Update history log

def update_history(H, chrono, epoch, train_acc, train_loss, val_acc, val_loss):
    H["epoch"].append(epoch)
    H["train_loss"].append(train_loss)
    H["train_acc"].append(train_acc)
    H["val_loss"].append(val_loss)
    H["val_acc"].append(val_acc)
    H["chrono"].append(chrono)
    return H

# %%
# -- Training Loop


def train(start_epoch, end_epoch, H):
    n_batches = len(train_data)
    prev_val_acc = -1
    print(f"*** Starting training loop...")
    print(f"--- Using: {ctx}")
    print(f"    num_batch: {n_batches}")
    epoch = start_epoch
    for epoch in range(start_epoch, end_epoch):
        # Init training loop
        tic = time.time()
        train_loss = 0
        metric.reset()
        # Start training loop
        for i, batch in enumerate(train_data):
            tic_batch = time.time()
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
            toc_batch = time.time()
            chrono_imgsec = len(batch[0]) / (toc_batch - tic_batch)
            print(
                f"[Epoch {epoch}], batch: {i:04d}, batch_loss: {batch_loss:0.5f}, lr: {trainer.learning_rate}, speed: {chrono_imgsec:0.2f} img/sec")
        # Get metrics
        train_loss /= n_batches
        _, train_acc = metric.get()
        val_acc, val_loss = test(net, val_data, ctx)
        toc = time.time()
        chrono = toc - tic
        # Updates history
        H = update_history(H, chrono, epoch, train_acc, train_loss, val_acc, val_loss)
        plot_history(H, model_name)
        # Save best model, based on val_acc
        if val_acc > prev_val_acc:
            net.export(model_name, epoch)
            prev_val_acc = val_acc
        # Save model and history every N epochs to be able to restart
        if epoch % 10 == 0:
            net.export(f"checkpoint_{model_name}", epoch)
            with open(f"{model_name}_hist.pkl", "wb") as fp:
                pkl.dump(H, fp)
        for k, v in H.items():
            print(f"{k:>20}: {float(v[-1]):0.5f}")
    # Save last epoch checkpoint
    net.export(f"checkpoint_{model_name}", epoch)

# %%
# -- Transfer learning

H = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "chrono": []}

scheduler = mx.lr_scheduler.FactorScheduler(base_lr=1e-3, factor=0.7, step=10 * len(train_data))
trainer = gluon.Trainer(net.collect_params(), "sgd", {"lr_scheduler": scheduler, "momentum": sgd_momentum, "wd": sgd_wd})

train(0, transfer_epochs, H)

# %%
# -- Finetune last N blocks of the network

pretrained_features = pretrained_net.features
# Allow update of weights for the last N blocks
for param in pretrained_features[24:].collect_params().values():
    param.grad_req = 'write'

# DEBUG
for index, param in enumerate(net.collect_params().values()):
    print(f"{param.name:>40}: {param.grad_req}")

# reset scheduler and trainer
scheduler = mx.lr_scheduler.FactorScheduler(base_lr=1e-5, factor=0.7, step=10 * len(train_data))
trainer = gluon.Trainer(net.collect_params(), "sgd", {"lr_scheduler": scheduler, "momentum": sgd_momentum, "wd": sgd_wd})

train(transfer_epochs, transfer_epochs + finetune_epochs, H)


# %%
# --
root_folder = "/home/gilbert/Development/git/mxnet-cookbook/src/mxnet_cookbook/transfer_learning"
epoch = 63
model_prefix = "minc_vgg16_202006110056"
symbol_file = os.path.join(root_folder, f"{model_prefix}-symbol.json")
param_file = os.path.join(root_folder, f"{model_prefix}-{epoch:04d}.params")
net = gluon.nn.SymbolBlock.imports(symbol_file, ["data"], param_file=param_file, ctx=ctx)

# %%
# --

epoch = 63
root_folder = "/home/gilbert/Development/git/mxnet-cookbook/src/mxnet_cookbook/transfer_learning"
model_prefix = "minc_vgg16_202006110056"
param_file = os.path.join(root_folder, f"{model_prefix}-{epoch:04d}.params")
net.load_parameters(param_file)
