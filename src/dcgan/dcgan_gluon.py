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
import datetime

import os
import tarfile
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon, init, autograd
from mxnet.gluon import utils
from mxnet.gluon.data.vision import transforms
import matplotlib.pyplot as plt
import model

# Set logging to see some output during training
logging.getLogger().setLevel(logging.DEBUG)

mx.random.seed(42)

# Set the compute context, GPU is available otherwise CPU
mx_ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

# %%
# -- Constants

EPOCHS = 1500
BATCH_SIZE = 32
INPUT_SHAPE = (64, 64, 3)
LATENT_Z_SIZE = 100
LEARNING_RATE = 0.0001
RESUME_SESSION = None  # 202001301354
BETA1 = 0.5


# %%
# -- Helper functions

def transform(data, shape):
    data = mx.image.imresize(data, shape[1], shape[0])
    data = nd.transpose(data, (2, 0, 1))
    data = data.astype(np.float32) / 127.5 - 1.0
    if data.shape[0] == 1:
        data = nd.tile(data, (3, 1, 1))
    return data.reshape((1, ) + data.shape)


def visualize(img_arr):
    plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis("off")


def save_samples(output_folder, epoch):
    num_images = 16
    for i in range(num_images):
        latent_z = mx.nd.random_normal(0, 1, shape=(1, LATENT_Z_SIZE, 1, 1), ctx=mx_ctx)
        img = netG(latent_z)
        plt.subplot(4, 4, i + 1)
        visualize(img[0])
    plt.savefig(f"{output_folder}/result{epoch:09d}.png")


# %%
# -- Get some data to train on

lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
data_path = os.path.join("data", "lfw_dataset")
os.makedirs(data_path, exist_ok=True)
data_file = utils.download(lfw_url)
with tarfile.open(data_file) as tar:
    tar.extractall(path=data_path)
img_list = []
for path, _, fnames in os.walk(data_path):
    for fname in fnames:
        if not fname.endswith(".jpg"):
            continue
        img_path = os.path.join(path, fname)
        img_arr = mx.image.imread(img_path)
        img_arr = transform(img_arr, INPUT_SHAPE)
        img_list.append(img_arr)

# %%
# -- Create session folder

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
output_folder = f"output/{timestamp}"
os.makedirs(output_folder, exist_ok=True)

# %%
# -- Create a data iterator that feeds batches

train_data = mx.io.NDArrayIter(data=nd.concatenate(img_list), batch_size=BATCH_SIZE, shuffle=False)


# %%
# -- Define the model

n_channels = INPUT_SHAPE[2]
n_filters = 64

netG = model.build_generator(n_filters, n_channels, mx_ctx)
netD = model.build_discriminator(n_filters, n_channels, mx_ctx)

if RESUME_SESSION is not None:
    netG.load_parameters(f"output/{RESUME_SESSION}/_net_g.params", ctx=mx_ctx)
    netD.load_parameters(f"output/{RESUME_SESSION}/_net_d.params", ctx=mx_ctx)

# %%
# -- Define loss function and optimizer 
loss_fn = gluon.loss.SigmoidBinaryCrossEntropyLoss() 
trainerG = gluon.Trainer(netG.collect_params(), "adam", {"learning_rate": LEARNING_RATE, "beta1": BETA1}) 
trainerD = gluon.Trainer(netD.collect_params(), "adam", {"learning_rate": LEARNING_RATE, "beta1": BETA1})


# %%
# -- Custom metric function

def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


metric = mx.metric.CustomMetric(facc)

# %%
# -- Train

real_label = nd.ones((BATCH_SIZE, ), ctx=mx_ctx)
fake_label = nd.zeros((BATCH_SIZE, ), ctx=mx_ctx)

for epoch in range(EPOCHS):
    tic = time.time()
    train_data.reset()
    step = 0
    for batch in train_data:
        tic_batch = time.time()
        # Update discriminator:
        # maximize log(D(x)) + log(1 - D(G(z))
        data = batch.data[0].as_in_context(mx_ctx)
        latent_z = mx.nd.random_normal(0, 1, shape=(BATCH_SIZE, LATENT_Z_SIZE, 1, 1), ctx=mx_ctx)
        with autograd.record():
            # Train discriminator with a real image
            output = netD(data).reshape((-1, 1))
            errD_real = loss_fn(output, real_label)
            metric.update([real_label, ], [output, ])
            # Train discriminator with a fake image
            fake = netG(latent_z).detach()
            output = netD(fake).reshape((-1, 1))
            errD_fake = loss_fn(output, fake_label)
            errD = errD_real + errD_fake
            errD.backward()
            metric.update([fake_label, ], [output, ])
        trainerD.step(batch.data[0].shape[0])
        # Update generator (run twice to prevent loss_D going to zero):
        # maximize log(D(G(z))
        with autograd.record():
            fake = netG(latent_z)
            output = netD(fake).reshape((-1, 1))
            errG = loss_fn(output, real_label)
            errG.backward()
        trainerG.step(batch.data[0].shape[0])
        # print log information every 10 batches
        chrono_batch = time.time() - tic_batch
        if step % 10 == 0:
            name, acc = metric.get()
            mean_errG = nd.mean(errG).asscalar()
            mean_errD = nd.mean(errD).asscalar()
            logging.info(f"speed: {BATCH_SIZE / chrono_batch} samples/s")
            logging.info(f"loss_D = {mean_errD}, loss_G = {mean_errG}, acc = {acc:0.4f} at step {step} epoch {epoch}, lr = {LEARNING_RATE}")
        step += 1

    save_samples(output_folder, epoch)
    netD.export(os.path.join(output_folder, f"_net_d"), epoch=epoch)
    netG.export(os.path.join(output_folder, f"_net_g"), epoch=epoch)
    name, acc = metric.get()
    metric.reset()

    # %%
    # -- Lower the learning rate after every N epochs

    if epoch % 50 == 0 and epoch > 10:
        LEARNING_RATE = LEARNING_RATE / 3
