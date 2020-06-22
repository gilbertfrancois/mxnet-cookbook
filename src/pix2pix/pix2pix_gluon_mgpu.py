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
#
# Image-to-Image Translation with Conditional Adversarial Networks
# Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros
# https://arxiv.org/abs/1611.07004
#
# This script is based on the Pixel2Pixel example from
# https://gluon.mxnet.io/chapter14_generative-adversarial-networks/pixel2pixel.html
#
# New features are:
# - Multiple GPU support
# - New feature based loss function resulting in better and cleaner generated images.

import logging
import os
import tarfile
import time
from datetime import datetime

import gluoncv
import matplotlib.pyplot as plt
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon import utils
from mxnet.gluon.nn import Activation
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import Conv2DTranspose
from mxnet.gluon.nn import Dropout
from mxnet.gluon.nn import HybridBlock
from mxnet.gluon.nn import HybridSequential
from mxnet.gluon.nn import LeakyReLU

# %%
# -- Settings

N_GPUS = 2
INPUT_SHAPE = (256, 256, 3)
EPOCHS = 101
BATCH_SIZE_PER_DEVICE = 10
BATCH_SIZE = BATCH_SIZE_PER_DEVICE * max(N_GPUS, 1)
LR = 0.0002 * N_GPUS
BETA1 = 0.5
BETA2 = 0.9
LAMBDA1 = BETA2 * 100
LAMBDA2 = (1-BETA2) * 1e-4
POOL_SIZE = 50 # // N_GPUS
DATASET = "facades"
DEBUG = False
LOSS_FEATURE_EXTRACTOR = "VGG19"
# LOSS_FEATURE_EXTRACTOR_LAYERS = [2, 7, 12, 19, 26]   # VGG16
LOSS_FEATURE_EXTRACTOR_LAYERS = [0, 5, 10]
# LOSS_FEATURE_EXTRACTOR_LAYERS = [25]
# LOSS_FEATURE_EXTRACTOR_LAYERS = [0, 5, 10, 19, 28]   # VGG19

mx_ctx = [mx.gpu(i) for i in range(N_GPUS)] if N_GPUS > 0 else [mx.cpu()]

data_root_folder = os.path.expanduser("~/Development/git/mxnet-cookbook/data")
train_image_path = os.path.join(data_root_folder, DATASET, "train")
val_image_path = os.path.join(data_root_folder, DATASET, "val")


# %%
# --

def download_data(root_folder, dataset):
    data_folder = os.path.join(root_folder, dataset)
    if os.path.exists(data_folder):
        return
    url = f"https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/{dataset}.tar.gz"
    os.makedirs(data_folder, exist_ok=True)
    data_file = utils.download(url)
    with tarfile.open(data_file) as tar:
        tar.extractall(path=data_folder)
    os.remove(data_file)


def load_data(path, batch_size, input_shape, is_reversed=False, shuffle=False):
    img_in_list = []
    img_out_list = []
    for path, _, fnames in os.walk(path):
        for fname in fnames:
            if not fname.endswith(".jpg"):
                continue
            img = os.path.join(path, fname)
            img_arr = mx.image.imread(img).astype(np.float32) / 127.5 - 1
            img_arr = mx.image.imresize(img_arr, input_shape[1] * 2, input_shape[0])
            # Crop input and output images
            img_arr_in = mx.image.fixed_crop(img_arr, 0, 0, input_shape[1], input_shape[0])
            img_arr_in = nd.transpose(img_arr_in, (2, 0, 1))
            img_arr_in = img_arr_in.reshape((1,) + img_arr_in.shape)
            img_arr_out = mx.image.fixed_crop(img_arr, input_shape[1], 0, input_shape[1], input_shape[0])
            img_arr_out = nd.transpose(img_arr_out, (2, 0, 1))
            img_arr_out = img_arr_out.reshape((1,) + img_arr_out.shape)
            img_in_list.append(img_arr_out if is_reversed else img_arr_in)
            img_out_list.append(img_arr_in if is_reversed else img_arr_out)

    img_in_arr = nd.concat(*img_in_list, dim=0)
    img_out_arr = nd.concat(*img_out_list, dim=0)

    return mx.io.NDArrayIter(data=[img_in_arr, img_out_arr], batch_size=batch_size, shuffle=shuffle)


# %%
# -- Load datasets

download_data(data_root_folder, DATASET)
train_data = load_data(train_image_path, BATCH_SIZE, INPUT_SHAPE, is_reversed=True, shuffle=True)
val_data = load_data(val_image_path, BATCH_SIZE, INPUT_SHAPE, is_reversed=True)


# %%
# -- Visualize some training images

def denormalize(img_arr):
    return ((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)


def visualize(img_arr):
    plt.imshow(denormalize(img_arr))
    plt.axis('off')


def preview_train_data(train_data):
    img_in_list, img_out_list = train_data.next().data
    for i in range(4):
        plt.subplot(2, 4, i + 1)
        visualize(img_in_list[i])
        plt.subplot(2, 4, i + 5)
        visualize(img_out_list[i])
    plt.show()


def preview_images(real_in_arr, real_out_arr, fake_out_arr, n_cols=4, title=""):
    fig, axs = plt.subplots(3, n_cols)
    for i in range(n_cols):
        axs[0][i].imshow(denormalize(real_in_arr[i]))
        axs[0][i].axis("off")
        axs[1][i].imshow(denormalize(real_out_arr[i]))
        axs[1][i].axis("off")
        axs[2][i].imshow(denormalize(fake_out_arr[i]))
        axs[2][i].axis("off")
    plt.suptitle(title)
    plt.show()


preview_train_data(train_data)


# %%
# -- Define the Generator (U-Net)

class UnetSkipUnit(HybridBlock):
    def __init__(self,
                 inner_channels,
                 outer_channels,
                 inner_block=None,
                 innermost=False,
                 outermost=False,
                 use_dropout=False,
                 use_bias=False):
        super(UnetSkipUnit, self).__init__()

        with self.name_scope():
            self.outermost = outermost
            en_conv = Conv2D(channels=inner_channels,
                             kernel_size=4,
                             strides=2,
                             padding=1,
                             in_channels=outer_channels,
                             use_bias=use_bias)
            en_relu = LeakyReLU(alpha=0.2)
            en_norm = BatchNorm(momentum=0.1, in_channels=inner_channels)
            de_relu = Activation(activation="relu")
            de_norm = BatchNorm(momentum=0.1, in_channels=outer_channels)

            if innermost:
                de_conv = Conv2DTranspose(channels=outer_channels,
                                          kernel_size=4,
                                          strides=2,
                                          padding=1,
                                          in_channels=outer_channels,
                                          use_bias=use_bias)
                encoder = [en_relu, en_conv]
                decoder = [de_relu, de_conv, de_norm]
                model = encoder + decoder
            elif outermost:
                de_conv = Conv2DTranspose(channels=outer_channels,
                                          kernel_size=4,
                                          strides=2,
                                          padding=1,
                                          in_channels=inner_channels * 2)
                encoder = [en_conv]
                decoder = [de_relu, de_conv, Activation(activation="tanh")]
                model = encoder + [inner_block] + decoder
            else:
                de_conv = Conv2DTranspose(channels=outer_channels,
                                          kernel_size=4,
                                          strides=2,
                                          padding=1,
                                          in_channels=inner_channels * 2,
                                          use_bias=use_bias)
                encoder = [en_relu, en_conv, en_norm]
                decoder = [de_relu, de_conv, de_norm]
                model = encoder + [inner_block] + decoder
            if use_dropout:
                model += [Dropout(rate=0.5)]

            self.model = HybridSequential()
            with self.model.name_scope():
                for block in model:
                    self.model.add(block)

    def hybrid_forward(self, F, x):
        if self.outermost:
            return self.model(x)
        else:
            return F.concat(self.model(x), x, dim=1)


class UnetGenerator(HybridBlock):
    def __init__(self, in_channels, num_downs, ngf=64, use_dropout=True):
        super().__init__()
        unet = UnetSkipUnit(ngf * 8, ngf * 8, innermost=True)
        for _ in range(num_downs - 5):
            unet = UnetSkipUnit(ngf * 8, ngf * 8, unet, use_dropout=use_dropout)
        unet = UnetSkipUnit(ngf * 8, ngf * 4, unet, use_dropout=use_dropout)
        unet = UnetSkipUnit(ngf * 4, ngf * 2, unet, use_dropout=use_dropout)
        unet = UnetSkipUnit(ngf * 2, ngf * 1, unet, use_dropout=use_dropout)
        unet = UnetSkipUnit(ngf, in_channels, unet, outermost=True)

        with self.name_scope():
            self.model = unet

    def hybrid_forward(self, F, x):
        return self.model(x)


# %%
# -- Define the discriminator

class Discriminator(HybridBlock):
    def __init__(self, in_channels, ndf=64, n_layers=3, use_sigmoid=False, use_bias=False):
        super(Discriminator, self).__init__()

        with self.name_scope():
            self.model = HybridSequential()
            kernel_size = 4
            padding = int(np.ceil((kernel_size - 1) / 2))
            self.model.add(Conv2D(channels=ndf, kernel_size=kernel_size, strides=2,
                                  padding=padding, in_channels=in_channels))
            self.model.add(LeakyReLU(alpha=0.2))

            nf_mult = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                self.model.add(Conv2D(channels=ndf * nf_mult, kernel_size=kernel_size, strides=2,
                                      padding=padding, in_channels=ndf * nf_mult_prev,
                                      use_bias=use_bias))
                self.model.add(BatchNorm(momentum=0.1, in_channels=ndf * nf_mult))
                self.model.add(LeakyReLU(alpha=0.2))

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 8)
            self.model.add(Conv2D(channels=ndf * nf_mult, kernel_size=kernel_size, strides=1,
                                  padding=padding, in_channels=ndf * nf_mult_prev,
                                  use_bias=use_bias))
            self.model.add(BatchNorm(momentum=0.1, in_channels=ndf * nf_mult))
            self.model.add(LeakyReLU(alpha=0.2))
            self.model.add(Conv2D(channels=1, kernel_size=kernel_size, strides=1,
                                  padding=padding, in_channels=ndf * nf_mult))
            if use_sigmoid:
                self.model.add(Activation(activation='sigmoid'))

    def hybrid_forward(self, F, x):
        return self.model(x)


# %%
# -- We use history image pool to help discriminator memorize history errors instead of just comparing current real
#    input and fake output.

class ImagePool():
    def __init__(self, pool_size, ctx):
        self.ctx = ctx
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.n_images = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        ret_images = []
        for i in range(images.shape[0]):
            image = nd.expand_dims(images[i], axis=0)
            if self.n_images < self.pool_size:
                self.n_images += 1
                self.images.append(image)
                ret_images.append(image)
            else:
                p = nd.random_uniform(0, 1, shape=(1,)).asscalar()
                if p > 0.5:
                    random_id = nd.random_uniform(0, self.pool_size - 1, shape=(1,)).astype(np.uint8).asscalar()
                    tmp = self.images[random_id].copy()
                    self.images[random_id] = image
                    ret_images.append(tmp)
                else:
                    ret_images.append(image)
        ret_images = nd.concat(*ret_images, dim=0)
        return ret_images


# %%
# -- Parameter initialisation


def param_init(param):
    if param.name.find("conv") != -1:
        if param.name.find("weight") != -1:
            param.initialize(init=mx.init.Normal(0.02))
        else:
            param.initialize(init=mx.init.Zero())
    elif param.name.find("batchnorm") != -1:
        param.initialize(init=mx.init.Zero())
        if param.name.find("gamma") != -1:
            shape = param.data().shape
            rn = nd.random.normal(1, 0.02, shape)
            param.set_data(rn)


def network_init(net):
    for param in net.collect_params().values():
        param_init(param)


# %%
# -- Create networks


def set_network(lr, beta1):
    # Pixel2Pixel network
    netG = UnetGenerator(in_channels=3, num_downs=8)
    netD = Discriminator(in_channels=6)

    # Initialize parameters
    network_init(netG)
    network_init(netD)

    x = nd.zeros(shape=(1, 3, 256, 256))

    netG(x)
    netD(nd.concat(x, x, dim=1))

    netG.hybridize()
    netD.hybridize()

    netG.collect_params().reset_ctx(mx_ctx)
    netD.collect_params().reset_ctx(mx_ctx)

    trainerG = gluon.Trainer(netG.collect_params(), "adam", {"learning_rate": LR, "beta1": BETA1})
    trainerD = gluon.Trainer(netD.collect_params(), "adam", {"learning_rate": LR, "beta1": BETA1})

    return netG, netD, trainerG, trainerD


netG, netD, trainerG, trainerD = set_network(LR, BETA1)

# %%
# -- VGG16 as loss function

feature_extractor = gluoncv.model_zoo.get_model(LOSS_FEATURE_EXTRACTOR, pretrained=True)
if LOSS_FEATURE_EXTRACTOR == "VGG19":
    feature_extractor = feature_extractor.features[:37]
elif LOSS_FEATURE_EXTRACTOR == "VGG16":
    feature_extractor = feature_extractor.features[:31]
else:
    feature_extractor = feature_extractor.features
feature_extractor.collect_params().reset_ctx(mx_ctx)
feature_extractor.hybridize()


# %%
# -- Loss

def loss_fe_normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    assert x.ndim == 4
    assert x.shape[1] == 3
    # Convert from [-1..1] to range [0..1]
    xn = (x + 1) * 0.5
    # Convert to normalization for pretrained imagenet
    xn = nd.transpose(xn, axes=(0, 2, 3, 1))
    xn = (xn - nd.array(mean, ctx=x.context, dtype=x.dtype)) / nd.array(std, ctx=x.context, dtype=x.dtype)
    xn = nd.transpose(xn, axes=(0, 3, 1, 2))
    return xn


def loss_fe_forward(x):
    xn = loss_fe_normalize(x)
    y = []
    for i in range(len(feature_extractor)):
        xn = feature_extractor[i](xn)
        if i in LOSS_FEATURE_EXTRACTOR_LAYERS:
            y.append(x)
    return y


def loss_fe_fn(data, label):
    y_data_list = loss_fe_forward(data)
    y_label_list = loss_fe_forward(label)
    loss = nd.zeros(shape=data.shape[0], ctx=data.context, dtype=data.dtype)
    for i in range(len(y_data_list)):
        loss = loss + nd.sum(nd.square(y_data_list[i] - y_label_list[i]), axis=[1, 2, 3])
    return loss


GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
L1_loss = gluon.loss.L1Loss()


# %%


def facc(label, pred):
    # if isinstance(label, list):
    #     label = [l.as_in_context(mx.cpu()) for l in label]
    #     label = nd.concat(*label, dim=0).asnumpy()
    # if isinstance(pred, list):
    #     pred = [p.as_in_context(mx.cpu()) for p in pred]
    #     pred = nd.concat(*pred, dim=0).asnumpy()
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


# %%
def validate(data_iterator, n_cols, title):
    data_iterator.reset()
    batch = data_iterator.next()
    real_in = batch.data[0]
    real_out = batch.data[1]
    fake_out = netG(real_in.as_in_context(mx_ctx[0]))
    preview_images(real_in, real_out, fake_out, n_cols, title)


def train():
    image_pool_list = [ImagePool(POOL_SIZE, ctx) for ctx in mx_ctx]
    metric = mx.metric.CustomMetric(facc)

    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    logging.basicConfig(level=logging.DEBUG)

    for epoch in range(EPOCHS):
        tic = time.time()
        train_data.reset()
        metric.reset()
        iter = 0
        for batch in train_data:
            btic = time.time()
            ############################
            # (1) Update D network: maximize log(D(x, y)) + log(1 - D(x, G(x, z)))
            ###########################
            real_in_list = gluoncv.utils.split_and_load(batch.data[0], ctx_list=mx_ctx, batch_axis=0, even_split=False)
            real_out_list = gluoncv.utils.split_and_load(batch.data[1], ctx_list=mx_ctx, batch_axis=0, even_split=False)
            fake_out_list = [netG(real_in) for real_in in real_in_list]
            if DEBUG:
                for i in range(len(mx_ctx)):
                    preview_images(real_in_list[i], real_out_list[i], fake_out_list[i], n_cols=4,
                                   title=f"Update Dis {i}")

            fake_concat_list = []
            for i, ctx in enumerate(mx_ctx):
                fake_concat = image_pool_list[i].query(nd.concat(real_in_list[i], fake_out_list[i], dim=1))
                fake_concat_list.append(fake_concat)
            with autograd.record():

                # Train with fake image
                output_fake_list = [netD(fake_concat) for fake_concat in fake_concat_list]
                fake_label_list = [nd.zeros(output.shape, ctx=output.context) for output in output_fake_list]
                errD_fake_list = [GAN_loss(output, fake_label) for output, fake_label in
                                  zip(output_fake_list, fake_label_list)]
                metric.update(fake_label_list, output_fake_list)

                # Train with real image
                real_concat_list = [nd.concat(real_in, real_out, dim=1) for real_in, real_out in
                                    zip(real_in_list, real_out_list)]
                output_real_list = [netD(real_concat) for real_concat in real_concat_list]
                real_label_list = [nd.ones(output.shape, ctx=ctx) for output, ctx in zip(output_real_list, mx_ctx)]
                errD_real_list = [GAN_loss(output, real_label) for output, real_label in
                                  zip(output_real_list, real_label_list)]

                # compute combined loss
                errD_list = []
                for errD_real, errD_fake in zip(errD_real_list, errD_fake_list):
                    errD = (errD_real + errD_fake) * 0.5
                    errD.backward()
                    errD_list.append(errD)
                metric.update(real_label_list, output_real_list)

            trainerD.step(batch.data[0].shape[0])

            ############################
            # (2) Update G network: maximize log(D(x, G(x, z))) - lambda1 * L1(y, G(x, z))
            ###########################
            with autograd.record():
                fake_out_list = [netG(real_in) for real_in in real_in_list]
                fake_concat_list = [nd.concat(real_in, fake_out, dim=1) for real_in, fake_out in
                                    zip(real_in_list, fake_out_list)]
                if DEBUG:
                    for i in range(len(mx_ctx)):
                        preview_images(real_in_list[i], real_out_list[i], fake_out_list[i], n_cols=4,
                                       title=f"Update Gen {i}")
                output_list = [netD(fake_concat) for fake_concat in fake_concat_list]
                real_label_list = [nd.ones(output.shape, ctx=ctx) for output, ctx in zip(output_list, mx_ctx)]
                errG_list = []
                for output, real_label, real_out, fake_out in zip(output_list, real_label_list, real_out_list,
                                                                  fake_out_list):
                    loss1 = GAN_loss(output, real_label)
                    loss2 = L1_loss(real_out, fake_out)
                    loss3 = loss_fe_fn(real_out, fake_out)
                    errG_list.append(loss1 + LAMBDA1 * loss2 + LAMBDA2 * loss3)
                    # errG_list.append(loss1 + LAMBDA1 * loss2)
                for errG in errG_list:
                    errG.backward()

            trainerG.step(batch.data[0].shape[0])
            btoc = time.time()
            # Print log infomation every ten batches
            if iter % 10 == 0:
                name, acc = metric.get()
                errD_mean = sum([errD.mean().asscalar() for errD in errD_list]) / len(errD_list)
                errG_mean = sum([errG.mean().asscalar() for errG in errG_list]) / len(errG_list)
                msg = [
                    f"epoch: {epoch}",
                    f"iter: {iter}",
                    f"d_loss: {errD_mean:0.5f}",
                    f"g_loss: {errG_mean:0.5f}",
                    f"acc: {acc:0.2f}",
                    f"speed: {(BATCH_SIZE / (btoc - btic)):0.2f} samples/s"
                ]
                logging.info(" ".join(msg))
            iter = iter + 1

        toc = time.time()
        name, acc = metric.get()
        logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
        logging.info('time: %f' % (toc - tic))

        # Visualize one generated image for each epoch
        if epoch % 10 == 0:
            validate(train_data, 4, f"{timestamp} train_data epoch: {epoch}")
            validate(val_data, 4, f"{timestamp} val_data epoch: {epoch}")
            netG.save_parameters(f"{timestamp}_{DATASET}_net_g.params")
            netD.save_parameters(f"{timestamp}_{DATASET}_net_d.params")


train()
