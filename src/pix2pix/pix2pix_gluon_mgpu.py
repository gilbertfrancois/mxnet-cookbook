# Image-to-Image Translation with Conditional Adversarial Networks
# Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros
# https://arxiv.org/abs/1611.07004
#
# This file follows closely the Pixel2Pixel example from
# https://gluon.mxnet.io/chapter14_generative-adversarial-networks/pixel2pixel.html

import os
import tarfile
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import mxnet as mx
import mxnet.ndarray as nd
from mxboard import SummaryWriter
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn, utils
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import Conv2DTranspose
from mxnet.gluon.nn import Activation
from mxnet.gluon.nn import Dense
from mxnet.gluon.nn import Dropout
from mxnet.gluon.nn import Flatten
from mxnet.gluon.nn import MaxPool2D
from mxnet.gluon.nn import HybridSequential
from mxnet.gluon.nn import HybridBlock
from mxnet.gluon.nn import LeakyReLU
from datetime import datetime
import time
import logging
import gluoncv

# %%
# -- Settings

N_GPUS = 1
INPUT_SHAPE = (256, 256, 3)
EPOCHS = 100
BATCH_SIZE_PER_DEVICE = 10
BATCH_SIZE = BATCH_SIZE_PER_DEVICE * max(N_GPUS, 1)
LR = 0.0002
BETA1 = 0.5
LAMBDA1 = 100
POOL_SIZE = 50
DATASET = "facades"

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


def load_data(path, batch_size, input_shape, is_reversed=False):
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

    res = mx.io.NDArrayIter(data=[nd.concat(*img_in_list, dim=0),
                                  nd.concat(*img_out_list, dim=0)],
                            batch_size=batch_size)
    return res


# %%
# -- Load datasets

download_data(data_root_folder, DATASET)
train_data = load_data(train_image_path, BATCH_SIZE, INPUT_SHAPE, is_reversed=True)
val_data = load_data(val_image_path, BATCH_SIZE, INPUT_SHAPE, is_reversed=True)


# %%
# -- Visualize some training images


def visualize(img_arr):
    plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')


def preview_train_data(train_data):
    img_in_list, img_out_list = train_data.next().data
    for i in range(4):
        plt.subplot(2, 4, i + 1)
        visualize(img_in_list[i])
        plt.subplot(2, 4, i + 5)
        visualize(img_out_list[i])
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
        out = self.model(x)
        return out


# %%
# -- We use history image pool to help discriminator memorize history errors instead of just comparing current real
#    input and fake output.

class ImagePool():
    def __init__(self, pool_size):
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

    x_test = nd.zeros(shape=(1,3,256,256))
    print(netD(nd.concat(x_test, x_test, dim=1)))
    print(netG(x_test))

    netG.hybridize()
    netD.hybridize()

    netG.collect_params().reset_ctx(mx_ctx)
    netD.collect_params().reset_ctx(mx_ctx)

    trainerG = gluon.Trainer(netG.collect_params(), "adam", {"learning_rate": LR, "beta1": BETA1})
    trainerD = gluon.Trainer(netD.collect_params(), "adam", {"learning_rate": LR, "beta1": BETA1})

    return netG, netD, trainerG, trainerD


netG, netD, trainerG, trainerD = set_network(LR, BETA1)

# %%
# -- Loss


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
    print("--- facc")
    print(type(label), type(pred))
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()

# %%


def train():
    image_pool = ImagePool(POOL_SIZE)
    # metric = mx.metric.CustomMetric(facc)

    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    logging.basicConfig(level=logging.DEBUG)

    for epoch in range(EPOCHS):
        tic = time.time()
        train_data.reset()
        iter = 0
        for batch in train_data:
            btic = time.time()
            ############################
            # (1) Update D network: maximize log(D(x, y)) + log(1 - D(x, G(x, z)))
            ###########################
            # real_in = batch.data[0].as_in_context(mx_ctx[0])
            # real_out = batch.data[1].as_in_context(mx_ctx[0])
            real_in_list = gluoncv.utils.split_and_load(batch.data[0], ctx_list=mx_ctx, batch_axis=0)
            real_out_list = gluoncv.utils.split_and_load(batch.data[1], ctx_list=mx_ctx, batch_axis=0)

            fake_out_list = [netG(real_in) for real_in in real_in_list]
            fake_concat_list = [image_pool.query(nd.concat(real_in, fake_out, dim=1)) for real_in, fake_out in zip(real_in_list, fake_out_list)]
            with autograd.record():
                # Train with fake image
                # Use image pooling to utilize history images
                output_fake_list = [netD(fake_concat) for fake_concat in fake_concat_list]
                fake_label_list = [nd.zeros(output.shape, ctx=ctx) for ctx, output in zip(mx_ctx, output_fake_list)]
                errD_fake_list = [GAN_loss(output, fake_label) for output, fake_label in zip(output_fake_list, fake_label_list)]
                # metric.update(fake_label_list, output_fake_list)
                # Train with real image
                real_concat_list = [nd.concat(real_in, real_out, dim=1) for real_in, real_out in zip(real_in_list, real_out_list)]
                output_real_list = [netD(real_concat) for real_concat in real_concat_list]
                real_label_list = [nd.ones(output.shape, ctx=ctx) for output, ctx in zip(output_real_list, mx_ctx)]
                errD_real_list = [GAN_loss(output, real_label) for output, real_label in zip(output_real_list, real_label_list)]
                errD_list = []
                for errD_real, errD_fake in zip(errD_real_list, errD_fake_list):
                    errD = (errD_real + errD_fake) * 0.5
                    errD.backward()
                    errD_list.append(errD)
                # metric.update(real_label_list, output_real_list)

            trainerD.step(batch.data[0].shape[0])

            ############################
            # (2) Update G network: maximize log(D(x, G(x, z))) - lambda1 * L1(y, G(x, z))
            ###########################
            with autograd.record():
                fake_out_list = [netG(real_in) for real_in in real_in_list]
                fake_concat_list = [nd.concat(real_in, fake_out, dim=1) for real_in, fake_out in zip(real_in_list, fake_out_list)]

                output_list = [netD(fake_concat) for fake_concat in fake_concat_list]
                real_label_list = [nd.ones(output.shape, ctx=ctx) for output, ctx in zip(output_list, mx_ctx)]
                errG_list = [GAN_loss(output, real_label) + L1_loss(real_out, fake_out) * LAMBDA1 for output, real_label, real_out, fake_out in zip(output_list, real_label_list, real_out_list, fake_out_list)]
                for errG in errG_list:
                    errG.backward()

            trainerG.step(batch.data[0].shape[0])

            # Print log infomation every ten batches
            if iter % 10 == 0:
                # @todo: compute loss over all devices, not just the first device
                # name, acc = metric.get()
                logging.info('speed: {} samples/s'.format(BATCH_SIZE / (time.time() - btic)))
                logging.info('discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d'
                         %(nd.mean(errD_list[0]).asscalar(),
                           nd.mean(errG_list[0]).asscalar(), 1, iter, epoch))
            iter = iter + 1
            btic = time.time()

        # name, acc = metric.get()
        # metric.reset()
        # logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
        logging.info('time: %f' % (time.time() - tic))

        # Visualize one generated image for each epoch
        fake_img = fake_out_list[0][0]
        visualize(fake_img)
        plt.show()
        netG.save_parameters(f"{timestamp}_net_g.params")
        netD.save_parameters(f"{timestamp}_net_d.params")

train()

