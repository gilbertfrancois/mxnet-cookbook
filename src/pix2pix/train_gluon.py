import os
import tarfile
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import mxnet as mx
import mxnet.ndarray as nd
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


def download_data(dataset):
    if os.path.exists(dataset):
        return
    url = f"https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/{dataset}.tar.gz"
    os.makedirs(dataset, exist_ok=True)
    data_file = utils.download(url)
    with tarfile.open(data_file) as tar:
        tar.extractall(path=".")
    os.remove(data_file)


def load_data(path, batch_size, INPUT_SHAPE, is_reversed=False):
    img_in_list = []
    img_out_list = []
    for path, _, fnames in os.walk(path):
        for fname in fnames:
            if not fname.endswith(".jpg"):
                continue
            img = os.path.join(path, fname)
            img_arr = mx.image.imread(img).astype(np.float32) / 127.5 - 1
            img_arr = mx.image.imresize(img_arr, INPUT_SHAPE[1] * 2, INPUT_SHAPE[0])
            # Crop input and output images
            img_arr_in = mx.image.fixed_crop(img_arr, 0, 0, INPUT_SHAPE[1], INPUT_SHAPE[0])
            img_arr_in = nd.transpose(img_arr_in, (2, 0, 1))
            img_arr_in = img_arr_in.reshape((1,) + img_arr_in.shape)
            img_arr_out = mx.image.fixed_crop(img_arr, INPUT_SHAPE[1], 0, INPUT_SHAPE[1], INPUT_SHAPE[0])
            img_arr_out = nd.transpose(img_arr_out, (2, 0, 1))
            img_arr_out = img_arr_out.reshape((1,) + img_arr_out.shape)
            img_in_list.append(img_arr_out if is_reversed else img_arr_in)
            img_out_list.append(img_arr_in if is_reversed else img_arr_out)

    res = mx.io.NDArrayIter(data=[nd.concat(*img_in_list, dim=0),
                                  nd.concat(*img_out_list, dim=0)],
                            batch_size=batch_size)
    return res


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


def main():
    # Set the compute context, GPU is available otherwise CPU
    mx_ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

    # %%
    # --  settings
    INPUT_SHAPE = (256, 256, 3)
    EPOCHS = 100
    BATCH_SIZE = 10
    LR = 0.0002
    BETA1 = 0.5
    LAMBDA1 = 100
    POOL_SIZE = 50

    dataset = "facades"

    train_image_path = f"./{dataset}/train"
    val_image_path = f"./{dataset}/val"

    download_data(dataset)
    train_data = load_data(train_image_path, BATCH_SIZE, INPUT_SHAPE, is_reversed=True)
    val_data = load_data(val_image_path, BATCH_SIZE, INPUT_SHAPE, is_reversed=True)

    preview_train_data(train_data)

    netG = UnetGenerator(in_channels=3, num_downs=8)


if __name__ == '__main__':
    main()
