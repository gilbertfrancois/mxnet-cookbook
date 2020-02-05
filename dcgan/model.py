import mxnet as mx
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import Conv2DTranspose
from mxnet.gluon.nn import Activation
from mxnet.gluon.nn import Dense
from mxnet.gluon.nn import Dropout
from mxnet.gluon.nn import Flatten
from mxnet.gluon.nn import MaxPool2D
from mxnet.gluon.nn import HybridSequential
from mxnet.gluon.nn import LeakyReLU


def build_generator(n_filters, n_channels, mx_ctx):
    netG = HybridSequential()
    with netG.name_scope():
        # Input is Z
        netG.add(Conv2DTranspose(n_filters * 8, kernel_size=4, strides=1, padding=0, use_bias=False))
        netG.add(BatchNorm())
        netG.add(Activation("relu"))

        netG.add(Conv2DTranspose(n_filters * 4, kernel_size=4, strides=2, padding=1, use_bias=False))
        netG.add(BatchNorm())
        netG.add(Activation("relu"))

        netG.add(Conv2DTranspose(n_filters * 2, kernel_size=4, strides=2, padding=1, use_bias=False))
        netG.add(BatchNorm())
        netG.add(Activation("relu"))

        netG.add(Conv2DTranspose(n_filters, kernel_size=4, strides=2, padding=1, use_bias=False))
        netG.add(BatchNorm())
        netG.add(Activation("relu"))

        netG.add(Conv2DTranspose(n_channels, kernel_size=4, strides=2, padding=1, use_bias=False))
        netG.add(BatchNorm())
        netG.add(Activation("tanh"))

    netG.initialize(mx.init.Normal(0.02), ctx=mx_ctx)
    netG.hybridize()
    return netG


def build_discriminator(n_filters, n_channels, mx_ctx):
    netD = HybridSequential()
    with netD.name_scope():
        # Input is n_channels * 64 * 64
        netD.add(Conv2D(n_filters, kernel_size=4, strides=2, padding=1, use_bias=False))
        netD.add(LeakyReLU(0.2))

        netD.add(Conv2D(n_filters * 2, kernel_size=4, strides=2, padding=1, use_bias=False))
        netD.add(BatchNorm())
        netD.add(LeakyReLU(0.2))

        netD.add(Conv2D(n_filters * 4, kernel_size=4, strides=2, padding=1, use_bias=False))
        netD.add(BatchNorm())
        netD.add(LeakyReLU(0.2))

        netD.add(Conv2D(n_filters * 8, kernel_size=4, strides=2, padding=1, use_bias=False))
        netD.add(BatchNorm())
        netD.add(LeakyReLU(0.2))

        netD.add(Conv2D(1, 4, 1, 0, use_bias=False))

    netD.initialize(mx.init.Normal(0.02), ctx=mx_ctx)
    netD.hybridize()
