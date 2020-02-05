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

train_image_path = f"./data/train/{dataset}"
val_image_path = f"./data/val/{dataset}"


def download_data(dataset):
    if os.path.exists(dataset):
        return
    url = f"https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/{dataset}.tar.gz"
    os.makedirs(dataset, exist_ok=True)
    data_file = utils.download(url)
    with tarfile.open(data_file) as tar:
        tar.extractall(path=".")
    os.remove(data_file)


def load_data(path, batch_size, is_reversed=False):
    img_in_list = []
    img_out_list = []
    for path, _, fnames in os.walk(path):
        for fname in fnames:
            if not fname.endswith(".jpg"):
                continue
            img = os.path.join(path, fname)
            img_arr  = mx.image.imread(img).astype(np.float32) / 127.5 - 1
            img_arr = mx.image.imresize(img_arr, INPUT_SHAPE[1] * 2, INPUT_SHAPE[0])
