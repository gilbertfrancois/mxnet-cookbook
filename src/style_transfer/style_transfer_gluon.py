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

# Based on the explanation in the book:
# "Dive into deep learning", A. Zhang, Z.C. Lipton, M. Li, A.J. Smola

import os
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet import image
from mxnet import init
from mxnet import np, npx
from mxnet.gluon import nn
import matplotlib.pyplot as plt

# %%
# -- Settings

npx.set_np()

IMAGE_SIZE = (225, 150)
N_EPOCHS = 2000
RGB_MEAN = np.array([0.485, 0.456, 0.406])
RGB_STD = np.array([0.229, 0.224, 0.225])
STYLE_LAYERS = [0, 5, 10, 19, 28]
CONTENT_LAYERS = [25]
LR = 0.8
LR_DECAY_EPOCH = 300
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 1e5
TV_WEIGHT = 50

mx_ctx = mx.gpu(0)

def find_root_folder(project_folder):
    folder_list = os.getcwd().split(sep="/")
    root_folder_list = []
    for folder in folder_list:
        if folder == project_folder:
            break
        else:
            root_folder_list.append(folder)
    root_folder_list.append(project_folder)
    return "/" + os.path.join(*root_folder_list)

ROOT_FOLDER = find_root_folder("mxnet-cookbook")

# %%
# -- Load images

content_image = image.imread(os.path.join(ROOT_FOLDER, "_resources", "IMG_3855.jpeg"))
style_image = image.imread(os.path.join(ROOT_FOLDER, "_resources", "Famous-Pablo-Picasso-Paintings-and-Art-Pieces21.jpg"))
# %%
# -- Show images

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].imshow(content_image.asnumpy())
axs[1].imshow(style_image.asnumpy())
plt.show()


# %%
# -- Preprocessing and postprocessing

def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32') / 255 - RGB_MEAN) / RGB_STD
    return np.expand_dims(img.transpose(2, 0, 1), axis=0)


def postprocess(img):
    img = img[0].as_in_ctx(RGB_STD.ctx)
    return (img.transpose(1, 2, 0) * RGB_STD + RGB_MEAN).clip(0, 1)


# %%
# -- Load pretrained net for feature extraction

pretrained_net = gluon.model_zoo.vision.vgg19(pretrained=True)

net = nn.Sequential()
for i in range(max(CONTENT_LAYERS + STYLE_LAYERS) + 1):
    net.add(pretrained_net.features[i])


# %%
# --
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles


def get_contents(image_shape, ctx):
    content_X = preprocess(content_image, image_shape).copyto(ctx)
    contents_Y, _ = extract_features(content_X, CONTENT_LAYERS, STYLE_LAYERS)
    return content_X, contents_Y


def get_styles(image_shape, ctx):
    style_X = preprocess(style_image, image_shape).copyto(ctx)
    _, styles_Y = extract_features(style_X, CONTENT_LAYERS, STYLE_LAYERS)
    return style_X, styles_Y


# %%
# -- Content loss

def content_loss(Y_hat, Y):
    return np.square(Y_hat, Y).mean()


# %%
# -- Style loss

def gram(X):
    num_channels = X.shape[1]
    n = X.size // X.shape[1]
    X = X.reshape(num_channels, n)
    return np.dot(X, X.T) / (num_channels * n)


def style_loss(Y_hat, gram_Y):
    return np.square(gram(Y_hat) - gram_Y).mean()


# %%
# -- Total Variance Loss (TV loss)

def tv_loss(Y_hat):
    return 0.5 * (np.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  np.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


# %%
# -- Total loss

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    contents_l = [content_loss(Y_hat, Y) * CONTENT_WEIGHT for Y_hat, Y in zip(contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * STYLE_WEIGHT for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * TV_WEIGHT
    l = sum(styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l


# %%
# -- Creating and initializing the composite image

class GeneratedImage(nn.Block):
    def __init__(self, img_shape, **kwargs):
        super(GeneratedImage, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=img_shape)

    def forward(self):
        return self.weight.data()


def get_inits(X, ctx, lr, styles_Y):
    gen_img = GeneratedImage(X.shape)
    gen_img.initialize(init.Constant(X), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(gen_img.collect_params(), 'adam', {'learning_rate': lr})
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer


# %%
# -- Train

def train(X, contents_Y, styles_Y, ctx, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, ctx, lr, styles_Y)
    for epoch in range(num_epochs):
        with autograd.record():
            contents_Y_hat, styles_Y_hat = extract_features(X, CONTENT_LAYERS, STYLE_LAYERS)
            contents_l, styles_l, tv_l, l = compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step(1)
        npx.waitall()
        if epoch % lr_decay_epoch == 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.3)
        if epoch % 100 == 0:
            msg = [
                f"Epoch: {epoch}",
                f"contents_l: {float(sum(contents_l)):0.3f}",
                f"style_l: {float(sum(styles_l)):0.3f}",
                f"tv_l: {float(tv_l):0.3f}",
                f"total_l: {float(l):0.3f}"
            ]
            msg = ", ".join(msg)
            print(msg)
            plt.imshow(postprocess(X).asnumpy())
            plt.show()
    return X


# %%
# -- Train (continued)

net.collect_params().reset_ctx(mx_ctx)

content_X, contents_Y = get_contents(IMAGE_SIZE, mx_ctx)
_, styles_Y = get_styles(IMAGE_SIZE, mx_ctx)

output = train(content_X, contents_Y, styles_Y, mx_ctx, LR, N_EPOCHS, LR_DECAY_EPOCH)
