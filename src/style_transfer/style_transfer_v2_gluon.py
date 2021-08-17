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
import time
import mxnet as mx
import cv2 as cv
import numpy
from mxnet import gluon
from mxnet import autograd
from mxnet import image
from mxnet import init
from mxnet import np, npx
from mxnet.gluon import nn
import glob
import matplotlib.pyplot as plt
from datetime import timedelta
from cartonifier import Cartonifier

# %%
# -- Settings

npx.set_np()


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


class GeneratedImage(nn.Block):
    def __init__(self, img_shape, **kwargs):
        super(GeneratedImage, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=img_shape)

    def forward(self):
        return self.weight.data()


class StyleTransferGF:
    def __init__(self, content_image, style_image, image_size, content_weight=1.0, style_weight=1.0e4, tv_weight=10.0,
                 lr=0.1, out_image_filepath=None):
        super(StyleTransferGF, self).__init__()
        self.IMAGE_SIZE = image_size
        self.N_EPOCHS = 600
        self.RGB_MEAN = np.array([0.485, 0.456, 0.406])
        self.RGB_STD = np.array([0.229, 0.224, 0.225])
        self.style_layers = [0, 5, 10, 19, 28]
        self.content_layers = [25]
        self.LR = lr
        self.LR_DECAY_EPOCH = 300
        self.CONTENT_WEIGHT = content_weight
        self.STYLE_WEIGHT = style_weight
        self.TV_WEIGHT = tv_weight
        self.mx_ctx = mx.gpu(0)
        self.out_image_filepath = out_image_filepath

        # Load and prepare images
        if isinstance(content_image, numpy.ndarray):
            self.content_image = self.as_nd_np(content_image)
        elif isinstance(content_image, str):
            self.content_image = image.imread(content_image)
        else:
            raise TypeError("Only numpy array or str are supported.")
        if isinstance(style_image, numpy.ndarray):
            self.style_image = self.as_nd_np(style_image)
        elif isinstance(style_image, str):
            self.style_image = image.imread(style_image)
        else:
            raise TypeError("Only numpy array or str are supported.")

        # Load and prepare feature extractor
        pretrained_net = gluon.model_zoo.vision.vgg19(pretrained=True)
        self.net = nn.Sequential()
        for i in range(max(self.content_layers + self.style_layers) + 1):
            self.net.add(pretrained_net.features[i])

    def smooth(self, src: mx.numpy.ndarray, d: int, sigma_color: int, sigma_space: int):
        img = image.imresize(src, *self.IMAGE_SIZE)
        dst = cv.bilateralFilter(img.asnumpy(), d, sigma_color, sigma_space)
        dst = self.as_nd_np(dst)
        return dst

    def as_nd_np(self, img):
        return mx.nd.array(img, dtype=np.int32).as_np_ndarray()

    def preprocess(self, img):
        img = image.imresize(img, *self.IMAGE_SIZE)
        img = (img.astype('float32') / 255 - self.RGB_MEAN) / self.RGB_STD
        return np.expand_dims(img.transpose(2, 0, 1), axis=0)

    def postprocess(self, img):
        img = img[0].as_in_ctx(self.RGB_STD.ctx)
        return (img.transpose(1, 2, 0) * self.RGB_STD + self.RGB_MEAN).clip(0, 1)

    def extract_features(self, x):
        contents = []
        styles = []
        for i in range(len(self.net)):
            x = self.net[i](x)
            if i in self.style_layers:
                styles.append(x)
            if i in self.content_layers:
                contents.append(x)
        return contents, styles

    def get_contents(self):
        content_x = self.preprocess(self.content_image).copyto(self.mx_ctx)
        contents_y, _ = self.extract_features(content_x)
        return content_x, contents_y

    def get_styles(self):
        style_x = self.preprocess(self.style_image).copyto(self.mx_ctx)
        _, styles_y = self.extract_features(style_x)
        return style_x, styles_y

    def get_inits(self, x, styles_y):
        gen_img = GeneratedImage(x.shape)
        gen_img.initialize(init.Constant(x), ctx=self.mx_ctx, force_reinit=True)
        trainer = gluon.Trainer(gen_img.collect_params(), 'adam', {'learning_rate': self.LR})
        styles_y_gram = [self.gram(y) for y in styles_y]
        return gen_img(), styles_y_gram, trainer

    @staticmethod
    def content_loss(y_hat, y):
        return np.square(y_hat, y).mean()

    @staticmethod
    def gram(x):
        num_channels = x.shape[1]
        n = x.size // x.shape[1]
        x = x.reshape(num_channels, n)
        return np.dot(x, x.T) / (num_channels * n)

    @staticmethod
    def style_loss(y_hat, gram_y):
        return np.square(StyleTransferGF.gram(y_hat) - gram_y).mean()

    @staticmethod
    def tv_loss(y_hat):
        return 0.5 * (np.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]).mean() +
                      np.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]).mean())

    def compute_loss(self, x, contents_y_hat, styles_y_hat, contents_y, styles_y_gram):
        contents_l = [StyleTransferGF.content_loss(y_hat, y) * self.CONTENT_WEIGHT for y_hat, y in
                      zip(contents_y_hat, contents_y)]
        styles_l = [StyleTransferGF.style_loss(y_hat, y) * self.STYLE_WEIGHT for y_hat, y in
                    zip(styles_y_hat, styles_y_gram)]
        tv_l = StyleTransferGF.tv_loss(x) * self.TV_WEIGHT
        l = sum(styles_l + contents_l + [tv_l])
        return contents_l, styles_l, tv_l, l

    def train(self):
        self.net.collect_params().reset_ctx(self.mx_ctx)
        content_x, contents_y = self.get_contents()
        _, styles_y = self.get_styles()
        x, styles_y_gram, trainer = self.get_inits(content_x, styles_y)
        styles_y_gram = [StyleTransferGF.gram(Y) for Y in styles_y]
        for epoch in range(self.N_EPOCHS):
            with autograd.record():
                contents_y_hat, styles_y_hat = self.extract_features(x)
                contents_l, styles_l, tv_l, l = self.compute_loss(x, contents_y_hat, styles_y_hat, contents_y,
                                                                  styles_y_gram)
            l.backward()
            trainer.step(1)
            npx.waitall()
            if epoch % self.LR_DECAY_EPOCH == 0:
                trainer.set_learning_rate(trainer.learning_rate * 0.3)
            if epoch % 100 == 0:
                msg = [
                    f"Size: {self.IMAGE_SIZE}",
                    f"Epoch: {epoch}",
                    f"contents_l: {float(sum(contents_l)):0.3f}",
                    f"style_l: {float(sum(styles_l)):0.3f}",
                    f"tv_l: {float(tv_l):0.3f}",
                    f"total_l: {float(l):0.3f}"
                ]
                msg = ", ".join(msg)
                print(msg)
                # plt.imshow(self.postprocess(x).asnumpy())
                # plt.show()
        out = self.postprocess(x).asnumpy()
        out = (out * 255).astype(numpy.uint8)
        if self.out_image_filepath is not None:
            cv.imwrite(self.out_image_filepath, cv.cvtColor(out, cv.COLOR_RGB2BGR))
        return out


# %%
# -- Train (continued)

def get_output_filepath(content_image_filepath, style_image_filepath, cw, sw, tw, output_folder):
    filename_noext1 = os.path.splitext(os.path.basename(content_image_filepath))[0]
    filename_noext2 = os.path.splitext(os.path.basename(style_image_filepath))[0]
    out = f"{filename_noext1}_{filename_noext2}_{cw}_{sw}_{tw}.png"
    out = os.path.join(output_folder, out)
    return out


def process_image(content_image_filepath, style_image_filepath, content_weight, style_weight, tv_weight, output_folder,
                  timestamp):
    print(f"[ ] Processing {os.path.basename(content_image_filepath)} with settings: {content_weight} {style_weight} {tv_weight}")
    alpha = 0.90
    scales = ((200, 150), (283, 212), (400, 300), (566, 424), (800, 600))
    lr_list = (0.7, 0.6, 0.5, 0.5, 0.5)
    # Prepare content image.
    original_image = cv.cvtColor(cv.imread(content_image_filepath), cv.COLOR_BGR2RGB)
    shape = original_image.shape
    ratio = shape[1] / shape[0]
    if ratio < 1:
        original_image = cv.rotate(original_image, cv.ROTATE_90_CLOCKWISE)
        is_rotated = True
    else:
        is_rotated = False
    content_image = cv.resize(original_image, scales[0], cv.INTER_CUBIC)
    # Prepare style image.
    original_style_image = cv.cvtColor(cv.imread(style_image_filepath), cv.COLOR_BGR2RGB)
    shape = original_style_image.shape
    ratio = shape[1] / shape[0]
    if ratio < 1:
        original_style_image = cv.rotate(original_style_image, cv.ROTATE_90_CLOCKWISE)
    style_image = cv.resize(original_style_image, scales[0], cv.INTER_CUBIC)


    index = 0
    for index, scale in enumerate(scales):
        if index > 0:
            src1 = cv.resize(original_image, dsize=scale, interpolation=cv.INTER_CUBIC)
            src2 = cv.resize(content_image, dsize=scale, interpolation=cv.INTER_CUBIC)
            src2 = cv.medianBlur(src2, ksize=3)
            src3 = cv.addWeighted(src2, alpha, src1, 1.0 - alpha, 0)
            content_image = src3
            style_image = cv.resize(original_style_image, dsize=scale, interpolation=cv.INTER_CUBIC)
        output_filepath = None
        lr = lr_list[index]
        style_transfer_gf = StyleTransferGF(content_image, style_image, scale, content_weight=content_weight,
                                            style_weight=style_weight, tv_weight=tv_weight,
                                            out_image_filepath=output_filepath)
        content_image = style_transfer_gf.train()
        del style_transfer_gf
        time.sleep(3)
    if is_rotated:
        content_image = cv.rotate(content_image, cv.ROTATE_90_COUNTERCLOCKWISE)
    output_filepath = get_output_filepath(content_image_filepath, style_image_filepath, content_weight, style_weight, tv_weight, output_folder)
    cv.imwrite(output_filepath, cv.cvtColor(content_image, cv.COLOR_RGB2BGR))


def main():
    root_folder = find_root_folder("mxnet-cookbook")
    output_folder = os.path.join(root_folder, "data", "output")
    os.makedirs(output_folder, exist_ok=True)
    timestamp = str(int(time.time()))

    content_weight_list = [1.0]
    style_weight_list = [1e4]
    tv_weight_list = [10]

    content_image_filepath_list = sorted(glob.glob(os.path.join(root_folder, "data", "input", "IMG_20201029*")))
    content_image_filepath_list = [content_image_filepath_list[0]]

    style_image_filepath_list = sorted([
        os.path.join(root_folder, "data", "style_transfer", "picasso_00009.jpg")
    ])


    for style_weight in style_weight_list:
        for content_weight in content_weight_list:
            for tv_weight in tv_weight_list:
                for content_image_filename in content_image_filepath_list:
                    for style_image_filename in style_image_filepath_list:
                        tic = time.time()
                        if not os.path.exists(style_image_filename):
                            raise FileNotFoundError(f"Cannot find {style_image_filename}")
                        if not os.path.exists(content_image_filename):
                            raise FileNotFoundError(f"Cannot find {content_image_filename}")
                        process_image(content_image_filename, style_image_filename, content_weight, style_weight,
                                      tv_weight, output_folder, timestamp)
                        toc = time.time()
                        print(f"Elapsed time: f{timedelta(seconds=(toc - tic))}")

if __name__ == '__main__':
    main()