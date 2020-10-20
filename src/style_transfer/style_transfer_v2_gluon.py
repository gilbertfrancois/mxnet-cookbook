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
    def __init__(self, content_image_filepath, style_image_filepath, out_image_filepath, image_size, content_weight=1.0,
                 style_weight=1.0e4, tv_weight=50.0, lr=0.1):
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
        if isinstance(content_image_filepath, numpy.ndarray):
            self.content_image = self.as_nd_np(content_image_filepath)
        elif isinstance(content_image_filepath, str):
            self.content_image = image.imread(content_image_filepath)
        else:
            raise TypeError("Only numpy array or str are supported.")
        # self.content_image = self.smooth(content_image, 25, 75, 75)
        self.style_image = image.imread(style_image_filepath)

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
        cv.imwrite(self.out_image_filepath, cv.cvtColor(out, cv.COLOR_RGB2BGR))
        return out


# %%
# -- Train (continued)

content_weight_list = [1.0]
style_weight_list = [1e4]
tv_weight_list = [10]

for content_weight in content_weight_list:
    for style_weight in style_weight_list:
        for tv_weight in tv_weight_list:
            tic = time.time()
            print(f"[ ] Starting processing with settings: {content_weight} {style_weight} {tv_weight}")
            alpha = 0.8
            kernel_size = 3
            sigma_color = 75
            sigma_space = 75
            timestamp = str(int(time.time()))
            ROOT_FOLDER = find_root_folder("mxnet-cookbook")
            content_image_filename = os.path.join(ROOT_FOLDER, "_resources", "IMG_5226.jpeg")
            style_image_filename = os.path.join(ROOT_FOLDER, "_resources", "cat1.jpg")
            # scales = ((200, 150), (300, 225), (400, 300), (600, 450), (800, 600))
            # scales = ((200, 150), (400, 300), (200, 150), (400, 300), (800, 600), (400, 300), (800, 600))
            scales = ((200, 150), (283, 212), (400, 300), (566, 424), (800, 600))
            lr_list = (0.7, 0.6, 0.5, 0.5, 0.5)
            # lr_list = (0.7, 0.5, 0.3)

            original_image = cv.cvtColor(cv.imread(content_image_filename), cv.COLOR_BGR2RGB)
            # cartonifier = Cartonifier()
            # original_image = cartonifier.process(original_image, 192)
            content_image = cv.resize(original_image, scales[0], cv.INTER_CUBIC)

            index = 0
            for index, scale in enumerate(scales):
                if index > 0:
                    src1 = cv.resize(original_image, dsize=scale, interpolation=cv.INTER_CUBIC)
                    # src1 = cartonifier.process(src1, 255)
                    # src1 = cv.bilateralFilter(src1, kernel_size, sigma_color, sigma_space)
                    src2 = cv.resize(content_image, dsize=scale, interpolation=cv.INTER_CUBIC)
                    src2 = cv.medianBlur(src2, ksize=3)
                    src3 = cv.addWeighted(src2, alpha, src1, 1.0 - alpha, 0)
                    content_image = src3
                out_image_filepath = os.path.join(ROOT_FOLDER, "_resources", f"{timestamp}_out{index}_{content_weight}_{style_weight}_{tv_weight}.png")
                print(out_image_filepath)
                lr = lr_list[index]
                style_transfer_gf = StyleTransferGF(content_image, style_image_filename, out_image_filepath, scale,
                                                    content_weight=content_weight, style_weight=style_weight, tv_weight=tv_weight)
                content_image = style_transfer_gf.train()
                # plt.figure()
                # plt.imshow(content_image)
                # plt.title(f"{timestamp} {scale}")
                # plt.show()
                style_transfer_gf = None
                del style_transfer_gf
                time.sleep(3)

            # content_image = cv.imread(os.path.join(ROOT_FOLDER, "_resources", "1602971485_out2.png"))
            # content_image = cv.cvtColor(content_image, cv.COLOR_BGR2RGB)
            tmp1 = cv.medianBlur(content_image, ksize=3)
            tmp2 = cv.medianBlur(content_image, ksize=5)
            kernel = numpy.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            tmp2 = cv.filter2D(tmp2, -1, kernel)
            tmp3 = cv.addWeighted(tmp1, 0.9, tmp2, 0.1, 0)
            out_image_filepath = os.path.join(ROOT_FOLDER, "_resources", f"{timestamp}_out{index + 1}_{content_weight}_{style_weight}_{tv_weight}.png")
            cv.imwrite(out_image_filepath, cv.cvtColor(tmp3, cv.COLOR_RGB2BGR))
            toc = time.time()
            print(f"Elapsed time: f{timedelta(seconds=(toc-tic))}")


