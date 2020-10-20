import cv2 as cv
import os
import numpy as np


class Cartonifier:

    def __init__(self, n_downsampling_steps=2, n_filtering_steps=7):
        self.num_down = n_downsampling_steps
        self.num_bilateral = n_filtering_steps

    # def process_folder(self, input_folder, output_folder):
    #     if not os.path.exists(input_folder):
    #         raise FileNotFoundError('Input folder {} not found'.format(input_folder))
    #     if not os.path.exists(output_folder):
    #         raise FileNotFoundError('Output folder {} not found'.format(output_folder))
    #     file_path_list = fu.get_absolute_path_list(input_folder)
    #     for file_path in file_path_list:
    #         self.process(file_path, output_folder)

    def process(self, image, max_value=200):
        img_rgb = image
        # downsample image using Gaussian pyramid
        img_color = img_rgb
        for _ in range(self.num_down):
            img_color = cv.pyrDown(img_color)
        # repeatedly apply small bilateral filter instead of
        # applying one large filter
        for _ in range(self.num_bilateral):
            img_color = cv.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
        # upsample image to original size
        for _ in range(self.num_down):
            img_color = cv.pyrUp(img_color)
        # convert to grayscale and apply median blur
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
        img_blur = cv.medianBlur(img_gray, 7)
        # detect and enhance edges
        img_edge = self.edge_detection_v1(img_blur, max_value)
        if img_color.shape[0] != img_edge.shape[0] or img_color.shape[1] != img_edge.shape[1]:
            img_color = cv.resize(img_color, (img_edge.shape[1], img_edge.shape[0]))
        img_cartoon = cv.bitwise_and(img_color, img_edge)

        return img_cartoon

    def edge_detection_v1(self, img_blur, max_value):
        img_edge = cv.adaptiveThreshold(img_blur, max_value,
                                        cv.ADAPTIVE_THRESH_MEAN_C,
                                        cv.THRESH_BINARY,
                                        blockSize=9,
                                        C=4)
        # convert back to color, bit-AND with color image
        img_edge = cv.cvtColor(img_edge, cv.COLOR_GRAY2RGB)
        return img_edge

    # def process_image(self, src):
    #     self.alpha += 0.01
    #     if self.alpha > 1:
    #         self.alpha = 0
    #         self.current_model += 1
    #         if self.current_model >= len(self.model_list):
    #             self.current_model = 1
    #
    #     # Edge detection
    #     img_edge = self.edge_detection_v2(src)
    #
    #     # Coloured image from ML models
    #     img_colors = self.feed_forward(src)
    #
    #     # Compose layers
    #     img_blend = np.clip(((1 - self.beta) * (img_colors - img_edge * 0.1) + self.beta * self.frame).astype(np.uint8),
    #                         0, 255)
    #
    #     # Blur for smooth effect
    #     dst = cv.GaussianBlur(img_blend, (5, 5), cv.BORDER_DEFAULT)
    #     return dst
    #
    # def edge_detection_v2(self, src):
    #     dst = cv.GaussianBlur(src, (5, 5), cv.BORDER_DEFAULT)
    #     dst = cv.Canny(dst, 50, 200)
    #     # dst = self.edge_detection_v1(dst)
    #     dst = cv.cvtColor(dst, cv.COLOR_GRAY2RGB)
    #     dst = np.ones_like(dst) * 255 - dst
    #     return dst


if __name__ == '__main__':
    c = Cartonifier()
    c.process("/Users/gilbert/Desktop/test.jpg", "/Users/gilbert/Desktop/out")