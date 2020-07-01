# MXNet Cookbook

## About

The *Cookbook.md* and code examples are my personal notes for how to do things in MXNet. The repo is intended to be a
"living" document that evolves over time. When possible, I try to have _one-to-one_ code comparisons between imperative
programming with Gluon and symbolic programming with Module.

*Note: the symbol API is deprecated in mxnet. From now on, this repo will focus on Gluon and hybridize methods.*

## Custom Loss

Shows how to implement a custom loss function and custom metrics function.

- Module
  - MakeLoss needs to be written in symbol api
  - MakeLoss returns the gradient of the loss function. For the metrics you need the forward output of the network, 
    and therefore you need to write an output that blocks the gradient and add it to the sym.Group() which contains
    the list of network outputs.
  - CustomMetric expects float values, not symbols. The function needs to be written with numpy arrays.

## DCGAN

This consists of a simple Deep Convolutional Generative Adversarial Network (dcgan), implemented in Gluon. As
training data, it uses the aligned images of the Labeled Faces in the Wild dataset.

## Faster R-CNN

Work in progress...


## Image Classifier

Code comparison between Module and Gluon for a simple image classifier, using a convolutional neural network. Some 
highlighted topics are how to load data to the GPU and run training of the model on GPU. 
- `mnist_module.py` Module implementation for CPU or single GPU. 
- `mnist_gluon.py` Gluon implementation for CPU or single GPU.
- `mnist_gluon_mxboard.py` Hybridized CPU or multi GPU implementation with TensorBoard writer.

Measurements are performed on the MNIST Fashion Image Classifier `mnist_gluon_mxboard.py`. Note that the model has been
hybridized before training. The used system is a 16 core Intel i9-9900K @ 3.60GHz and 2 NVidia GeForce GTX 1070 Ti with
8Gb:

| device  | time per epoch (s) | total training time (s)  |
|:---      |---:               |---:                      |
| cpu     | 162.6              | 1716.7                   |
| gpu x 1 | 6.1                | 67.1                     |
| gpu x 2 | 3.7                | 41.1                     |

## Linear Regression

Code comparison between Module and Gluon for a simple linear regression problem. There are 2 versions for the *Module*
API. The file `linear_regression_module.py` uses the high-level `fit()` function. The file 
`linear_regression_module_intermediate.py` does the steps of `fit()` more verbose. 

## Pix2Pix

This is an implementation from the method described in the paper: ["Image-to-Image Translation with Conditional Adversarial Networks", Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros](https://arxiv.org/abs/1611.07004). This script is based on the Pixel2Pixel example from [Gluon, the straight dope](https://gluon.mxnet.io/chapter14_generative-adversarial-networks/pixel2pixel.html)
New features are:

- Multiple GPU support
- New feature based loss function resulting in better and cleaner generated images.

## Transfer Learning

Shows how to do transfer learning and combined transfer learning with fine tuning. Highlighted topics are how to replace
the final layer(s) and how to freeze the feature layers, if needed.

Todo:
- [ ] Fix batch normalization.
- [ ] Freeze first layer in code.


