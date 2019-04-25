# MXNet Cookbook

## About

The *Cookbook.md* and code examples are my personal notes for how to do things in MXNet. The repo is intended to be a
"living" document that evolves over time. When possible, I try to have _one-to-one_ code comparisons between imperative
programming with Gluon and symbolic programming with Module.

## Linear Regression

Code comparison between Module and Gluon for a simple linear regression problem. There are 2 versions for the *Module*
API. The file `linear_regression_module.py` uses the high-level `fit()` function. The file `linear_regression_module_intermediate.py`
does the steps of `fit()` more verbose. 

## Image Classifier

Code comparison between Module and Gluon for a simple image classifier, using a convolutional neural network. Some 
highlighted topics are how to load data to the GPU and run training of the model on GPU. 

## Custom Loss

Shows how to implement a custom loss function and custom metrics function.
- Module
    - MakeLoss needs to be written in symbol api
    - MakeLoss returns the gradient of the loss function. For the metrics you need the forward output of the network, 
    and therefore you need to write an output that blocks the gradient and add it to the sym.Group() which contains
    the list of network outputs.
    - CustomMetric expects float values, not symbols. The function needs to be written with numpy arrays.
    