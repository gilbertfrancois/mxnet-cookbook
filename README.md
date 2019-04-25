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
