import mxnet as mx
from mxnet.gluon import nn


#%%

layer = nn.Dense(2)
layer

#%%

layer.initialize()

#%%

x = mx.nd.random.uniform(-1, 1, shape=(3, 4))

#%%

layer(x)
#%%

layer.weight.data()

#%%