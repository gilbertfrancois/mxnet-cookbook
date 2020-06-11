import datetime
import os
import pickle as pkl
import time

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from gluoncv import model_zoo
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

# %%

x = nd.random.randn(1, 3, 224, 224)

# %%
net1 = model_zoo.get_model("VGG16", pretrained=True)
net1.hybridize()

# %%

# %%

net1.export("/tmp/net", epoch=0)

# %%

symbol_file = "/tmp/net-symbol.json"
param_file = "/tmp/net-0000.params"

# At the moment, you cannot automatically convert a SymbolBlock to a (Hybrid)Sequential.
net2 = gluon.nn.SymbolBlock.imports(symbol_file, ["data"], param_file=param_file)

# %%

y1 = net1(x)
y2 = net2(x)
np.allclose(y1.asnumpy(), y2.asnumpy())

# %%

