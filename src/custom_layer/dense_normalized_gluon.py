import os
import mxnet as mx
from mxnet import nd
from mxnet import autograd
import mxnet.gluon.nn as nn
import numpy as np

os.environ["MXNET_CUDNN_LIB_CHECKING"] = "0"

class DenseNormalized(nn.HybridBlock):

    def __init__(self, in_units:int, units:int, with_norm:bool, **kwargs):
        """ Constructor

        Parameters
        ----------
        in_units: int
            Input dimension
        units: int
            Ouput dimension
        with_normalization: bool
            Normalize weight and input data before fully connected operation.
        """
        super().__init__(**kwargs)
        self.units = units
        self.with_norm = with_norm
        init_W = mx.init.Constant(5)
        with self.name_scope():
            self.w = self.params.get("weight", shape=(units, in_units), init=init_W, dtype="float32")
        
    def hybrid_forward(self, F, x, w, *args, **kwargs):
        """ Layer forward function

        Parameters
        ----------
        x: (Symbol or NDArray)
            The first input tensor.
        w: Symbol or NDArray
            Self defined weight parameter, defined in the init function. This value is filled in automatically by the 
            super class for its current ctx.
        *args: List[Symbol] or List[NDArray]
            Additional input tensors.
        **kwargs: dict
            Additional arguments
            
        Returns
        -------
        Symbol or NDArray
            y = nd.dot(x, W.T) or nd.dot( ||x||, ||W.T|| )
          
        """
        if self.with_norm:
            # Normalize the input and weight. Note that w after normalization is a copy, not a reference to 
            # self.w.data()
            x = F.L2Normalization(x, mode="instance", name="embedding_norm")
            id_w_tic = id(w)
            w = F.L2Normalization(w, mode="instance", name="weight_norm")
            id_w_toc = id(w)
            assert id_w_tic != id_w_toc
            y = F.FullyConnected(data=x, weight=w, no_bias=True, num_hidden=self.units, name="fc_norm")
        else:
            y = F.FullyConnected(data=x, weight=w, no_bias=True, num_hidden=self.units, name="fc")
        return y

    def __repr__(self):
        s = '{name}({layout})'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        layout='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]))

print("================================================================================")
print("= Test regular FullyConnected")
print("================================================================================")
# %%
# Define network, containing the DenseNormalized only.
fc1 = DenseNormalized(3, 2, with_norm=False)
fc1.collect_params().initialize()
 
# %%
# Define input vector.
x = nd.array([[1, 2, 3]])

# %%
# Compute forward and gradient
with autograd.record():
    y = fc1(x)
y.backward()

# %%
# Verify gradient:
#     y = x * W.T + b.T
# dy/dW = x * dW/dW + dx/dW * W, where dx/dW = 0
dydw = fc1.w.grad()
dydw_ = nd.broadcast_mul(x, nd.ones_like(fc1.w.data()))

print("\nx", x)
print("\nw", fc1.w.data())
print("\ndydw", dydw)

if nd.sum(dydw).asscalar() > 1 and np.allclose(dydw.asnumpy(), dydw_.asnumpy()):
    print("Success!")

print("================================================================================")
print("= Test normalized FullyConnected")
print("================================================================================")

# %%
# Define network, containing the DenseNormalized only.
fc1 = DenseNormalized(3, 2, with_norm=True)
fc1.collect_params().initialize()
 
# %%
# Define input vector.
x = nd.array([[1, 2, 3]])

# %%
# Compute forward and gradient
scale = 5
with autograd.record():
    y = fc1(x)
    y = y*scale
y.backward()

# %%
# Verify gradient:
#     y = x * W.T + b.T
# dy/dW = x * dW/dW + dx/dW * W, where dx/dW = 0
dydw = fc1.w.grad()
dydw_ = nd.broadcast_mul(x, nd.ones_like(fc1.w.data()))

print("\nx", x)
print("\nw", fc1.w.data())
print("\ndydw", dydw)

if nd.sum(dydw).asscalar() < 1 and not np.allclose(dydw.asnumpy(), dydw_.asnumpy()):
    print("Success!")




