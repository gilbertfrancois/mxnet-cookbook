# MXNet Cheatsheet

## Load / save a model

### Load a model

_Module_
```python
sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 0)
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
```

_Gluon_
```python
net = gluon.nn.SymbolBlock.imports('resnet18-symbol.json', ['data'], param_file='resnet18-0000.params', ctx=mx.gpu())
```

https://discuss.mxnet.io/t/load-params-from-symbol-models-to-gluon/804

## Update model from checkpoint

_Module_
```python
sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 3)
mod.fit(train_data=train_dataiter, eval_data=val_dataiter, 
    optimizer='sgd',
    optimizer_params={'learning_rate':0.01, 'momentum': 0.9},
    arg_params=arg_params, aux_params=aux_params,
    eval_metric='acc', num_epoch=10, begin_epoch=3)
```

_Gluon_
```python

```

## Save after every epoch

_Module_
```python
# construct a callback function to save checkpoints
model_prefix = 'mymodel'
checkpoint = mx.callback.do_checkpoint(model_prefix)

mod = mx.mod.Module(symbol=net)
mod.fit(train_iter, num_epoch=5, epoch_end_callback=checkpoint)
```

_Gluon_
```python
for epoch in epochs:
    # Train here...
    net.export(f"{model_name}", epoch)  # Saves sym and params files
    # or
    net.save_parameters(f"{model_name}.params")   # Saves params only, model defined in code.
```

## Get all layers after loading

_Module_

```python
sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, epoch=epoch)
all_layers = sym.get_internals()
sym = all_layers['fc1_output']
model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
model.bind(for_training=False, data_shapes=[('data', (1, 3, self.image_size, self.image_size))])
model.set_params(arg_params, aux_params)
```

_Gluon_

```python
net = gluoncv.model_zoo.get_model("ResNet50V2", pretrained=True)
all_params = net.collect_params()
print(net)
# Get one layer by using the index name/number from the output above
W1 = net.features[6][0].conv1.weight.data()  # returns NDArray
W1params = all_params.get("resnetv20_stage2_conv1_weight")  # Params from same layer as above.
```




## Data
### Data
Combine data and labels in a Dataset object
```python
mx.random.seed(42) # Fix the seed for reproducibility
X = mx.random.uniform(shape=(10, 3))
y = mx.random.uniform(shape=(10, 1))
dataset = mxnet.gluon.data.ArrayDataset(X, y)
```
Randomly reading mini-batches
```python
data_iter = mxnet.gluon.data.DataLoader(dataset, batch_size, shuffle=True)
# or with transformer
data_iter = mxnet.gluon.data.DataLoader(dataset.transform_first(transformer), batch_size, shuffle=True)
```
### To tensor

Converts an image NDArray of shape $(H \times W \times C)$ in the range $[0, 255]$ to a float32 tensor NDArray of shape $(C \times H \times W)$ in the range $[0, 1)$.

```python
transformer = mxnet.gluon.data.vision.transform.ToTensor()
```

## Transfer learning
### Freeze parameters of layers

_Gluon_

You can set grad_req attribute to 'null' (it is a string) to prevent changes of this parameter. Here is the example. I define a set of parameter names I want to freeze and freeze them after creating my model, but before the initialization.

```python
layers_to_freeze = set(['dense0_weight', 'dense0_bias', 'dense1_weight', 'dense1_bias'])    
for p in net.collect_params().values():
    if p.name in layers_to_freeze:
        p.grad_req = 'null'
```

or

```
for param in net[24:].collect_params().values():
    param.grad_req = 'write'
```

https://discuss.mxnet.io/t/gluon-access-layer-weights/1160/2


## Layer and parameter access in a pretrained model

https://discuss.mxnet.io/t/layer-access-in-a-pre-trained-model/2248

https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/blocks/parameters.html

_Gluon_

Given VGG16:

```python
>>> net = model_zoo.get_model("VGG16", pretrained=True)
>>> net
Out[50]: 
VGG(
  (features): HybridSequential(
    (0): Conv2D(3 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): Activation(relu)
    (2): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): Activation(relu)
    (4): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
    (5): Conv2D(64 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): Activation(relu)
    (7): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): Activation(relu)
    (9): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
    (10): Conv2D(128 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): Activation(relu)
    (12): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): Activation(relu)
    (14): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): Activation(relu)
    (16): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
    (17): Conv2D(256 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): Activation(relu)
    (19): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): Activation(relu)
    (21): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): Activation(relu)
    (23): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
    (24): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): Activation(relu)
    (26): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): Activation(relu)
    (28): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): Activation(relu)
    (30): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
    (31): Dense(25088 -> 4096, Activation(relu))
    (32): Dropout(p = 0.5, axes=())
    (33): Dense(4096 -> 4096, Activation(relu))
    (34): Dropout(p = 0.5, axes=())
  )
  (output): Dense(4096 -> 1000, linear)
)
```

Get the weight of the first convolutional layer:

```python
W0 = net.features[0].weight.data()
```

or

```python
W0 = net.collect_params().get("conv0_weight").data()
```

Note: it can be that the listed key contains a prefix. E.g. `net.collect_params().keys()` gives `['vgg0_conv0_weight', 'vgg0_conv0_bias', ...]`. When quering the value by key with the `get()` function, you have to remove the prefix first.

or 

```python
for p in pretrained_net.collect_params().values():
    print(p.name, p.data())
```