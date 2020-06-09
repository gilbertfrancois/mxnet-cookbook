# MXNet Cheatsheet







## Load / save a model

### Load a model

_Module_
```python
sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 0)
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
        label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
```

_Gluon_
```python

```


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
### Freeze parameters of layers (Gluon)

You can set grad_req attribute to 'null' (it is a string) to prevent changes of this parameter. Here is the example. I define a set of parameter names I want to freeze and freeze them after creating my model, but before the initialization.

```python
num_hidden = 10
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_outputs))

layers_to_freeze = set(['sequential1_dense0_weight', 'sequential1_dense0_bias', 'sequential1_dense1_weight', 'sequential1_dense1_bias'])    

for p in net.collect_params().values():
    if p.name in layers_to_freeze:
        p.grad_req = 'null'

net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
```

https://discuss.mxnet.io/t/gluon-access-layer-weights/1160/2
