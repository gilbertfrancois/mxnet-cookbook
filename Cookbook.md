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



## Data
### Data
Combine the features and labels of the training data
```python
dataset = mxnet.gluon.data.ArrayDataset(features, labels)
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