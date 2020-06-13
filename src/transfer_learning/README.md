# Transfer Learning Notes

## Experiment 1

- Pretrained ResNet50 v2
- Freeze all feature layers
- Add Dense(1024)
```python
scheduler = mx.lr_scheduler.FactorScheduler(base_lr=0.001, factor=0.75, step=8*len(train_data), stop_factor_lr=1e-8)
trainer = gluon.Trainer(net.collect_params(), "sgd", {"lr_scheduler": scheduler, "momentum": momentum, "wd": wd})
```