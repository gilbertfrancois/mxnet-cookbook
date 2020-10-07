#    Copyright 2019 Gilbert Francois Duivesteijn
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import time

import mxnet as mx
from mxnet import gluon, init, autograd
from mxnet.gluon.data.vision import transforms
from mxnet.gluon import nn
import mxnet.ndarray as nd

# Set logging to see some output during training
logging.getLogger().setLevel(logging.DEBUG)

# Fix the seed
mx.random.seed(42)

# Set the compute context, GPU is available otherwise CPU
mx_ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
#mx_ctx = mx.cpu()

# %%
# -- Constants

EPOCHS = 10
BATCH_SIZE = 256

# %%
# -- Get some data to train on

mnist_train = gluon.data.vision.datasets.FashionMNIST(train=True)
mnist_valid = gluon.data.vision.datasets.FashionMNIST(train=False)

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.286, 0.353)
])

# %%
# -- Create a data iterator that feeds batches

c
train_data = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=2)

eval_data = gluon.data.DataLoader(mnist_valid.transform_first(transformer),
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=2
                                  )


# %%
# -- Define the model

class MNISTNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(MNISTNet, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(
            # layer 1
            nn.Conv2D(channels=32, kernel_size=(5, 5), padding=(2, 2)),
            nn.Activation("relu"),
            nn.BatchNorm(axis=1, momentum=0.9, epsilon=1e-5),
            # layer 2
            nn.Conv2D(channels=32, kernel_size=(5, 5), padding=(2, 2)),
            nn.Activation("relu"),
            nn.BatchNorm(axis=1, momentum=0.9, epsilon=1e-5),
            nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            # layer 3
            nn.Conv2D(channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.Activation("relu"),
            nn.BatchNorm(axis=1, momentum=0.9, epsilon=1e-5),
            # layer 4
            nn.Conv2D(channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.Activation("relu"),
            nn.BatchNorm(axis=1, momentum=0.9, epsilon=1e-5),
            nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        )
        self.embeddings = nn.HybridSequential()
        self.embeddings.add(
            nn.Flatten(),
            nn.Dense(128),
            nn.BatchNorm(axis=1, momentum=0.9, epsilon=1e-5),
        )
        self.output = nn.Dense(10)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.embeddings(x)
        x = self.output(x)
        return x

net = MNISTNet()

# %%
# -- Initialize parameters

net.initialize(init=init.Xavier(), ctx=mx_ctx)

for name, param in net.collect_params().items():
    print(name)

# %%
# -- Define loss function and optimizer

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'Adam', {'learning_rate': 0.001})


# %%%
# -- Custom metric function

def acc(output, label):
    return (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()

# %%
# -- test inference

x = nd.ones(shape=(1, 1, 28, 28), ctx=mx_ctx)
y = net(x)
print(y.shape)

# %%
# -- Train

t0 = time.time()

for epoch in range(EPOCHS):
    train_loss = 0
    train_acc = 0
    eval_acc = 0

    for data, label in train_data:
        data = data.as_in_context(mx_ctx)
        label = label.as_in_context(mx_ctx)
        with autograd.record():
            output = net(data)
            loss = loss_fn(output, label)
        loss.backward()
        trainer.step(batch_size=BATCH_SIZE)
        train_loss += loss.mean().asscalar()
        train_acc += acc(output, label)

    for data, label in eval_data:
        data = data.as_in_context(mx_ctx)
        label = label.as_in_context(mx_ctx)
        eval_acc += acc(net(data), label)

    print("epoch {}, loss {:0.3f}, acc {:0.3f}, val_acc {:0.3f}".format(
        epoch, train_loss / len(train_data), train_acc / len(train_data), eval_acc / len(eval_data)))

print("Elapsed time {:02f} seconds".format(time.time() - t0))

# %%
# -- Save parameters

net.save_parameters('fashion_mnist_gluon.params')
