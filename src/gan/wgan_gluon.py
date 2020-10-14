#    Copyright 2020 Gilbert Francois Duivesteijn
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

import matplotlib.pyplot as plt
import mxnet as mx
import mxnet.gluon.nn as nn
import mxnet.ndarray as nd
import numpy as np
from mxboard import SummaryWriter
from mxnet import gluon, autograd
from mxnet import initializer
from mxnet.gluon.data.vision import transforms

# Set logging to see some output during training
logging.getLogger().setLevel(logging.DEBUG)

# Hyperparameters
NAME = "wgan"
N_GPUS = 2
mx_ctx = [mx.gpu(i) for i in range(N_GPUS)] if N_GPUS > 0 else [mx.cpu()]
N_WORKERS = 4
Z_DIM = 64
EPOCHS = 200
PER_DEVICE_BATCH_SIZE = 128
PER_DEVICE_LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999
C_LAMBDA = 10
CRIT_REPEATS = 5

LEARNING_RATE = PER_DEVICE_LEARNING_RATE * max(len(mx_ctx), 1)
BATCH_SIZE = PER_DEVICE_BATCH_SIZE * max(len(mx_ctx), 1)


# %%
# -- Helper function for visualization

def plot_image_tensor(image_tensor, rows, cols):
    grid = make_grid(image_tensor, rows)
    plt.figure(figsize=(rows * 2, cols * 2))
    plt.imshow(grid)
    plt.show()


def make_grid(image_tensor, rows):
    cols = image_tensor.shape[0] // rows
    if image_tensor.ndim == 2:
        image_tensor = image_tensor.reshape(-1, 1, 28, 28)
    if image_tensor.ndim != 4:
        raise ValueError(f"Image tensor has wrong dimension. Expected 4, actual {image_tensor.ndim}")
    n, c, h, w = image_tensor.shape
    image_tensor = (image_tensor + 1) / 2
    assert nd.max(image_tensor) <= 1
    assert nd.min(image_tensor) >= 0
    grid = image_tensor.reshape(rows, cols, c, h, w)
    grid = grid.transpose(axes=(0, 3, 1, 4, 2))
    grid = grid.reshape(rows * h, cols * w, c).asnumpy()
    if grid.ndim == 3 and grid.shape[2] == 1:
        grid = grid.squeeze()
    return grid


# %%
# Fix the seed
mx.random.seed(42)
np.random.seed(42)

# %%
# -- Get some data to train on

mnist_train = gluon.data.vision.datasets.MNIST(train=True)
mnist_valid = gluon.data.vision.datasets.MNIST(train=False)

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# %%
# -- Create a data iterator that feeds batches

train_data = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   last_batch="rollover",
                                   num_workers=N_WORKERS)

eval_data = gluon.data.DataLoader(mnist_valid.transform_first(transformer),
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  last_batch="rollover",
                                  num_workers=N_WORKERS)


# %%
# -- Define the generator

class Generator(nn.HybridBlock):
    def __init__(self, z_dim=10, im_channel=1, hidden_dim=64):
        super().__init__()
        self.z_dim = z_dim
        with self.name_scope():
            self.gen = nn.HybridSequential()
            self.gen.add(
                self.get_generator_block(z_dim, hidden_dim * 4, kernel_size=3, strides=2),
                self.get_generator_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, strides=1),
                self.get_generator_block(hidden_dim * 2, hidden_dim * 1, kernel_size=3, strides=2),
                self.get_generator_block(hidden_dim * 1, im_channel, kernel_size=4, strides=2, final_layer=True),
            )

    def get_generator_block(self, input_dim, output_dim, kernel_size=3, strides=2, final_layer=False):
        layer = nn.HybridSequential()
        if not final_layer:
            layer.add(
                nn.Conv2DTranspose(in_channels=input_dim, channels=output_dim, kernel_size=kernel_size, strides=strides,
                                   use_bias=False),
                nn.BatchNorm(in_channels=output_dim),
                nn.Activation(activation="relu")
            )
        else:
            layer.add(
                nn.Conv2DTranspose(in_channels=input_dim, channels=output_dim, kernel_size=kernel_size, strides=strides,
                                   use_bias=False),
                nn.Activation(activation="tanh")
            )
        return layer

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.gen(x)


# %%

class Critic(nn.HybridBlock):
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.HybridSequential()
        self.crit.add(
            self.make_crit_block(im_chan, hidden_dim),
            self.make_crit_block(hidden_dim, hidden_dim * 2),
            self.make_crit_block(hidden_dim * 2, 1, final_layer=True)
        )

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.crit(x)

    def make_crit_block(self, input_dim, output_dim, kernel_size=4, strides=2, final_layer=False):
        layer = nn.HybridSequential()
        if not final_layer:
            layer.add(
                nn.Conv2D(in_channels=input_dim, channels=output_dim, kernel_size=kernel_size, strides=strides,
                          use_bias=False),
                nn.BatchNorm(in_channels=output_dim),
                nn.LeakyReLU(alpha=0.2)
            )
        else:
            layer.add(
                nn.Conv2D(in_channels=input_dim, channels=output_dim, kernel_size=kernel_size, strides=strides)
            )
        return layer


# %%


def get_gradient(crit, real, fake, epsilon):
    mixed_images = epsilon * real + (1 - epsilon) * fake
    mixed_images.attach_grad()
    with autograd.record():
        mixed_scores = crit(mixed_images)
    grad = autograd.grad(mixed_scores, [mixed_images], retain_graph=True)[0]
    return grad


def gradient_penalty(gradient):
    gradient = gradient.reshape(gradient.shape[0], -1)
    gradient_norm = nd.norm(gradient, ord=2, axis=1)
    penalty = nd.mean((gradient_norm - 1) ** 2)
    return penalty


# %%
# -- Initialize parameters

gen = Generator(z_dim=Z_DIM)
gen.initialize(init=initializer.Normal(0.02), ctx=mx_ctx)
# %%

crit = Critic()
crit.initialize(init=initializer.Normal(0.02), ctx=mx_ctx)

# %%
# -- Print summary before hybridizing

z = nd.random.randn(1, Z_DIM, 1, 1, ctx=mx_ctx[0])
print(gen)
gen.summary(z)
# %%

xhat = nd.random.randn(1, 1, 28, 28, ctx=mx_ctx[0])
print(crit)
crit.summary(xhat)

# %%
# -- Hybridize and run a forward pass once to generate a symbol which will be used later
#    for plotting the network.

gen.hybridize()
crit.hybridize()
#
gen(z)
crit(xhat)

# %%
# -- Define loss function and optimizer

loss_fn = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
gen_trainer = gluon.Trainer(gen.collect_params(), 'adam', {"learning_rate": LEARNING_RATE,
                                                           "beta1": BETA1,
                                                           "beta2": BETA2})
crit_trainer = gluon.Trainer(crit.collect_params(), 'adam', {"learning_rate": LEARNING_RATE,
                                                             "beta1": BETA1,
                                                             "beta2": BETA2})


# %%

def gen_loss_fn(crit_fake_pred):
    gen_loss = -nd.mean(crit_fake_pred)
    return gen_loss


def crit_loss_fn(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = nd.mean(crit_fake_pred) - nd.mean(crit_real_pred) + c_lambda * gp
    return crit_loss


# %%
# -- Discriminator's loss function

def get_crit_loss(gen, crit, X, batch_size, z_dim, ctx):
    # loss from real images
    # loss from fake images
    z = nd.random.randn(batch_size, z_dim, 1, 1, ctx=ctx)
    Xhat = gen(z).detach()
    y_pred_fake = crit(Xhat).reshape(X.shape[0], -1)
    y_pred_real = crit(X).reshape(X.shape[0], -1)
    epsilon = np.random.rand(len(X), 1, 1, 1)
    epsilon = nd.array(epsilon, ctx=ctx)
    grad = get_gradient(crit, X, Xhat.detach(), epsilon)
    gp = gradient_penalty(grad)
    crit_loss = crit_loss_fn(y_pred_fake, y_pred_real, gp, C_LAMBDA)
    return crit_loss


# %%
# -- Generator's loss function

def get_gen_loss(gen, crit, batch_size, z_dim, ctx):
    z = nd.random.randn(batch_size, z_dim, 1, 1, ctx=ctx)
    xhat = gen(z)
    y_pred = crit(xhat).reshape(xhat.shape[0], -1)
    gen_loss = gen_loss_fn(y_pred)
    return gen_loss


# %%
# -- Train

t0 = time.time()
timestamp = str(int(t0))
param_file_prefix = f"{timestamp}_{NAME}"

iter = 0

with SummaryWriter(logdir=f"../../log/{timestamp}_{NAME}", flush_secs=5) as sw:
    for epoch in range(EPOCHS):
        gen_train_loss = 0
        crit_train_loss = 0
        # Forward pass and update weights
        tic_epoch = time.time()
        for i, batch in enumerate(train_data):
            tic_batch = time.time()

            data = gluon.utils.split_and_load(batch[0], ctx_list=mx_ctx, batch_axis=0, even_split=False)
            # label = gluon.utils.split_and_load(batch[1], ctx_list=mx_ctx, batch_axis=0, even_split=False)

            # Update critic
            for _ in range(CRIT_REPEATS):
                with autograd.record():
                    crit_loss_list = [get_crit_loss(gen, crit, X, PER_DEVICE_BATCH_SIZE, Z_DIM, mx_ctx[i])
                                      for i, X in enumerate(data)]
                for crit_loss in crit_loss_list:
                    crit_loss.backward()
                crit_trainer.step(batch_size=BATCH_SIZE)
                crit_train_loss += sum([loss.mean().asscalar() for loss in crit_loss_list]) / len(crit_loss_list) / CRIT_REPEATS

            # Update generator. Do multiple steps to keep up with the discriminator.
            with autograd.record():
                gen_loss_list = [get_gen_loss(gen, crit, PER_DEVICE_BATCH_SIZE, Z_DIM, mx_ctx[i])
                                 for i, X in enumerate(data)]

            for gen_loss in gen_loss_list:
                gen_loss.backward()
            gen_trainer.step(batch_size=BATCH_SIZE)
            gen_train_loss += sum([loss.mean().asscalar() for loss in gen_loss_list]) / len(gen_loss_list)

            toc_batch = time.time()

            if iter % 500 == 0:
                print("Plotting...")
                with autograd.record(False):
                    xhat = gen(nd.random.randn(25, Z_DIM, 1, 1, ctx=mx_ctx[0]))
                grid = make_grid(xhat, 5)
                sw.add_image("xhat", grid, iter)
                plt.figure(figsize=(10, 10))
                plt.imshow(grid)
                plt.show()
                time.sleep(0.1)
            iter += 1

        # Epoch statistics
        toc_epoch = time.time()
        chrono_epoch = toc_epoch - tic_epoch
        crit_train_loss /= len(train_data)
        gen_train_loss /= len(train_data)
        samples_per_sec = len(mnist_train) / chrono_epoch

        # Print info to console
        msg = f"epoch: {epoch} " \
              f"iter: {iter} " \
              f"disc_train_loss: {crit_train_loss:0.3f} " \
              f"gen_train_loss: {gen_train_loss:0.3f} " \
              f"lr: {LEARNING_RATE} " \
              f"samples/s: {samples_per_sec:0.2f} "
        print(msg)

        # Add some values to mxboard
        sw.add_scalar("Loss", ("gen_train_loss", gen_train_loss), global_step=epoch)
        sw.add_scalar("Loss", ("disc_train_loss", crit_train_loss), global_step=epoch)

print("=== Total training time: {:02f} seconds".format(time.time() - t0))

# %%
# -- Save parameters

gen.save_parameters(f"{param_file_prefix}_gen_final.params")
crit.save_parameters(f"{param_file_prefix}_disc_final.params")
