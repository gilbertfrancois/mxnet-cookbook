import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time

# %%
# -- Constants

NUM_INPUTS = 2
NUM_OUTPUTS = 1
NUM_EPOCHS = 10
BATCH_SIZE = 4

# %%
# -- Create some data to train on

def get_x(num_examples, num_inputs):
    return torch.randn(num_examples, num_inputs)


def get_y(X):
    y = 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2
    noise = 0.01 * torch.randn(len(X))
    return (y + noise).reshape(len(X), 1)

# %%
# -- Split train / eval data

X_train = get_x(9000, NUM_INPUTS)
y_train = get_y(X_train)

X_eval = get_x(1000, NUM_INPUTS)
y_eval = get_y(X_eval)

X_test = get_x(1000, NUM_INPUTS)

# %%
# -- Create a data loader that feeds batches

train_data = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
eval_data = DataLoader(TensorDataset(X_eval, y_eval), batch_size=BATCH_SIZE, shuffle=False)

# %%
# -- Define model

# layer 1
net = torch.nn.Linear(2, 1)

# %%
# -- Initialize parameters

torch.nn.init.normal_(net.weight)

# %%
# -- Define loss function and optimizer

loss_fn = F.mse_loss
trainer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

t0 = time.time()

loss_sequence = []
for epoch in range(NUM_EPOCHS):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        output = net(data)
        loss = loss_fn(output, label)
        loss.backward()
        trainer.step()
        trainer.zero_grad()
        cumulative_loss += torch.mean(loss).detach().numpy()
    print("Epoch {}, loss: {}".format(epoch, cumulative_loss / len(X_train)))
    loss_sequence.append(cumulative_loss)

print("Elapsed time: {:0.2f} seconds".format(time.time() - t0))

# %%
# -- Plot training loss on a log scale

plt.plot(np.log10(loss_sequence))
plt.savefig("lr_torch_loss.png")
# plt.show()

# %%
# -- Predict y for the test data and plot the result as a point cloud

y_pred = net(X_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0].detach().numpy(), X_test[:, 1].detach().numpy(), y_pred.detach().numpy())
plt.savefig("lr_torch_test.png")
# fig.show()

# %%
# -- Take a peak into the trained weights and bias

for param in net.parameters():
    print(param.data)
