from mxnet import nd
from mxnet import autograd

# %%
# Helper function 
def log_info(w, loss):
    print(f"Epoch: {epoch}, Weight: {w[0][0].asscalar():.3f}, Loss: {loss[0].asscalar():.3f}, dW: {w.grad[0][0].asscalar():.3f}")

# %% 
# Configuration
EPOCHS = 50
LR = 0.005

# %% 
# Target function to train:
# f: y = xW = 2x
#
# Data x and label y
x = nd.array([[-2], [-1], [0], [1], [2], [3], [4], [5], [6], [7], [8]])
y = 2*x

# %%
# Trainable parameter
W = nd.array([[0.01]], dtype="float32") 
W.attach_grad()

# %% 
# Network as a forward function
def forward(x):
    return nd.dot(x, W)

# %% 
# Loss function: MSE
def loss_fn(y_pred, y):
    return nd.mean(nd.square(y_pred - y))

# %%
# Training loop
for epoch in range(EPOCHS):
    with autograd.record():
        # Compute f(x) = Wx
        y_pred = forward(x)
        # Compute loss
        loss = loss_fn(y_pred, y)
    # Compute dL/dW 
    loss.backward()
    # Show intermediate values to screen
    log_info(W, loss)
    # Update weights
    W -= LR * W.grad
    
# %%
# Test the model: f(5) = 2*5 = 10
x_test = nd.array([[5]], dtype="float32")
print(f"f(5) = {forward(x_test)}")

