from imports import *


def init_params(i_sz, h1, h2, o_sz):
    W1 = np.random.randn(i_sz,   h1) * np.sqrt(2.0 / i_sz)
    W2 = np.random.randn(  h1,   h2) * np.sqrt(2.0 /   h1)
    W3 = np.random.randn(  h2, o_sz) * np.sqrt(2.0 /   h2)

    b1 = np.zeros((1,   h1))
    b2 = np.zeros((1,   h2))
    b3 = np.zeros((1, o_sz))

    return W1, b1, W2, b2, W3, b3

def relu(Z):
    return np.maximum(0, Z)

def relu_gate(Z):
    return (Z > 0).astype('float')    # 0 / 1 : relu derivative

def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def forward(X_batch, W1, b1, W2, b2, W3, b3):
    Z1 = X_batch @ W1 + b1      # (100, 128)
    A1 = relu(Z1)               # (100, 128)

    Z2 = A1 @ W2 + b2           # (100, 64)
    A2 = relu(Z2)               # (100, 64)

    Z3 = A2 @ W3 + b3           # (100, 10)
    A3 = softmax(Z3)            # (100, 10)  = Ŷ

    cache = (X_batch, Z1, A1, Z2, A2, Z3, A3)
    return A3, cache

def compute_loss(Y_hat, Y_batch):
    m = Y_batch.shape[0]
    Y_hat_clipped = np.clip(Y_hat, 1e-12, 1 - 1e-12)
    loss = -np.sum(Y_batch * np.log(Y_hat_clipped)) / m
    return loss

def backward(Y_batch, cache, W2, W3):
    X_batch, Z1, A1, Z2, A2, Z3, A3 = cache
    m = Y_batch.shape[0]

    dZ3 = (A3 - Y_batch) / m       # (100, 10)   entry point
    dW3 = A2.T @ dZ3               # (64,  10)
    db3 = np.sum(dZ3, axis=0, keepdims=True)  # (1, 10)

    dA2 = dZ3 @ W3.T               # (100, 64)
    dZ2 = dA2 * relu_gate(Z2)      # (100, 64)   ReLU gate
    dW2 = A1.T @ dZ2               # (128, 64)
    db2 = np.sum(dZ2, axis=0, keepdims=True)  # (1, 64)

    dA1 = dZ2 @ W2.T               # (100, 128)
    dZ1 = dA1 * relu_gate(Z1)      # (100, 128)  ReLU gate
    dW1 = X_batch.T @ dZ1          # (784, 128)
    db1 = np.sum(dZ1, axis=0, keepdims=True)  # (1, 128)

    grads = (dW1, db1, dW2, db2, dW3, db3)
    return grads

def init_adam(W1, b1, W2, b2, W3, b3):
    mW1 = np.zeros_like(W1);  vW1 = np.zeros_like(W1)
    mb1 = np.zeros_like(b1);  vb1 = np.zeros_like(b1)
    mW2 = np.zeros_like(W2);  vW2 = np.zeros_like(W2)
    mb2 = np.zeros_like(b2);  vb2 = np.zeros_like(b2)
    mW3 = np.zeros_like(W3);  vW3 = np.zeros_like(W3)
    mb3 = np.zeros_like(b3);  vb3 = np.zeros_like(b3)
    return (mW1,vW1, mb1,vb1, mW2,vW2, mb2,vb2, mW3,vW3, mb3,vb3)

def adam_update(params, grads, adam_buffers, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):

    W1, b1, W2, b2, W3, b3                          = params
    dW1, db1, dW2, db2, dW3, db3                    = grads
    mW1,vW1, mb1,vb1, mW2,vW2, mb2,vb2, mW3,vW3, mb3,vb3 = adam_buffers

    def step(param, grad, m, v):
        m = beta1 * m + (1 - beta1) * grad          # momentum
        v = beta2 * v + (1 - beta2) * grad**2       # variance
        m_hat = m / (1 - beta1**t)                  # bias correction
        v_hat = v / (1 - beta2**t)                  # bias correction
        param = param - (lr / (np.sqrt(v_hat) + eps)) * m_hat
        return param, m, v

    W1, mW1, vW1 = step(W1, dW1, mW1, vW1)
    b1, mb1, vb1 = step(b1, db1, mb1, vb1)
    W2, mW2, vW2 = step(W2, dW2, mW2, vW2)
    b2, mb2, vb2 = step(b2, db2, mb2, vb2)
    W3, mW3, vW3 = step(W3, dW3, mW3, vW3)
    b3, mb3, vb3 = step(b3, db3, mb3, vb3)

    params       = (W1, b1, W2, b2, W3, b3)
    adam_buffers = (mW1,vW1, mb1,vb1, mW2,vW2, mb2,vb2, mW3,vW3, mb3,vb3)
    return params, adam_buffers

def accuracy(X, Y, W1, b1, W2, b2, W3, b3):
    Y_hat, _    = forward(X, W1, b1, W2, b2, W3, b3)
    preds       = np.argmax(Y_hat, axis=1)
    labels      = np.argmax(Y,     axis=1)
    return np.mean(preds == labels)
