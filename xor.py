# XOR value prediction

import numpy as np 
import time

# Variables
n_hidden = 10
n_in = 10

# Outputs
n_out = 10

# Sample data
n_sample = 300

# Hyperparameters
learning_rate = 0.01
momentum = 0.9

# non deterministic seeding
np.random.seed(0)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def tanh_prime(x):
    return 1 - np.tanh(x)**2

# Input data, transpose, layer 1, layer 2, biases
def train(x, t, V, W, E, bv, bw):
    # Forward -- matrix multiplication + biases
    A = np.dot(x, V) + bv
    Z = np.tanh(A)

    B = np.dot(Z, W) + bw
    Y = sigmoid(B)

    # Backward
    Ew = Y - t
    Ev = tanh_prime(A) * np.dot(W, Ew)

    # Predict out loss
    dW = np.outer(Z, Ew)
    dV = np.outer(x, Ev)

    # Cross entropy
    loss = -np.mean(t * np.log(Y) + (1 - t) * np.log(1 - Y))

    return loss, (dV, dW, Ev, Ew)

def predict(x, V, W, bv, bw):
    A = np.dot(x, V) + bv
    B = np.dot(np.tanh(A), W) + bw
    return (sigmoid(B) > 0.5).astype(int)

# Create layers
V = np.random.normal(scale=0.1, size=(n_in, n_hidden))
W = np.random.normal(scale=0.1, size=(n_hidden, n_out))

bv = np.zeros(n_hidden)
bw = np.zeros(n_out)

params = [V, W, bv, bw]

# Generate data
X = np.random.binomial(1, 0.5, (n_sample, n_in))
T = X ^ 1

# Training
for epoch in range(100):
    err = []
    upd = [0] * len(params)

    t0 = time.clock()
    # for each data point, update weights
    for i in range(X.shape[0]):
        loss,grad = train(X[i], T[i], *params, bw)
        #update loss
        for j in range(len(params)):
            params[j] -= upd[j]
        
        for j in range(len(params)):
            upd[j] = learning_rate * grad[j] * momentum * upd[j]
        
        err.append(loss)

    print('Epoch: %d, Loss: %.8f, Time: %.4fs'%(
        epoch, np.mean(err), time.clock()-t0
    ))

# Predict
x = np.random.binomial(1, 0.5, n_in)
print('XOR prediction')
print(x)
print(predict(x, *params))