import numpy as np
import pickle
import os

# --- Data loading ---
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_batch(batch_filename):
    batch = unpickle(batch_filename)
    X = batch[b'data']
    Y = np.array(batch[b'labels'])
    X = X.reshape(-1, 3, 32, 32).astype(np.float32)
    return X, Y

def load_cifar10(data_dir):
    xs, ys = [], []
    for i in range(1, 6):
        filename = os.path.join(data_dir, f'data_batch_{i}')
        X, Y = load_cifar10_batch(filename)
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)
    Y_train = np.concatenate(ys)
    X_test, Y_test = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    return X_train, Y_train, X_test, Y_test

# --- Utility functions ---
def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(probs, labels):
    N = probs.shape[0]
    correct_logprobs = -np.log(probs[range(N), labels] + 1e-8)
    return np.sum(correct_logprobs) / N

def accuracy(preds, labels):
    return np.mean(preds == labels)

# --- Layers ---
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        self.b = np.zeros(out_channels)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        F, _, HH, WW = self.W.shape
        pad = self.padding
        stride = self.stride
        H_out = (H + 2 * pad - HH) // stride + 1
        W_out = (W + 2 * pad - WW) // stride + 1
        x_padded = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
        out = np.zeros((N, F, H_out, W_out))
        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        window = x_padded[n, :, h_start:h_start+HH, w_start:w_start+WW]
                        out[n, f, i, j] = np.sum(window * self.W[f]) + self.b[f]
        self.x_padded = x_padded
        return out

    def backward(self, dout):
        N, C, H, W = self.x.shape
        F, _, HH, WW = self.W.shape
        pad = self.padding
        stride = self.stride
        H_out = (H + 2 * pad - HH) // stride + 1
        W_out = (W + 2 * pad - WW) // stride + 1
        dx = np.zeros_like(self.x_padded)
        self.dW.fill(0)
        self.db.fill(0)
        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        window = self.x_padded[n, :, h_start:h_start+HH, w_start:w_start+WW]
                        self.dW[f] += dout[n, f, i, j] * window
                        self.db[f] += dout[n, f, i, j]
                        dx[n, :, h_start:h_start+HH, w_start:w_start+WW] += dout[n, f, i, j] * self.W[f]
        dx = dx[:, :, pad:pad+self.x.shape[2], pad:pad+self.x.shape[3]]
        return dx

class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        N, C, H, W = x.shape
        HH, WW = self.kernel_size, self.kernel_size
        H_out = (H - HH) // self.stride + 1
        W_out = (W - WW) // self.stride + 1
        out = np.zeros((N, C, H_out, W_out))
        self.x = x
        self.argmax = np.zeros_like(out, dtype=np.int32)
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        window = x[n, c, h_start:h_start+HH, w_start:w_start+WW]
                        out[n, c, i, j] = np.max(window)
                        self.argmax[n, c, i, j] = np.argmax(window)
        return out

    def backward(self, dout):
        N, C, H, W = self.x.shape
        HH, WW = self.kernel_size, self.kernel_size
        H_out = (H - HH) // self.stride + 1
        W_out = (W - WW) // self.stride + 1
        dx = np.zeros_like(self.x)
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        idx = self.argmax[n, c, i, j]
                        h_idx = h_start + idx // WW
                        w_idx = w_start + idx % WW
                        dx[n, c, h_idx, w_idx] += dout[n, c, i, j]
        return dx

class Flatten:
    def forward(self, x):
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.orig_shape)

class Dense:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2. / in_dim)
        self.b = np.zeros(out_dim)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.W.T

# --- Model ---
class SimpleCNN:
    def __init__(self):
        self.conv1 = Conv2D(3, 16, 3, stride=1, padding=1)
        self.relu1 = relu
        self.pool1 = MaxPool2D(2, 2)
        self.conv2 = Conv2D(16, 32, 3, stride=1, padding=1)
        self.relu2 = relu
        self.pool2 = MaxPool2D(2, 2)
        self.flatten = Flatten()
        self.fc1 = Dense(8*8*32, 64)
        self.relu3 = relu
        self.fc2 = Dense(64, 10)

    def forward(self, x):
        x = self.conv1.forward(x)
        self.conv1_out = x
        x = self.relu1(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        self.conv2_out = x
        x = self.relu2(x)
        x = self.pool2.forward(x)
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu3(x)
        self.fc1_out = x  # right after relu
        x = self.fc2.forward(x)
        return x

    def backward(self, dout):
        dout = self.fc2.backward(dout)
        dout = relu_grad(self.fc1_out) * dout
        dout = self.fc1.backward(dout)
        dout = self.flatten.backward(dout)
        dout = self.pool2.backward(dout)
        dout = relu_grad(self.conv2_out) * dout
        dout = self.conv2.backward(dout)
        dout = self.pool1.backward(dout)
        dout = relu_grad(self.conv1_out) * dout
        dout = self.conv1.backward(dout)
        return dout

    def params_and_grads(self):
        for layer in [self.conv1, self.conv2, self.fc1, self.fc2]:
            yield layer.W, layer.dW
            yield layer.b, layer.db

# --- Optimizer ---
class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
        self.params = params

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# --- Training ---
def train(model, optimizer, X_train, Y_train, X_val, Y_val, epochs=10, batch_size=64):
    for epoch in range(epochs):
        idx = np.random.permutation(len(X_train))
        X_train, Y_train = X_train[idx], Y_train[idx]
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            Y_batch = Y_train[i:i+batch_size]
            logits = model.forward(X_batch)
            probs = softmax(logits)
            loss = cross_entropy_loss(probs, Y_batch)
            dout = probs
            dout[range(len(Y_batch)), Y_batch] -= 1
            dout /= len(Y_batch)
            model.backward(dout)
            # update
            params = [p for p, _ in model.params_and_grads()]
            grads = [g for _, g in model.params_and_grads()]
            optimizer.step(grads)
        # Evaluate
        val_logits = model.forward(X_val)
        val_preds = np.argmax(val_logits, axis=1)
        val_acc = accuracy(val_preds, Y_val)
        print(f"Epoch {epoch+1}, Val Accuracy: {val_acc*100:.2f}%")

# --- Main ---
if __name__ == "__main__":
    # Download and extract CIFAR-10 python version to ./cifar-10-batches-py
    X_train, Y_train, X_test, Y_test = load_cifar10('./cifar-10-batches-py')
    # Normalize
    X_train /= 255.0
    X_test /= 255.0
    # Use a validation split
    X_val, Y_val = X_train[-5000:], Y_train[-5000:]
    X_train, Y_train = X_train[:-5000], Y_train[:-5000]
    model = SimpleCNN()
    params = [p for p, _ in model.params_and_grads()]
    optimizer = Adam(params, lr=1e-3)
    train(model, optimizer, X_train, Y_train, X_val, Y_val, epochs=10, batch_size=64)
    # Test accuracy
    test_logits = model.forward(X_test)
    test_preds = np.argmax(test_logits, axis=1)
    test_acc = accuracy(test_preds, Y_test)
    print(f"Test Accuracy: {test_acc*100:.2f}%")

