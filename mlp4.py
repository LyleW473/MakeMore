# Exercises from: https://www.youtube.com/watch?v=q8SA3rM6ckI

import torch
import torch.nn.functional as F
from string import ascii_lowercase as string_ascii_lowercase

# Load words
words = open("names.txt", "r").read().splitlines()

# Build look-up tables for all characters:
s_to_i = {l:index + 1 for index, l in enumerate(list(string_ascii_lowercase))}
s_to_i["."] = 0
i_to_s = {index:l for l, index in s_to_i.items()}

# Building dataset + splitting the dataset into splits: (80%, 10%, 10% split)
def build_dataset(words, block_size):

    X, Y = [], []
    
    for w in words:

        """
        Outputs:
        ... ---> e
        ..e ---> m
        .em ---> m
        emm ---> a
        mma ---> .
        """
        context = [0] * block_size
        for char in w + ".":
            idx = s_to_i[char]
            X.append(context)
            Y.append(idx)
            context = context[1:] + [idx]

    X = torch.tensor(X)
    Y = torch.tensor(Y) 

    return X, Y


# Splitting data into splits
import random
random.seed(42)
random.shuffle(words) 
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

block_size = 6 

Xtr, Ytr = build_dataset(words[0:n1], block_size = block_size) 
Xdev, Ydev = build_dataset(words[n1:n2], block_size = block_size)
Xte, Yte = build_dataset(words[n2:], block_size = block_size)

print(f"Train: X: {Xtr.shape} Y: {Ytr.shape}")
print(f"Dev: X: {Xdev.shape} Y: {Ydev.shape}")
print(f"Test: X: {Xte.shape} Y: {Yte.shape}")

# Utility function used to compare manual gradients to PyTorch gradients
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f"{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}")


# Creating layers:
g = torch.Generator().manual_seed(2147483647)
n_embedding = 10 
n_hidden = 200
fan_in = n_embedding * block_size
standard_deviation = ((5/3) / (fan_in ** 0.5)) 

C = torch.randn((27, n_embedding), generator = g)

# Hidden
W1 = torch.randn((fan_in, n_hidden), generator = g) * standard_deviation
B1 = torch.randn(n_hidden, generator = g) * 0.1 # Usually not required when a batch normalisation layer is used

# Output
W2 = torch.randn((n_hidden, 27), generator = g) * 0.1
B2 = torch.randn(27, generator = g) * 0.1

# Batch normalisation parameters
bn_gain = torch.ones((1, n_hidden)) * 0.1 + 1.0
bn_bias = torch.zeros((1, n_hidden)) * 0.1
epsilon = 1e-5

# Note: Parameters are not initialised in the standard ways

# All parameters
parameters = [C, W1, B1, W2, B2, bn_gain, bn_bias]
for p in parameters:
    p.requires_grad = True

print(f"Total number of parameters: {sum(p.nelement() for p in parameters)}")

steps = 200000
batch_size = 32

# Training

for i in range(steps):

    # Generate mini batch
    mini_b_idxs = torch.randint(0, Xtr.shape[0], (batch_size,), generator = g)

    # Forward pass:
    embedding = C[Xtr[mini_b_idxs]]
    embedding_concat = embedding.view(embedding.shape[0], -1)

    # Linear layer 1
    h_pre_batch_norm = embedding_concat @ W1

    # ----------------------------------------------
    # Batch normalisation layer

    bn_mean_i = 1 / (batch_size * h_pre_batch_norm.sum(0, keepdim = True))
    bn_diff = h_pre_batch_norm - bn_mean_i
    bn_diff2 = bn_diff ** 2

    bn_variance = 1 / ((batch_size - 1) * (bn_diff2).sum(0, keepdim = True)) # Bessel's correction (dividing by n-1 not n)
    bn_variance_inv = (bn_variance + epsilon) ** -0.5
    bn_raw = bn_diff * bn_variance_inv
    h_pre_activation = (bn_gain * bn_raw) + bn_bias

    # ----------------------------------------------
    # Hidden layer

    H = torch.tanh(h_pre_activation) 

    # ----------------------------------------------
    # Output layer (Linear layer 2)

    logits = H @ W2 + B2
    
    # ----------------------------------------------
    # Cross entropy loss (Same as loss = F.cross_entropy(logits, Ytr[mini_b_idxs]))

    logit_maxes = logits.max(1, keepdim = True).values
    norm_logits = logits - logit_maxes # Subtract maxes for numerical stability
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdims = True)
    counts_sum_inv = counts_sum ** -1
    probs = counts * counts_sum_inv
    log_probs = probs.log()
    loss = -log_probs[range(batch_size), Ytr[mini_b_idxs]].mean()

    # ----------------------------------------------
    # Backpropagation

    # Zero-grad
    for p in parameters:
        p.grad = None

    # Retain gradients of all intermediate variables
    for t in [
            log_probs, probs, counts_sum_inv, counts_sum, counts, norm_logits, logit_maxes, logits, H, h_pre_activation, bn_raw, bn_variance_inv, 
            bn_variance, bn_diff2, bn_diff, bn_mean_i, h_pre_batch_norm, embedding_concat, embedding]:
        t.retain_grad()
    
    loss.backward()