# Py-torch-ified MLP with batch normalisation

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from string import ascii_lowercase as string_ascii_lowercase

g = torch.Generator().manual_seed(2147483647)

# Linear layer
class Linear:

    def __init__(self, fan_in, fan_out, bias = True): # fan_in = no.inputs, fan_out = no.outputs
        self.weight = torch.randn((fan_in, fan_out), generator = g) / (fan_in ** 0.5)
        self.bias = torch.zeros(fan_out) if bias else None  

    def __call__(self, x):

        # Wx
        self.out = x @ self.weight

        # + b
        if self.bias != None:
            self.out += self.bias

        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

# Batch normalisation layer
class BatchNorm1d:

    def __init__(self, dim, eps = 1e-5, momentum = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True # When training, update the running means, if evaluating, use the existing running means
        
        # Parameters (Trained with backpropagation)
        self.gamma = torch.ones(dim) # Batch normalisation gain
        self.beta = torch.zeros(dim) # Batch normalisation bias

        # Buffers (Trained with a running "momentum update")
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    
    def __call__(self, x):

        # Training
        if self.training:
            xmean = x.mean(0, keepdim = True) # Batch mean
            xvar = x.var(0, keepdim = True) # Batch variance
        
        # Evaluation
        else:
            xmean = self.running_mean
            xvar = self.running_var

        # Normalise to unit variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        # Update buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = ((1 - self.momentum) * self.running_mean) + (self.momentum * xmean)
                self.running_var = ((1 - self.momentum) * self.running_var) + (self.momentum * xvar)

        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]

# Tanh activation
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []


# Load words
words = open("names.txt", "r").read().splitlines()

# Build look-up tables for all characters:
s_to_i = {l:index + 1 for index, l in enumerate(list(string_ascii_lowercase))}
s_to_i["."] = 0
i_to_s = {index:l for l, index in s_to_i.items()}

# Building dataset + splitting the dataset into splits:
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

import random
random.seed(42)
random.shuffle(words) 
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

block_size = 3

Xtr, Ytr = build_dataset(words[0:n1], block_size = block_size) 
Xdev, Ydev = build_dataset(words[n1:n2], block_size = block_size)
Xte, Yte = build_dataset(words[n2:], block_size = block_size)

print(f"Train: X: {Xtr.shape} Y: {Ytr.shape}")
print(f"Dev: X: {Xdev.shape} Y: {Ydev.shape}")
print(f"Test: X: {Xte.shape} Y: {Yte.shape}")


# Creating layers:
n_dimensions = 10 # Number of columns for each character embedding vector
n_hidden = 100 # In the hidden layer
C = torch.randn((27, n_dimensions), generator = g)

layers = [

        Linear(fan_in = (n_dimensions * block_size), fan_out = n_hidden), 
        Tanh(),

        Linear(fan_in = n_hidden, fan_out = n_hidden),
        Tanh(),

        Linear(fan_in = n_hidden, fan_out = n_hidden),
        Tanh(),

        Linear(fan_in = n_hidden, fan_out = n_hidden),
        Tanh(),
        
        Linear(fan_in = n_hidden, fan_out = n_hidden),
        Tanh(),

        Linear(fan_in = n_hidden, fan_out = 27),

        ]   

# Initialising layers:
with torch.no_grad():

    # Make the last layer less confident
    layers[-1].weight *= 0.1

    # Apply gain to all other layers that are Linear:
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5/3 # The ideal gain for a tanh unlinearity (via. Kai Ming initialisation paper)
    
# Embedding lookup table + all the parameters in all of the layers of the NN
parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(f"Total number of parameters: {sum(p.nelement() for p in parameters)}")

for p in parameters:
    p.requires_grad = True



# Training:
steps = 200_000
mini_batch_size = 32
losses_i = []
ud = [] # Update to grad:data ratio

for i in range(steps):

    # Generate mini batch
    mini_b_idxs = torch.randint(0, Xtr.shape[0], (mini_batch_size,), generator = g)

    # ----------------------------------------------
    # Forward pass

    embedding = C[Xtr[mini_b_idxs]] # Embed characters into vectors
    embedding_concat = embedding.view(embedding.shape[0], -1) # Concatenate all vectors
    
    # Batch normalisation, find activations, etc..
    for layer in layers:
        embedding_concat = layer(embedding_concat)

    # Softmax (Classication)
    loss = F.cross_entropy(embedding_concat, Ytr[mini_b_idxs]) # embedding_concat will be the logits at this point

    # ----------------------------------------------
    # Backpropagation

    for layer in layers:
        layer.out.retain_grad() # AFTER_DEBUG: Retain the gradients of all outputs after the forward pass

    # Zero-grad
    for p in parameters:
        p.grad = None

    # Fill in gradients
    loss.backward()

    # ----------------------------------------------
    # Update weights + learning rate

    learning_rate = 0.1 if i < (steps / 2) else 0.01

    for p in parameters:
        p.data += -(learning_rate * p.grad)
    
    # ----------------------------------------------
    # Track stats

    losses_i.append(loss.log10().item())
    with torch.no_grad():
        # learning_rate * p.grad).std() = Standard deviation of the update that was applied
        # p.data.std() = Standard deviation of the contents of the parameter
        # .log10() for a nicer visualisation
        # .item() to extract the float
        ud.append([((learning_rate * p.grad).std() / p.data.std()).log10().item() for p in parameters])
    
    if i == 1000:
        break # AFTER_DEBUG: Remove after optimisation

print(loss)

# Diagnostics:

# Activations distribution
# Note: Want to avoid activations from compressing on the x axis (a standard deviation that shrinks) [Shrinking to 0 on the x-axis]
# - The use of the "5/3" gain for the tanh activation counteracts this squashing property

plt.figure(figsize=(20,4)) # Width and height of the plot
legends = []
print("Activation distributions")
for i, layer in enumerate(layers[:-1]): # All layers except the last
    if isinstance(layer, Tanh):
        t = layer.out # Activations
        
        print("layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%" % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean() * 100))

        # Plots the distribution of activations in each layer (The number of values in each tensor that take on the values in the x axis)
        hy, hx = torch.histogram(t, density = True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i} ({layer.__class__.__name__}")

print()
plt.legend(legends)
plt.title("Activations distribution")
plt.show()

# Gradients distribution
# Note: Want to avoid shrinking gradients or exploding gradients (All layers should have roughly the same gradients)

plt.figure(figsize=(20,4))
legends = []
print("Gradient distributions")
for i, layer in enumerate(layers[:-1]): # All layers except the last
    if isinstance(layer, Tanh):
        t = layer.out.grad # Gradients

        print("layer %d (%10s): mean %+f, std %e" % (i, layer.__class__.__name__, t.mean(), t.std()))

        # Plots the distribution of gradients in each layer
        hy, hx = torch.histogram(t, density = True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i} ({layer.__class__.__name__}")

print()
plt.legend(legends)
plt.title("Gradients distribution")
plt.show()

# Weights:Gradient distribution

plt.figure(figsize=(20,4))
legends = []
print("Weights:Gradients distribution")
for i, p in enumerate(parameters):
    t = p.grad

    # Only weights
    if p.ndim == 2:
        
        # grad:data ratio = Scale of the gradient compared to the scale of the data
        # Note: If the gradient is too large compared to the data, this is a bad.
        print("weights %10s | mean %+f | std %e | grad:data ratio %e" % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))

        hy, hx = torch.histogram(t, density = True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"{i} {tuple(p.shape)}")

plt.legend(legends)
plt.title("Weights:Gradients distribution")
plt.show()

# Update:data ratio
# Note: Shows the rate at which each layer learns over time
# - Updates that are below -3.0 on the plot are 10^3 times smaller than the size of the data inside each tensor
plt.figure(figsize=(20,4))
legends = []

for i, p in enumerate(parameters):

    # Only weights
    if p.ndim == 2:
        # Plots the update ratios over time
        plt.plot([ud[j][i] for j in range(len(ud))])
        legends.append("param %d" % i)

plt.plot([0, len(ud)], [-3, -3], "k") # These ratios should be 1e-3 (Indicated on plot at -3 with a black line)
plt.legend(legends)
plt.show()