import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from string import ascii_lowercase as string_ascii_lowercase

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

g = torch.Generator().manual_seed(2147483647)

# Creating layers:
n_dimensions = 10 # Number of columns for each character embedding vector
C = torch.randn((27, n_dimensions), generator = g)

# Hidden
num_neurons = 200 # In the hidden layer
W1 = torch.randn((block_size * n_dimensions, num_neurons), generator = g) * 0.2 # Scaling down to prevent tanh saturation (Where at initialisation these weights may be in the flat regions of tanh, so learn slower)
B1 = torch.randn(num_neurons, generator = g) * 0.01 # Scaling down to prevent tanh saturation (Where at initialisation these biases may be in the flat regions of tanh, so learn slower)

# Output
W2 = torch.randn((num_neurons, 27), generator = g) * 0.01 # Scale down at initialisation to get logits to be near 0 to prevent very high losses at initialisation
B2 = torch.randn(27, generator = g) * 0 # logits = H @ W2 + B2, initialise biases to 0 at initialisation

# All parameters
parameters = [C, W1, B1, W2, B2]
for p in parameters:
    p.requires_grad = True

print(f"Total number of parameters: {sum(p.nelement() for p in parameters)}")

steps = 200000 # 300000

# Learning rate tweaking:

learning_rate_exponents = torch.linspace(-3, 0, steps) 
learning_rates = 10 ** learning_rate_exponents

learning_rate_i = []
losses_i = []

mini_batch_size = 32

# Training

for i in range(steps):

    # Generate mini batch
    mini_b_idxs = torch.randint(0, Xtr.shape[0], (mini_batch_size,))

    # Forward pass:
    embedding = C[Xtr[mini_b_idxs]] # Embed characters into vectors
    embedding_concat = embedding.view(embedding.shape[0], - 1) # Concatenate all vectors
    h_pre_activation = embedding_concat @ W1 + B1 # Hidden layer pre-activation
    H = torch.tanh(h_pre_activation) # Hidden layer
    logits = H @ W2 + B2 # Output layer

    # Softmax (Classication)
    loss = F.cross_entropy(logits, Ytr[mini_b_idxs])

    # Backpropagation:

    # Zero-grad
    for p in parameters:
        p.grad = None
    loss.backward()

    # Update weights
    learning_rate = 0.1 if i < (steps / 2) else 0.01

    for p in parameters:
        p.data += -(learning_rate * p.grad)
    
    losses_i.append(loss.log10().item())

print(f"TrainingLoss {loss.item()}")

# Plotting the loss over steps
plt.plot([i for i in range(steps)], losses_i)
plt.show()

# Plotting the activations in the hidden layer (White = True (In the flat tail of tanh, so the gradient would be vanishing), black = False)
# If an entire column is white (meaning that all input examples don't land in the active section of the tanh curve), this is a dead neuron, meaning that it will never learn (weights and biases would not change)
plt.figure(figsize = (20, 10))
plt.imshow(H.abs() > 0.99, cmap = "gray", interpolation = "nearest")
plt.show()

# Plotting the pre-activations of the hidden layer
plt.hist(h_pre_activation.view(-1).tolist(), 50)
plt.show()

# Plotting the activatons of the hidden layer
plt.hist(H.view(-1).tolist(), 50)
plt.show()


@torch.no_grad() # Disables gradient tracking
def split_loss(inputs, targets):
    embedding = C[inputs]
    H = torch.tanh(embedding.view(embedding.shape[0], n_dimensions * block_size) @ W1 + B1)
    logits = H @ W2 + B2
    loss = F.cross_entropy(logits, targets)
    return loss.item()

# Dev:
print(f"DevLoss:{split_loss(inputs = Xdev, targets = Ydev)}")

# Test:
print(f"TestLoss:{split_loss(inputs = Xte, targets = Yte)}")

def create_samples(num_samples, block_size, embedding_lookup_table):
    g = torch.Generator().manual_seed(2147483647 + 10)

    samples = []

    for _ in range(num_samples):
        word = ""
        context = [0] * block_size # Initialise with special case character "."

        while True:
            embedding = embedding_lookup_table[torch.tensor([context])] # [1, block_size, d]
            hidden = torch.tanh(embedding.view(1, -1) @ W1 + B1) # -1 will find the number of inputs automatically

            logits = hidden @ W2 + B2 # Find predictions
            probabilities = F.softmax(logits, dim = 1) # Exponentiates for counts and then normalises them (to sum to 1)

            # Generate the next character using the probabiltiy distribution outputted by the NN
            idx = torch.multinomial(probabilities, num_samples = 1, generator = g).item()
            
            # Update the context
            context = context[1:] + [idx]
            
            # Add character corresponding to the index
            word += i_to_s[idx]

            # Found the special character "."
            if idx == 0:
                break
        
        samples.append(word)

    return samples

print(f"Samples: {create_samples(num_samples = 30, block_size = block_size, embedding_lookup_table = C)}")
