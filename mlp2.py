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

# Note: We want the mean and standard deviation for the inputs and pre-activations to be roughly the same to preserve the gaussian / normal distribution
fan_in = n_dimensions * block_size # Number of input elements
standard_deviation = ((5/3) / (fan_in ** 0.5)) # standard deviation = gain / sqrt(fan_in) [gain is 5/3 for tanh linearity based on Kai Ming paper]

W1 = torch.randn((fan_in, num_neurons), generator = g) * standard_deviation # Scaling down to prevent tanh saturation (Where at initialisation these weights may be in the flat regions of tanh, so learn slower)
# B1 = torch.randn(num_neurons, generator = g) * 0.01 # Scaling down to prevent tanh saturation (Where at initialisation these biases may be in the flat regions of tanh, so learn slower)

# Output
W2 = torch.randn((num_neurons, 27), generator = g) * 0.01 # Scale down at initialisation to get logits to be near 0 to prevent very high losses at initialisation
B2 = torch.randn(27, generator = g) * 0 # logits = H @ W2 + B2, initialise biases to 0 at initialisation

# Batch normalisation (bn = batch normalisation)
bn_gain = torch.ones((1, num_neurons))
bn_bias = torch.zeros((1, num_neurons))

# Running as it can be used at inference (testing / eval) without having to estimate it again after training (The replaced section) [Used to calibrate the batch normalisation]
bn_mean_running = torch.zeros((1, num_neurons)) # h_pre_activation will be unit gaussian, meaning the mean and std will be roughly 0
bn_std_running = torch.ones((1, num_neurons)) # h_pre_activation will be unit gaussian, meaning the std will be roughly 1

# All parameters
parameters = [C, W1, W2, B2, bn_gain, bn_bias]
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

    # ----------------------------------------------
    # Batch normalisation layer

    # To make activation states to be ROUGHLY gaussian (As in tanh, small numbers would result in inactivity but large numbers would result in saturation so gradient does not flow)

    h_pre_activation = embedding_concat @ W1 # Hidden layer pre-activation

    bn_mean_i = h_pre_activation.mean(0, keepdim = True) # Find mean across the 0th dimension (Mean over all examples in the batch)
    bn_std_i = h_pre_activation.std(0, keepdim = True) # Find standard deviation across the 0th dimension (std over all examples in the batch)

    # Note: Every neuron's firing rate will be unit gaussian on this mini-batch at initialisation
    # In backpropagation, bn_gain and bn_bias will be altered (used so that the neurons are not forced to be gaussian for every batch)
    # (h_pre_activation - bn_mean_i) / bn_std_i) = Normalise
    # bn_gain, bn_bias = scale and shift
    # Main purpose: Centers the batch to be unit gaussian and then offsetting / scaling by the bn_bias and bn_gain
    h_pre_activation = (bn_gain * ((h_pre_activation - bn_mean_i) / bn_std_i)) + bn_bias

    with torch.no_grad(): # Update without building a graph
        # Update running batch normalisation mean and standard deviation
        bn_mean_running = 0.999 * bn_mean_running + (0.001 * bn_mean_i)
        bn_std_running = 0.999 * bn_std_running + (0.001 * bn_std_i)

    # ----------------------------------------------
    # Hidden layer

    H = torch.tanh(h_pre_activation) 
    # ----------------------------------------------
    # Output layer

    logits = H @ W2 + B2

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

# Calibrate the batch normalisation at the end of training [Replaced with bn_mean_Running and bn_std_running]
# with torch.no_grad():
#     # Pass the training set through
#     embedding = C[Xtr]
#     embedding_concat = embedding.view(embedding.shape[0], - 1)
#     h_pre_activation = embedding_concat @ W1 + B1

#     # Measure the mean and standard deviation over the entire training set 
#     bn_mean = h_pre_activation.mean(0, keepdim = True)
#     bn_std = h_pre_activation.std(0, keepdim = True)

@torch.no_grad() # Disables gradient tracking
def split_loss(inputs, targets):

    embedding = C[inputs]
    embedding_concat = embedding.view(embedding.shape[0], - 1)

    h_pre_activation = embedding_concat @ W1

    # mean = h_pre_activation.mean(0, keepdim = True)
    # standard_deviation = h_pre_activation.std(0, keepdim = True)
    # h_pre_activation = (bn_gain * ((h_pre_activation - mean) / standard_deviation)) + bn_bias
    # By doing the following, you can input a single example into the network (Since the NN was trained on batches and will expect batches)
    h_pre_activation = (bn_gain * ((h_pre_activation - bn_mean_running) / bn_std_running)) + bn_bias

    H = torch.tanh(h_pre_activation)

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
            hidden = torch.tanh(embedding.view(1, -1) @ W1) # -1 will find the number of inputs automatically

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
