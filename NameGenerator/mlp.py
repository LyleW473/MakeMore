import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from string import ascii_lowercase as string_ascii_lowercase

# Load words
words = open("names.txt", "r").read().splitlines()


# Build look-up tables for all characters:

# Letter to index
s_to_i = {l:index + 1 for index, l in enumerate(list(string_ascii_lowercase))}
s_to_i["."] = 0

# Index to letter
i_to_s = {index:l for l, index in s_to_i.items()}


# Building dataset + splitting the dataset into splits:

def build_dataset(words, block_size):

    # block_size = Number of characters used to predict the next character

    X, Y = [], [] # X = inputs, Y = targets / labels
    
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

            #print("".join(i_to_s[i] for i in context), "--->", i_to_s[idx])

            context = context[1:] + [idx]

    X = torch.tensor(X) # X is a tensor of arrays containing the indexes of each character e.g. ['.', '.', 'e'] == [0, 0, 5]
    Y = torch.tensor(Y) # Y is a tensor of integers containing the index of the actual next character at i. (e.g. "m" comes after "..e", so the value at i will be 13)

    return X, Y

# Training, dev/validation split, test split: 80%, 10%, 10%
# Training = Used for optimising the parameters of the model with gradient descent
# Dev/validation = Used for development of the hyperparameters of the model
# Test = Testing the performance of the model (Smaller percentage so that the model does not become overfitted on the testing data)

import random
random.seed(42)
random.shuffle(words) 
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

block_size = 3 # round(sum(len(word) for word in words) / len(words))

Xtr, Ytr = build_dataset(words[0:n1], block_size = block_size) # 80% training split
Xdev, Ydev = build_dataset(words[n1:n2], block_size = block_size) # 10% dev / validation split
Xte, Yte = build_dataset(words[n2:], block_size = block_size) # 10% test split

print(f"Train: X: {Xtr.shape} Y: {Ytr.shape}")
print(f"Dev: X: {Xdev.shape} Y: {Ydev.shape}")
print(f"Test: X: {Xte.shape} Y: {Yte.shape}")

g = torch.Generator().manual_seed(2147483647)

# C = embedding look-up table 
n_dimensions = 10 # Each of the 27 characters will have a embedding with "n_dimensions"
vocab_size = 27 # 27 possible characters (Letters a - z and the "."" character)
C = torch.randn((vocab_size, n_dimensions), generator = g)

# Hidden layer (= H) :
num_neurons = 200
W1 = torch.randn((block_size * n_dimensions, num_neurons), generator = g) # (block x n_dimensinal), num_neurons for this layer
B1 = torch.randn(num_neurons, generator = g) # num_neurons biases

# Output layer:
W2 = torch.randn((num_neurons, vocab_size), generator = g) # 27 possible outputs
B2 = torch.randn(27, generator = g)

# All parameters
parameters = [C, W1, B1, W2, B2]
for p in parameters:
    p.requires_grad = True

print(f"Total number of parameters: {sum(p.nelement() for p in parameters)}")

steps = 300000

# Learning rate tweaking:
learning_rate_exponents = torch.linspace(-3, 0, steps) 
learning_rates = 10 ** learning_rate_exponents # 0.001 learning rate --- > 10^^-3, 1 learning rate ---> 10^0

learning_rate_i = [] # Used learning rates
losses_i = [] # Losses for each learning rate

mini_batch_size = 32

for i in range(steps):

    # Generate mini batch (Stochastic gradient descent for faster convergence to find a local minimum to minimise the loss)
    mini_b_idxs = torch.randint(0, Xtr.shape[0], (mini_batch_size,), generator = g) # Generate indexes between 0 and X.shape[0], 32 indexes inside the list (Chooses 32 examples out of the 228146 examples in the dataset)

    # Forward pass:
    # C[X] = entire data set, C[X[mini_b_idxs]] = Training on mini batch
    # Y[mini_b_idxs] = Training on mini batch, Y = Training on entire dataset (All 228146 examples at once)


    """  
    embedding.shape = [batch size, block size, number of dimensions in the embedding for each letter in the sequence]
    If mini_batch_size = 32, block_size = 3, n_dimensions = 10

    Xtr[mini_b_idxs].shape = [32, 3] (Finds the corresponding embedding vector for each letter in each sequence, for each sequence in the batch)
    embedding.shape = [32, 3, 10]
    """
    embedding = C[Xtr[mini_b_idxs]]

    """
    Convert [32, 3, 10] -- > [32, (3 * 10)] = [32, 30] for matrix multiplication with weights and biases

    Method 1:(Creates new memory for concatenation)
    torch.cat([embedding[:, 0, :], embedding[:, 1, :], embedding[:, 2, :]], dim = 1) # Concatenate across dimension 1 (the columns)

    Method 2: (Creates new memory for concatenation)
    - Produces the same results as the code above but if block_size changes from 3, the code above would not work
    torch.cat(torch.unbind(embedding, 1), dim = 1)) 

    Method 3: (Most efficient)
    embedding.view(32, 6) @ W1 + B1
    - Tanh is used as the activation function to get numbers between -1 and 1 
    - H.shape = [batch_size, number of neurons in this hidden layer]
    If batch_size = 32 and num_neurons = 200, H.shape = [32, 200]
    """
    H = torch.tanh(embedding.view(embedding.shape[0], n_dimensions * block_size) @ W1 + B1)
    
    # Output layer, find the output in the form of logits (Shape will be the number of neurons in the output layer, in this case being vocab_size)
    logits = H @ W2 + B2

    # Softmax (Classication)
    """
    - Negative loss likelihood, smaller values = better performance
    
    Method 1:
    counts = logits.exp()
    prob = counts / counts.sum(1, keepdims = True)

    - torch.arange(32) creates a tensor of shape [batch_size] from numbers 0 to batch_size
    - Ytr[mini_b_idxs] is a tensor of shape [batch_size]
    - prob[torch.arange(32), Ytr[mini_b_idxs]] finds the probability predicted by the NN for each letter coming next for each sequence in the batch (DEMO below)

    For a single example in a batch:
    prob[0] = A vector of length 27, containing the probability predicted for each character coming next in the sequence
    prob[0][y] = y is the ith character in the vector Y. if y = 5 (which would be the character "e"), this would find the probability assigned by the model that "e" will come next in the sequence
    loss = -(prob[torch.arange(32), Ytr[mini_b_idxs]].log().mean())


    Explanation for how the model learns from its predictions:
    This means that if the probability assigned by the model to the actual next character in the sequence was:
    - High = The loss would be lower
    - Low = The loss would be higher
    For example, let x = the probability assigned to a single example by the model, where loss = -mean(log(x) + ... + ...)
    If the probability assigned for a single example was 0.001, log(0.001) = -3, meaning that the loss generated from this example would be bigger
    Whereas, if the probability assigned for the example was 0.999, log(0.999) = -0.00043451177, generating a smaller loss
    Therefore, the model is encouraged to assign a higher probability to the actual next expected character in the sequence , via backpropagation, in order to minimise the loss


    Method 2:
    Reasons for usage of cross_entropy:
    - Method 1 creates intermediary tensors in memory whereas cross_entropy does not
    - Complex expressions are simplified for the backward pass
    - Ensures that very positive numbers do not result in a probability of "nan" due to e^num being out of range
    """
    loss = F.cross_entropy(logits, Ytr[mini_b_idxs])

    # Backpropagation:

    # Zero-grad
    for p in parameters:
        p.grad = None
    

    loss.backward()

    # Update weights
    learning_rate = 0.1 if i < (steps / 2) else 0.01

    for p in parameters:
        # p.data += -(learning_rate * p.grad)
        p.data += -(learning_rate * p.grad)
    
    # # Track stats:
    # learning_rate_i.append(learning_rate_exponents[i])
    losses_i.append(loss.log10().item())

print(f"TrainingLoss {loss.item()}")

# Finding a good initial learning rate:
# # Plot learning rate exponents on x axis, losses on y axis 
# # (Using the plot, look for a learning rate exponent such that the loss is minimised) [In this case, it is around 0.1]
# plt.plot(learning_rate_i, losses_i)
# plt.show()

# Plotting the loss over steps
plt.plot([i for i in range(steps)], losses_i)
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
            embedding = embedding_lookup_table[torch.tensor([context])] # [1, block_size, n_dimenions]
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
