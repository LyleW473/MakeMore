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

def build_dataset(words):
    block_size = 3 # Number of characters used to predict the next character
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

Xtr, Ytr = build_dataset(words[0:n1]) # 80% training split
Xdev, Ydev = build_dataset(words[n1:n2]) # 10% dev / validation split
Xte, Yte = build_dataset(words[n2:]) # 10% test split

print(Xtr.shape, Ytr.shape)
print(Xdev.shape, Ydev.shape)
print(Xte.shape, Yte.shape)

g = torch.Generator().manual_seed(2147483647)

# 27 possible characters (a - z + .)
# C = embedding look-up table 
C = torch.randn((27, 10), generator = g) # Each of the 27 characters will have a 2-D embedding

# Hidden layer (= H) :
W1 = torch.randn((30, 200), generator = g) # 6 inputs because (3 x 2 dimensional embeddings), Neurons = 300 (subject to change)
B1 = torch.randn(200, generator = g) # 300 biases

# Output layer:
W2 = torch.randn((200, 27), generator = g) # 27 possible outputs
B2 = torch.randn(27, generator = g)

# All parameters
parameters = [C, W1, B1, W2, B2]
for p in parameters:
    p.requires_grad = True

print(f"Total number of parameters: {sum(p.nelement() for p in parameters)}")

steps = 200000

# Learning rate tweaking:
learning_rate_exponents = torch.linspace(-3, 0, steps) 
learning_rates = 10 ** learning_rate_exponents # 0.001 learning rate --- > 10^^-3, 1 learning rate ---> 10^0

learning_rate_i = [] # Used learning rates
losses_i = [] # Losses for each learning rate

mini_batch_size = 32

for i in range(steps):

    # Generate mini batch (Stochastic gradient descent for faster convergence to find a local minimum to minimise the loss)
    mini_b_idxs = torch.randint(0, Xtr.shape[0], (mini_batch_size,)) # Generate indexes between 0 and X.shape[0], 32 indexes inside the list (Chooses 32 examples out of the 228146 examples in the dataset)

    # Forward pass:
    # C[X] = entire data set, C[X[mini_b_idxs]] = training on mini batch

    embedding = C[Xtr[mini_b_idxs]] # Shape = torch.Size([32, 3, 2]) Note: First number = number of examples in X (Changes with input size, i.e. words + length of words)

    # Convert [32, 3, 2] -- > [32, 6] for matrix multiplication with weights and biases [Use torch.cat]
    # All of these 3 have a shape of [32, 2]

    # Method 1:(Creates new memory for concatenation)
    # torch.cat([embedding[:, 0, :], embedding[:, 1, :], embedding[:, 2, :]], dim = 1) # Concatenate across dimension 1 (the columns)

    # Method 2: (Creates new memory for concatenation)
    # torch.cat(torch.unbind(embedding, 1), dim = 1)) # Same as the code above but if block_size changes from 3, the code above would not work

    # Method 3: (Most efficient)
    # embedding.view(32, 6) @ W1 + B1

    H = torch.tanh(embedding.view(embedding.shape[0], 30) @ W1 + B1) # Tanh to get numbers between -1 and 1 # H.shape = [num_examples, num_neurons] (num_neurons for each example)

    logits = H @ W2 + B2 # Find output in the form of logits

    # Softmax (Classication)
    # Negative loss likelihood, smaller = better performance
    
    # Method 1:
    # counts = logits.exp()
    # prob = counts / counts.sum(1, keepdims = True)

    # # prob[torch.arange(32), Y].shape creates a [num_example] tensor that states the probability predicted by the NN for the actual expected next letter for each actual expected next letter in Y
    # loss = -(prob[torch.arange(32), Y].log().mean()) # The loss

    # Method 2:
    # Reasons for usage of cross_entropy:
    # - Method 1 creates intermediary tensors in memory whereas cross_entropy does not
    # - Complex expressions are simplified for the backward pass
    # - Ensures that very positive numbers do not result in a probability of "nan" due to e^num being out of range
    # Y[mini_b_idxs] = Training on mini batch, Y = Training on entire dataset (All 228146 examples at once)
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

# Dev:
embedding = C[Xdev]
H = torch.tanh(embedding.view(embedding.shape[0], 30) @ W1 + B1)
logits = H @ W2 + B2
loss = F.cross_entropy(logits, Ydev)
print(f"DevLoss:{loss}")

# Test:
embedding = C[Xte]
H = torch.tanh(embedding.view(embedding.shape[0], 30) @ W1 + B1)
logits = H @ W2 + B2
loss = F.cross_entropy(logits, Yte)
print(f"TestLoss:{loss}")

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

print(f"Samples: {create_samples(num_samples = 30, block_size = 3, embedding_lookup_table = C)}")
