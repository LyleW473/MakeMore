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


# Building training dataset:
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
print(X)
print(Y)

g = torch.Generator().manual_seed(2147483647)

# 27 possible characters (a - z + .)
# C = embedding look-up table 
C = torch.randn((27, 2), generator = g) # Each of the 27 characters will have a 2-D embedding

# Hidden layer (= H) :
W1 = torch.randn((6, 100), generator = g) # 6 inputs because (3 x 2 dimensional embeddings), Neurons = 100 (subject to change)
B1 = torch.randn(100, generator = g) # 100 biases

# Output layer:
W2 = torch.randn((100, 27), generator = g) # 27 possible outputs
B2 = torch.randn(27, generator = g)

# All parameters
parameters = [C, W1, B1, W2, B2]
for p in parameters:
    p.requires_grad = True


steps = 10000

# Learning rate tweaking:
learning_rate_exponents = torch.linspace(-3, 0, steps) 
learning_rates = 10 ** learning_rate_exponents # 0.001 learning rate --- > 10^^-3, 1 learning rate ---> 10^0

learning_rate_i = [] # Used learning rates
losses_i = [] # Losses for each learning rate

for i in range(steps):

    # Generate mini batch (Stochastic gradient descent for faster convergence to find a local minimum to minimise the loss)
    mini_b_idxs = torch.randint(0, X.shape[0], (32,)) # Generate indexes between 0 and X.shape[0], 32 indexes inside the list (Chooses 32 examples out of the 228146 examples in the dataset)

    # Forward pass:
    # C[X] = entire data set, C[X[mini_b_idxs]] = training on mini batch

    embedding = C[X[mini_b_idxs]] # Shape = torch.Size([32, 3, 2]) Note: First number = number of examples in X (Changes with input size, i.e. words + length of words)

    # Convert [32, 3, 2] -- > [32, 6] for matrix multiplication with weights and biases [Use torch.cat]
    # All of these 3 have a shape of [32, 2]

    # Method 1:(Creates new memory for concatenation)
    # torch.cat([embedding[:, 0, :], embedding[:, 1, :], embedding[:, 2, :]], dim = 1) # Concatenate across dimension 1 (the columns)

    # Method 2: (Creates new memory for concatenation)
    # torch.cat(torch.unbind(embedding, 1), dim = 1)) # Same as the code above but if block_size changes from 3, the code above would not work

    # Method 3: (Most efficient)
    # embedding.view(32, 6) @ W1 + B1

    H = torch.tanh(embedding.view(embedding.shape[0], 6) @ W1 + B1) # Tanh to get numbers between -1 and 1 # H.shape = [num_examples, num_neurons] (num_neurons for each example)

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
    loss = F.cross_entropy(logits, Y[mini_b_idxs])


    # Backpropagation:

    # Zero-grad
    for p in parameters:
        p.grad = None
        

    loss.backward()

    # Update weights
    learning_rate = learning_rates[i]

    for p in parameters:
        # p.data += -(learning_rate * p.grad)
        p.data += -(0.1 * p.grad)
    
    # # Track stats:
    # learning_rate_i.append(learning_rate_exponents[i])
    # losses_i.append(loss.item())

    print(loss.item())

# Finding a good initial learning rate:
# # Plot learning rate exponents on x axis, losses on y axis 
# # (Using the plot, look for a learning rate exponent such that the loss is minimised) [In this case, it is around 0.1]
# plt.plot(learning_rate_i, losses_i)
# plt.show()