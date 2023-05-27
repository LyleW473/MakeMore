import torch
from string import ascii_lowercase as string_ascii_lowercase
import torch.nn.functional as F

# Neural network approach to bigram (The final loss after training should be roughly the same as the loss produced in bigram.py)

# Notes:
# - Goal is to adjust the weights such that the NN can output an accurate probability for each possible second character, describing the likelihood for this character to come after an input character.
#   - This in turn would minimise the negative log likelihood (the loss)
# probabilities.shape should be of size: [num examples for each input, possible outputs]
# The size of the loss measures the quality of the neural net (low loss means high quality)


# 1. Creating dataset
# Load dataset
words = open("names.txt", "r").read().splitlines()

# Create training set of the bigrams (x, y) x = inputs, y = targets:
xs, ys = [], []

# Build look-up table (From letter to index)
s_to_i = {l:index + 1 for index, l in enumerate(list(string_ascii_lowercase))}
s_to_i["."] = 0

for w in words:
    chars = ["."] + list(w) + ["."] # "emma" = [".","e", "m", "m", "a", "."], "." = start or end of the string
    for i in range(0, len(chars) - 1):
        index_1 = s_to_i[chars[i]] # First char
        index_2 = s_to_i[chars[i + 1]] # Second char

        # Increase frequency of char2 coming after char1
        xs.append(index_1)
        ys.append(index_2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num_examples = xs.nelement()

# 2. Initialising the network:

# Initialise weights for 27 neurons with each neuron receiving 27 inputs
generator = torch.Generator().manual_seed(2147483647)
W = torch.randn(size = (27, 27), generator = generator, requires_grad = True) # Second num = number of neurons

# 3. Gradient descent:
for i in range(100):

    # Forward pass:

    # One-hot encoding to convert inputs into vectors (Row = current example, Column = 1 if this is the "current_example" column else 0), which can then be passed to the neural net
    # Note: num_classes = 27 as there are 27 possible inputs, (the alphabet and the special char ".")
    xenc = F.one_hot(xs, num_classes = 27).float() # Convert to floats for the neural net

    # For each input example, output a probability distribution which states the prediction of the NN as to how likely each of the 27 characters are likely to come next after the input example character
    # Multiply weights by inputs (Matrix multiplication)
    logits = xenc @ W # Predict log-counts

    # Softmax activation function:
    counts = logits.exp() # Equivalent to the counts for each pattern in bigram.py (To convert log-counts into something that looks like counts)
    probabilities = counts / counts.sum(1, keepdims = True) # Normalise rows to get the probability distribution for each row
    
    # Finding loss:

    # probabilities[torch.arange(5), ys] = probabilities that the NN outputs to the correct next character e.g. for "em", it outputs the probabilities that "m" will come after "e" [But for all input examples]
    # .log() to find the log likelihood
    # "-" to find the negative log likelihood
    # .mean() to find the average negative log likelihood

    # + (W **2).mean() = Regulurisation loss for model smoothing to achieve a more uniform probability distribution by incentivising weights to be 0
    #   - Achieves 0 loss if W is exactly 0, but for non-zero numbers, more loss is accumulated
    #   - Adjusting the multiplier will determine the regularisation strength (Increasing = smoother)
    loss = -(probabilities[torch.arange(num_examples), ys].log().mean()) + (0.01 * (W ** 2).mean())
     


    # Backward pass:

    # Set gradients to be 0, in Pytorch, setting to None is more efficient and will automatically assign to 0
    W.grad = None 

    # Set new gradients gradients
    # Each W.grad[i][j] states the influence of that weight on the loss function
    loss.backward()

    # Update weights to minimise loss
    W.data += - (60 * W.grad)

    print(f"Loss: {loss.item()}")