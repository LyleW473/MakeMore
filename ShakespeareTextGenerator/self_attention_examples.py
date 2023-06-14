# Examples of self-attention

import torch
from torch.nn import functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 2 # Batch, Time(i.e. token number), Channels
x = torch.randn(B, T, C)

# Self-attention (weighted aggregation)
# - Set xbag_of_words[b, t] as the mean of x[b:t + 1] where b is the batch index and t is the current time step (Takes the mean of previous context and itself)
# - All methods produce the same result
# - The goal is that tokens in the past cannot "communicate" with tokens in the future (as we are trying to predict future tokens)

# Set the weights as:
# [[1., 0., 0.]
# [1., 1., 0.]
# [1., 1., 1.]]
weights = torch.tril(torch.ones(T, T))

# Method 1: [Simple for loop]
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t + 1]
        xbow[b,t] = torch.mean(xprev, 0)


# Method 2: [Matrix multiplication]
# Set weights as the mean / average of the past context
# weights = weights / weights.sum(1, keepdim = True)
# xbow2= weights @ x # (T, T) @ (B, T, C) --> (B, T, T) @ (B, T, C) ---> Creates (B, T, C)


# Method 3: [Matrix multiplication + softmax]
tril = torch.tril(torch.ones(T, T))
weights = torch.zeros(T, T)
weights = weights.masked_fill(tril == 0, float("-inf")) # Wherever there is a 0 in tril, make it negative infinity [Ensures that past tokens cannot communicate with future tokens]
weights = F.softmax(weights, dim = -1) # Normalise each weight
xbow3 = weights @ x

# print(torch.allclose(xbow, xbow3))