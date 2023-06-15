# Examples of self-attention

import torch
import torch.nn as nn
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
weights = F.softmax(weights, dim = -1) # Exponentiate and normalise weight rows
xbow3 = weights @ x

# print(torch.allclose(xbow, xbow3))

# Method 4: [Self-attention]

tril = torch.tril(torch.ones(T, T))


# We don't want weights = torch.zeros(T, T) as we don't want all tokens to be uniform but rather it be data dependent, self-attention achieves this by:
# Every token at each position emits two vectors, a key and a query
# Query = What the token is looking for
# Key = What does this token contain
# Matrix multiplying the query with the keys will become weights (To allow tokens to interact with each other, making it data dependent)

# Single Head
head_size = 16
key = nn.Linear(C, head_size, bias = False)
query = nn.Linear(C, head_size, bias = False)
k = key(x) # (B, T, head_size)
q = query(x) # (B, T, head_size)

# (B, T, head_size) @ (B, head_size, T) --> (B, T, T)
# Transpose last 2 dimensions of k
weights = q @ k.transpose(-2, -1) * (head_size ** - 0.5) # Apply scaled attention to 

weights = weights.masked_fill(tril == 0, float("-inf")) # Wherever there is a 0 in tril, make it negative infinity [Ensures that past tokens cannot communicate with future tokens]
weights = F.softmax(weights, dim = -1) # Exponentiate and normalise weight rows

# At this stage, the weights will indicate how much information to aggregate from past tokens

# X is private information to a token, the value will contain the information for past tokens if the current token is "interested enough" in it
value = nn.Linear(C, head_size, bias = False)
v = value(x)

out = weights @ v

print(out.shape) # [B, T, head_size]
print(out)

# Notes:
# - Attention is a communication mechanism. 
#   - Every token has a vector of information which can aggregate information using a weighted sum from all tokens that point to them using data dependent weights

# - Attention has no notion of space as it only acts over a set of vectors. This is why positional encodding is used for the tokens.

# - Each batch example is processed completely independently from other batch examples (and so never communicate with each other)

# - An encoder attention block is when no masking is used (i.e. tril), which allows all tokens to communicate with each other (used in e.g. sentiment analysis)
# - A decoder attention block is when masking used (tril is used), which means past tokens cannot communicate with future tokens

# - Self-attention is when the keys, queries and values come from the same source (in this case, "x")
# - Cross-attention is when e.g. the queries are produced from "x" but the keys and values come from an external source (e.g. other encoder blocks)

# - Scaled attention: Attention(Q, K, V) = softmax(QK^T / sqrt(head_size))
# - Scaled attention is used to preserve the variance before and after the dot product of q and k
# - Preserving the variance is important because the weights go into softmax:
#   - If the weights before softmax has very positive and very negative values, exponentiating and normalising will make the weights converge to one-hot vectors
#   - This is not ideal as for a given token, we want to aggregate information from several tokens, not just a single one