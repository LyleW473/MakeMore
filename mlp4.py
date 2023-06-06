# Exercises from: https://www.youtube.com/watch?v=q8SA3rM6ckI

import torch
import torch.nn.functional as F
from string import ascii_lowercase as string_ascii_lowercase#

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

block_size = 3

Xtr, Ytr = build_dataset(words[0:n1], block_size = block_size) 
Xdev, Ydev = build_dataset(words[n1:n2], block_size = block_size)
Xte, Yte = build_dataset(words[n2:], block_size = block_size)

print(f"Train: X: {Xtr.shape} Y: {Ytr.shape}")
print(f"Dev: X: {Xdev.shape} Y: {Ydev.shape}")
print(f"Test: X: {Xte.shape} Y: {Yte.shape}")

# Utility function used to compare manual gradients to PyTorch gradients
def cmp(s, dt, t):
    # Check if all of our manual gradients are the same as all the gradients outputted by PyTorch
    ex = torch.all(dt == t.grad).item() # True if all are the same
    # If it isn't the same, check if the gradients are close (Floating point issues)
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
B1 = torch.randn(n_hidden, generator = g) * 0.1 # Usually not required when a batch normalisation layer is used [Used for manual backpropagation for fun]

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
    h_pre_batch_norm = embedding_concat @ W1 + B1

    # ----------------------------------------------
    # Batch normalisation layer

    bn_mean_i = (1 / batch_size) * h_pre_batch_norm.sum(0, keepdim = True)
    bn_diff = h_pre_batch_norm - bn_mean_i
    bn_diff2 = bn_diff ** 2

    # Note: Bessel's correction (dividing by n-1 not n) 
    # - n-1 = unbiased variance, n = biased variance, which is Used to get better estimate for variance for small samples for a population
    #- Used here because we are using mini-batches (small sample for a large population)
    bn_variance = (1 / (batch_size - 1)) * (bn_diff2).sum(0, keepdim = True)
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

    logit_maxes = logits.max(1, keepdim = True).values # Find the maximum in each row
    norm_logits = logits - logit_maxes # Subtract maxes for numerical stability so that we don't exponentiate logits with extreme values 

    # Softmax
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdims = True)
    counts_sum_inv = counts_sum ** -1 # Same as (1.0 / counts_sum), but sometimes yields incorrect results when performing backpropagation with PyTorch
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
    break

# Manual backpropagation: (Calculated by hand) [Exercise 1]

# loss = -log_probs[range(batch_size), Ytr[mini_b_idxs]].mean()
# dLoss/dLogProbs = -1/batch_size (for all items in log_probs)
d_log_probs = torch.zeros_like(log_probs)
# Set for all elements in d_log_probs at the indexes in Ytr(mini_b_idxs), the derivative of the loss in respect to this element
d_log_probs[range(batch_size), Ytr[mini_b_idxs]] = -1.0/batch_size
cmp("log_probs", d_log_probs, log_probs)

# dLoss/dProbs = (dLoss / dLogProbs) * (dLogProbs / dProbs)
# log_probs = probs.log() == ln(probs)
# dLogProbs / dProbs = 1/probs
# dLoss/dLogProbs = -1/batch_size
# dLoss/dProbs = (1/probs) * (-1/batch_size) --> -1/(probs * batch_size) [Calculation is broadcasted below]
d_probs = (1.0 / probs) * d_log_probs
cmp("probs", d_probs, probs)

# Note: counts_sum_inv.shape = (32, 1), counts.shape = (32, 27), d_probs.shape = (32, 27)

# dLoss/dCountsSumInv = (dLoss / dProbs) * (dProbs / dCountsSumInv)
# probs = counts * counts_sum_inv
# dProbs/dCountsSumInv = counts
# dLoss/dCountsSumInv = (-1/(probs * batch_size)) * counts
# The correct thing to do to to sum all the gradients for each row to make d_counts_sum_inv of shape (32, 1) (Because all gradients that arrive at any of these elements should sum)
d_counts_sum_inv = (d_probs * counts).sum(1, keepdim = True)
cmp("counts_sum_inv", d_counts_sum_inv, counts_sum_inv)


# Note: counts was used in "probs" and "counts_sum", so "cmp" is not used here yet

# dLoss/dCounts = (dLoss / dProbs) * (dProbs / dCounts)
# probs = counts * counts_sum_inv
# dProbs / dCounts = counts_sum_inv
# dLoss/dCounts = (-1/(probs * batch_size)) * counts_sum_inv
d_counts = d_probs * counts_sum_inv # Will matrix multiply [ (32, 7) * (32, 1) ---> (32, 7)]

# dLoss/dCountsSum = (dLoss / dCountsSumInv) * (dCountsSumInv / dCountsSum)
# counts_sum_inv = counts_sum ** -1
# dCountsSumInv / dCountsSum = -1 * (counts_sum ** -2) === -1 / (counts_sum ** 2) ---> -(counts_sum ** -2)
# dLoss/dCountsSum = (-1/(probs * batch_size)) * counts) * (-1 / (counts_sum ** 2))
d_counts_sum = d_counts_sum_inv * (-(counts_sum ** -2))
cmp("counts_sum", d_counts_sum, counts_sum)


# dLoss/dCounts = (dLoss / dCountsSum) * (dCountsSum / dCounts)
# counts_sum = counts.sum(1, keepdims = True)
# dCountsSum/dCounts = 1
# dLoss/dCounts = ((-1/(probs * batch_size)) * counts) * (-1 / (counts_sum ** 2))) * 1
# Notes:
# - ones_like as dCountsSum/dCounts = 1
# - counts.shape = (32, 27), counts_sum.shape = (32, 1)
# - All of the derivatives of counts_sum (i.e. d_counts_sum) are replicated 27 times for d_counts

# - Each row of counts_sum is a summation of the corresponding row in counts, so counts_sum[0][0] = counts[0][0] + counts[0][1] + counts[0][2]... +counts[0][27]
# - Therefore the derivative of counts_sum[0][0] w.r.t counts[0][x] = 1 (so this means the derivatives of [dLoss/dCountsSum] will just flow to each cell of d_counts)
# - However the derivative of counts_sum[0][0] w.r.t counts[(any other row that isn't the same as the row in counts_sum)][x] = 0 (as these other elements were not involved in the summation)
d_counts += torch.ones_like(counts) * d_counts_sum # += As the gradients of d_counts_sum should be added to the existing gradients in d_counts, calculated earlier
cmp("counts", d_counts, counts)

# dLoss/dNormLogits = (dLoss / dCounts) * (dCounts / dNormLogits)
# counts = norm_logits.exp()
# dCounts/dNormLogits = norm_logits.exp() === e^norm_logits
# dLoss/dNormLogits = ((-1/(probs * batch_size)) * counts) * (-1 / (counts_sum ** 2))) * e^norm_logits
d_norm_logits = d_counts * counts # counts holds the value of norm_logits.exp(), so re-using it here, otherwise, replace with norm_logits.exp()
cmp("norm_logits", d_norm_logits, norm_logits)

# Note: Logits was used in multiple operations (Same process as when "counts" was used multiple times)

# dLoss/dLogits = (dLoss / dNormLogits) * (dNormLogits / dLogits)
# norm_logits = logits - logit_maxes
# dNormLogits/dLogits = 1
# dLoss/dLogits = (((-1/(probs * batch_size)) * counts) * (-1 / (counts_sum ** 2))) * e^norm_logits) * (1)
d_logits = d_norm_logits.clone() # Copy just in case

# dLoss/dLogitMaxes = (dLoss / dNormLogits) * (dNormLogits / dLogitMaxes)
# norm_logits = logits - logit_maxes
# dNormLogits/dLogitMaxes = -1
# dLoss/dLogitMaxes = (((-1/(probs * batch_size)) * counts) * (-1 / (counts_sum ** 2))) * e^norm_logits) * (-1)
d_logit_maxes = (-d_norm_logits).sum(1, keepdim = True)
cmp("logit_maxes", d_logit_maxes, logit_maxes)


# dLoss/dLogits = (dLoss/ dLogitMaxes) * (dLogitMaxes / dLogits)
# logit_maxes = logits.max(1, keepdim = True)

# To find dLogitMaxes / dLogits:
indexes = logits.max(1).indices # Contains indexes of the column / index of the maximum value in each row of logits
logit_maxes_oh = F.one_hot(indexes, num_classes = 27) # Convert indexes to one-hot vectors (for matrix mult)

# dLoss/dLogits = (-((-1/(probs * batch_size)) * counts) * (-1 / (counts_sum ** 2))) * e^norm_logits) * (logit_maxes_oh)
d_logits += d_logit_maxes * logit_maxes_oh
cmp("logits", d_logits, logits)

# Note: The local derivatives of the variables involved in equation, logits = H @ W2 + B2, were derived on paper (Check video for example)
# - The backward pass of a matrix multiplication, is a matrix multiplication

# - One way of finding the derivatives quickly is by looking at the shapes of all the variables, and finding the matrices that can matrix multiply to make the correct shape: (IMPORTANT)
# - For example: d_H needs to be of shape (32, 200). 
# - This can only be produced from a matrix multiplication of d_logits (32, 27) and W2.T (27, 200)

# logits = H @ W2 + B2
# dLoss/dH = (dLoss/dLogits) @ W2.Transpose 
# 
d_H = d_logits @ W2.T

cmp("H", d_H, H)

# logits = H @ W2 + B2
# [200, 32] @ [32, 27] ---> [200, 27]
# dLoss/dW2 = H.Transpose @ (dLoss/dLogits)
d_W2 = H.T @ d_logits
cmp("W2", d_W2, W2)

# logits = H @ W2 + B2
# dLoss/dB2 = dLoss/dLogits.sum(0) [Sum across the columns]
# [32, 27].sum(columns) ---> [1, 27] --> [27]
d_B2 = d_logits.sum(0, keepdim = False) # keepdim = False to convert from [1, 27] to [27] vector
cmp("B2", d_B2, B2)

# dLoss/dHPreActivation = (dLoss/d_H) * (dH/dHPreActivation)
# H = torch.tanh(h_pre_activation)
# dH/dHPreActivation = 1 - (H ** 2) 
# dLoss/dHPreActivation = d_H * (1 - (H ** 2))
# a = tanh(z), da/dz = 1 - (a ** 2)
d_h_pre_activation = d_H * (1.0 - (H ** 2))
cmp("h_pre_activation", d_h_pre_activation, h_pre_activation)

# dLoss/dBnGain = (dLoss/dHPreActivation) * (dHPreActivation/dBnGain)
# h_pre_activation = (bn_gain * bn_raw) + bn_bias
# dHPreActivation/dBnGain = bn_raw
# dLoss/dBnGain = d_h_pre_activation * bn_raw
d_bn_gain = (d_h_pre_activation * bn_raw).sum(0, keepdim = True)
cmp("bn_gain", d_bn_gain, bn_gain)

# dLoss/dBnRaw = (dLoss/dHPreActivation) * (dHPreActivation/dBnRaw)
# h_pre_activation = (bn_gain * bn_raw) + bn_bias
# dHPreActivation/dBnRaw = bn_gain
# dLoss/dBnRaw = d_h_pre_activation * bn_gain
d_bn_raw = (d_h_pre_activation * bn_gain)
cmp("bn_raw", d_bn_raw, bn_raw)

# dLoss/dBnBias= (dLoss/dHPreActivation) * (dHPreActivation/dBnBias)
# h_pre_activation = (bn_gain * bn_raw) + bn_bias
# dHPreActivation/dBnBias = 1
# dLoss/dBnBias = d_h_pre_activation * 1
d_bn_bias = d_h_pre_activation.sum(0, keepdim = True)
cmp("bn_bias", d_bn_bias, bn_bias)


# dLoss/dBnDiff = (dLoss/dBnRaw) * (dBnRaw/dBnDiff)
# bn_raw = bn_diff * bn_variance_inv
# dBnRaw/dBnDiff = bn_variance_inv
# dLoss/dBnDiff = d_bn_raw * bn_variance_inv
d_bn_diff = d_bn_raw * bn_variance_inv # Note: bn_diff used multiple times so cmp not used yet

# dLoss/dBnVarianceInv = (dLoss/dBnRaw) * (dBnRaw/dBnVarianceInv)
# bn_raw = bn_diff * bn_variance_inv
# dBnRaw/dBnVarianceInv = bn_diff
# dLoss/dBnVarianceInv = d_bn_raw * bn_diff
d_bn_variance_inv = (d_bn_raw * bn_diff).sum(0, keepdim = True)
cmp("bn_variance_inv", d_bn_variance_inv, bn_variance_inv)

# dLoss/dBnVariance = (dLoss/dBnVarianceInv) * (dBnVarianceInv/dBnVariance)
# bn_variance_inv = (bn_variance + epsilon) ** -0.5
# dBnVarianceInv/dBnVariance = -0.5 * ((bn_variance + epsilon) ** (-3/2)) * 1
# dLoss/dBnVariance = d_bn_variance_inv * 0.5
d_bn_variance = d_bn_variance_inv * (-0.5 * ((bn_variance + epsilon) ** (-1.5)))
cmp("bn_variance", d_bn_variance, bn_variance)

# dLoss/dBnDiff2 = (dLoss/dbnVariance) * (dBnVariance/dBnDiff2)
# bn_variance = 1 / ((batch_size - 1) * (bn_diff2).sum(0, keepdim = True))
# dBnVariance/dBnDiff2 = 1 / (batch_size - 1)
# dLoss/dBnDiff2 = d_bn_variance * (1 / (batch_size - 1))
# Note:
# - torch.ones_like(bn_diff2) because bn_diff2.shape = (32, 200), d_bn_variance.shape = (1, 200)
# - d_bn.variance multiplied by scalar (1 / (batch_size - 1)) ---> (1, 200) shape, hence why torch.ones_like is used
d_bn_diff2 = d_bn_variance * ((1.0 / (batch_size - 1)) * torch.ones_like(bn_diff2))
cmp("bn_diff2", d_bn_diff2, bn_diff2)

# dLoss/dBnDiff = (dLoss/dBnDiff2) * (dBnDiff2/dBnDiff)
# bn_diff2 = bn_diff ** 2
# dBnDiff2/dBnDiff = 2 * bn_diff
# dLoss/dBnDiff = d_bn_diff2 * (2 * bn_diff)
d_bn_diff += d_bn_diff2 * (2 * bn_diff)
cmp("bn_diff", d_bn_diff, bn_diff)


# dLoss/dHPreBatchNorm = (dLoss/dBnDiff) * (dBnDiff/dHPreBatchNorm)
# bn_diff = h_pre_batch_norm - bn_mean_i 
# dBnDiff/dHPreBatchNorm = 1
# dLoss/dHPreBatchNorm = d_bn_diff * 1
d_h_pre_batch_norm = d_bn_diff.clone()

# dLoss/dBnMeanI= (dLoss/dBnDiff) * (dBnDiff/dBnMeanI)
# bn_diff = h_pre_batch_norm - bn_mean_i 
# dBnDiff/dBnMeanI = -1
# dLoss/dHPreBatchNorm = d_bn_diff * -1
d_bn_mean_i = -d_bn_diff.sum(0, keepdim = True)

cmp("bn_mean_i", d_bn_mean_i, bn_mean_i)

# dLoss/dHPreBatchNorm = (dLoss/dBnMeanI) * (dBnMeanI * dHPreBatchNorm)
# bn_mean_i = (1 / batch_size) * h_pre_batch_norm.sum(0, keepdim = True)
# dBnMeanI * dHPreBatchNorm = (1 / batch_size)
# dLoss/dHPreBatchNorm = d_bn_mean_i * (1 / batch_size)
d_h_pre_batch_norm += d_bn_mean_i * (1.0 / batch_size) * torch.ones_like(h_pre_batch_norm) # use of torch.ones_like explained above
cmp("h_pre_batch_norm", d_h_pre_batch_norm, h_pre_batch_norm)

# dLoss/dEmbeddingConcat = (dLoss/dHPreBatchNorm) * (dHPreBatchNorm/dEmbeddingConcat)
# h_pre_batch_norm = embedding_concat @ W1
# h_pre_batch_norm.shape = (32, 200), embedding_concat.shape = (32, 60), W1.shape = (60, 200)
# d_embedding_concat.shape needs to be (32, 60) so:
# - W1.T.shape = (200, 60), d_h_pre_batch_norm.shape = (32, 200),  (32, 200) @ (200, 60) ---> (32, 60)
# dLoss/dEmbeddingConcat = d_h_pre_batch_norm @ W1.T 
d_embedding_concat = d_h_pre_batch_norm @ W1.T
cmp("embedding_concat", d_embedding_concat, embedding_concat)

# dLoss/dW1 = (dLoss/dHPreBatchNorm) * (dHPreBatchNorm/dW1)
# - Using the same logic as above: d_W1.shape == W1.shape
# embedding_concat.T = (60, 32), h_pre_batch_norm.shape = (32, 200), W1.shape = (60, 200)
# embedding_concat.T @ d_h_pre_batch_norm = (60, 32) @ (32, 200) ---> (60, 200)
# dLoss/dW1 = embedding_concat.T @ d_h_pre_batch_norm
d_W1 = embedding_concat.T @ d_h_pre_batch_norm
cmp("W1", d_W1, W1)

# dLoss/dB1 = (dLoss/dHPreBatchNorm) * (dHPreBatchNorm/dB1)
# B1.shape = (200), W1.shape = (60, 200), embedding_concat.shape = (32, 60), d_h_pre_batch_norm.shape = (32, 200)
d_B1 = d_h_pre_batch_norm.sum(0, keepdim = False) # keepdim = False to convert (32, 200) --> (200)
cmp("B1", d_B1, B1)


# To reverse this, you simply re-view the concatenated vectors as the original shape
# embedding_concat = embedding.view(embedding.shape[0], -1)
d_embedding = d_embedding_concat.view(embedding.shape)
cmp("embedding", d_embedding, embedding)

# embedding = C[Xtr[mini_b_idxs]]
# C.shape = (27, 10) [All characters have a 10-dimensional embedding]
# Xtr[mini_b_idxs].shape = (32, 3) # 32 examples with a block size of 3, with each item in each row being an index to the character embedding
# embedding.shape = (32, 3, 10) = # 32 examples with all 3 characters having a 10 dimensional embedding 
d_C = torch.zeros_like(C) # (27, 10) tensor with all zeroes
for i in range(Xtr[mini_b_idxs].shape[0]):
    for j in range(Xtr[mini_b_idxs].shape[1]):
        # idx = The jth index of the character inside the ith example
        idx = Xtr[mini_b_idxs][i][j]
        # d_embedding[i][j] = A single row of the gradients of the 10 dimensional embedding for this character (e.g. "a")
        d_C[idx] += d_embedding[i][j]

cmp("d_C", d_C, C)


# Cross entropy loss backward pass [Exercise 2]
# The derivative for items at the Y labels with respect to the loss is: (e^a / (e^a + e^b + e^c + e^d)) - 1 if a is in "y" labels, otherwise it is (e^a / (e^a + e^b + e^c + e^d))

loss_fast = F.cross_entropy(logits, Ytr[mini_b_idxs])
print(loss_fast.item(), "difference:", (loss_fast - loss).item())

# Set derivative of all logits as (e^a / (e^a + e^b + e^c + e^d)) [as in either case where "a" is in or not in y labels, the derivative of the loss with respect to each logit is (e^a / (e^a + e^b + e^c + e^d))]
# Derivative of the loss with respect to each logit regardless of if the logit("a") is in the y-labels is (e^a / (e^a + e^b + e^c + e^d))
d_logits = F.softmax(logits, 1) # Softmax across rows

#  Derivative of the loss with respect to a logit inside of the y-labels is (e^a / (e^a + e^b + e^c + e^d)) - 1
d_logits[range(batch_size), Ytr[mini_b_idxs]] -= 1 
# loss = the average of the losses, so d_logits needs to be scaled down by batch_size
d_logits /= batch_size
cmp("logits", d_logits, logits)

# Full derivation:
# log_probs = probs.log()
# loss = -log_probs[range(batch_size), Ytr[mini_b_idxs]].mean()


# Simplified:
# loss = -log(e^a / (e^a + e^b + e^c + e^d))
# loss = -ln(e^a / (e^a + e^b + e^c + e^d)) [ln as .log() in PyTorch is the natural logarithm]
# loss = ln( (e^a + e^b + e^c + e^d) / e^a )
# Using the quotient log rule: ln(x/y) = ln(x) − ln(y)
# loss = ln(e^a + e^b + e^c + e^d) - ln(e^a)
# Using the rule: ln(e^x) = x
# loss = ln(e^a + e^b + e^c + e^d) - a

# To find derivative of the loss with respect to any variable i:


# y = ln(e^a + e^b + e^c + e^d) 
# u = e^a + e^b + e^c + e^d

# If i == a: (i.e. "a" is an y-label)

# du/di = e^a
# dy/di = (du/da) / u [The rule where if y = ln(u), dy/du = u'/u]
# dy/di = e^a / (e^a + e^b + e^c + e^d)

# Therefore if loss = ln(e^a + e^b + e^c + e^d) - a: 
# dLoss/di = dy/di + d/da(-a)
# dLoss/di = (e^a / (e^a + e^b + e^c + e^d)) + (-1)
# dLoss/di = (e^a / (e^a + e^b + e^c + e^d)) - 1

# If i != a:

# du/di = e^a
# dy/di = (du/da) / u [The rule where if y = ln(u), dy/du = u'/u]
# dy/di = e^a / (e^a + e^b + e^c + e^d)

# dLoss/di = dy/di
# dLoss/di = (e^a / (e^a + e^b + e^c + e^d))

# Hence to find the derivative of the loss with respect to any variable "a", use (e^a / (e^a + e^b + e^c + e^d)) - 1 (If i == "a" ["a" is a y-label]) else (e^a / (e^a + e^b + e^c + e^d))