import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyper parameters:
batch_size = 32 # Number of independent sequences to process in parallel
block_size = 8 # Context length for predictions
steps = 5000
learning_rate = 1e-3 # 1e-3 is the typical setting for the learning rate when using the Adam optimiser
device = "cuda" if torch.cuda.is_available() else "cpu" # Runs on the GPU rather than CPU if available

eval_interval = 200 # The interval to start evaluating the average loss
eval_iterations = 300 # Number of iterations for evaluating the average loss
n_embedding_dimensions = 32 # Dimensionality of each embedding

# Load data
with open("data.txt", "r", encoding = "utf-8") as data_file:
    text = data_file.read()

print(f"No.Characters:{len(text)}")

# Unique characters that appear in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mappings from characters to integers and vice versa
s_to_i = {char: i for i, char in enumerate(chars)}
i_to_s  = {i:char for i, char in enumerate(chars)}

encode = lambda s: [s_to_i[char] for char in s] # Takes a string and outputs a list of integers
decode = lambda l: [i_to_s[i] for i in l] # Takes a list of integers and outputs a string

# Entire data set as a tensor
data = torch.tensor(encode(text), dtype = torch.long)

# Splitting dataset into train + validation splits (90%, 10%)
n = int(0.9 * len(data))
train_data = data[:n]
validation_data = data[n:]

# Generating (mini) batches of examples
torch.manual_seed(1337)

def get_batch(split):
    # Note:
    # x and y will have the same shape
    # Because each target in y[a] will be the next correct character for the sequence of everything up to that index i (a = row number, i = column number in row a)
    # i.e. y[a][i] === the next correct character for x[a][:i]

    data = train_data if split == "Train" else validation_data

    # Creates a list of batch_size containing the indexes to start the batch examples from
    # E.g. idx = tensor([76049, 234249, 934904, 560986])
    idx = torch.randint(len(data) - block_size, (batch_size,))

    # Notes:
    # x_test = [data[idx[0]:idx[0] + block_size]] = Creates a list of the encoded characters starting from index i, up to the i + block_size
    # y_test = [data[idx[0] + 1:idx[0] + 1 + block_size]] = Creates a list of the correct next character following a previous sequence
    # torch.stack compiles all the lists for each index in idx into a single list (all lists into rows)

    # Examples (The contexts)
    x = torch.stack([data[i:i + block_size] for i in idx])
    # Targets (The expected next characters after a context)
    y = torch.stack([data[i + 1: i + 1 + block_size] for i in idx]) # Starts at data[i + 1] as the item after data[i] will be the correct next haracter

    # Move to CPU / GPU
    x, y = x.to(device), y.to(device)

    return x, y

Xb, Yb = get_batch("Train")
print("Inputs:")
print(Xb)
print(Xb.shape)

print("Targets")
print(Yb)
print(Yb.shape)

@torch.no_grad()
def estimate_loss(): # Estimates the average loss over a number of batches for both splits

    # Hashmap for average loss
    out = {}
    # Set model to evaluation mode
    model.eval()

    # Finding losses
    for split in ["Train", "Val"]:

        losses = torch.zeros(eval_iterations)

        for x in range(eval_iterations):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[x] = loss.item()
        out[split] = losses.mean()

    # Set model back to training mode
    model.train()

    return out

class Head(nn.Module):

    """ Single head of self attention """
    def __init__(self, head_size):
        super().__init__()
        
        self.key = nn.Linear(n_embedding_dimensions, head_size, bias = False) # What this token contains
        self.query = nn.Linear(n_embedding_dimensions, head_size, bias = False) # What this token is looking for
        self.value = nn.Linear(n_embedding_dimensions, head_size, bias = False) # Information from tokens that can be aggregated
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))) # Tril for encoder

    def forward(self, x):
        B, T, C = x.shape # C = head_size

        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # Compute attention scores ("affinities" across tokens)
        weights = q @ k.transpose(-2, -1) * (C ** - 0.5) # (B, T, C) @ (B, C, T) --> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        weights = F.softmax(weights, dim = -1)

        # Perform weighted aggregation of the values
        v = self.value(x)
        out = weights @ v # (B, T, T) @ (B, T, C) ---> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel 

    - Multi-head attention applies multiple attentions in parallel and concatenates the results

    """
    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size = head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim = -1) # -1 = channel dimension (C)
    
class FeedForward(nn.Module):

    """
    - FeedForward is an MLP
    - A simple linear layer followed by a non-linearity 
    """

    def __init__(self, n_embedding_dimensions):
        super().__init__()
        self.net = nn.Sequential(
                                nn.Linear(n_embedding_dimensions, n_embedding_dimensions),
                                nn.ReLU()
                                )
    def forward(self, x):
        return self.net(x)


# Bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Create embedding table of shape (vocab_size, n_embedding_dimensions)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding_dimensions)

        # Positional encoding embedding table
        self.position_embedding_table = nn.Embedding(block_size, n_embedding_dimensions)

        # Multi-head attention
        # Note: Better than a single attention head as multiple independent channels of communication can mean that the tokens can gather lots of different data
        self.self_attention_heads = MultiHeadAttention(4, n_embedding_dimensions // 4) # i.e. 4 heads of (n_embedding_dimensions // 4)-dimensional self-attention

        # Feed forward
        self.feed_forward = FeedForward(n_embedding_dimensions = n_embedding_dimensions)

        # Linear layer used to convert the token embeddings to logits
        self.language_modeling_head = nn.Linear(n_embedding_dimensions, vocab_size)

    def forward(self, idx, targets = None):
        
        B, T = idx.shape
        
        # idx and targets are both (B, T) tensor of integers (Encodes idx with the token embeddings)
        token_embeddings = self.token_embedding_table(idx) # (B, T, C) [Batch, Time, Channels] (C = n_embedding_dimensions)

        # Positional encoding (Encodes idx with the position of the token)
        # Note: Need to positionally encode tokens because attention has no notion of space as it acts over a set of vectors
        # Integers from 0 to T - 1
        positional_embeddings = self.position_embedding_table(torch.arange(T, device = device)) # (T, C) (C = n_embedding_dimensions)
        x = token_embeddings + positional_embeddings # (B, T, C)

        # Apply self_attention to all heads
        x = self.self_attention_heads(x) # (B, T, C)

        # Feed forward 
        # - Allows each independent token to "think" on the information aggregated after the self-attention heads
        x = self.feed_forward(x) # (B, T, C)

        # Convert token embeddings to logits
        logits = self.language_modeling_head(x) # (B, T, vocab_size)
        
        # Find loss:
        if targets == None:
            loss = None
        else:
            # Convert shapes for compatibility with PyTorch's cross_entropy function
            B, T, C = logits.shape
            logits = logits.view(B * T, C) # 2-D
            targets = targets.view(B * T) # 1-D

            # Cross-entropy loss
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):

        # idx is (B, T) array of indices in the current context [Batch size, Time step] (at initialisation)
       
        for _ in range(max_new_tokens):
            
            # Crop idx after adding positional encoding
            # If idx is more than block size, the positional embedding table will run out of scope (as it only has embeddings up to block size)
            # idx[:, -block_size:] = Only gets predictions from the last 8 characters in idx 
            idx_cropped = idx[:, -block_size:]

            # Get predictions
            logits, _ = self(idx_cropped)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] # From (B, T, C) --> (B, C)

            # Apply softmax to find the probability distribution of the logits
            probabilities = F.softmax(logits, dim = -1) # (B, C)

            # Sample from the outputted probability distribution
            idx_next = torch.multinomial(probabilities, num_samples = 1) # (B, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T + 1)
        
        return idx

model = BigramLanguageModel()
model = model.to(device) # Move model parameters to device (so calculations are executed on GPU if available)

output, loss = model(idx = Xb, targets = Yb)
print(output.shape)

# Test model
# context_text = torch.zeros((1, 1), dtype = torch.long, device = device) # First param = batch size, Second parameter = current time step
# print("".join(decode(model.generate(context_text, max_new_tokens = 100)[0].tolist()))) # [0] to pluck out the single batch dimension, tolist() to convert from Tensor to Python list

# PyTorch optimiser
optimiser = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for i in range(steps):

    # Evaluate loss on the train and validation splits every once in a while
    if i % eval_iterations == 0:
        losses = estimate_loss()
        print(f"Step: {i} | Training loss: {losses['Train']:.4f} | Validation loss: {losses['Val']:.4f}")

    # Generate mini-batch of data
    Xb, Yb = get_batch("Train")

    # Forward pass
    logits, loss = model(idx = Xb, targets = Yb)

    # Backward pass
    optimiser.zero_grad(set_to_none = True)
    loss.backward()
    optimiser.step()

context_text = torch.zeros((1, 1), dtype = torch.long, device = device) # First param = batch size, Second parameter = current time step
print("".join(decode(model.generate(context_text, max_new_tokens = 100)[0].tolist())))