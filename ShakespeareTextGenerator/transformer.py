import torch

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
batch_size = 4 # Number of independent sequences to process in parallel
block_size = 8 # Context length for predictions

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

    return x, y

Xb, Yb = get_batch("Train")
print("Inputs:")
print(Xb)
print(Xb.shape)

print("Targets")
print(Yb)
print(Yb.shape)