import torch
from string import ascii_lowercase as string_ascii_lowercase
import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()

# Convert into 2D array where the row is the first char and the column is the second char and array[row][column] = frequency of the second char following the first char
# 27 = 26 letters of the alphabet + "."
count = torch.zeros((27, 27), dtype = torch.int32)

# Build look-up table (From letter to index)
s_to_i = {l:index + 1 for index, l in enumerate(list(string_ascii_lowercase))}
s_to_i["."] = 0

# From index to letter
i_to_s = {index:l for l, index in s_to_i.items()}

for w in words:
    chars = ["."] + list(w) + ["."] # "emma" = [".","e", "m", "m", "a", "."], "." = start or end of the string
    for i in range(0, len(chars) - 1):
        index_1 = s_to_i[chars[i]] # First char
        index_2 = s_to_i[chars[i + 1]] # Second char

        # Increase frequency of char2 coming after char1
        count[index_1][index_2] += 1 


plt.figure(figsize = (16, 16))
plt.imshow(count, cmap = "Blues")
for i in range(27):
    for j in range(27):
        # Plot the pattern e.g. "eg", "an", etc...
        char_string = i_to_s[i] + i_to_s[j]
        plt.text(j, i, char_string, ha = "center", va = "bottom", color = "gray")
        # Plot the number of times this pattern has occured
        plt.text(j, i, count[i, j].item(), ha = "center", va = "top", color = "gray")
plt.axis("off")
plt.show()

# Generating samples
generator = torch.Generator().manual_seed(2147483647)

# Build probabilities array
# Note: Probabilities is a 2D array where each row will contain a probability distribution of all the possible characters that can come after the first character (i.e. the character at index "row")
probabilities = (count + 1).float() # Copy of count array but converted to floats, +1 for model smoothing so that patterns that occurred 0 times in the training set will not result in infinite loss, higher values makes the model more smooth 
# Normalise each row by summing the entire row, keepdims = True so that we can normalise the rows, not the columns
probabilities /= probabilities.sum(dim = 1, keepdims = True) # Dim = The dimension we want to sum over, keepdims = True will return a [dim, input_size] tensor

for _ in range(10):
    word = ""
    idx = 0
    while True:
        # Row with the probabilites of all characters coming after the first character at idx
        probability = probabilities[idx]

        # Generate a random index based on the probability distribution of the row of probabilities for the first character
        # E.g. if there was a 60% chance that the character at the index 1 follows this first character at idx, 60% of the items will be 1
        # Replacement paramter means that if we drew an element, it can be drawn again
        idx = torch.multinomial(probability, num_samples = 1, replacement = True, generator = generator).item() # Item to convert tensor([index]) into just the index
        
        word += i_to_s[idx]

        if idx == 0: # The "." token
            break
    
    print(word)

# Evaluating the model (using maximum likelihood estimation)
# Steps:
# - Maximise likelihood of the data with respect to the model parameters (e.g. the parameters in the count table) [Statistical modeling]
# - Equivalent to maximising the log likelihood (log is monotonic)
# - Equivalent to minimising the the negative log likelihood
# - Equivalent to minimising the average negative log likelihood
# The lower the negative log likelihood, the better the model is. The lowest it can be is 0

log_likelihood = 0.0
n = 0
for w in words:
    chars = ["."] + list(w) + ["."] # "emma" = [".","e", "m", "m", "a", "."], "." = start or end of the string
    for i in range(0, len(chars) - 1):

       
        index_1 = s_to_i[chars[i]] # First char
        index_2 = s_to_i[chars[i + 1]] # Second char

        # Find the likelihood
        prob = probabilities[index_1, index_2]
        log_prob = torch.log(prob) # High probabilities are closer to 0 and low probabilities become more and more negative
        log_likelihood += log_prob

        n += 1

        #print(f"{chars[i]}{chars[i + 1]}: {prob:.4f} {log_prob:.4f}")

print(f"{log_likelihood=}")
negative_log_likelihood = -log_likelihood
print(f"{negative_log_likelihood=}")
average_log_likelihood = negative_log_likelihood / n
print(f"{average_log_likelihood=}") # Typically used as the loss function, the aim of training would be to minimise this