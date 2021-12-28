import numpy as np
import pandas as pd

sentence = """Thomas Jefferson began buildong Montichello at th age of 26."""
token_sequence = str.split(sentence)
vocab = sorted(set(token_sequence))
print(', '.join(vocab))
num_tokens = len(token_sequence)
vocab_size =  len(vocab)
one_hot_vectors = np.zeros((num_tokens,vocab_size),int)
print(one_hot_vectors)
for i, word in enumerate(token_sequence):
	one_hot_vectors[i,vocab.index(word)] = 1
print(' '.join(vocab))
print(one_hot_vectors)
text_df = pd.DataFrame(one_hot_vectors, columns=vocab)
print(text_df)

## bag of words

df = pd.DataFrame(pd.Series(dict([(token,1) for token in token_sequence])),columns=['sent']).T
print(df)
