import numpy as np

# First we'll load the text file and convert it into integers for our network to use.
# Encoding the characters as integers makes it easier to use as input in the network.

def readAndEncode():
    with open('Poems_JohnKeats.txt', 'r') as f:
        text = f.read()
    # vocab contains all characters appear in text
    vocab = sorted(set(text))
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
    return vocab, vocab_to_int, int_to_vocab, encoded
