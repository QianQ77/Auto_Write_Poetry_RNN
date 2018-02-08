
import numpy as np
import tensorflow as tf
from RNNmodel import CharRNN
import ReadInputTxt as readInput
import string
import random

# The network gives us predictions for each character
# Only choose a new character from the top N most likely characters
def pick_top_n(preds, vocab_size, top_n=3):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


vocab, vocab_to_int, int_to_vocab, encoded = readInput.readAndEncode()

# We pass in a character, then the network will predict the next character
# We can use the new one, to predict the next one. And we keep doing this to generate all new text
def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="F"):
    samples = [c for c in prime]
    model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0 ,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0 ,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])

    return ''.join(samples)

checkpoint = tf.train.latest_checkpoint('checkpoints')

# choose a random letter to start with
randomLetter = random.choice(string.ascii_letters)

samp = sample(checkpoint, 600, 512, len(vocab), prime=randomLetter)



with open("Output.txt", "w") as text_file:
    print(samp, file=text_file)
