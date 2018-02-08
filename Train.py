import numpy as np
import time
import tensorflow as tf
from RNNmodel import CharRNN
import ReadInputTxt as readInput

# returns batches of size n_seqs x n_steps from arr
# n_seqs: batch size
# n_steps: sequence length
def get_batches(arr, n_seqs, n_steps):
    # Get the number of characters per batch and number of batches we can make
    characters_per_batch = n_seqs * n_steps
    n_batches = len(arr) // characters_per_batch

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * characters_per_batch]

    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs, -1))

    for n in range(0, arr.shape[1], n_steps):
        # The features
        x = arr[:, n:n + n_steps]
        # The targets, shifted by one
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y


vocab, vocab_to_int, int_to_vocab, encoded = readInput.readAndEncode()

batch_size = 30        # Sequences per batch
num_steps = 50         # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001   # Learning rate
keep_prob = 0.5         # Dropout keep probability


# ## for training
epochs = 20
# Save every N iterations
save_every_n = 200

model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers,
                learning_rate=learning_rate)

saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # load a checkpoint and resume training
    counter = 0
    for e in range(epochs):
        # Train network
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in get_batches(encoded, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.loss,
                                                 model.final_state,
                                                 model.optimizer],
                                                feed_dict=feed)

            end = time.time()
            print('Epoch: {}/{}... '.format(e + 1, epochs),
                  'Training Step: {}... '.format(counter),
                  'Training loss: {:.4f}... '.format(batch_loss),
                  '{:.4f} sec/batch'.format((end - start)))

            if (counter % save_every_n == 0):
                print('checkpoints/i{}_l{}.ckpt'.format(counter, lstm_size))
                saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))


