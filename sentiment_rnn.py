import datetime
import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with open('sentiment_set_rnn2.pickle', 'rb') as f:
    train_x, train_y, test_x, test_y = pickle.load(f)

hm_epochs = 50
n_classes = 2
batch_size = 128
rnn_size = 60
n_words = len(train_x[0])
stacked_layers = 2
word_vec_dim = len(train_x[0][0])  # dim_wordvec

# height x width
x = tf.placeholder('float', [None, n_words, word_vec_dim])
y = tf.placeholder('float', [None, n_classes])


def seql(sequence):
    return  tf.count_nonzero(tf.reduce_max(sequence, 2), 1)


def drop(layer, num_layers):
    return 1 if num_layers == 1 else 0.5 + 0.5 * layer / (num_layers - 1)


def dyn_rnn(inputs):
    cells = [rnn.GRUCell(rnn_size) for i in range(stacked_layers)]
    cells = [tf.contrib.rnn.DropoutWrapper(
        cell=cell, output_keep_prob=drop(i, stacked_layers)) for i, cell in enumerate(cells)]
    multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
    _, states = tf.nn.dynamic_rnn(
        multi_cell, inputs, dtype=tf.float32, sequence_length=seql(inputs))
    output = tf.layers.dense(inputs=states[-1], units=n_classes)
    return output


def bi_dyn_rnn(inputs):
    cells = [rnn.GRUCell(rnn_size) for i in range(stacked_layers)]
    cells = [tf.contrib.rnn.DropoutWrapper(
        cell=cell, output_keep_prob=drop(i, stacked_layers)) for i, cell in enumerate(cells)]
    # multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
    _, states = tf.nn.bidirectional_dynamic_rnn(
        cells[0], cells[1], inputs, dtype=tf.float32, sequence_length=seql(inputs))
    output1 = tf.layers.dense(inputs=states[0], units=n_classes)
    output2 = tf.layers.dense(inputs=states[1], units=n_classes)

    return output1 + output2


def train_neural_network(inputs):
    prediction = dyn_rnn(inputs)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(epsilon=1e-12, learning_rate=1e-3).minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={
                                inputs: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            print("Epoch ", epoch, " completed out of ", hm_epochs, " loss: ", epoch_loss, 'Accuracy: ',
                  accuracy.eval({inputs: test_x, y: test_y}))

train_neural_network(x)
