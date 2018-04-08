import os
import pickle

import numpy as np
import tensorflow as tf

with open('sentiment_set_rnn2.pickle', 'rb') as f:
    train_x, train_y, test_x, test_y = pickle.load(f)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
hm_epochs = 50
n_classes = 2
batch_size = 32

# width and heigth of sentence matrix
dim_word = len(train_x[0][0])
max_length = len(train_x[0])

x = tf.placeholder('float', [None, max_length, dim_word])  # x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float', [None, n_classes])


def neural_network_model(data):
    input_layer = tf.expand_dims(data, tf.rank(data))
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=8,
        kernel_size=[5,5],
        strides=2,
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1, padding="same")
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=8,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=1, padding="same")
    pool2_flat = tf.reshape(pool2, [batch_size, -1])
    dense = tf.layers.dense(inputs=pool2_flat, units=32, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense, units=32, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense2, rate=0.4, training=True)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    return logits


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

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

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print("Epoch ", epoch, " completed out of ", hm_epochs, " loss: ", epoch_loss, 'Accuracy: ',
                  accuracy.eval({x: test_x, y: test_y}))
            # print('Accuracy: ', accuracy.eval({x: test_x, y: test_y}))


train_neural_network(x)
