import os
import tensorflow as tf
import numpy as np
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with open('sentiment_set.pickle', 'rb') as f:
    train_x, train_y, test_x, test_y = pickle.load(f)

n_nodes_hl1 = 423
n_nodes_hl2 = 300
n_nodes_hl3 = 200
n_nodes_hl4 = 150
n_nodes_hl5 = 75
hm_epochs = 140
n_classes = 2
batch_size = 64

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float', [None, n_classes])


def neural_network_model(data):
    hidden_1_layer = tf.layers.dense(inputs=data, units=n_nodes_hl1, trainable=True, use_bias=True, activation=tf.nn.crelu)
    dropout = tf.layers.dropout(inputs=hidden_1_layer, rate=0.4)
    hidden_2_layer = tf.layers.dense(inputs=dropout, units=n_nodes_hl2, trainable=True, use_bias=True, activation=tf.nn.relu)
    hidden_3_layer = tf.layers.dense(inputs=hidden_2_layer, units=n_nodes_hl3, trainable=True, use_bias=True, activation=tf.nn.relu)
    hidden_4_layer = tf.layers.dense(inputs=hidden_3_layer, units=n_nodes_hl4, trainable=True, use_bias=True, activation=tf.nn.relu)
    hidden_5_layer = tf.layers.dense(inputs=hidden_4_layer, units=n_nodes_hl5, trainable=True, use_bias=True, activation=tf.nn.relu)
    output = tf.layers.dense(inputs=hidden_4_layer, units=n_classes, trainable=True, activation=tf.nn.relu)

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

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
            print("Epoch ", epoch, " completed out of ", hm_epochs, " loss: ", epoch_loss, 'Accuracy: ', accuracy.eval({x: test_x, y: test_y}))

train_neural_network(x)
