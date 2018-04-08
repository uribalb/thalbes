import os
import tensorflow as tf
import numpy as np
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
n_nodes_hl = 128
hm_epochs = 140
n_classes = 2
batch_size = 64


with open('sentiment_set_rnn2.pickle', 'rb') as f:
    train_x, train_y, test_x, test_y = pickle.load(f)

x = tf.placeholder('float32', [None, len(train_x[0]), len(train_x[0][0])])
y = tf.placeholder('float32', [None, n_classes])


def neural_network_model(data):
    inputs = tf.reshape(data, (-1, len(train_x[0]) * len(train_x[0][0])))
    hidden_layer = tf.layers.dense(inputs=inputs, units=n_nodes_hl, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=hidden_layer, rate=0.5)
    output_layer = tf.layers.dense(inputs=dropout, units=n_classes)

    return output_layer


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=prediction, labels=y))
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
                _, c = sess.run([optimizer, cost], feed_dict={
                                x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print("Epoch ", epoch, " completed out of ", hm_epochs, " loss: ",
                  epoch_loss, 'Accuracy: ', accuracy.eval({x: test_x, y: test_y}))

train_neural_network(x)
