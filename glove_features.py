import csv
import pickle
import random

import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000
dim_wordvec = 50
wordvec = pd.read_csv('glove.6B.50d.txt', sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)


def sample_handling(sample, classification):
    featureset = []
    with open(sample, 'r') as file:
        contents = file.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(shape=(62, dim_wordvec))
            for _, word in enumerate(current_words):
                if word.lower() in wordvec.index:
                    features[_] = wordvec.loc[word.lower()].values
            featureset.append([features, classification])
    return featureset


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    features = []
    features += sample_handling(pos, [1, 0])
    features += sample_handling(neg, [0, 1])
    random.shuffle(features)
    features = np.array(features)
    testing_size = int(test_size * len(features))

    trainx = list(features[:, 0][:-testing_size])
    trainy = list(features[:, 1][:-testing_size])
    testx = list(features[:, 0][:-testing_size:])
    testy = list(features[:, 1][:-testing_size:])
    return trainx, trainy, testx, testy


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    with open('sentiment_set_rnn.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)


































