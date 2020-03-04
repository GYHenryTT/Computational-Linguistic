# CS114 Spring 2020 Programming Assignment 2
# Naive Bayes Classifier and Evaluation

import os
import numpy as np
from collections import defaultdict
import re
import pandas as pd
import matplotlib.pyplot as plt


class NaiveBayes:

    def __init__(self):
        # be sure to use the right class_dict for each data set
        self.class_dict = {0: 'neg', 1: 'pos'}
        # self.class_dict = {0: 'action', 1: 'comedy'}
        self.features = set()
        self.prior = None
        self.likelihood = None
        self.class_count = defaultdict(int)

    '''
    Trains a multinomial Naive Bayes classifier on a training set.
    Specifically, fills in self.prior and self.likelihood such that:
    self.prior[class] = log(P(class))
    self.likelihood[class][feature] = log(P(feature|class))
    '''

    def train(self, train_set, feature_select_rate=1.0):
        self.features = self.select_features(train_set, feature_select_rate)
        # Initialize word count dictionary
        word_count = {0: defaultdict(int), 1: defaultdict(int)}

        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    word_list = re.split('[^A-Za-z\']+', f.read())

                    # collect feature counts (the class counts has been done in feature selection)
                    if root.find('pos') != -1:
                        for word in word_list:
                            if word not in self.features:
                                continue
                            word_count[1][word] += 1

                    elif root.find('neg') != -1:
                        for word in word_list:
                            if word not in self.features:
                                continue
                            word_count[0][word] += 1

                    else:
                        raise ValueError('The class names are not in file root')

        # normalize counts to probabilities, and take logs
        n_doc = sum(self.class_count.values())
        logprior_pos = np.log(self.class_count['pos'] / n_doc)
        logprior_neg = np.log(self.class_count['neg'] / n_doc)
        # add the log prior to a numpy array
        self.prior = np.array([logprior_neg, logprior_pos])

        # get total word counts in each class using add-1 method
        neg_total = len(self.features) + sum(word_count[0].values())
        pos_total = len(self.features) + sum(word_count[1].values())

        # calculate likelihoods with add-1 method
        self.likelihood = np.array([[], []])
        for word in self.features:
            neg_count = word_count[0][word] + 1
            pos_count = word_count[1][word] + 1
            self.likelihood = np.append(self.likelihood, [[neg_count / neg_total], [pos_count / pos_total]], axis=1)
        self.likelihood = np.log(self.likelihood)
        return

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''

    def test(self, dev_set):
        results = defaultdict(dict)
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    word_list = re.split('[^A-Za-z\']+', f.read())
                    # create feature vectors for each document
                    word_count = defaultdict(int)
                    for word in word_list:
                        if word not in self.features:
                            continue
                        word_count[word] += 1

                    # create a feature vector
                    doc_vector = np.array([])
                    for word in self.features:
                        doc_vector = np.append(doc_vector, word_count[word])

                    # get the correct class
                    if root.find('pos') != -1:
                        results[name]['correct'] = 1
                    elif root.find('neg') != -1:
                        results[name]['correct'] = 0
                    else:
                        raise ValueError('The class names are not in file root')

                # get most likely class
                log_prop = (doc_vector * self.likelihood).sum(axis=1) + self.prior
                results[name]['predicted'] = log_prop.argmax()

        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''

    def evaluate(self, results, mode='print'):
        # you may find this helpful
        confusion_matrix = np.zeros((len(self.class_dict), len(self.class_dict)))
        for _, result_dict in results.items():
            confusion_matrix[result_dict['correct'], result_dict['predicted']] += 1

        precision_pos = confusion_matrix[1, 1] / confusion_matrix[:, 1].sum()
        recall_pos = confusion_matrix[1, 1] / confusion_matrix[1, :].sum()
        f1_pos = (2 * precision_pos * recall_pos) / (precision_pos + recall_pos)
        accuracy = np.diag(confusion_matrix).sum() / confusion_matrix.sum()

        precision_neg = confusion_matrix[0, 0] / confusion_matrix[:, 0].sum()
        recall_neg = confusion_matrix[0, 0] / confusion_matrix[0, :].sum()
        f1_neg = (2 * precision_neg * recall_neg) / (precision_neg + recall_neg)

        if mode == 'print':
            print('------- Evaluation for positive class -------\n'
                  + f'Precision: {round(precision_pos, 3)}\nRecall: {round(recall_pos, 3)}\n'
                  + f'F1 Score: {round(f1_pos, 3)}\nAccuracy: {round(accuracy, 3)}\n'
                  + '------- Evaluation for negative class -------\n'
                  + f'Precision: {round(precision_neg, 3)}\nRecall: {round(recall_neg, 3)}\n'
                  + f'F1 Score: {round(f1_neg, 3)}\nAccuracy: {round(accuracy, 3)}\n')
        else:
            return round(f1_neg, 3), round(f1_pos, 3), round(accuracy, 3)

    '''
    Performs feature selection.
    Returns a dictionary of features.
    '''

    def select_features(self, train_set, select_rate):
        # initialize a new conditional class count dictionary
        conditional_class = defaultdict(dict)
        # initialize word counts, vocabulary and mutual information dictionary
        word_count = defaultdict(int)
        vocabulary = set()
        mutual_info = {}

        # Use mutual information to select features
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    word_list = re.split('[^A-Za-z\']+', f.read())

                    # collect class counts
                    if root.find('pos') != -1:
                        self.class_count['pos'] += 1
                        doc_class = 1
                    elif root.find('neg') != -1:
                        self.class_count['neg'] += 1
                        doc_class = 0
                    else:
                        raise ValueError('The class names are not in file root')

                    # collect word counts and conditional class count
                    visited_word = set()
                    for word in word_list:
                        if word == '':
                            continue
                        word_count[word] += 1

                        if word not in vocabulary:
                            conditional_class[word] = defaultdict(int)
                            vocabulary.add(word)

                        if word not in visited_word:
                            conditional_class[word][doc_class] += 1
                            visited_word.add(word)

        # function to calculate information entropy (in this case, cross entropy)
        def entropy(p):
            if p == 0 or p == 1:
                return 0
            else:
                return -(p * np.log(p) + (1 - p) * np.log(1 - p))

        pos_prop = self.class_count['pos'] / sum(self.class_count.values())
        class_entropy = entropy(pos_prop)

        total_words = sum(word_count.values())
        for word in vocabulary:
            word_in_pos_class = conditional_class[word][1] / sum(conditional_class[word].values())  # P(c|w)

            not_word_in_pos_class = (self.class_count['pos'] - conditional_class[word][1]) / (
                    sum(self.class_count.values()) - sum(conditional_class[word].values()))  # P(c|not w)

            word_prop = word_count[word] / total_words  # P(w)
            conditional_entropy = (word_prop * entropy(word_in_pos_class)
                                   + (1 - word_prop) * entropy(not_word_in_pos_class))

            mutual_info[word] = class_entropy - conditional_entropy

        number_features = int(len(vocabulary) * select_rate)
        mutual_info = sorted(mutual_info.items(), key=lambda x: x[1], reverse=True)[0:number_features]
        features_selected = set([feature[0] for feature in mutual_info])

        return features_selected

    # grid search on different selection rates
    def grid_search(self, train_set, validation_set, select_rates):
        grid_results = pd.DataFrame(columns=['neg_f1', 'pos_f1', 'accuracy'])
        for select_rate in select_rates:
            self.__init__()
            self.train(train_set, feature_select_rate=select_rate)
            results_valid = self.test(validation_set)
            (grid_results.loc[select_rate, 'neg_f1'], grid_results.loc[select_rate, 'pos_f1'],
             grid_results.loc[select_rate, 'accuracy']) = self.evaluate(results_valid, mode='grid_search')

        return grid_results


if __name__ == '__main__':
    # select_rates = [x / 100 for x in range(1, 100, 5)]
    # nb = NaiveBayes()
    # gs_result = nb.grid_search('movie_reviews/train', 'movie_reviews/dev', select_rates=select_rates)
    # print(gs_result)
    # gs_result.plot()
    # plt.show()

    # make sure these point to the right directories
    nb = NaiveBayes()
    nb.train('movie_reviews/train', feature_select_rate=0.026)
    # nb.train('movie_reviews_small/train')
    results = nb.test('movie_reviews/dev')
    # results = nb.test('movie_reviews_small/test')
    nb.evaluate(results)
