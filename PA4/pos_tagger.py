# CS114 Spring 2020 Programming Assignment 4
# Part-of-speech Tagging with Hidden Markov Models

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import lil_matrix


class POSTagger:

    def __init__(self, k=1.0):
        self.pos_dict = {}
        self.reversed_pos_dict = {}
        self.word_dict = {}
        self.initial = None
        self.transition = None
        self.emission = None
        self.total = None
        self.UNK = '<UNK>'
        self.k = k

    '''
    Trains a supervised hidden Markov model on a training set.
    self.initial[POS] = log(P(the initial tag is POS))
    self.transition[POS1][POS2] =
    log(P(the current tag is POS2|the previous tag is POS1))
    self.emission[POS][word] =
    log(P(the current word is word|the current tag is POS))
    '''
    def train(self, train_set):
        word_set = set()
        pos_set = set()
        initial = defaultdict(int)
        transition = defaultdict(lambda : defaultdict(int))
        emission = defaultdict(lambda : defaultdict(int))

        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # be sure to split documents into sentences here
                    sentences = f.read().split('\n')

                    for sentence in sentences:
                        if not sentence:
                            continue
                        word_pos_list = sentence.split()
                        for i, word_pos in enumerate(word_pos_list):
                            # Write like this because the word may contain '/'
                            pos = tuple(word_pos.split('/'))[-1]
                            word = '/'.join(word_pos.split('/')[:-1])
                            # fill word set and pos set
                            word_set.add(word)
                            pos_set.add(pos)
                            if i == 0:
                                initial[pos] += 1  # count of the initial pos
                            else:
                                pre_word_pos = word_pos_list[i-1]  # previous word/tag
                                pre_pos = tuple(pre_word_pos.split('/'))[-1]
                                transition[pre_pos][pos] += 1
                                emission[pos][word] += 1

        # Set the unknown word
        for pos in emission:
            emission[pos][self.UNK] = 1

        # Set the word dict and pos dict. These are reversed dict (value: index)
        # because in this project we usually need to find the index of the word or pos
        word_set.add(self.UNK)
        self.word_dict = {k: i for i, k in enumerate(sorted(word_set))}
        self.pos_dict = {k: i for i, k in enumerate(sorted(pos_set))}
        self.reversed_pos_dict = {i: k for i, k in enumerate(sorted(pos_set))}

        # Generate needed matrix with add-k smoothing
        # initial probability distribution and transition matrix A
        self.initial = np.ndarray((len(pos_set), 1))
        self.transition = np.ndarray((len(pos_set), len(pos_set)))
        for pos1, i in self.pos_dict.items():
            self.initial[i] = initial[pos1]
            for pos2, j in self.pos_dict.items():
                self.transition[i, j] = transition[pos1][pos2]

        self.initial += self.k
        self.initial = np.log(self.initial / self.initial.sum())
        self.transition += self.k
        self.transition = np.log(self.transition / self.transition.sum(axis=1))

        # emission matrix B (use sparse matrix because the array shape is too large)
        self.emission = lil_matrix((len(pos_set), len(word_set)))
        temp_matrix = lil_matrix((len(pos_set), len(word_set)))
        for pos, word_dict in emission.items():
            for word, num in word_dict.items():
                i = self.pos_dict[pos]
                j = self.word_dict[word]
                temp_matrix[i, j] = num
                self.emission[i, j] = num + self.k
        self.total = temp_matrix.sum(axis=1) + self.k*len(word_set)
        self.emission = self.emission.multiply(1 / self.total).tolil()

        return

    '''
    Implements the Viterbi algorithm.
    Use v and backpointer to find the best_path.
    '''
    def viterbi(self, sentence):
        # the input sentence should be a word index list
        # Note that the emission matrix is sparse matrix (without add-k method on 0 values)and did not take log
        v = None
        backpointer = None
        # initialization step
        length_pos = len(self.pos_dict)
        b = self.emission[:, sentence[0]].toarray()
        v = np.log(np.where(b == 0, self.k/self.total, b)).reshape((length_pos, 1)) + self.initial
        backpointer = np.zeros((length_pos, 1))

        # recursion step
        for t_step, word_index in enumerate(sentence):
            if t_step == 0:
                continue
            b = self.emission[:, word_index].toarray()
            b = np.log(np.where(b == 0, self.k/self.total, b)).reshape((1, length_pos))
            v_temp = v[:, t_step-1].reshape((length_pos, 1)) + self.transition + b
            v_t = v_temp.max(axis=0).reshape((length_pos, 1))
            backpointer_t = v_temp.argmax(axis=0).reshape((length_pos, 1))
            v = np.append(v, v_t, axis=1)
            backpointer = np.append(backpointer, backpointer_t, axis=1)

        # termination step
        best_path = []
        best_path_pointer = int(v[:, -1].argmax())
        best_path.append(self.reversed_pos_dict[best_path_pointer])
        t = len(sentence) - 1
        while t > 0:
            best_path_pointer = int(backpointer[best_path_pointer, t])
            best_path.append(self.reversed_pos_dict[best_path_pointer])
            t -= 1

        # reverse the path
        best_path = best_path[::-1]

        return best_path

    '''
    Tests the tagger on a development or test set.
    Returns a dictionary of sentence_ids mapped to their correct and predicted
    sequences of POS tags such that:
    results[sentence_id]['correct'] = correct sequence of POS tags
    results[sentence_id]['predicted'] = predicted sequence of POS tags
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    sentences = f.read().split('\n')
                    sentence_index = 0
                    for sentence in sentences:

                        if not sentence:
                            continue
                        word_list = []
                        pos_list = []
                        word_pos_list = sentence.split()
                        for word_pos in word_pos_list:
                            pos = tuple(word_pos.split('/'))[-1]
                            word = '/'.join(word_pos.split('/')[:-1])
                            # Deal with unknown word
                            if word not in self.word_dict:
                                word = self.UNK
                            pos_list.append(pos)
                            word_list.append(self.word_dict[word])

                        sentence_id = name + '_' + str(sentence_index)
                        results[sentence_id]['correct'] = pos_list
                        results[sentence_id]['predicted'] = self.viterbi(word_list)
                        sentence_index += 1

        return results

    '''
    Given results, calculates overall accuracy.
    '''
    def evaluate(self, results):
        correct_word = 0
        total_word = 0

        for sentence_result in results.values():
            total_word += len(sentence_result['predicted'])
            correct_word += np.sum(np.array(sentence_result['predicted']) == np.array(sentence_result['correct']))

        accuracy = correct_word / total_word
        return accuracy

    '''
    Grid search to find the best k
    '''
    def grid_search(self, k_list):
        result_df = pd.DataFrame(columns=['k', 'accuracy'])
        for i, k in enumerate(k_list):
            self.__init__(k)
            self.train(r"C:\Users\Henry's Smart Brick\Desktop\NLP\PA4\brown\train")
            results = self.test(r"C:\Users\Henry's Smart Brick\Desktop\NLP\PA4\brown\dev")
            result_df.loc[i, 'k'] = k
            result_df.loc[i, 'accuracy'] = self.evaluate(results)
        print(result_df)


if __name__ == '__main__':
    pos = POSTagger(0.08)
    # make sure these point to the right directories
    pos.train(r"C:\Users\Henry's Smart Brick\Desktop\NLP\PA4\brown\train")
    results = pos.test(r"C:\Users\Henry's Smart Brick\Desktop\NLP\PA4\brown\dev")
    print('Accuracy:', pos.evaluate(results))
    # k_list = [x/100 for x in range(1, 11)]
    # pos.grid_search(k_list)
