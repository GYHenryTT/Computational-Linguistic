# CS114 Spring 2020 Programming Assignment 3
# N-gram Language Models

from collections import defaultdict
from languageModel import LanguageModel
import numpy as np
from scipy.sparse import lil_matrix


class BigramAddK(LanguageModel):

    def __init__(self, k=0.002):
        self.word_dict = {}
        self.total = None
        self.prob_counter = None
        self.k = k
    
    '''
    Trains a bigram-add-k language model on a training set.
    '''
    def train(self, trainingSentences):
        word_counts = defaultdict(lambda: defaultdict(int))

        # iterate over training sentences
        for sentence in trainingSentences:
            for i, word in enumerate(sentence):
                if i == 0:
                    word_counts[LanguageModel.START][word] += 1
                else:
                    word_counts[sentence[i - 1]][word] += 1  # word_counts[previous word][word]
            word_counts[sentence[-1]][LanguageModel.STOP] += 1

        # Deal with the unknown word counts
        for previous_word in list(word_counts.keys()):
            word_counts[previous_word][LanguageModel.UNK] += 1
            word_counts[LanguageModel.UNK][previous_word] += 1  # this will add an extra value [UNKNOWN][START]
        word_counts[LanguageModel.UNK][LanguageModel.UNK] += 1
        del word_counts[LanguageModel.UNK][LanguageModel.START]

        self.prob_counter = lil_matrix((len(word_counts), len(word_counts)))
        # sort words alphabetically
        # To simplify the procedure, set index 0 for START and STOP (will delete START afterwards)
        self.word_dict[LanguageModel.STOP] = 0
        self.word_dict[LanguageModel.START] = 0
        # set index for other words
        temp_word_list = sorted(word_counts)
        temp_word_list.remove(LanguageModel.START)
        for i, index_word in enumerate(temp_word_list):
            self.word_dict[index_word] = i + 1

        prob_counter_temp = lil_matrix((len(word_counts), len(word_counts)))
        for previous_word, word_dict in word_counts.items():
            for word in word_dict:
                i = self.word_dict[previous_word]
                j = self.word_dict[word]
                prob_counter_temp[i, j] = word_counts[previous_word][word]
                self.prob_counter[i, j] = word_counts[previous_word][word] + self.k  # add-k smoothing

        del self.word_dict[LanguageModel.START]  # START should not occur in word dict

        # Add k|V| to total count for previous words
        self.total = prob_counter_temp.sum(axis=1) + self.k * len(word_counts)
        # normalize counts to probabilities
        # to keep matrix sparse, use multiplication instead of division
        # also convert matrix back to lil format
        self.prob_counter = self.prob_counter.multiply(1 / self.total).tolil()
        return

    '''
    Returns the probability of the word at index, according to the model, within
    the specified sentence.
    '''
    def getWordProbability(self, sentence, index):
        # Note that START is not in self.word_dict
        previous_word_index = None
        if index == len(sentence):
            word = LanguageModel.STOP
            previous_word = sentence[-1]
        else:
            word = sentence[index]
            if index == 0:
                previous_word = LanguageModel.START
                previous_word_index = 0
            else:
                previous_word = sentence[index - 1]

        if word not in self.word_dict:
            word = LanguageModel.UNK

        word_index = self.word_dict[word]

        if (previous_word not in self.word_dict) and (previous_word != LanguageModel.START):
            previous_word = LanguageModel.UNK

        if previous_word_index is None:
            previous_word_index = self.word_dict[previous_word]

        prob = self.prob_counter[previous_word_index, word_index]

        # Use add-k smoothing here
        if prob == 0:
            prob = self.k / self.total[previous_word_index]

        return prob

    '''
    Returns, for a given context, a random word, according to the probabilities
    in the model.
    '''
    def generateWord(self, context):
        if context:
            previous_word = context[-1]
        else:
            previous_word = LanguageModel.START

        if (previous_word not in self.word_dict) and (previous_word != LanguageModel.START):
            previous_word = LanguageModel.UNK

        if previous_word == LanguageModel.START:
            previous_word_index = 0
        else:
            previous_word_index = self.word_dict[previous_word]

        probs = self.prob_counter[previous_word_index].toarray()
        probs = np.where(probs == 0, self.k / self.total[previous_word_index], probs).ravel()
        word_list = sorted(self.word_dict.items(), key=lambda item: item[1])
        word_list = [k[0] for k in word_list]

        return np.random.choice(word_list, p=probs)
