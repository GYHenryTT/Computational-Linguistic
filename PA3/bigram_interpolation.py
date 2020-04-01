# CS114 Spring 2020 Programming Assignment 3
# N-gram Language Models

from collections import defaultdict
from languageModel import LanguageModel
import numpy as np
from unigram import Unigram
from bigram import Bigram


class BigramInterpolation(LanguageModel):

    def __init__(self, lambda_1=0.67):
        self.unigram = Unigram()
        self.bigram = Bigram()
        # just needed for languageModel.py to work
        self.word_dict = self.bigram.word_dict
        self.lambda_1 = lambda_1
        self.lambda_2 = 1 - lambda_1
    
    '''
    Trains a bigram-interpolation language model on a training set.
    '''
    def train(self, trainingSentences):
        self.unigram.train(trainingSentences)
        self.bigram.train(trainingSentences)
    
    '''
    Returns the probability of the word at index, according to the model, within
    the specified sentence.
    '''
    def getWordProbability(self, sentence, index):
        return (self.lambda_1*self.bigram.getWordProbability(sentence, index)
                +self.lambda_2*self.unigram.getWordProbability(sentence, index))

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

        probs_bigram = self.bigram.prob_counter[previous_word_index].toarray().ravel()
        probs_unigram = self.unigram.prob_counter[0].toarray().ravel()

        # Because the unigram model and bigram model have different word index for STOP, I need to make some adjustment
        stop_index = self.unigram.word_dict[LanguageModel.STOP]
        # move STOP probability to the first element of probs_unigram and leave the others unchanged
        stop_prob = probs_unigram[stop_index]
        probs_unigram = np.append(stop_prob, np.delete(probs_unigram, stop_index))
        probs = self.lambda_1*probs_bigram + self.lambda_2*probs_unigram  # Get the interpolation probability

        word_list = sorted(self.word_dict.items(), key=lambda item: item[1])
        word_list = [k[0] for k in word_list]

        return np.random.choice(word_list, p=probs)
