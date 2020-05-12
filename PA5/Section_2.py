import numpy as np
import pandas as pd
from scipy.linalg import norm
import random


class EmbeddingNew:
    def __init__(self):
        self.word_dict = {}
        self.reversed_word_dict = {}
        self.test_set = None
        self.test_correct = None
        self.semantic_matrix = None

    def create_test_set(self, data_path):
        # Read the synonym verb data
        synonym_df = pd.read_csv(data_path, sep='\t')
        synonym_df = synonym_df.applymap(lambda x: x.split('_')[-1] if x != '0' else x)
        synonym_df = synonym_df[synonym_df['Answer.suggestion'] != '0'].drop_duplicates().reset_index()
        test_set = pd.DataFrame(columns=['word', 'multiple_choices'])

        # Generate 1000 synonym multiple-choice questions
        index_choices = random.choices(range(len(synonym_df)), k=1000)
        for i in index_choices:
            word = synonym_df.loc[i, 'Input.word']
            word_choices = [synonym_df.loc[i, 'Answer.suggestion']]
            # choose 4 words from non-synonymous words
            multiple_choices = random.choices(synonym_df[synonym_df['Input.word'] != word].index, k=4)
            for j in multiple_choices:
                word_choices.append(synonym_df.loc[j, 'Answer.suggestion'])

            test_set = test_set.append({'word': word, 'multiple_choices': word_choices}, ignore_index=True)

        self.test_set = test_set
        self.test_correct = synonym_df[['Input.word', 'Answer.suggestion']]

    def load_test_set(self, test_set, test_correct):
        self.test_set = pd.read_csv(test_set)
        self.test_set['multiple_choices'] = self.test_set['multiple_choices'].apply(lambda x: eval(x))
        self.test_correct = pd.read_csv(test_correct)

    # Generate the word semantic matrix from word vectors files
    def generate_semantic_matrix(self, file_path):
        word_vector = pd.read_csv(file_path, sep=' ', header=None, quoting=3)
        self.word_dict = word_vector[0].to_dict()
        self.reversed_word_dict = {i: k for k, i in self.word_dict.items()}
        self.semantic_matrix = word_vector.loc[:, 1:].to_numpy()

    def get_word_vector(self, word):
        # deal with the unknown words
        if word in self.reversed_word_dict:
            return self.semantic_matrix[self.reversed_word_dict[word], ]
        else:
            # set zero vector to unknown word
            return np.zeros(len(self.semantic_matrix[0])) + 1e-16

    def test(self, method):
        accurate_num = 0
        for _, row in self.test_set.iterrows():
            base_word = row['word']
            base_word_vector = self.get_word_vector(base_word)
            choice_words = row['multiple_choices']

            # find the shortest distance (most synonymous)
            distance = np.inf
            prediction = None
            for choice in choice_words:
                choice_word_vector = self.get_word_vector(choice)
                # Euclidean distance
                if method == 'eu':
                    temp_distance = norm(choice_word_vector-base_word_vector)
                # cosine similarity (here used cosine distance which is 1-cosine similarity)
                elif method == 'cos':
                    temp_distance = 1 - np.dot(base_word_vector, choice_word_vector) \
                                    / (norm(base_word_vector)*norm(choice_word_vector))
                else:
                    raise ValueError("The method should be in {'eu', 'cos'}")
                if temp_distance < distance:
                    prediction = choice
                    distance = temp_distance

            # check if the prediction is correct
            answers = self.test_correct.loc[self.test_correct['Input.word'] == base_word, 'Answer.suggestion'].values
            if prediction in answers:
                accurate_num += 1

        accuracy = accurate_num / len(self.test_set)
        return accuracy


class Analogy(EmbeddingNew):
    def __init__(self):
        super().__init__()

    def create_test_set(self, data_path):
        self.test_set = pd.DataFrame(columns=['base', 'choices', 'answer'])
        word_pairs = []
        with open(data_path) as f:
            i = 0
            for line in f.readlines():
                if line.startswith('#'):  # skip comments
                    continue
                elif not line.strip():  # skip blank line
                    continue
                elif i % 8 == 0:  # each question has 8 lines, skip the first line with introduction
                    i += 1
                    continue

                if i % 8 == 7:  # last line is the answer
                    answer = line.strip()
                    self.test_set = self.test_set.append({'base': word_pairs[0],
                                                          'choices': word_pairs[1:],
                                                          'answer': answer}, ignore_index=True)
                    word_pairs = []
                else:
                    word_pairs.append(line.strip().split())
                i += 1
        return

    def create_vector(self, word_pair=(), method=None):
        word_1, word_2 = word_pair
        return method(self.get_word_vector(word_1), self.get_word_vector(word_2))

    def test(self, method):
        accurate_num = 0
        alphabet = 'abcdefg'
        for _, row in self.test_set.iterrows():
            base_word_pair = tuple(row['base'][0:2])
            base_word_vector = self.create_vector(base_word_pair, method)

            # find the shortest cosine distance
            choice_pairs = row['choices']
            distance = np.inf
            prediction = None
            for i, choice_pair in enumerate(choice_pairs):
                choice_word_vector = self.create_vector(tuple(choice_pair[0:2]), method)
                # cosine distance
                temp_distance = 1 - np.dot(base_word_vector, choice_word_vector)\
                                / (norm(base_word_vector) * norm(choice_word_vector))
                if temp_distance < distance:
                    prediction = alphabet[i]
                    distance = temp_distance

            # check if the prediction is right
            if prediction == row['answer']:
                accurate_num += 1

        accuracy = accurate_num / len(self.test_set)
        return accuracy


if __name__ == '__main__':
    embedding_new = EmbeddingNew()
    # Generate test set and save as csv file
    # embedding_new.create_test_set(r"C:\Users\Henry's Smart Brick\Desktop\NLP\PA5\EN_syn_verb.txt")
    # embedding_new.test_set.to_csv('test_set.csv', index=None)
    # embedding_new.test_correct.to_csv('test_correct.csv', index=None)

    embedding_new.load_test_set('test_set.csv', 'test_correct.csv')
    result_df = pd.DataFrame(columns=['word_vector', 'method', 'accuracy'])

    # COMPOSES word vectors
    embedding_new.generate_semantic_matrix(r"C:\Users\Henry's Smart Brick\Desktop\NLP\PA5"
                                           r"\EN-wform.w.2.ppmi.svd.500.rcv_vocab.txt")
    # euclidean distance
    result_df = result_df.append({'word_vector': 'COMPOSES',
                      'method': 'Euclidean distance',
                      'accuracy': embedding_new.test('eu')}, ignore_index=True)
    # cosine similarity
    result_df = result_df.append({'word_vector': 'COMPOSES',
                      'method': 'cosine similarity',
                      'accuracy': embedding_new.test('cos')}, ignore_index=True)
    # word2vec
    embedding_new.generate_semantic_matrix(r"C:\Users\Henry's Smart Brick\Desktop\NLP\PA5"
                                           r"\GoogleNews-vectors-rcv_vocab.txt")
    # euclidean distance
    result_df = result_df.append({'word_vector': 'word2vec',
                      'method': 'Euclidean distance',
                      'accuracy': embedding_new.test('eu')}, ignore_index=True)
    # cosine similarity
    result_df = result_df.append({'word_vector': 'word2vec',
                      'method': 'cosine similarity',
                      'accuracy': embedding_new.test('cos')}, ignore_index=True)

    result_df = result_df.set_index(['word_vector', 'method']).unstack(-1)
    print(result_df)

    # SAT Analogy
    analogy = Analogy()
    analogy.create_test_set(r"C:\Users\Henry's Smart Brick\Desktop\NLP\PA5\SAT-package-V3.txt")
    analogy.generate_semantic_matrix(r"C:\Users\Henry's Smart Brick\Desktop\NLP\PA5\GoogleNews-vectors-rcv_vocab.txt")

    # subtraction method
    def subtraction(vector_1, vector_2):
        return vector_1 - vector_2

    # adding
    def add(vector_1, vector_2):
        return vector_1 + vector_2

    # multiplying
    def multiply(vector_1, vector_2):
        return vector_1 * vector_2

    # concatenating
    def concatenate(vector_1, vector_2):
        return np.concatenate((vector_1, vector_2), axis=None)


    result_df = pd.DataFrame(columns=['word_vector', 'method', 'accuracy'])
    result_df = result_df.append({'word_vector': 'word2vec',
                                  'method': 'subtraction',
                                  'accuracy': analogy.test(subtraction)}, ignore_index=True)
    result_df = result_df.append({'word_vector': 'word2vec',
                                  'method': 'add',
                                  'accuracy': analogy.test(add)}, ignore_index=True)
    result_df = result_df.append({'word_vector': 'word2vec',
                                  'method': 'multiply',
                                  'accuracy': analogy.test(multiply)}, ignore_index=True)
    result_df = result_df.append({'word_vector': 'word2vec',
                                  'method': 'concatenate',
                                  'accuracy': analogy.test(concatenate)}, ignore_index=True)

    print(result_df)