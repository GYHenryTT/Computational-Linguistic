import numpy as np
import pandas as pd
from scipy.linalg import norm, svd
from collections import defaultdict


# The embedding method used in Section 1
class Embedding:
    def __init__(self):
        self.co_occurrence = None
        self.word_dict = None
        self.reversed_word_dict = None
        self.ppmi = None
        self.word_count = None

    def word_to_vector(self, train_set):
        word_count = defaultdict(int)
        word_co_occurrence = defaultdict(lambda: defaultdict(int))

        with open(train_set) as f:
            for sentence in f.readlines():
                if sentence == '\n':
                    continue

                # Generate the word count dict
                words = sentence.split()
                for i, word in enumerate(words):
                    if word == '\n':
                        continue
                    elif i == 0:
                        word_count[word] += 1
                        continue

                    word_count[word] += 1
                    word_co_occurrence[word][words[i-1]] += 1
                    word_co_occurrence[words[i-1]][word] += 1

        self.word_count = word_count
        # Generate word dict
        vocabulary = sorted(word_count)
        v_length = len(vocabulary)
        self.word_dict = {i: k for i, k in enumerate(vocabulary)}
        self.reversed_word_dict = {i: k for k, i in self.word_dict.items()}

        # Create co_occurrence matrix from word_count
        self.co_occurrence = np.ndarray((v_length, v_length))
        for i in range(v_length):
            for j in range(v_length):
                self.co_occurrence[i, j] = word_co_occurrence[self.word_dict[i]][self.word_dict[j]]

        # Multiply the co-occurrence matrix by 10 and apply add-1 smoothing
        self.co_occurrence = self.co_occurrence*10 + 1

        # Compute the PPMI matrix
        p_w = self.co_occurrence.sum(axis=0) / self.co_occurrence.sum()
        p_c = self.co_occurrence.sum(axis=1) / self.co_occurrence.sum()
        p_cw = self.co_occurrence / self.co_occurrence.sum()
        self.ppmi = np.log2(p_cw / p_w.reshape((1, v_length)) / p_c.reshape((v_length, 1)))
        self.ppmi = np.where(self.ppmi < 0, 0, self.ppmi)

    def get_word_vector(self, word):
        word_index = self.reversed_word_dict[word]
        return self.co_occurrence[:, word_index], self.ppmi[:, word_index]

    def euclidean_distance(self, word_1, word_2):
        _, vector_1 = self.get_word_vector(word_1)
        _, vector_2 = self.get_word_vector(word_2)
        return norm(vector_1-vector_2)


if __name__ == '__main__':
    embedding = Embedding()
    embedding.word_to_vector(r"C:\Users\Henry's Smart Brick\Desktop\NLP\PA5\dist_sim_data.txt")

    # vector for dogs
    print('\nword dict: ')
    print(embedding.word_dict)
    count_vector, ppmi_vector = embedding.get_word_vector('dogs')
    print(f'\nCount vector for dogs:')
    print(count_vector)
    print(f'PPMI vector for dogs: ')
    print(ppmi_vector)

    # euclidean distance
    distance_df = pd.DataFrame(columns=['Word_pairs', 'Euclidean_distance'])
    word_pairs = [['women', 'men'],
                  ['women', 'dogs'],
                  ['men', 'dogs'],
                  ['feed', 'like'],
                  ['feed', 'bite'],
                  ['like', 'bite']]
    for i, word_pair in enumerate(word_pairs):
        distance_df.loc[i, 'Word_pairs'] = word_pair
        distance_df.loc[i, 'Euclidean_distance'] = embedding.euclidean_distance(word_pair[0], word_pair[1])

    print('\nDistance between word pairs:')
    print(distance_df)

    # SVD
    print('\nOriginal PPMI matrix:')
    print(embedding.ppmi)
    U, E, V_t = svd(embedding.ppmi)
    U = np.matrix(U)  # compute U
    E = np.matrix(np.diag(E)) # compute E
    V_t = np.matrix(V_t)
    V = V_t.T
    # Recover the matrix
    print('\nRecovered PPMI matrix by multiply U, E, and Vt:')
    print(np.dot(U, np.dot(E, V_t)))

    # Reduce the dimensions of PPMI matrix
    reduced_PPMI = embedding.ppmi * V[:, 0:3]
    embedding.ppmi = reduced_PPMI.T
    distance_df = pd.DataFrame(columns=['Word_pairs', 'Euclidean_distance'])
    for i, word_pair in enumerate(word_pairs):
        distance_df.loc[i, 'Word_pairs'] = word_pair
        distance_df.loc[i, 'Euclidean_distance'] = embedding.euclidean_distance(word_pair[0], word_pair[1])

    print('\nnew PPMI vector in dimensions of 3:')
    print(reduced_PPMI)
    print('\nDistance between word pairs: ')
    print(distance_df)