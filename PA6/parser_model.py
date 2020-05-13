#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS114 Spring 2020 Programming Assignment 6
Neural Transition-Based Dependency Parsing
Adapted from:
CS224N 2019-20: Homework 3
parser_model.py: Feed-Forward Neural Network for Dependency Parsing
Sahil Chopra <schopra8@stanford.edu>
Haoshen Hong <haoshen@stanford.edu>
"""
import argparse
import numpy as np

def softmax(x):
    stable_x = np.exp(x - np.max(x))
    y = stable_x / np.sum(stable_x, axis=1)[:, None]
    return y

def relu(x):
    """ Compute the ReLU function.

    @param x (ndarray): tensor of z (scores)

    @return y (ndarray): tensor of ReLU(z) for each z in x
    """
    ### YOUR CODE HERE (1 Line)
    ### TODO:
    ###     Compute the ReLU function: ReLU(z) = max(z, 0).
    ###     Be sure to take advantage of Numpy universal functions!



    ### END YOUR CODE
    return y

class ParserModel():
    """ Feedforward neural network with an embedding layer and two hidden layers.
    The ParserModel will predict which transition should be applied to a
    given partial parse configuration.
    """
    def __init__(self, embeddings, n_features=36,
        hidden_size=200, n_classes=3, lr=0.0005):
        """ Initialize the parser model.

        @param embeddings (ndarray): word embeddings (num_words, embedding_size)
        @param n_features (int): number of input features
        @param hidden_size (int): number of hidden units
        @param n_classes (int): number of output classes
        @param lr (float): Learning rate
        """
        super(ParserModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.embeddings = embeddings / np.maximum(np.max(embeddings), 1)

        ### YOUR CODE HERE (~3 Lines)
        # 1) Initialize weights for each layer according to a random uniform distribution.
        #    It is strongly recommended that you initialize weights within the following intervals:
        #        1st hidden layer: [-0.05, 0.05]
        #        2nd hidden layer: [-0.1, 0.1]
        #        Output layer: [-0.1, 0.1]
        self.w1 = np.random.uniform(-0.05, 0.05, (self.embed_size, self.hidden_size))
        self.w2 = np.random.uniform(-0.1, 0.1, (self.hidden_size, self.hidden_size))
        self.u = np.random.uniform(-0.1, 0.1, (self.hidden_size, self.n_classes))
        ### END YOUR CODE

    def embedding_lookup(self, w):
        """ Utilize `w` to select embeddings from embedding matrix `self.embeddings`
            @param w (ndarray): input tensor of word indices (batch_size, n_features)

            @return x (ndarray): tensor of embeddings for words represented in w
                                 (batch_size, n_features * embed_size)
        """

        ### YOUR CODE HERE (~1-3 Lines)
        ### TODO:
        ###     1) For each index `i` in `w`, select `i`th vector from self.embeddings
        ###     2) Reshape the tensor using `reshape` function if necessary
        ###
        ### Note: All embedding vectors are stacked and stored as a matrix. The model receives
        ###       a list of indices representing a sequence of words, then it calls this lookup
        ###       function to map indices to sequence of embeddings.
        ###
        ###       Pay attention to tensor shapes and reshape if necessary.
        ###       Make sure you know each tensor's shape before you run the code!

        ### END YOUR CODE
        return x


    def forward(self, w):
        """ Run the model forward.

        @param w (ndarray): input tensor of tokens (batch_size, n_features)

        @return outputs (list of ndarray): list of tensors of outputs of each layer of the network
                                           The last element of this list should be the
                                           tensor of predictions (output after applying the layers of the network)
                                           (batch_size, n_classes)
        """
        ### YOUR CODE HERE (~11 Lines)
        ### TODO:
        ###     Complete the forward computation as described in write-up.



        ### END YOUR CODE
        return outputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simple sanity check for parser_model.py')
    parser.add_argument('-e', '--embedding', action='store_true', help='sanity check for embeding_lookup function')
    parser.add_argument('-f', '--forward', action='store_true', help='sanity check for forward function')
    args = parser.parse_args()

    embeddings = np.zeros((100, 30), dtype=np.float32)
    model = ParserModel(embeddings)

    def check_embedding():
        inds = np.random.randint(0, 100, (4, 36), dtype=np.int32)
        selected = model.embedding_lookup(inds)
        assert np.all(selected == 0), "The result of embedding lookup: " \
                                      + repr(selected) + " contains non-zero elements."

    def check_forward():
        inputs = np.random.randint(0, 100, (4, 36), dtype=np.int32)
        out = model.forward(inputs)[-1]
        expected_out_shape = (4, 3)
        assert out.shape == expected_out_shape, "The result shape of forward is: " + repr(out.shape) + \
                                                " which doesn't match expected " + repr(expected_out_shape)

    if args.embedding:
        check_embedding()
        print("Embedding_lookup sanity check passes!")

    if args.forward:
        check_forward()
        print("Forward sanity check passes!")
