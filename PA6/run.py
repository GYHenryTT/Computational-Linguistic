#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS114 Spring 2020 Programming Assignment 6
Neural Transition-Based Dependency Parsing
Adapted from:
CS224N 2019-20: Homework 3
run.py: Run the dependency parser.
Sahil Chopra <schopra8@stanford.edu>
Haoshen Hong <haoshen@stanford.edu>
"""
from datetime import datetime
import os
import pickle
import math
import time
import argparse
import numpy as np

from parser_model import ParserModel
from utils.parser_utils import minibatches, load_and_preprocess_data, AverageMeter

parser = argparse.ArgumentParser(description='Train neural dependency parser in python')
parser.add_argument('-d', '--debug', action='store_true', help='whether to enter debug mode')
args = parser.parse_args()

def d_relu(x):
    """ Compute the derivative of the ReLU function.

    @param x (ndarray): tensor of ReLU outputs

    @return y (ndarray): tensor of derivatives at each ReLU output in x
    """
    ### YOUR CODE HERE (~2 Lines)
    ###     Compute the derivative of the ReLU function.
    ###     Be sure to take advantage of Numpy universal functions!
    ###     Note that by convention, we take the derivative of ReLU(z) at z = 0 to be 0.
    y = np.where(x > 0, 1, 0)
    ### END YOUR CODE
    return y

# -----------------
# Primary Functions
# -----------------
def train(parser, train_data, dev_data, batch_size=1024, n_epochs=10, lr=0.0005):
    """ Train the neural dependency parser.

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param batch_size (int): Number of examples in a single batch
    @param n_epochs (int): Number of training epochs
    @param lr (float): Learning rate
    """

    parser.lr = lr
    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        dev_UAS = train_for_epoch(parser, train_data, dev_data, batch_size)
        print("")


def train_for_epoch(parser, train_data, dev_data, batch_size):
    """ Train the neural dependency parser for single epoch.

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param batch_size (int): batch size

    @return dev_UAS (float): Unlabeled Attachment Score (UAS) for dev data
    """
    n_minibatches = math.ceil(len(train_data) / batch_size)
    loss_meter = AverageMeter()

    for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
        loss = 0. # store loss for this batch here

        ### YOUR CODE HERE (~11+ Lines)
        ###      1) Run train_x forward through model to produce outputs
        ###      2) Calculate the cross-entropy loss
        ###      3) Backprop losses
        ###      4) Update the model weights
        model = parser.model
        hidden1_output, hidden2_output, y_hat = model.forward(train_x)
        x_input = np.insert(train_x, train_x.shape[1], 1, axis=1)
        # cross-entropy loss
        loss -= np.sum(train_y * np.log(y_hat), axis=1)
        # Backprop losses
        outputs_delta = y_hat - train_y
        output_gradient = np.dot(hidden2_output.T, outputs_delta)
        hidden2_delta = np.dot(np.dot(outputs_delta, model.output_weights.T),
                               d_relu(np.dot(hidden1_output, model.hidden_weights2)))
        hidden2_gradient = np.dot(hidden1_output.T, hidden2_delta)
        hidden1_delta = np.dot(np.dot(hidden2_delta, model.hidden_weights2.T),
                               d_relu(np.dot(x_input, model.hidden_weights1)))
        hidden1_gradient = np.dot(x_input.T, hidden1_delta)
        # Update the model weights
        model.output_weights -= model.lr * output_gradient
        model.hidden_weights2 -= model.lr * hidden2_gradient
        model.hidden_weights1 -= model.lr * hidden1_gradient
        ### END YOUR CODE
        loss_meter.update(loss)

    print ("Average Train Loss: {}".format(loss_meter.avg))

    print("Evaluating on dev set",)
    dev_UAS, _ = parser.parse(dev_data)
    print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
    return dev_UAS


if __name__ == "__main__":
    debug = args.debug

    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)

    start = time.time()
    model = ParserModel(embeddings)
    parser.model = model
    print("took {:.2f} seconds\n".format(time.time() - start))

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")

    train(parser, train_data, dev_data, batch_size=1024, n_epochs=10, lr=0.0005)

    if not debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Final evaluation on test set",)
        UAS, dependencies = parser.parse(test_data)
        print("- test UAS: {:.2f}".format(UAS * 100.0))
        print("Done!")
