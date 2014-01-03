'''
Created on Jan 1, 2014

@author: sumanravuri
'''

import numpy as np
from read_pfile_v2 import read_pfile
import scipy.io as sp
#import scipy.linalg as sl
#import scipy.optimize as sopt
#import math
import copy
#import argparse
from context_window import context_window
from SVM_HMM_Weight import SVM_HMM_Weight
from SVM_HMM_base import SVM_HMM_base
import forward_cython

class SVM_HMM(SVM_HMM_base):
    def __init__(self, feature_file_name, weight_name = None, label_file_name = None, num_labels = None, context_window = 1):
        super(SVM_HMM,self).__init__(feature_file_name, weight_name, label_file_name, num_labels, context_window)
    
    def find_most_violated_sentence_labels(self, sent_features, labels, loss_matrix):
#        emission_features = np.dot(sent_features, self.weights.feature_weights) + self.weights.bias
        emission_features = self.classify_parallel_dot(sent_features, self.weights.feature_weights, self.weights.bias)
        loss_scores = np.zeros(emission_features.shape)
#        num_labels = self.weights.time_weights.shape[0]
        label = labels[0]
        loss_scores[0] = loss_matrix[label] + emission_features[0] + self.weights.start_time_weights
        argmax_features = np.zeros(emission_features.shape)
        num_emission_features = emission_features.shape[0]
        outputs = np.empty((num_emission_features,), dtype=int)
        for feature_index in range(1,num_emission_features):
            previous_time_feature = loss_scores[feature_index-1]
            current_emission_feature = emission_features[feature_index]
            label = labels[feature_index]
            current_loss_scores = previous_time_feature.T[:,np.newaxis] + self.weights.time_weights + (current_emission_feature + loss_matrix[label])[np.newaxis,:]
            argmax_features[feature_index] = np.argmax(current_loss_scores, axis=0)
            loss_scores[feature_index] = np.max(current_loss_scores, axis=0)
        
        outputs[-1] = np.argmax(loss_scores[-1])
        for feature_index in range(num_emission_features-1, 0, -1):
            output = outputs[feature_index]
            outputs[feature_index-1] = argmax_features[feature_index, output]
#        print outputs
        return outputs, loss_scores
    
    def update_gradient(self, sent_features, sent_labels, most_violated_sequence, gradient):
        num_sent_features = sent_features.shape[0]
        
        gradient.feature_weights[:,sent_labels[0]] -= sent_features[0]
        gradient.feature_weights[:,most_violated_sequence[0]] += sent_features[0]
        
        gradient.bias[sent_labels[0]] -= 1.0
        gradient.bias[most_violated_sequence[0]] += 1.0
        
        gradient.start_time_weights[sent_labels[0]] -= 1.0
        gradient.start_time_weights[most_violated_sequence[0]] += 1.0
        
#        previous_label = sent_labels[0]
#        previous_violated_label = most_violated_sequence[0]
        for observation_index in range(1,num_sent_features):
            feature = sent_features[observation_index]
            current_label = sent_labels[observation_index]
            current_violated_label = most_violated_sequence[observation_index]
#            print current_violated_label, current_label
            gradient.feature_weights[:,current_label] -= feature
            gradient.feature_weights[:,current_violated_label] += feature
            gradient.bias[current_label] -= 1.0
            gradient.bias[current_violated_label] += 1.0
#            previous_label = current_label
#            previous_violated_label = current_violated_label
        
        gradient.time_weights[sent_labels[:-1], sent_labels[1:]] -= 1.0
        gradient.time_weights[most_violated_sequence[:-1], most_violated_sequence[1:]] += 1.0
        
        return gradient
    
    def train(self, lambda_const = 0.5, batch_size = 128, num_epochs = 100):
        """train using structured Pegasos algorithm
        """
        batch_size = min(batch_size, self.num_examples)
        gradient = SVM_HMM_Weight(self.num_dims, self.num_labels, init_zero_weights = True)
        weight_norm = self.weights.norm()
        if weight_norm > 1. / np.sqrt(lambda_const):
#            print "projecting"
            self.weights.two_norm_project(new_norm_size = 1. / np.sqrt(lambda_const) )
        self.classify(self.frame_table, sent_indices=range(self.num_examples))
        for epoch_num in range(1,num_epochs+1):
            print "At epoch number", epoch_num
            learning_rate = 1. / (lambda_const * epoch_num)
            sentence_indices = np.random.permutation(self.num_examples)[:batch_size]
            gradient *= 0.0
            for sent_index in sentence_indices:
                start_frame = self.frame_table[sent_index]
                end_frame = self.frame_table[sent_index+1]
                num_observations = end_frame - start_frame
                sent_labels = self.labels[start_frame:end_frame]
                sent_features, last_sent_index = self.return_sequence_chunk(self.frame_table, sent_index, num_observations)
                most_violated_sequence, loss_scores = self.find_most_violated_sentence_labels(sent_features, self.labels, self.loss_matrix)
                gradient = self.update_gradient(sent_features, sent_labels, most_violated_sequence, gradient)
            self.weights = self.weights * (1. - 1. / epoch_num) - gradient * (learning_rate / batch_size)
#            self.weights.time_weights = (1. - 1. / epoch_num) * gradient.feature_weights + learning_rate / batch_size * gradient.feature_weights
            weight_norm = self.weights.norm()
#            print weight_norm
            if weight_norm > 1. / np.sqrt(lambda_const):
#                print "projecting"
                self.weights.two_norm_project(new_norm_size = 1. / np.sqrt(lambda_const) )
            self.classify(self.frame_table, sent_indices=range(self.num_examples))