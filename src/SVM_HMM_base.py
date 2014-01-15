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
import forward_cython

class SVM_HMM_base(object):
    def __init__(self, feature_file_name, weight_name = None, label_file_name = None, num_labels = None, context_window = 1, random_seed = 0):
        self.feature_file_name = feature_file_name
        self.num_dims, self.frame_table = self.read_feature_file_stats(self.feature_file_name)
#        self.num_dims += 1 #add 1 for bias
        self.num_examples = self.frame_table.size - 1
        
        if label_file_name != None:
            self.label_file_name = label_file_name
            self.labels = self.read_label_file(self.label_file_name)
            self.num_labels = np.max(self.labels) + 1
        
        if weight_name != None:
            self.weight_name = weight_name
            self.weights = SVM_HMM_Weight(weight_name = self.weight_name)
        else:
            self.weights = SVM_HMM_Weight(num_dims = self.num_dims, num_labels = self.num_labels, random_seed = random_seed)
        
        self.context_window = context_window
        
    def read_label_file(self, label_file_name):
        try:
            data, frames, labels, frame_table = read_pfile(label_file_name)
        except IOError:
            raise IOError('Cannot Open File %s' % label_file_name)
        return labels
    
    def read_feature_file(self, feature_file_name):
        try:
            data, frames, labels, frame_table = read_pfile(feature_file_name)
        except IOError:
            raise IOError('Cannot Open File %s' % feature_file_name)
        return data
    
    def read_feature_file_stats(self, feature_file_name):
        try:
            return read_pfile(self.feature_file_name, True)
        except IOError:
            raise IOError('Cannot Open File %s' % feature_file_name)
        
    def return_sequence_chunk(self, frame_table, current_sent_index, chunk_size):
        """
        """
        try:
            last_sent_index = np.where(frame_table > chunk_size + frame_table[current_sent_index])[0][0] - 1
        except IndexError: #means that we can read the whole frame table
            last_sent_index = len(frame_table) - 1
        data, _, _, _ = read_pfile(self.feature_file_name, sent_indices = (current_sent_index, last_sent_index))
        if self.context_window > 1:
            cw_data = context_window(data, self.context_window, False, None, False)
        else:
            cw_data = data
        return cw_data, last_sent_index
    
    def classify_parallel(self, frame_table = None, sent_indices = None, num_sequences_per_forward = 1000):
        if frame_table is None:
            frame_table = self.frame_table
        if sent_indices is None:
            sent_indices = [0,len(frame_table) - 1]
        
        feature_sequence_lens = np.diff(frame_table[sent_indices[0]:sent_indices[1]+1])
        num_examples = frame_table[-1] - frame_table[0]
#        num_sequences = len(frame_table) - 1
        outputs = np.empty((num_examples,))
        last_sent_index = sent_indices[-1]
        first_sent_index = sent_indices[0]
        current_frame = 0
        for chunk_first_sent_index in range(first_sent_index, last_sent_index + 1, num_sequences_per_forward):
            chunk_last_sent_index = min(chunk_first_sent_index + num_sequences_per_forward, last_sent_index)
            chunk_size = frame_table[chunk_last_sent_index] - frame_table[chunk_first_sent_index]
            sent_features, _ = self.return_sequence_chunk(frame_table, chunk_first_sent_index, chunk_size)
            end_frame = current_frame + chunk_size
            outputs[current_frame:end_frame]  = self.find_best_sentence_labels_parallel(sent_features, feature_sequence_lens[chunk_first_sent_index:chunk_last_sent_index])
            current_frame = end_frame
#            print outputs

        try:
            num_correct = np.sum(self.labels == outputs)
            percentage_correct = float(num_correct) / self.labels.size * 100
            print "Got %d of %d correct: %.2f%%" % (num_correct, self.labels.size, percentage_correct)
        except AttributeError:
            pass
        return outputs

    def find_best_sentence_labels_parallel(self, features, feature_sequence_lens):
        emission_features = self.classify_parallel_dot(features, self.weights.feature_weights, self.weights.bias)
        outputs = np.empty((sum(feature_sequence_lens),))
        current_frame = 0
        for sequence_len in feature_sequence_lens:
            end_frame = current_frame + sequence_len
            label_scores, argmax_features = forward_cython.fast_forward_time_chunk(self.weights.time_weights, 
                                                                                   emission_features[current_frame:end_frame], 
                                                                                   self.weights.start_time_weights,
                                                                                   self.weights.end_time_weights)
            outputs[current_frame:end_frame] = self.naive_backtrace(label_scores, argmax_features)
            current_frame = end_frame
        return outputs
    
    def classify_parallel_dot(self, features, weights, bias):
        return np.dot(features, weights) + bias
    
    def naive_backtrace(self, label_scores, argmax_features):
        num_emission_features = argmax_features.shape[0]
        outputs = np.empty((num_emission_features,), dtype=np.int)
        outputs[-1] = np.argmax(label_scores[-1])
        
        for feature_index in range(num_emission_features-1, 0, -1):
            output = outputs[feature_index]
            outputs[feature_index-1] = argmax_features[feature_index, output]
        return outputs

    
    
    

        
