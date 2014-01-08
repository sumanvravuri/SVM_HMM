'''
Created on Oct 12, 2013

@author: sumanravuri
'''

import numpy as np
from read_pfile_v2 import read_pfile
import scipy.io as sp
import sys
#import scipy.linalg as sl
#import scipy.optimize as sopt
#import math
import copy
#import argparse
from context_window import context_window
from SVM_HMM_Weight import SVM_HMM_Weight


class SVM_HMM(object):
    def __init__(self, feature_file_name, weight_name = None, label_file_name = None, num_labels = None, context_window = 1, seed = 0):
        np.random.seed(seed)
        self.feature_file_name = feature_file_name
        self.num_dims, self.frame_table = self.read_feature_file_stats(self.feature_file_name)
        self.num_dims += 1 #add 1 for bias
        self.num_examples = self.frame_table.size - 1
        
        if label_file_name != None:
            self.label_file_name = label_file_name
            self.labels = self.read_label_file(self.label_file_name)
            self.num_labels = np.max(self.labels) + 1
            self.loss_matrix = np.ones((self.num_labels, self.num_labels)) - np.identity(self.num_labels) #0-1 loss for now
        
        if weight_name != None:
            self.weight_name = weight_name
            self.weights = SVM_HMM_Weight(weight_name = self.weight_name)
        else:
            self.weights = SVM_HMM_Weight(num_dims = self.num_dims, num_labels = self.num_labels)
        
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
        data, sub_frames, sub_labels, sub_frame_table = read_pfile(self.feature_file_name, sent_indices = (current_sent_index, last_sent_index))
        if self.context_window > 1:
            new_data = context_window(data, self.context_window, False, None, False)
        else:
            new_data = data
        num_examples = new_data.shape[0]
        new_data = np.concatenate((new_data, np.ones((num_examples, 1))), axis=1)
        return new_data, last_sent_index

    def classify_parallel(self, frame_table = None, sent_indices = None):
        if frame_table is None:
            frame_table = self.frame_table
        if sent_indices is None:
            sent_indices = range(len(frame_table) - 1)
        num_examples = frame_table[-1] - frame_table[0]
#        num_sequences = len(frame_table) - 1
        outputs = np.empty((num_examples,))
        current_frame = 0
        for sent_index in sent_indices:
            chunk_size = frame_table[sent_index + 1] - frame_table[sent_index]
            sent_features, last_sent_index = self.return_sequence_chunk(frame_table, sent_index, chunk_size)
            current_frame = frame_table[sent_index]
            end_frame = frame_table[last_sent_index]#current_frame + frames.shape[0]
            outputs[current_frame:end_frame], loss_scores = self.find_best_sentence_labels(sent_features)
#            print outputs[current_frame:end_frame]
#	print np.unique(outputs)
        try:
#            num_sents_correct = 0
#            num_sents = len(sent_indices)
#            for sent_index in sent_indices:
#                chunk_size = frame_table[sent_index + 1] - frame_table[sent_index]
#                start_frame = frame_table[sent_index]
#                end_frame = frame_table[last_sent_index]
#                print self.labels[start_frame:end_frame] == outputs[current_frame:end_frame], self.labels[start_frame:end_frame].size
#                if np.sum(self.labels[start_frame:end_frame] == outputs[current_frame:end_frame]) == self.labels[start_frame:end_frame].size:
#                    num_sents_correct += 1
            num_correct = np.sum(self.labels == outputs)
            percentage_correct = float(num_correct) / self.labels.size * 100
            sys.stderr.write("Got %d of %d correct: %.2f%%\n" % (num_correct, self.labels.size, percentage_correct))
#            percentage_sents_correct = float(num_sents_correct) / num_sents * 100
#            print "Got %d of %d correct: %.2f%%" % (num_sents_correct, num_sents, percentage_sents_correct)
        except AttributeError:
            pass
#        current_sent_index = 0
#        while current_sent_index < num_sequences:
#            feature_chunk, last_sent_index = self.return_sequence_chunk(frame_table, current_sent_index, chunk_size)
#            feature_sequence_lens = np.diff(frame_table[current_sent_index:last_sent_index])
#            feature_output_chunk = self.reshape_chunk(np.dot(feature_chunk, self.weights.feature_weights), feature_sequence_lens)
#            feature_output_chunk = np.dot(feature_chunk, self.weights.feature_weights)
#            current_sent_index = last_sent_index
        return outputs
    
    def find_best_training_sentence_labels(self, sent_features, sent_labels):
        emission_features = np.dot(sent_features, self.weights.feature_weights)
        loss_scores = np.zeros(emission_features.shape)
        loss_scores[0] = emission_features[0] + self.weights.start_time_weights
        argmax_features = np.zeros(emission_features.shape, dtype=int)
        num_emission_features = emission_features.shape[0]
        outputs = np.empty((num_emission_features,), dtype=int)
        for feature_index in range(1,num_emission_features):
            previous_time_feature = loss_scores[feature_index-1]
            current_emission_feature = emission_features[feature_index]
            current_loss_scores = previous_time_feature.T[:,np.newaxis] + self.weights.time_weights + emission_features[feature_index]
            argmax_features[feature_index] = np.argmax(current_loss_scores, axis=0)
            loss_scores[feature_index] = np.max(current_loss_scores, axis=0)
        
        outputs[-1] = np.argmax(loss_scores[-1])
        for feature_index in range(num_emission_features-1, 0, -1):
            output = outputs[feature_index]
            outputs[feature_index-1] = argmax_features[feature_index, output]
        if outputs != sent_labels: #sum(outputs == sent_labels) != sent_labels.size:
            return outputs, loss_scores + 1.0

        #In this case, it means that the best output sequence is correct, but we want to make sure that we didn't suffer a margin violation
        current_loss_scores += np.ones(current_loss_scores.shape)
        current_loss_scores[sent_labels[-2], sent_labels[-1]] -= 1.0
        loss_scores[-1] = np.max(current_loss_scores, axis = 0)
        argmax_features[-1] = np.argmax(current_loss_scores, axis = 0)

        outputs[-1] = np.argmax(loss_scores[-1])
        for feature_index in range(num_emission_features-1, 0, -1):
            output = outputs[feature_index]
            outputs[feature_index-1] = argmax_features[feature_index, output]

#        print outputs
        return outputs, loss_scores

    def find_best_sentence_labels(self, sent_features):
        emission_features = np.dot(sent_features, self.weights.feature_weights)
        label_scores = np.zeros(emission_features.shape)
        label_scores[0] = emission_features[0] + self.weights.start_time_weights
        argmax_features = np.zeros(emission_features.shape)
        num_emission_features = emission_features.shape[0]
        outputs = np.empty((num_emission_features,), dtype=int)
        for feature_index in range(1,num_emission_features):
            previous_time_feature = label_scores[feature_index-1]
            current_emission_feature = emission_features[feature_index]
            current_time_scores = previous_time_feature.T[:,np.newaxis] + self.weights.time_weights + current_emission_feature[np.newaxis,:]
            label_scores[feature_index] = np.max(current_time_scores, axis=0)
            argmax_features[feature_index] = np.argmax(current_time_scores, axis=0)
        
        outputs[-1] = np.argmax(label_scores[-1])
        for feature_index in range(num_emission_features-1, 0, -1):
            output = outputs[feature_index]
            outputs[feature_index-1] = argmax_features[feature_index, output]
        return outputs, label_scores

    def update_gradient(self, sent_features, sent_labels, most_violated_sequence, gradient):
        num_sent_features = sent_features.shape[0]
        gradient.feature_weights[:,sent_labels[0]] -= sent_features[0]
        gradient.feature_weights[:,most_violated_sequence[0]] += sent_features[0]

        gradient.start_time_weights[sent_labels[0]] -= 1.0
        gradient.start_time_weights[most_violated_sequence[0]] += 1.0

        previous_label = sent_labels[0]
        previous_violated_label = most_violated_sequence[0]
        for observation_index in range(1,num_sent_features):
            feature = sent_features[observation_index]
            current_label = sent_labels[observation_index]
            current_violated_label = most_violated_sequence[observation_index]
            gradient.feature_weights[:,current_label] -= feature
            gradient.feature_weights[:,current_violated_label] += feature

        gradient.time_weights[sent_labels[:-1], sent_labels[1:]] -= 1.0
        gradient.time_weights[most_violated_sequence[:-1], most_violated_sequence[1:]] += 1.0

        return gradient
    
    def calculate_prediction_loss(self, features, prediction_sequence, labels):
        loss_sequence = np.zeros((len(prediction_sequence),))
        loss = 0.0
        prediction = prediction_sequence[0]
        feature = features[0]
        label = labels[0]
        loss += self.weights.start_time_weights[prediction] + np.dot(feature, self.weights.feature_weights)[prediction]
        loss += self.loss_matrix[prediction, label]
        prev_prediction = prediction
        loss_sequence[0] = loss
        for prev_seq_index, prediction in enumerate(prediction_sequence[1:]):
            seq_index = prev_seq_index + 1
            feature = features[seq_index]
            prediction = prediction_sequence[seq_index]
            label = labels[seq_index]
            loss += self.weights.time_weights[prev_prediction, prediction] + np.dot(feature, self.weights.feature_weights)[prediction]
            loss += self.loss_matrix[prediction, label]
            prev_prediction = prediction
            loss_sequence[seq_index] = loss
        return loss, loss_sequence

    def train(self, lambda_const = 0.5, batch_size = 128, start_epoch = 1, num_epochs=100, save_epoch_prefix=None):
        """train using structured Pegasos algorithm
        """
        batch_size = min(batch_size, self.num_examples)
        gradient = SVM_HMM_Weight(self.num_dims, self.num_labels, init_zero_weights = True)
        weight_norm = self.weights.norm()
        projection_norm = 1. / np.sqrt(lambda_const)
        if weight_norm > projection_norm:
#            print "projecting"
            self.weights.two_norm_project(new_norm_size = projection_norm )
#        self.classify(self.frame_table, sent_indices=range(self.num_examples))
        end_epoch = start_epoch + num_epochs
        for epoch_num in range(start_epoch, end_epoch+1):
            print "At epoch number", epoch_num
            learning_rate = 1. / (lambda_const * epoch_num)
#            print learning_rate
            sentence_indices = np.random.permutation(self.num_examples)[:batch_size]
            gradient *= 0.0
            for sent_index in sentence_indices:
                start_frame = self.frame_table[sent_index]
                end_frame = self.frame_table[sent_index+1]
                num_observations = end_frame - start_frame
                sent_labels = self.labels[start_frame:end_frame]
                sent_features, last_sent_index = self.return_sequence_chunk(self.frame_table, sent_index, num_observations)
                most_violated_sequence, loss_scores = self.find_best_training_sentence_labels(sent_features, self.labels)
#                best_sequence, sequences_scores = self.find_best_sentence_labels(sent_features)
                gradient = self.update_gradient(sent_features, sent_labels, most_violated_sequence, gradient)
            self.weights = self.weights * (1. - 1. / epoch_num) - gradient * (learning_rate / batch_size)
#            self.weights.time_weights = self.weights.time_weights * (1. - 1. / epoch_num) - gradient.time_weights * (learning_rate / batch_size)
  
            weight_norm = self.weights.norm()
            if weight_norm > projection_norm:
#                print "projecting"
                self.weights.two_norm_project(new_norm_size = projection_norm )
#            self.classify(self.frame_table, sent_indices=range(self.num_examples))
            if save_epoch_prefix != None:
                file_name = save_epoch_prefix + "_epoch_" + str(epoch_num)
                print "saving weights to", file_name
                self.weights.save_weights(weight_name=file_name)
    
    

        
    