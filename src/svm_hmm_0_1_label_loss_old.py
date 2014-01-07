'''
Created on Oct 12, 2013

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
import forward_cython

class SVM_HMM_Weight(object):
    def __init__(self, num_dims=None, num_labels=None, weight_name=None, init_zero_weights = False, random_seed = 0):
        if weight_name is not None:
            self.feature_weights, self.time_weights = self.load_weights(weight_name)
            return
        if num_dims == None or num_labels == None:
            raise ValueError('Since weight_name is not defined, you will need to define both num_dims and num_labels to get weights')
        if init_zero_weights:
            self.feature_weights = np.zeros((num_dims, num_labels))
            self.time_weights = np.zeros((num_labels, num_labels))
        else:
            np.random.seed(random_seed)
            self.feature_weights = np.random.rand(num_dims, num_labels)
            self.time_weights = np.random.rand(num_labels, num_labels)
    def load_weights(self, weight_name):
        weight_dict = sp.loadmat(weight_name)
        return np.array(weight_dict['feature_weights'], order='C'), np.array(weight_dict['time_weights'], order='C')
    def save_weights(self, weight_name):
        sp.savemat(weight_name, {'feature_weights' : self.feature_weights, 'time_weights' : self.time_weights}, oned_as='column')
    def norm(self):
        return np.sqrt(np.sum(self.feature_weights ** 2) + np.sum(self.time_weights ** 2))
    def two_norm_project(self, norm_size = 1.0):
        current_norm = self.norm()
        self.feature_weights *= norm_size / current_norm
        self.time_weights *= norm_size / current_norm
    def __add__(self, addend):
        output = copy.deepcopy(self)
        if type(addend) is SVM_HMM_Weight:
            output.feature_weights = self.feature_weights + addend.feature_weights
            output.time_weights = self.time_weights + addend.time_weights
        else:
            output.feature_weights = self.feature_weights + addend
            output.time_weights = self.time_weights + addend
        return output
    def __sub__(self, subtrahend):
        output = copy.deepcopy(self)
        if type(subtrahend) is SVM_HMM_Weight:
            output.feature_weights = self.feature_weights - subtrahend.feature_weights
            output.time_weights = self.time_weights - subtrahend.time_weights
        else:
            output.feature_weights = self.feature_weights - subtrahend
            output.time_weights = self.time_weights - subtrahend
        return output
    def __mul__(self, multiplier):
        output = copy.deepcopy(self)
        if type(multiplier) is SVM_HMM_Weight:
            output.feature_weights = self.feature_weights * multiplier.feature_weights
            output.time_weights = self.time_weights * multiplier.time_weights
        else:
            output.feature_weights = self.feature_weights * multiplier
            output.time_weights = self.time_weights * multiplier
        return output
    def __div__(self, divisor):
        output = copy.deepcopy(self)
        if type(divisor) is SVM_HMM_Weight:
            output.feature_weights = self.feature_weights / divisor.feature_weights
            output.time_weights = self.time_weights / divisor.time_weights
        else:
            output.feature_weights = self.feature_weights / divisor
            output.time_weights = self.time_weights / divisor
        return output
    def __imul__(self, multiplier):
        self.feature_weights *= multiplier
        self.time_weights *= multiplier
        return self

class SVM_HMM(object):
    def __init__(self, feature_file_name, weight_name = None, label_file_name = None, num_labels = None, context_window = 1):
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
            cw_data = context_window(data, self.context_window, False, None, False)
        else:
            cw_data = data
#        print cw_data.flags
        new_data = self.add_bias_to_array(cw_data)
#        new_data = np.concatenate((new_data, np.ones((num_examples, 1))), axis=1)
#        print new_data.flags
        return new_data, last_sent_index
    def add_bias_to_array(self, data):
        num_examples, num_dims = data.shape
        new_data = np.empty((num_examples, num_dims+1))
        new_data[:,:-1] = data
        new_data[:,-1] = 1.0
        return new_data
     
    def return_parallel_sequence_chunk(self, frame_table, current_sent_index, chunk_size):
        """
        """
        try:
            last_sent_index = np.where(frame_table > chunk_size + frame_table[current_sent_index])[0][0] - 1
        except IndexError: #means that we can read the whole frame table
            last_sent_index = len(frame_table) - 1
        data, sub_frames, sub_labels, sub_frame_table = read_pfile(self.feature_file_name, sent_indices = (current_sent_index, last_sent_index))
        if self.context_window > 1:
            cw_data = context_window(data, self.context_window, False, None, False)
        else:
            cw_data = data
            
        num_examples, num_dims = cw_data.shape
        new_data = np.empty((num_examples, num_dims+1))
        new_data[:,:-1] = cw_data
        new_data[:,-1] = 1.0
#        new_data = np.concatenate((new_data, np.ones((num_examples, 1))), axis=1)
#        print new_data.flags
        return new_data, last_sent_index
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
            sent_features, last_sent_index = self.return_sequence_chunk(frame_table, chunk_first_sent_index, chunk_size)
            end_frame = current_frame + chunk_size
            outputs[current_frame:end_frame]  = self.find_best_sentence_labels_parallel(sent_features, feature_sequence_lens)
            current_frame = end_frame
#            print outputs

        try:
            num_correct = np.sum(self.labels == outputs)
            percentage_correct = float(num_correct) / self.labels.size * 100
            print "Got %d of %d correct: %.2f%%" % (num_correct, self.labels.size, percentage_correct)
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
    
    def classify(self, frame_table = None, sent_indices = None):
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
#            print outputs

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
            print "Got %d of %d correct: %.2f%%" % (num_correct, self.labels.size, percentage_correct)
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
    
    def find_most_violated_sentence_labels(self, sent_features, labels, loss_matrix):
        emission_features = np.dot(sent_features, self.weights.feature_weights)
        loss_scores = np.zeros(emission_features.shape)
#        num_labels = self.weights.time_weights.shape[0]
        label = labels[0]
        loss_scores[0] = loss_matrix[label] + emission_features[0]
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
    
    def find_best_sentence_labels(self, sent_features, do_dot = True):
        if do_dot:
            emission_features = self.classify_dot(sent_features, self.weights.feature_weights)
        else:
            emission_features = sent_features
#        num_emission_features = emission_features.shape[0]
        
        label_scores, argmax_features = self.naive_forward_time_chunk(emission_features)
        outputs = self.naive_backtrace(label_scores, argmax_features)
        
        
        return outputs, label_scores
    
    def naive_backtrace(self, label_scores, argmax_features):
        num_emission_features = argmax_features.shape[0]
        outputs = np.empty((num_emission_features,))
        outputs[-1] = np.argmax(label_scores[-1])
        
        for feature_index in range(num_emission_features-1, 0, -1):
            output = outputs[feature_index]
            outputs[feature_index-1] = argmax_features[feature_index, output]
        return outputs
    
    def naive_forward_time_chunk(self, emission_features):
        label_scores = np.zeros(emission_features.shape)
        label_scores[0] = emission_features[0]
        argmax_features = np.zeros(emission_features.shape)
        num_emission_features = emission_features.shape[0]
        
        for feature_index in range(1,num_emission_features):
#            previous_time_feature = label_scores[feature_index-1]
#            current_emission_feature = emission_features[feature_index]
            current_time_scores = np.add.outer(label_scores[feature_index-1], emission_features[feature_index])
            current_time_scores += self.weights.time_weights
#            current_time_scores += previous_time_feature.T[:,np.newaxis]
            label_scores[feature_index] = np.max(current_time_scores, axis=0)
            argmax_features[feature_index] = np.argmax(current_time_scores, axis=0)
        return label_scores, argmax_features
    
    def update_gradient(self, sent_features, sent_labels, most_violated_sequence, gradient):
        num_sent_features = sent_features.shape[0]
        gradient.feature_weights[:,sent_labels[0]] -= sent_features[0]
        gradient.feature_weights[:,most_violated_sequence[0]] += sent_features[0]
        previous_label = sent_labels[0]
        previous_violated_label = most_violated_sequence[0]
        for observation_index in range(1,num_sent_features):
            feature = sent_features[observation_index]
            current_label = sent_labels[observation_index]
            current_violated_label = most_violated_sequence[observation_index]
#            print current_violated_label, current_label
            gradient.feature_weights[:,current_label] -= feature
            gradient.feature_weights[:,current_violated_label] += feature
            gradient.time_weights[previous_label, current_label] -= 1.0
            gradient.time_weights[previous_violated_label, current_violated_label] += 1.0
            previous_label = current_label
            previous_violated_label = current_violated_label
        return gradient
    
    def train(self, lambda_const = 0.5, batch_size = 128, num_epochs = 100):
        """train using structured Pegasos algorithm
        """
        batch_size = min(batch_size, self.num_examples)
        gradient = SVM_HMM_Weight(self.num_dims, self.num_labels, init_zero_weights = True)
        weight_norm = self.weights.norm()
        if weight_norm > 1. / np.sqrt(lambda_const):
#            print "projecting"
            self.weights.two_norm_project(norm_size = 1. / np.sqrt(lambda_const) )
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
                self.weights.two_norm_project(norm_size = 1. / np.sqrt(lambda_const) )
            self.classify(self.frame_table, sent_indices=range(self.num_examples))
    
    
    
    
    
    
    
    
    ##### NOT USED AT THE MOMENT #######
    def classify_dot(self, features, weights):
        return np.dot(features, weights) 
    def classify_parallel_dot(self, features, weights):
#        print features.flags
#        print ""
#        print weights.flags
        return np.dot(features, weights)
    def find_best_sentence_labels_parallel(self, features, feature_sequence_lens):
        emission_features = self.classify_parallel_dot(features, self.weights.feature_weights)
        outputs = np.empty((sum(feature_sequence_lens),))
        current_frame = 0
        for sequence_len in feature_sequence_lens:
            end_frame = current_frame + sequence_len
#            label_scores, argmax_features = self.naive_forward_time_chunk(emission_features[current_frame:end_frame])
            label_scores, argmax_features = forward_cython.fast_forward_time_chunk(self.weights.time_weights, emission_features[current_frame:end_frame])
            outputs[current_frame:end_frame] = self.naive_backtrace(label_scores, argmax_features)
#            outputs[current_frame:end_frame] = self.naive_forward_time_chunk(emission_features)
            current_frame = end_frame
        return outputs
#        feature_chunk = self.reshape_chunk(emission_features, feature_sequence_lens)
#        forward_chunk, argmax_chunk = self.parallel_time_forward_chunk(feature_chunk)
#        back_forward_chunk, back_argmax_chunk = self.move_chunks_to_end(forward_chunk, argmax_chunk, feature_sequence_lens)
#        chunk_best_labels = self.fast_backtrace_parallel_sentence_labels(back_forward_chunk, back_argmax_chunk, feature_sequence_lens)
#        return self.flatten_labels_from_back(chunk_best_labels, feature_sequence_lens)
#        chunk_best_labels = self.backtrace_parallel_sentence_labels(forward_chunk, argmax_chunk, feature_sequence_lens)
        
#        return self.flatten_labels(chunk_best_labels, feature_sequence_lens)
    
    def backtrace_parallel_sentence_labels(self, forward_chunk, argmax_chunk, feature_sequence_lens):
        max_sequence_len = np.max(feature_sequence_lens).astype(int)
        num_sequences = feature_sequence_lens.size
        best_labels = -np.ones((num_sequences, max_sequence_len), dtype=np.int32)
        for sequence_index, sequence_len in enumerate(feature_sequence_lens):
            sequence_time_index = sequence_len - 1
#            print forward_chunk[sequence_time_index,:,sequence_index]
            best_labels[sequence_index, sequence_time_index] = np.argmax(forward_chunk[sequence_time_index,:,sequence_index])
            for time_index in range(sequence_time_index,0,-1):
                best_label = best_labels[sequence_index, sequence_time_index]
                best_labels[sequence_index, time_index -1] = argmax_chunk[time_index, best_label, sequence_index]
        
        return best_labels
    
    def fast_backtrace_parallel_sentence_labels(self, forward_chunk, argmax_chunk, feature_sequence_lens):
        max_sequence_len, num_dims, num_sequences = forward_chunk.shape
        num_sequence_index = range(num_sequences)
        num_sequences = feature_sequence_lens.size
        best_labels = -np.ones((num_sequences, max_sequence_len), dtype=np.int32)
        best_labels[:,-1] = np.argmax(forward_chunk[-1,:,:], axis=0)
        for time_index in range(max_sequence_len-1, 0, -1):
            best_labels[:, time_index - 1] = argmax_chunk[time_index, best_labels[:,time_index], num_sequence_index]
        
        return best_labels
    
    def parallel_time_forward_chunk(self, feature_chunk):
        max_sequence_len = feature_chunk.shape[0]
        forward_chunk = np.zeros(feature_chunk.shape)
        argmax_chunk = -np.ones(feature_chunk.shape)
        forward_chunk[:,:,0] = feature_chunk[:,:,0]
        for sequence_index in range(1,max_sequence_len):
            previous_forward_chunk = forward_chunk[sequence_index-1,:,:]
            current_feature_chunk = feature_chunk[sequence_index,:,:]
            time_i_trellis = np.transpose(previous_forward_chunk[:,np.newaxis,:] + self.weights.time_weights[:,:,np.newaxis], (1,0,2)) + current_feature_chunk[:,np.newaxis,:]
            forward_chunk[sequence_index,:,:] = np.max(time_i_trellis,axis=1)
            argmax_chunk[sequence_index,:,:] = np.argmax(time_i_trellis,axis=1)

        return forward_chunk, argmax_chunk
    
    def move_chunks_to_end(self, forward_chunk, argmax_chunk, feature_sequence_lens):
        max_sequence_len, num_dims, num_sequences = forward_chunk.shape
        
        assert max_sequence_len == max(feature_sequence_lens) and num_sequences == len(feature_sequence_lens)
        new_argmax_chunk = -np.ones(argmax_chunk.shape, dtype=np.int32)
        new_forward_chunk = np.zeros(forward_chunk.shape)
        for batch_index, sequence_len in enumerate(feature_sequence_lens):
            new_forward_chunk[(max_sequence_len-sequence_len):,:,batch_index] = forward_chunk[:sequence_len, :, batch_index]
            new_argmax_chunk[(max_sequence_len-sequence_len):,:,batch_index] = argmax_chunk[:sequence_len, :, batch_index]
        
        return new_forward_chunk, new_argmax_chunk
    
    def reshape_chunk(self, features, feature_sequence_lens):
        num_dims = features.shape[1]
        num_sequences = len(feature_sequence_lens)
        max_sequence_len = np.max(feature_sequence_lens)
        output_features = np.zeros((max_sequence_len, num_dims, num_sequences))
        frame_index = 0
        for sent_index, sequence_len in enumerate(feature_sequence_lens):
            end_index = frame_index + sequence_len
            output_features[:sequence_len, :, sent_index] = features[frame_index:end_index, :]
            frame_index = end_index
        return output_features
    
    def flatten_labels(self, labels, feature_sequence_lens):
#        num_sequences, max_sequence_len = labels.shape
        flat_labels = -np.ones(sum(feature_sequence_lens), dtype = np.int32)
        current_frame = 0
        
        for sequence_index, sequence_len in enumerate(feature_sequence_lens):
            flat_labels[current_frame:current_frame+sequence_len] = labels[sequence_index, :sequence_len]
            current_frame += sequence_len
        
        return flat_labels
    
    def flatten_labels_from_back(self, labels, feature_sequence_lens):
        num_sequences, max_sequence_len = labels.shape
        flat_labels = -np.ones(sum(feature_sequence_lens), dtype = np.int32)
        current_frame = 0
        
        for sequence_index, sequence_len in enumerate(feature_sequence_lens):
            flat_labels[current_frame:current_frame+sequence_len] = labels[sequence_index, (max_sequence_len - sequence_len):]
            current_frame += sequence_len
        
        return flat_labels
        
        
    