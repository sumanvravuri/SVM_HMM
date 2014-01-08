'''
Created on Jan 1, 2014

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

class SVM_HMM_Weight(object):
    def __init__(self, num_dims=None, num_labels=None, weight_name=None, init_zero_weights = False, random_seed = 0):
        if weight_name is not None:
            self.feature_weights, self.time_weights, self.start_time_weights, self.end_time_weights, self.bias = self.load_weights(weight_name)
            return
        if num_dims == None or num_labels == None:
            raise ValueError('Since weight_name is not defined, you will need to define both num_dims and num_labels to get weights')
        
        if init_zero_weights:
            self.feature_weights = np.zeros((num_dims, num_labels))
            self.bias = np.zeros((num_labels,))
            self.time_weights = np.zeros((num_labels, num_labels))
            self.start_time_weights = np.zeros((num_labels,))
            self.end_time_weights = np.zeros((num_labels,))
        else:
            np.random.seed(random_seed)
            self.feature_weights = np.random.rand(num_dims, num_labels)
            self.bias = np.random.rand(num_labels)
            self.time_weights = np.ones((num_labels, num_labels)) / float(num_labels)
            self.start_time_weights = np.random.rand(num_labels)
            self.end_time_weights = np.random.rand(num_labels)

    def load_weights(self, weight_name):
        weight_dict = sp.loadmat(weight_name)
        feature_weights = np.empty(weight_dict['feature_weights'].shape, dtype=np.float, order='C')
        bias = np.empty((weight_dict['bias'].size,), dtype=np.float, order='C')
        time_weights = np.empty(weight_dict['time_weights'].shape, dtype=np.float, order='C')
        start_time_weights = np.empty((weight_dict['start_time_weights'].size,), dtype=np.float, order='C')
        end_time_weights = np.empty((weight_dict['end_time_weights'].size,), dtype=np.float, order='C')
        
        feature_weights[:] = weight_dict['feature_weights'][:]
        bias[:] = weight_dict['bias'].T[:]
        time_weights[:] = weight_dict['time_weights'][:]
        start_time_weights[:] = weight_dict['start_time_weights'].T[:]
        end_time_weights[:] = weight_dict['end_time_weights'].T[:]
        
        return feature_weights, time_weights, start_time_weights, end_time_weights, bias

    def save_weights(self, weight_name):
        sp.savemat(weight_name, {'feature_weights' : self.feature_weights, 'time_weights' : self.time_weights, 
                                 'start_time_weights' : self.start_time_weights, 'end_time_weights' : self.end_time_weights, 
                                 'bias' : self.bias}, 
                   oned_as='column')

    def norm(self):
        return np.sqrt(np.sum(self.feature_weights ** 2) + np.sum(self.time_weights ** 2) + 
                       np.sum(self.start_time_weights ** 2) + np.sum(self.end_time_weights ** 2) +
                       np.sum(self.bias ** 2))

    def two_norm_project(self, new_norm_size = 1.0):
        current_norm = self.norm()
        if current_norm < new_norm_size:
            return
#        self.feature_weights *= new_norm_size / current_norm
#        self.time_weights *= new_norm_size / current_norm
#        self.start_time_weights *= new_norm_size / current_norm
#        self.bias *= new_norm_size / current_norm
        self *= new_norm_size / current_norm

    def __add__(self, addend):
        output = copy.deepcopy(self)
        if type(addend) is SVM_HMM_Weight:
            output.feature_weights = self.feature_weights + addend.feature_weights
            output.time_weights = self.time_weights + addend.time_weights
            output.start_time_weights = self.start_time_weights + addend.start_time_weights
            output.end_time_weights = self.end_time_weights + addend.end_time_weights
            output.bias = self.bias + addend.bias
        else:
            output.feature_weights = self.feature_weights + addend
            output.time_weights = self.time_weights + addend
            output.start_time_weights = self.start_time_weights + addend
            output.end_time_weights = self.end_time_weights + addend
            output.bias = self.bias + addend
        return output

    def __sub__(self, subtrahend):
        output = copy.deepcopy(self)
        if type(subtrahend) is SVM_HMM_Weight:
            output.feature_weights = self.feature_weights - subtrahend.feature_weights
            output.time_weights = self.time_weights - subtrahend.time_weights
            output.start_time_weights = self.start_time_weights - subtrahend.start_time_weights
            output.end_time_weights = self.end_time_weights - subtrahend.end_time_weights
            output.bias = self.bias - subtrahend.bias
        else:
            output.feature_weights = self.feature_weights - subtrahend
            output.time_weights = self.time_weights - subtrahend
            output.start_time_weights = self.start_time_weights - subtrahend
            output.end_time_weights = self.end_time_weights - subtrahend
            output.bias = self.bias - subtrahend
        return output

    def __mul__(self, multiplier):
        output = copy.deepcopy(self)
        if type(multiplier) is SVM_HMM_Weight:
            output.feature_weights = self.feature_weights * multiplier.feature_weights
            output.time_weights = self.time_weights * multiplier.time_weights
            output.start_time_weights = self.start_time_weights * multiplier.start_time_weights
            output.end_time_weights = self.end_time_weights * multiplier.end_time_weights
            output.bias = self.bias * multiplier.bias
        else:
            output.feature_weights = self.feature_weights * multiplier
            output.time_weights = self.time_weights * multiplier
            output.start_time_weights = self.start_time_weights * multiplier
            output.end_time_weights = self.end_time_weights * multiplier
            output.bias = self.bias * multiplier
        return output

    def __div__(self, divisor):
        output = copy.deepcopy(self)
        if type(divisor) is SVM_HMM_Weight:
            output.feature_weights = self.feature_weights / divisor.feature_weights
            output.time_weights = self.time_weights / divisor.time_weights
            output.start_time_weights = self.start_time_weights / divisor.start_time_weights
            output.end_time_weights = self.end_time_weights / divisor.end_time_weights
            output.bias = self.bias / divisor.bias
        else:
            output.feature_weights = self.feature_weights / divisor
            output.time_weights = self.time_weights / divisor
            output.start_time_weights = self.start_time_weights / divisor
            output.end_time_weights = self.end_time_weights / divisor
            output.bias = self.bias / divisor
        return output

    def __imul__(self, multiplier):
        self.feature_weights *= multiplier
        self.time_weights *= multiplier
        self.start_time_weights *= multiplier
        self.end_time_weights *= multiplier
        self.bias *= multiplier
        return self