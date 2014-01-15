from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

DTYPEINT = np.int
ctypedef np.int_t DTYPEINT_t

DTYPEFLOAT = np.float
ctypedef np.float_t DTYPEFLOAT_t

#DTYPEFLOAT32 = np.float32
#ctypedef np.float32_t DTYPEFLOAT32_t

#@cython.wraparound(False)
@cython.boundscheck(False)
def fast_forward_time_chunk(np.ndarray[DTYPEFLOAT_t, ndim=2] time_weights, np.ndarray[DTYPEFLOAT_t, ndim=2] emission_features, 
                            np.ndarray[DTYPEFLOAT_t, ndim=1] start_time_weights, np.ndarray[DTYPEFLOAT_t, ndim=1] end_time_weights):
    cdef int num_emission_features = emission_features.shape[0]
    cdef int num_labels = emission_features.shape[1]
    
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] label_scores = np.zeros([num_emission_features, num_labels], dtype=DTYPEFLOAT)
    cdef np.ndarray[DTYPEINT_t, ndim=2] argmax_features = np.zeros((num_emission_features, num_labels), dtype = DTYPEINT)
    cdef float current_time_scores
    cdef int feature_index, prev_label_index, current_label_index
    
    for current_label_index in range(num_labels):
        label_scores[0, current_label_index] = emission_features[0, current_label_index] + start_time_weights[current_label_index]
        
    for feature_index in range(1, num_emission_features):
        for current_label_index in range(num_labels):
            prev_label_index = 0
            label_scores[feature_index, current_label_index] = (label_scores[feature_index-1, prev_label_index] + 
                                                                emission_features[feature_index, current_label_index] + 
                                                                time_weights[prev_label_index, current_label_index])
            if feature_index == num_emission_features - 1:
                label_scores[feature_index, current_label_index] += end_time_weights[current_label_index]
            argmax_features[feature_index, current_label_index] = prev_label_index
            
            for prev_label_index in range(1, num_labels):
                current_time_scores = (label_scores[feature_index-1, prev_label_index] + 
                                       emission_features[feature_index, current_label_index] + 
                                       time_weights[prev_label_index, current_label_index])
                if feature_index == num_emission_features - 1:
                    current_time_scores += end_time_weights[current_label_index]
                if current_time_scores > label_scores[feature_index, current_label_index]:
                    label_scores[feature_index, current_label_index] = current_time_scores
                    argmax_features[feature_index, current_label_index] = prev_label_index
                
#    for current_label_index in range(num_labels):
#        label_scores[num_emission_features - 1, current_label_index] += end_time_weights[current_label_index]
            
    return label_scores, argmax_features

@cython.boundscheck(False)
def add_bias_to_array(np.ndarray[DTYPEFLOAT_t, ndim=2] data):
    cdef int num_examples = data.shape[0]
    cdef int num_dims = data.shape[1]
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] new_data = np.ones((num_examples, num_dims+1), dtype = DTYPEFLOAT)
    cdef int row_index, col_index
    
    for row_index in range(num_examples):
        for col_index in range(num_dims):
            new_data[row_index, col_index] = data[row_index, col_index]
    return new_data

@cython.boundscheck(False)
def update_gradient(np.ndarray[DTYPEFLOAT_t, ndim=2] sent_features, np.ndarray[DTYPEINT_t, ndim=1] sent_labels, np.ndarray[DTYPEINT_t, ndim=1] most_violated_sequence, 
                    np.ndarray[DTYPEFLOAT_t, ndim=2] gradient_feature_weights, np.ndarray[DTYPEFLOAT_t, ndim=1] gradient_bias, 
                    np.ndarray[DTYPEFLOAT_t, ndim=2] gradient_time_weights, np.ndarray[DTYPEFLOAT_t, ndim=1] gradient_start_time_weights,
                    np.ndarray[DTYPEFLOAT_t, ndim=1] gradient_end_time_weights):
    cdef int num_sent_features = sent_features.shape[0]
    cdef int num_dims = sent_features.shape[1]
    cdef int dim_index, current_violated_label, current_label, observation_index, previous_violated_label, previous_label
    
    for dim_index in range(num_dims):
        gradient_feature_weights[dim_index,sent_labels[0]] -= sent_features[0, dim_index]
        gradient_feature_weights[dim_index, most_violated_sequence[0]] += sent_features[0, dim_index]
    
    gradient_bias[sent_labels[0]] -= 1.0
    gradient_bias[most_violated_sequence[0]] += 1.0
    
    gradient_start_time_weights[sent_labels[0]] -= 1.0
    gradient_start_time_weights[most_violated_sequence[0]] += 1.0
    
    previous_label = sent_labels[0]
    previous_violated_label = most_violated_sequence[0]
    
    for observation_index in range(1,num_sent_features):
#        feature = sent_features[observation_index]
        current_label = sent_labels[observation_index]
        current_violated_label = most_violated_sequence[observation_index]
#            print current_violated_label, current_label
        for dim_index in range(num_dims):
            gradient_feature_weights[dim_index,current_label] -= sent_features[observation_index, dim_index]
            gradient_feature_weights[dim_index,current_violated_label] += sent_features[observation_index, dim_index]
        
        gradient_bias[current_label] -= 1.0
        gradient_bias[current_violated_label] += 1.0
        
        gradient_time_weights[previous_label, current_label] -= 1.0
        gradient_time_weights[previous_violated_label, current_violated_label] += 1.0
        
        previous_label = current_label
        previous_violated_label = current_violated_label
        
    gradient_end_time_weights[sent_labels[num_sent_features-1]] -= 1.0
    gradient_end_time_weights[most_violated_sequence[num_sent_features-1]] += 1.0
#    return gradient_feature_weights, gradient_bias, gradient_time_weights, gradient_start_time_weights

