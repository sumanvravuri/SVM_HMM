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
def fast_forward_time_chunk(np.ndarray[DTYPEFLOAT_t, ndim=2] time_weights, np.ndarray[DTYPEFLOAT_t, ndim=2] emission_features, np.ndarray[DTYPEFLOAT_t, ndim=1] start_time_weights):
    cdef int num_emission_features = emission_features.shape[0]
    cdef int num_labels = emission_features.shape[1]
    
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] label_scores = np.zeros([num_emission_features, num_labels], dtype=DTYPEFLOAT)
    cdef np.ndarray[DTYPEINT_t, ndim=2] argmax_features = np.zeros((num_emission_features, num_labels), dtype = DTYPEINT)
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] current_time_scores = np.zeros((num_labels, num_labels), dtype = DTYPEFLOAT)
    cdef int feature_index, a, b
    
    for a in range(num_labels):
        label_scores[0, a] = emission_features[0, a] + start_time_weights[a]
        
    for feature_index in range(1,num_emission_features):
        for a in range(num_labels):
            for b in range(num_labels):
                current_time_scores[a,b] = label_scores[feature_index-1,a] + emission_features[feature_index, b] + time_weights[a,b]
        for a in range(num_labels):
            label_scores[feature_index, a] = current_time_scores[0,a]
            argmax_features[feature_index, a] = 0
            for b in range(1,num_labels):
                if current_time_scores[b,a] > label_scores[feature_index,a]:
                    label_scores[feature_index, a] = current_time_scores[b,a]
                    argmax_features[feature_index, a] = b

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
                    np.ndarray[DTYPEFLOAT_t, ndim=2] gradient_time_weights, np.ndarray[DTYPEFLOAT_t, ndim=1] gradient_start_time_weights):
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
    
#    return gradient_feature_weights, gradient_bias, gradient_time_weights, gradient_start_time_weights

