'''
Created on Oct 16, 2013

@author: sumanravuri
'''

import svm_hmm

if __name__ == "__main__":
    svm_hmm_obj = svm_hmm.SVM_HMM(feature_file_name = 'declaration_of_independence_feats.pfile', 
                                  label_file_name = 'declaration_of_independence_labels.pfile')
    svm_hmm_obj.train(lambda_const = 0.25, batch_size = 128)
    svm_hmm_obj.weights.save_weights(weight_name='declaration_of_independence_weights.mat')
    
    svm_hmm_test_obj = svm_hmm.SVM_HMM(feature_file_name = 'gettysburg_feats.pfile', 
                                       label_file_name = 'gettysburg_labels.pfile', 
                                       weight_name = 'declaration_of_independence_weights.mat')
    
    svm_hmm_test_obj.classify()