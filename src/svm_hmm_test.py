'''
Created on Oct 16, 2013

@author: sumanravuri
'''

import svm_hmm_0_1_sequence_loss as svm_hmm
import time

if __name__ == "__main__":
    svm_hmm_obj = svm_hmm.SVM_HMM(feature_file_name = '../test_data/declaration_of_independence_feats.pfile', 
                                  label_file_name = '../test_data/declaration_of_independence_labels.pfile')
    start_time = time.time()
    svm_hmm_obj.train(lambda_const = 0.125, batch_size = 128, num_epochs = 100)
    print "%d epochs of training completed %f secs" % (100, time.time() - start_time)
    svm_hmm_obj.weights.save_weights(weight_name='../test_data/declaration_of_independence_weights.mat')
    
    svm_hmm_test_obj = svm_hmm.SVM_HMM(feature_file_name = '../test_data/gettysburg_feats.pfile', 
                                       label_file_name = '../test_data/gettysburg_labels.pfile', 
                                       weight_name = '../test_data/declaration_of_independence_weights.mat')
    
#    outputs = svm_hmm_test_obj.classify()
    num_passes = 100
    
#    start_time = time.time()
#    for pass_idx in range(num_passes):
#        outputs = svm_hmm_test_obj.classify()
#    print "%d runs of serial forward pass takes %f secs" % (num_passes, time.time() - start_time)
    
    start_time = time.time()
    for pass_idx in range(num_passes):
        outputs_parallel = svm_hmm_test_obj.classify_parallel()
    print "%d runs of parallel forward pass takes %f secs" % (num_passes, time.time() - start_time)
    
    
#    print "%d of %d outputs the same" % (sum(outputs == outputs_parallel), len(outputs))
#    print outputs
#    print outputs_parallel