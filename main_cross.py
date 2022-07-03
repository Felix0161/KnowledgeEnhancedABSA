import tensorflow as tf
import lcrModelAlt_hierarchical_v4
from loadData import *

#import parameter configuration and data paths
from config import *

#import modules
import numpy as np
import sys


# main function
def main(_):
    loadData = False
    runLCRROTALT = True

    BASE_train = "data/programGeneratedData/crossValidation"+str(FLAGS.year)+'/cross_train_'
    BASE_val = "data/programGeneratedData/crossValidation"+str(FLAGS.year)+'/cross_val_'


    # Number of k-fold cross validations
    split_size = 10
    
    # retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadCrossValidation(FLAGS, split_size)
    remaining_size = 248
    accuracyOnt = 0.87


    acc=[]
    #k-fold cross validation
    for i in range(split_size):
        acc1, _, _, _, _, _ = lcrModelAlt_hierarchical_v4.main(BASE_train+str(i)+'.txt',BASE_val+str(i)+'.txt', accuracyOnt, test_size[i], remaining_size)
        acc.append(acc1)
        tf.reset_default_graph()
        print('iteration: '+ str(i))
    with open("cross_results_"+str(FLAGS.year)+"/LCRROT_ALT_"+str(FLAGS.year)+'.txt', 'w') as result:
        result.write(str(acc))
        result.write('Accuracy: {}, St Dev:{} /n'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))
        print(str(split_size)+'-fold cross validation results')
        print('Accuracy: {}, St Dev:{}'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))


    print('Finished program succesfully')

if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()
