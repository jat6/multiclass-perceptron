import numpy as np
from datainput import read_data, binary_preprocessing, multi_preprocessing, random_arrays
from binaryperceptron import perceptron_train
from multiclassperceptron import m_perceptron_train

binary_classification = False

# Initialise parameters, weights of the perceptron and file data structures
epochs = 20
features_count = 4
num_classes = 3
reg_term = 0.01
raw_test, raw_training, test_data, train_data = ([] for i in range(4))

# Read data, preprocess into numeric values and output as numpy arrays, use list of lists for n classes into multiclass-classification
if(binary_classification == True):
    raw_test, raw_training = read_data('test.data', 'train.data')
    test_data, train_data = binary_preprocessing(raw_test, raw_training, 4, '2', '1')
    instances = len(train_data)
    examples = len(test_data)
else:
    for i in range(num_classes):
        raw_test, raw_training = read_data('test.data', 'train.data')
        test_multi, train_multi = multi_preprocessing(raw_test, raw_training, 4, i + 1)
        test_data.append(test_multi)
        train_data.append(train_multi)
    instances = len(train_data[0])
    examples = len(test_data[0])
    train_data = random_arrays(num_classes, train_data)
    test_data = random_arrays(num_classes, test_data)

# Structure feature and target vectors, call binary perceptron training loop and testing
def binary_perceptron():
    train_x = np.delete(train_data, np.s_[-1:5], 1).astype(np.float)
    test_x = np.delete(test_data, np.s_[-1:5], 1).astype(np.float)
    train_y = np.delete(train_data, np.s_[0:-1], 1).astype(np.float)
    test_y = np.delete(test_data, np.s_[0:-1], 1).astype(np.float)
    bias = 0
    w = np.zeros([1, features_count]) 
    perceptron_train(epochs, instances, examples, features_count, train_x, train_y, test_x, test_y, w, bias)
    
def multiclass_perceptron(train_data):
    bias = np.zeros([num_classes, 1]) 
    w = np.zeros([num_classes, features_count])
    m_perceptron_train(epochs, instances, examples, features_count, num_classes, train_data, test_data, w, bias, reg_term)
    
if(binary_classification == True):    
    binary_perceptron()
else:
    multiclass_perceptron(train_data)
