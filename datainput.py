import numpy as np

# Import the test and training data into raw text form
def read_data(test_file, train_file):
   raw_test = open('test.data', "r")
   raw_training = open('train.data', "r") 
   return raw_test, raw_training

# Manipulate string to be in form of python list - convert target variables to be numeric
def binary_preprocessing(raw_test, raw_training, target_element,  removed_class, positive_class):
    test_data = []
    train_data = []
    for row in raw_test:
        temp = row.split(",")
        temp[target_element] = ''.join([i for i in temp[target_element] if i.isdigit()])
        if not(removed_class) in temp[target_element]:     
            if (positive_class) in temp[target_element]:  
                temp[target_element] = 1
            else:
                temp[target_element] = -1
            test_data.append(temp)
            
    for row in raw_training:
        temp = row.split(",")
        temp[target_element] = ''.join([i for i in temp[target_element] if i.isdigit()])
        if not(removed_class) in temp[target_element]:     
            if (positive_class) in temp[target_element]:  
                temp[target_element] = 1
            else:
                temp[target_element] = -1
            train_data.append(temp)
            
    # Convert lists into numpy arrays     
    np.asarray(test_data)
    np.asarray(train_data)

    # Shuffle order of training set
    np.random.shuffle(test_data)
    np.random.shuffle(train_data)
    
    return test_data, train_data

# Manipulate string to be in form of python list - convert target variables to be numeric
def multi_preprocessing(raw_test, raw_training, target_element, positive_class):    
    test_data = []
    train_data = []
    test_targets = []
    train_targets = []
    
    for row in raw_test:
        temp = row.split(",")
        temp[target_element] = ''.join([i for i in temp[target_element] if i.isdigit()])    
        if positive_class == int(temp[target_element]):  
            temp[target_element] = 1
        else:
            temp[target_element] = -1
        test_data.append(temp)
        test_targets.append(temp[target_element])
    
    
    for row in raw_training:
        temp = row.split(",")
        temp[target_element] = ''.join([i for i in temp[target_element] if i.isdigit()])
        if positive_class == int(temp[target_element]):   
            temp[target_element] = 1
        else:
            temp[target_element] = -1
        train_data.append(temp)
        train_targets.append(temp[target_element])
            
    # Convert lists into numpy arrays     
    np.asarray(test_data)
    np.asarray(train_data)
    np.asarray(test_targets)
    np.asarray(train_targets)
    
    
    return np.asarray(test_data), np.asarray(train_data)

def random_arrays(classes, data):
    s = np.arange(data[0].shape[0])
    np.random.shuffle(s)
    for i in range(classes):
        data[i] = data[i][s].astype(np.float)
    return data
        

    