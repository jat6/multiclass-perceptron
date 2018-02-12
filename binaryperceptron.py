import numpy as np

# Perceptron training loop
def perceptron_train(epochs, instances, examples, features_count, train_x, train_y, test_x, test_y, w, bias):
    for i in range(epochs):    
        errors = 0  
        for j in range(instances):    
            activation = 0
            y = np.ndarray.item(train_y[j])
            activation = bias + np.dot(train_x[j,:], w[0,:])
            #print("Predicted: " + str(activation) + ", Expected: " + str(y))
            if((y*activation) <= 0):
                #print("Iteration: " + str(j) + ", Predicted: " + str(activation) + ", Expected: " + str(y))
                errors += 1
                bias += y
                for d in range(features_count):
                    w[0,d] += (y * train_x[j, d].astype(np.float))
                    #w[0,d] =  w[0,d] + 0.05*(y - activation) * train_x[j,d]
                #print(w)
        print("TRAINING: " + "Epoch: " + str(i + 1) + ": " + str(errors) + "/" + str(instances) + " misclassified instances\n")
        perceptron_test(examples, test_x, test_y, w, bias)
    return w, bias
        
# Perceptron testing
def perceptron_test(examples, test_x, test_y, w, bias):
    errors = 0
    for i in range(examples):
        y = np.ndarray.item(test_y[i])
        activation = bias + np.dot(test_x[i,:], w[0,:])
        #print("Predicted: " + str(activation) + ", Expected: " + str(y))
        if((y*activation) <= 0):
            #print("WRONG")
            errors = errors + 1
    print("PREDICTION: " + str(errors) + "/" + str(examples) + " misclassified instances\n") 