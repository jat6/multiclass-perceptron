import numpy as np
import matplotlib.pyplot as plt

plt.ion()

# Perceptron training loop
def m_perceptron_train(epochs, instances, examples, features_count, classes, train_data, test_data, w, bias, reg_term):
    for i in range(epochs):   
        errors = 0  
        for j in range(instances):
            for p in range(classes):
                activation = 0
                data = train_data[p]
                y = float(data.item(j, classes + 1))
                activation = float(bias.item(p)) + np.dot(data[j,:-1], w[p,:])
                if((y*activation) <= 0):
                    errors += 1
                    bias[p] += y
                    #w[p] +=  0.05 * (y - activation) * data[j,:-1]
                    w[p] += (y * data[j,:-1]) + reg_term * np.sum(w[p])
                    #print(w)
                    #for d in range(features_count):
                        #w[p,d] += (y * data[j,d])
                        #w[p,d] =  w[p,d] + 0.05*(y - activation) * data[j,d]
        print("epoch: " + str(i) + " " + str(errors) + "/" + str(instances * 3))
        m_perceptron_test(classes, features_count, examples, test_data, w, bias, i)
        
# Perceptron testing
def m_perceptron_test(classes, features_count, examples, test_data, w, bias, epoch):
    errors = 0
    activation = np.zeros([classes, 1])
    predictions = np.zeros([classes, 1])
    y = np.zeros([classes, 1])
    for i in range(examples):
        for p in range(classes):
            data = test_data[p]
            y[p,0] = float(data.item(i, classes + 1))
            activation[p,0] = float(bias.item(p)) + np.dot(data[i,:-1], w[p,:])
            #print("Predicted: " + str(activation) + ", Expected: " + str(y))
            predictions = y*activation
        #print(predictions)
        argmax = np.argmax(abs(0 - predictions))
        #print("Predict: " + str(activation[argmax]) + "Actual: " + str(y[argmax]))
        #print(y)
        #print(y[argmax])
        #print(str(predictions[argmax]))
        if(predictions[argmax] <= 0):
            errors += 1
            #print("WRONG")
    print("PREDICTION: " + str(errors) + "/" + str(examples) + " misclassified instances\n") 
    #plt.scatter(epoch, errors)
    #plt.pause(0.05)
    #print(w)