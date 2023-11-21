#-------------------------------------------------------------------------
# AUTHOR: Gabriel Fok
# FILENAME: perceptron.py
# SPECIFICATION: This program reads the optdigits.tra and optdigits.tes files and uses the Perceptron and MLPClassifier algorithms to classify the data.
# FOR: CS 4210- Assignment #4
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test


p_highest_accuracy = 0
mlp_highest_accuracy = 0

for learning_rate in n: #iterates over n

    for shuffle in r: #iterates over r

        #iterates over both algorithms
        #-->add your Pyhton code here
        algorithms = ['Perceptron', 'MLPClassifier']

        for algo in algorithms: #iterates over the algorithms

            #Create a Neural Network classifier
            #if Perceptron then
            #   clf = Perceptron()    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            #else:
            #   clf = MLPClassifier() #use those hyperparameters: activation='logistic', learning_rate_init = learning rate,
            #                          hidden_layer_sizes = number of neurons in the ith hidden layer - use 1 hidden layer with 25 neurons,
            #                          shuffle = shuffle the training data, max_iter=1000
            #-->add your Pyhton code here
            if algo == 'Perceptron':
                clf = Perceptron(eta0=learning_rate, shuffle=shuffle, max_iter=1000)
            else:
                clf = MLPClassifier(activation='logistic', learning_rate_init=learning_rate, hidden_layer_sizes=25, shuffle=shuffle, max_iter=1000)

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            #for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #--> add your Python code here
            accuracy = 0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                clf.predict([x_testSample])
                if clf.predict([x_testSample]) == y_testSample:
                    accuracy += 1

            accuracy = accuracy / len(X_test)

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            #--> add your Python code here
            if algo == 'Perceptron':
                if accuracy > p_highest_accuracy:
                    p_highest_accuracy = accuracy
                    print("Highest " + algo + " accuracy so far: " + str(p_highest_accuracy) + ", Parameters: learning rate=" + str(learning_rate) + ", shuffle=" + str(shuffle))
            else:
                if accuracy > mlp_highest_accuracy:
                    mlp_highest_accuracy = accuracy
                    print("Highest " + algo + " accuracy so far: " + str(mlp_highest_accuracy) + ", Parameters: learning rate=" + str(learning_rate) + ", shuffle=" + str(shuffle))