#Network Template
#Made to be useful with any dataset
#Variable number of layers and such
#All classification problems should work
import random

import numpy as np

class Network(object):

    #Sizes is a matrix containg the number of neurons in each layer
    #Ex. [3,4,5] would be a network with 3 layers containing 3, 4 and 5 neurons in each layer respectively
    def __init__(self, sizes):
        self.layers  = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand(y) for y in sizes[1:]]
        self.weights = [np.random.rand(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, inputs):
        #Each input is one matrix
        #When iterating through the multidimensional weights array, dot the matrix with each one and add the bias
        #This will give the raw activation, then do the sigmoid function
        activations = []
        for i in inputs:
            for b, w in zip(self.biases, self.weights):
                a = np.dot(w, i)+ b
                ins = self.sigmoid(a)
                i = ins
            activations.append(ins)
        return np.array(activations)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def der_sigmoid(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def cost_derivative(self, y, y_hat):
        return y-y_hat

    #def epoch(self):
        #Shuffle training data
        #Pick a sample
        #Feedforward

    #Fix this up in a format to to be useful for feedforward once again
    def update_network(self, training_data, outputs):
        for i in range(10000):
            delta_w = [np.zeros(w.shape) for w in self.weights]
            delta_b = [np.zeros(b.shape) for b in self.biases]
            for x,y in zip(training_data, outputs):
                update_weigths, update_biases = self.backprop(x, y)
                for w, w_u in zip(delta_w, update_weigths):
                    w += w_u
                for b, b_u in zip(delta_b, update_biases):
                    b += b_u
            for weights, delta in zip(self.weights, delta_w):
                weights += delta
            for biases, delta in zip(self.biases, delta_b):
                biases += delta

    def backprop(self, x, y):

        #Fix the feedforward so that the activations append to a matrix
        #Bring in this activations to compare to the training data
        #Calculate the gradient so that the change in weights and
        #biases can be calculated from here
        activations = [x]
        activation = x
        zs = []
        for b, w in zip(self.biases, self.weights):
            a = np.dot(w, x) + b
            zs.append(a)
            ins = self.sigmoid(a)
            activations.append(ins)
            x = ins

        activations = np.array(activations)
        #print("Activations: ", activations)


        #Set dummy matrices to store the gradient in
        der_b1 = [np.zeros(b.shape) for b in self.biases]
        der_w1 = [np.zeros(w.shape) for w in self.weights]

        der_b2 = []
        der_w2 = []
        for b, w in zip(self.biases, self.weights):
            der_b2.append(np.zeros(np.shape(b)))
            der_w2.append(np.zeros(np.shape(w)))

        ##Calculate the gradient
        delta = self.cost_derivative(y, activations[-1]) * self.der_sigmoid(zs[-1])
        der_b2[-1] = delta
        der_w2[-1] = np.dot(np.array([delta]).T, np.array([activations[-2]]))

        #Now calculate the rest of the deltas for the network
        delta = np.array([delta]).T
        for t in range(2, len(self.sizes)):
            z = zs[-t]
            sp = self.der_sigmoid(z)
            delta = np.dot(np.array(self.weights[-t+1]).T, delta)
            delta = delta * np.array([sp]).T
            der_b2[-t] = delta
            der_w2[-t] = np.dot(delta, np.array([activations[-t-1]]))

        return der_w2, der_b2
        ##Adjust learning rate

    #def update(self):3
        #Adjust the weights and biases according to gradient

Net = Network([3,1])
training_data = [[1,0,0], [1,0,1], [1,0,0], [1,1,1]]
y = [[0], [1], [0], [1]]
Initial = Net.feedforward(training_data)
#print("Initial Outputs Residual: ", y-Initial)
Net.update_network(training_data, y)
After = Net.feedforward(training_data)
print("After Training Outputs Residual: ", y-After)
Test = Net.feedforward([[0,1,1]])
print("Test Set: ", Test)