import sys,os
import numpy as np

sys.path.append("./common/")

from functions import *

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * \
                            np.random.randn( input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * \
                            np.random.randn( hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)


    def Predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1  = np.dot(x, W1) + b1
        z1  = Sigmoid(a1)

        a2  = np.dot(z1, W2) + b2
        y   = SoftMax(a2)

        return(y)
    
    def Loss( self, x, t):
        y = self.Predict(x)

        return(CrossEntropyError(y, t))

    def Accuracy( self, x, t):
        y = self.Predict(x)
        y = np.argmax(y , axis = 1)
        t = np.argmax(t , axis = 1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return(accuracy)

    def NumericalGradient(self, x, t):
        loss_W = lambda W: self.Loss(x, t)
        grads = {}

        grads["W1"] = NumericalGradient(loss_W, self.params["W1"])
        grads["b1"] = NumericalGradient(loss_W, self.params["b1"])
        grads["W2"] = NumericalGradient(loss_W, self.params["W2"])
        grads["b2"] = NumericalGradient(loss_W, self.params["b2"])

        return(grads)


net = TwoLayerNet(input_size = 784, hidden_size=100, output_size=10)




