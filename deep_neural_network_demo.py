import random
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import time

class Readfile(object):
    def __init__(self):
        (X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()
        self.feature_list = np.array([X_train_image[i].reshape(-1) / 255.0 for i in range(0, X_train_image.shape[0])])
        self.output_list = self.one_hot_encoding(np.array(y_train_label))
        self.test_feature_list = np.array([X_test_image[i].reshape(-1) / 255.0 for i in range(0, X_test_image.shape[0])])
        self.test_output_list = self.one_hot_encoding(np.array(y_test_label))

    def one_hot_encoding(self, output_list):
        output_list = np_utils.to_categorical(output_list)
        return output_list

class BPNN(object):
    def __init__(self, dataset, learning_rate=0.1, batch_size=200, epoch=10, momentum=0.99):
        self.feature_list = dataset.feature_list
        self.output_list_OHE = dataset.output_list
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.momentum = momentum
        self.set_nn_architecture()

    # step1: definite of network architecture
    def set_nn_architecture(self):
        # amount of nodes
        self.input_node = self.feature_list.shape[1]
        self.output_node = self.output_list_OHE.shape[1]
        self.hidden1_node = 200
        self.hidden2_node = 200
        self.hidden3_node = 200

        # bias
        self.bias_weight_h1 = np.random.uniform(-1.0, 1.0, size=self.hidden1_node)
        self.bias_weight_h2 = np.random.uniform(-1.0, 1.0, size=self.hidden2_node)
        self.bias_weight_h3 = np.random.uniform(-1.0, 1.0, size=self.hidden3_node)
        self.bias_weight_o = np.random.uniform(-1.0, 1.0, size=self.output_node)

        # weight
        self.weight_list_h1 = np.random.uniform(-1.0, 1.0, size=(self.input_node, self.hidden1_node))
        self.weight_list_h2 = np.random.uniform(-1.0, 1.0, size=(self.hidden1_node, self.hidden2_node))
        self.weight_list_h3 = np.random.uniform(-1.0, 1.0, size=(self.hidden2_node, self.hidden3_node))
        self.weight_list_o = np.random.uniform(-1.0, 1.0, size=(self.hidden3_node, self.output_node))

        self.pre_delta_o = 0
        self.pre_delta_h1 = 0
        self.pre_delta_h2 = 0
        self.pre_delta_h3 = 0

        self.pre_delta_o_bias = 0
        self.pre_delta_h1_bias = 0
        self.pre_delta_h2_bias = 0
        self.pre_delta_h3_bias = 0

    def forward(self, mini_x):
        self.after_sigmoid_h1 = 1 / (1 + np.exp(-(np.dot(mini_x, self.weight_list_h1) + self.bias_weight_h1)))
        self.after_sigmoid_h2 = 1 / (1 + np.exp(-(np.dot(self.after_sigmoid_h1, self.weight_list_h2) + self.bias_weight_h2)))
        self.after_sigmoid_h3 = 1 / (1 + np.exp(-(np.dot(self.after_sigmoid_h2, self.weight_list_h3) + self.bias_weight_h3)))
        self.after_sigmoid_o = 1 / (1 + np.exp(-(np.dot(self.after_sigmoid_h3, self.weight_list_o) + self.bias_weight_o)))

    # step2: feed_forward stage (learning function: sigmoid)
    def predict(self, mini_x, mini_y):
        self.forward(mini_x)
        self.accuracy = 0
        for i in range(0, mini_y.shape[0]):
            calculate_y = np.argmax(self.after_sigmoid_o[i])
            practical_y = np.argmax(mini_y[i])
            if (practical_y == calculate_y):
                self.accuracy = self.accuracy + 1
        self.accuracy = self.accuracy / mini_y.shape[0] * 100

    # step3: back propagation neural network
    def backend(self, mini_x, mini_y):
        E = (mini_y - self.after_sigmoid_o)
        delta_y = E * self.after_sigmoid_o * (1 - self.after_sigmoid_o)
        delta_h3 = (1 - self.after_sigmoid_h3) * self.after_sigmoid_h3 * np.dot(delta_y, self.weight_list_o.T)
        delta_h2 = (1 - self.after_sigmoid_h2) * self.after_sigmoid_h2 * np.dot(delta_h3, self.weight_list_h3.T)
        delta_h1 = (1 - self.after_sigmoid_h1) * self.after_sigmoid_h1 * np.dot(delta_h2, self.weight_list_h2.T)

        self.weight_list_o += self.learning_rate * self.after_sigmoid_h3.T.dot(delta_y) + self.momentum * self.pre_delta_o
        self.weight_list_h3 += self.learning_rate * self.after_sigmoid_h2.T.dot(delta_h3) + self.momentum * self.pre_delta_h3
        self.weight_list_h2 += self.learning_rate * self.after_sigmoid_h1.T.dot(delta_h2) + self.momentum * self.pre_delta_h2
        self.weight_list_h1 += self.learning_rate * mini_x.T.dot(delta_h1) + self.momentum * self.pre_delta_h1

        self.bias_weight_o += self.learning_rate * delta_y.sum() + self.momentum * self.pre_delta_o_bias
        self.bias_weight_h3 += self.learning_rate * delta_h3.sum() + self.momentum * self.pre_delta_h3_bias
        self.bias_weight_h2 += self.learning_rate * delta_h2.sum() + self.momentum * self.pre_delta_h2_bias
        self.bias_weight_h1 += self.learning_rate * delta_h1.sum() + self.momentum * self.pre_delta_h1_bias

        self.pre_delta_o = self.learning_rate * self.after_sigmoid_h3.T.dot(delta_y)
        self.pre_delta_h3 = self.learning_rate * self.after_sigmoid_h2.T.dot(delta_h3)
        self.pre_delta_h2 = self.learning_rate * self.after_sigmoid_h1.T.dot(delta_h2)
        self.pre_delta_h1 = self.learning_rate * mini_x.T.dot(delta_h1)

        self.pre_delta_o_bias = self.learning_rate * delta_y.sum()
        self.pre_delta_h3_bias = self.learning_rate * delta_h3.sum()
        self.pre_delta_h2_bias = self.learning_rate * delta_h2.sum()
        self.pre_delta_h1_bias = self.learning_rate * delta_h1.sum()

    def train(self):
        self.accuracy_list = []
        for _iter in range(0, self.epoch):
            for i in range(0, self.output_list_OHE.shape[0], self.batch_size):
                mini_x = self.feature_list[i: i+self.batch_size]
                mini_y = self.output_list_OHE[i: i+self.batch_size]
                self.predict(mini_x, mini_y)
                self.backend(mini_x, mini_y)
            self.predict(self.feature_list, self.output_list_OHE)

            if(_iter % 5 == 0):
                print("Epoch = {} , train accuracy = {:.2f}".format(_iter, self.accuracy))

            if (self.accuracy >= 98):
                print("Epoch = {} , train accuracy = {:.2f}".format(_iter, self.accuracy))
                break

if __name__ == "__main__":
    start_time = time.time()
    dataset = Readfile()
    bpnn = BPNN(dataset, learning_rate=0.002, batch_size=300, epoch=500, momentum=0.99)
    bpnn.train()
    bpnn.predict(dataset.test_feature_list, dataset.test_output_list)
    end_time = time.time()

    print("test accuracy = {:.2f}".format(bpnn.accuracy))
    print("time = {}".format(end_time - start_time))
