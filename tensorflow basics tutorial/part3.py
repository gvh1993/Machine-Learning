# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:20:16 2017

@author: Gert-Jan
"""



'''

input > weight > hidden layer 1 (activation function) > weights > hidden layer 2
(activation function) > weights > output layer

feed forward = put it straight through to our neural network

at the end: compare output to intended output > cost function (cross entropy)

optimization function (optimizer) > minimize cost (AdamOptimizer, Stochastic gradient descent, AdaGrad)

go back and manipulate the weights = backpropagation

feedforward + backprop = epoch

'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#10 classes = 0,9

'''
one hot to off:
    0 = 0
    1 = 1
    2 = 2
    n = n
    
one hot to True:
    0 = [1,0,0,0,0,0,0,0,0,0]
    1 = [0,1,0,0,0,0,0,0,0,0]
    2 = [0,0,1,0,0,0,0,0,0,0]
    3 = [0,0,0,1,0,0,0,0,0,0]
'''

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784]) #28 pixels * 28 pixels = 784
y = tf.placeholder('float')

def neural_network_model(data):
    # (input_data * weights) + biases
    # biases are coming in handy if all weights are 0 and no neurons are fired
    
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])), #your weights for hidden layer 1 are randomly chosen which are 784 by 500
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))} # biases are added in after the weights. 
    
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), #your weights for hidden layer 1 are randomly chosen which are 784 by 500
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))} # biases are added in after the weights. 
    
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), #your weights for hidden layer 1 are randomly chosen which are 784 by 500
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))} # biases are added in after the weights. 
    
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), #your weights for hidden layer 1 are randomly chosen which are 784 by 500
                      'biases': tf.Variable(tf.random_normal([n_classes]))} # biases are added in after the weights. 
    
    
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    
    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
                
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
                
        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x)