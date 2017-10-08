#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow_derivative.inputs import Inputs
from tensorflow_derivative.outputs import Outputs
from tensorflow_derivative.derivatives import Derivatives

def neural_network(inputs):
    w = tf.get_variable('w', initializer=np.array([[1.0], [1.0]], dtype=np.float32))
    b = tf.get_variable('b', initializer=np.array([1.0, 0.0], dtype=np.float32))
    return tf.add(b,tf.matmul(inputs,w), name='neural_network')

def main():
    """
    Set up input vector for neural network
    """
    inputs_names = ['x', 'y']
    inputs = Inputs(inputs_names)
    placeholders = inputs.placeholders

    """
    Add a simple neural network on top of the inputs and create the outputs structure
    """
    net = neural_network(placeholders)

    outputs_names = ['a', 'b']
    outputs = Outputs(net, outputs_names)

    """
    Make derivative of neural network output respective to the inputs
    """
    derivatives = Derivatives(inputs, outputs)

    """
    Calculate network output and first derivatives for given input
    """
    example = np.zeros((1,2), dtype=np.float32)
    example[0,:] = [1.0, 2.0]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        response = sess.run([net], feed_dict={placeholders:example})
        print("Response: {}".format(response))

if __name__ == "__main__":
    main()
