#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow_derivative.inputs import Inputs
from tensorflow_derivative.outputs import Outputs
from tensorflow_derivative.derivatives import Derivatives


def neural_network(inputs):
    w = tf.get_variable(
        'w', initializer=np.array([[1.0, 0.0], [1.0, -1.0]], dtype=np.float32))
    b = tf.get_variable(
        'b', initializer=np.array([1.0, -1.0], dtype=np.float32))
    return tf.sigmoid(tf.add(b, tf.matmul(inputs, w)), name='neural_network')


def main():
    # Set up input vector for neural network
    inputs = Inputs(['x', 'y'])

    # Add a simple neural network on top of the inputs and create the outputs structure
    net = neural_network(inputs.placeholders)
    outputs = Outputs(net, ['a', 'b'])

    # Make derivative of neural network output respective to the inputs
    derivatives = Derivatives(inputs, outputs)
    da_dx = derivatives.get('a', ['x'])
    da_dxdy = derivatives.get('a', ['x', 'y'])

    # Calculate network output, first derivatives and second derivatives for given input
    example = np.zeros((1, 2), dtype=np.float32)
    example[0, :] = [0.0, 0.0]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        r, r_da_dx, r_da_dxdy = sess.run(
            [net, da_dx, da_dxdy], feed_dict={inputs.placeholders: example})

    print("Response a: {}".format(r[0]))
    print("Response da_dx: {}".format(r_da_dx[0]))
    print("Response da_dxdy: {}".format(r_da_dxdy[0]))


if __name__ == "__main__":
    main()
