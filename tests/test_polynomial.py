#!/usr/bin/env python

import unittest
import numpy as np
import tensorflow as tf
from tensorflow_derivative.inputs import Inputs
from tensorflow_derivative.outputs import Outputs
from tensorflow_derivative.derivatives import Derivatives


def f1(inputs, a, b):
    with tf.variable_scope('f1'):
        a = tf.get_variable('a', initializer=np.array([[a]], dtype=np.float32))
        b = tf.get_variable('b', initializer=np.array([b], dtype=np.float32))
        return tf.sigmoid(tf.add(b, tf.matmul(inputs, a)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))


class TestPolynomial(unittest.TestCase):
    def test_f1(self):
        inputs = Inputs(['x'])

        a = 2.0
        b = -3.0
        f = f1(inputs.placeholders, a, b)
        outputs = Outputs(f, ['f'])

        derivatives = Derivatives(inputs, outputs)
        df_dx = derivatives.get('f', ['x'])
        df_dxdx = derivatives.get('f', ['x', 'x'])

        x = np.zeros((1, 1), dtype=np.float32)
        x[0, :] = [1.0]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            f_, df_dx_, df_dxdx_ = sess.run(
                [f, df_dx, df_dxdx], feed_dict={inputs.placeholders: x})

        # f = sigmoid(b+a*x)
        f_test = sigmoid(b + a * x)
        self.assertEqual(f_[0][0], f_test)

        # df_dx = f*(1.0-f)*a
        df_dx_test = f_test * (1.0 - f_test) * a
        self.assertEqual(df_dx_[0][0], df_dx_test)

        # df_dxdx = a*a*(f*(1.0-f)*(1.0-f)-f*(1.0-f))
        df_dxdx_test = a**2 * (f_test * (1.0 - f_test)**2 - f_test**2 *
                               (1.0 - f_test))
        self.assertEqual(df_dxdx_[0][0], df_dxdx_test)


if __name__ == "__main__":
    unittest.main()
