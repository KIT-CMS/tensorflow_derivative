import tensorflow as tf

class Derivatives(object):
    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs
