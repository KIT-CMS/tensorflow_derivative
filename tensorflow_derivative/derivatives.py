import tensorflow as tf


class Derivatives(object):
    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs

    def get(self, output, inputs):
        if not output in self._outputs.names:
            raise Exception('Output {} is not in list {}.'.format(
                output, self._outputs.names))
        for name in inputs:
            if not name in self._inputs.names:
                raise Exception('Input {} is not in list {}.'.format(
                    name, self._inputs.names))

        print('inputs', inputs)
        ops = [self._outputs.outputs_dict[output]]
        for name in inputs:
            print('before', ops)
            ops.append(
                tf.gradients(ops[-1], self._inputs.placeholders_dict[name]))
            print('after', name, ops)
        return ops[-1]
