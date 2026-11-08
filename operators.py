import numpy as np


def andneuron(fuzzy_output, weights):
    # Implements AND logic here
    return np.prod(np.array(fuzzy_output) * weights)


def orneuron(fuzzy_output, weights):
    # Implements OR logic here
    return 1 - np.prod(1 - np.array(fuzzy_output))


def unineuron(
    self, fuzzy_output, weights, g_value, neuron_index, UniNormOperator
):  # Add neuron_index as argument
    # Initialize UniNormOperator with random g value
    self.uni_op = UniNormOperator(g_value)
    output = fuzzy_output[0] * weights[0]
    for i in range(1, len(fuzzy_output)):
        output = self.uni_op(output, fuzzy_output[i] * weights[i])

    self.neuron_details[neuron_index] = {
        "g": g_value,
        "operation_type": self.uni_op.operation_type,
    }
    return output
