import re

import numpy as np
from sklearn.model_selection import train_test_split

from tasks import XOR, XOR_XOR, grammar, delayed_classifiation, temporal_continuous_XOR, narma
from tasks.utils import grammar_graphs


def sine(input):
    return np.sin(input)


# def XOR(input):
#     assert len(input.shape) == 2 and input.shape[1] == 2, "The input for the XOR tasks needs to be two dimensional"
#     unique_vals = np.unique(input)
#     assert unique_vals[0] == 0 and unique_vals[1] == 1, "The input should contain only ones and zeros"
#
#     return np.logical_xor(input[:, 0], input[:, 1]).astype(int)


def delayed_signal_and_states(signal, states, delay):
    if delay == 0:
        undelayed_signal = signal
        delayed_signal = signal
        delayed_states = states
    else:
        undelayed_signal = signal[delay:]
        delayed_signal = signal[:-delay]
        delayed_states = states[delay:]

    return undelayed_signal, delayed_signal, delayed_states


def train_test_splitx(array, test_ratio):
    steps = array.shape[0]
    test_size = int(steps*test_ratio)
    # train_array = array[:-test_size]
    # test_array = array[-test_size:]
    train_array = array[test_size:]
    test_array = array[:test_size]

    return train_array, test_array


def get_task(taskname, steps, input_path=None):
    if input_path is not None:
        inputs = np.load(input_path)
    else:
        inputs = None

    if "_DELAY=" in taskname:
        delay_re = re.search('.+_DELAY=(\d+)', taskname.upper())
        delay = int(delay_re.group(1))
    else:
        delay = 0

    if taskname.upper() == 'XOR' or taskname.upper().startswith('XOR_DELAY'):
        if (len(inputs.shape) == 2 and 1 in inputs.shape):
            inputs = np.reshape(inputs, -1)
        if inputs is not None and (len(inputs.shape)==1):
            if np.max(inputs) > 3:
                print('WARNING: the input values have more than 2 bits but only the first two will be used!')
                input_binary_strings = [f'{iv:b}'.rjust(4, '0') for iv in inputs]
            else:
                input_binary_strings = [f'{iv:b}'.rjust(2, '0') for iv in inputs]
            inputs = np.array([[int(ibs[0]), int(ibs[1])] for ibs in input_binary_strings])
        # task =  XOR.XOR(steps, inputs)
        task = XOR.XOR(steps, inputs, delay=delay)
    # elif taskname.upper().startswith('XOR_DELAY'):
    #     delay_re = re.search('.+_DELAY=(\d+)', taskname.upper())
    #     delay = int(delay_re.group(1))
    #     if inputs is not None and len(inputs.shape) == 1:
    #         if np.max(inputs) > 3:
    #             print('WARNING: the input values have more than 2 bits but only the first two will be used!')
    #             input_binary_strings = [f'{iv:b}'.rjust(4, '0') for iv in inputs]
    #         else:
    #             input_binary_strings = [f'{iv:b}'.rjust(2, '0') for iv in inputs]
    #         inputs = np.array([[int(ibs[0]), int(ibs[1])] for ibs in input_binary_strings])
    #     task = XOR.XOR(steps, inputs, delay=delay)
    elif taskname.upper() == 'XORXOR' or taskname.upper().startswith('XORXOR_DELAY'):
        if inputs is not None and len(inputs.shape)==1:
            input_binary_strings = [f'{iv:b}'.rjust(4, '0') for iv in inputs]
            inputs = np.array([[int(ibs[0]), int(ibs[1]), int(ibs[2]), int(ibs[3])] for ibs in input_binary_strings])
        task = XOR_XOR.XORXOR(steps, inputs, delay=delay)
    # elif taskname.upper().startswith('XORXOR_DELAY'):
    #     if inputs is not None and len(inputs.shape) == 1:
    #         input_binary_strings = [f'{iv:b}'.rjust(4, '0') for iv in inputs]
    #         inputs = np.array([[int(ibs[0]), int(ibs[1]), int(ibs[2]), int(ibs[3])] for ibs in input_binary_strings])
    #     task = XOR_XOR.XORXOR(steps, inputs)
    elif taskname.upper() == '4STREAMSXOR' or taskname.upper().startswith('4STREAMSXOR_DELAY'):
        if inputs is not None and len(inputs.shape) == 1:
            input_binary_strings = [f'{iv:b}'.rjust(4, '0') for iv in inputs]
            for ibs in input_binary_strings:
                assert ibs[0] != ibs[1], "the first two input streams need to be different"
                assert ibs[2] != ibs[3], "the last two input streams need to be different"
            # inputs = np.array([[int(ibs[0]), int(ibs[1]), int(ibs[2]), int(ibs[3])] for ibs in input_binary_strings])
            inputs = np.array([[1 if ibs[0] == '1' else 0, 1 if ibs[2] == '1' else 0] for ibs in input_binary_strings])
        task = XOR.XOR(steps, inputs, delay=delay)
    elif taskname.upper() == "TEMPORAL_CONTINUOUS_XOR" or taskname.upper() == "TEMPORAL_XOR":
        task = temporal_continuous_XOR.TemporalContinuousXOR(steps=steps, inputs=inputs)
    elif taskname.upper().startswith('NARMA'):
        narma_n_re = re.search('NARMA(\d+)', taskname.upper())
        narma_n = int(narma_n_re.group(1))
        task = narma.NARMA(steps, inputs, n=narma_n)
    elif taskname.upper() == 'REBER':
        # task = grammar.GrammarTask(args.steps, grammar_graph=grammar_graphs.get_reber_graph())
        task = grammar.GrammarTask(steps, grammar_graph=grammar_graphs.get_multyreber_graph(n_subgraphs=1))
        # task = grammar.GrammarTask(args.steps, grammar_graph=grammar_graphs.get_graph_A())
        # task = grammar.GrammarTask(args.steps, grammar_graph=grammar_graphs.get_multyreber_plus_graphA(n_rebergraphs=1)
    elif taskname.upper().startswith('DELAYED-CLASSIFICATION'):
        # delay_re = re.search('.+_DELAY=(\d+)', taskname.upper())
        # if not delay_re:
        #     delay = 0
        # else:
        #     delay = int(delay_re.group(1))
        if inputs is None:
            n_classes = 10
        else:
            n_classes = len(np.unique(inputs))
        task = delayed_classifiation.DelayedClassification(steps, n_classes=n_classes, delay=delay, input=inputs)
    else:
        raise NotImplementedError(f'The task {taskname} is not implemented')

    return task
