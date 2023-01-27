import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from tasks.base_task import BaseTask
from evaluators.binary_evaluator import BinaryEvaluator
from tasks.utils import grammar_graphs  # import get_reber_graph, get_multyreber_graph, concatenate_digraphs


class GrammarTask(BaseTask):

    def __init__(self, n_sequences, grammar_graph, inputs=None):
        super().__init__()

        assert 'start' in grammar_graph.nodes and 'end' in grammar_graph.nodes, 'The grammar graph needs to have a "start" and and "end" node.'

        self.n_sequences = n_sequences
        self.grammar_graph = grammar_graph
        self.reversed_grammar_graph = self.grammar_graph.reverse(copy=True)
        self.alphabet = np.unique([edge[2]['label'] for edge in list(self.grammar_graph.edges(data=True))])
        self.input_dimension = len(self.alphabet)
        self.target = self._generate_target()
        self.input_labels, self.readout_indices, self.sequence_lengths = self._generate_all_sequences()
        self.input = self._generate_input()

    def draw_grammar_graph(self, ax=None, positions=None):
        if positions is None:
            positions = nx.spring_layout(self.grammar_graph)
        if ax is None:
            fig, ax = plt.subplots()
        nx.draw(self.grammar_graph, positions, ax=ax, with_labels=True)
        edge_labels = dict([((node1, node2) , properties['label']) for node1, node2, properties in self.grammar_graph.edges(data=True)])
        nx.draw_networkx_edge_labels(self.grammar_graph, positions, ax=ax, edge_labels=edge_labels)

        return ax

    def _generate_all_sequences(self, use_simple_invalid=False):
        labels = []
        readout_indices = []
        sequence_lengths = []
        current_index = -1

        for target in self.target:
            if target == 1:
                seq = self._generate_valid_sequence()
            elif use_simple_invalid:
                # seq = self._generate_invalid_sequence_based_on_valid()
                size = np.random.randint(12, 22)  # TODO: do this with correct power law
                seq = self._generate_invalid_sequence_simple(size=size)
            else:
                # seq = self._generate_invalid_sequence_based_on_valid()
                seq = self._generate_invalid_sequence_only_change_first_symbols()
            labels.extend(seq)
            sequence_lengths.append(len(seq))
            current_index += len(seq)
            readout_indices.append(current_index)

        return labels, readout_indices, sequence_lengths

    def _generate_valid_sequence(self):
        # sequence = ['start']
        sequence = []
        current_node = 'start'
        while current_node != 'end':
            neighbor_data = dict(self.grammar_graph[current_node])
            neighbors = list(neighbor_data.keys())
            if 'probability' in neighbor_data[neighbors[0]].keys():
                probabilities = [neighbor_data[n]['probability'] for n in neighbors]
                next_node = np.random.choice(neighbors, size=1, p=probabilities)[0]
            else:
                next_node = np.random.choice(neighbors, size=1)[0]
            sequence.append(neighbor_data[next_node]['label'])
            current_node = next_node
        # sequence.append('end')

        return sequence

    def _generate_invalid_sequence_based_on_valid(self, max_num_shuffles=5):
        sequence = self._generate_valid_sequence()
        counter = 0
        while self._ends_with_valid_sequence(sequence):
            if counter > max_num_shuffles:
                sequence = self._generate_valid_sequence()
                counter = 0
            np.random.shuffle(sequence)
            counter += 1

        return sequence

    def _generate_invalid_sequence_only_change_first_symbols(self, max_num_symbols_changed=5, max_num_shuffles=10):
        sequence = self._generate_valid_sequence()
        counter = 0
        while self._ends_with_valid_sequence(sequence):
            if counter > max_num_shuffles:
                sequence = self._generate_valid_sequence()
                counter = 0
            num_symbols_changed = min(len(sequence), max_num_symbols_changed)
            sequence[:num_symbols_changed] = np.random.choice(self.alphabet, size=num_symbols_changed)
            counter += 1

        return sequence

    def _generate_invalid_sequence_simple(self, size, max_num_shuffles=3):
        sequence = np.random.choice(self.alphabet, size=size, replace=True)
        counter = 0
        while self._ends_with_valid_sequence(sequence):
            if counter > max_num_shuffles:
                sequence = np.random.choice(self.alphabet, size=size, replace=True)
                counter = 0
            np.random.shuffle(sequence)
            counter += 1

        return sequence

    def _ends_with_valid_sequence(self, sequence):
        current_node = 'end'
        valid_end = False
        for label in sequence[::-1]:
            predecessor_data = dict(self.reversed_grammar_graph[current_node])
            predecessors = list(predecessor_data.keys())
            edge_labels = [pd['label'] for pd in predecessor_data.values()]
            if label in edge_labels:
                assert edge_labels.count(label) == 1, "Checking whether a sequence ends with a valid sequence does not yet work properly with graphs that have nodes that have multiple incomming edges with the same label!"
                current_node = predecessors[edge_labels.index(label)]
                if current_node == 'start':
                    valid_end = True
                    break
            else:
                valid_end = False
                break

        return valid_end

    def _check_input(self, inputs):
        pass

    def _generate_input(self):
        # one-hot-encoding
        inputs = np.zeros(shape=(len(self.input_labels), self.input_dimension))
        for input_idx, label in enumerate(self.input_labels):
            label_idx = list(self.alphabet).index(label)
            inputs[input_idx, label_idx] = 1

        return inputs

    def _generate_target(self):
        target = np.random.uniform(-1, 1, self.n_sequences)
        target[target>0] = 1
        target[target<=0] = 0

        return target

    def get_default_evaluator(self):
        return BinaryEvaluator()

    def get_state_filter(self):
        filter_map = np.zeros(len(self.input_labels), dtype=np.bool)
        filter_map[self.readout_indices] = True

        return filter_map


if __name__ == "__main__":
    # task = GrammarTask(n_sequences=1000, grammar_graph=get_reber_graph())
    # task = GrammarTask(n_sequences=1000, grammar_graph=grammar_graphs.get_multyreber_graph(n_subgraphs=3))
    # task = GrammarTask(n_sequences=1000, grammar_graph=grammar_graphs.get_graph_A())
    # task = GrammarTask(n_sequences=1000, grammar_graph=grammar_graphs.concatenate_digraphs(grammar_graphs.get_graph_A(), grammar_graphs.add_to_all_node_labels(grammar_graphs.get_graph_A(), string_to_add='_1')))
    task = GrammarTask(n_sequences=1000, grammar_graph=grammar_graphs.get_multyreber_plus_graphA())
    v1 = task._generate_valid_sequence()
    v2 = task._generate_valid_sequence()
    v3 = task._generate_valid_sequence()
    i1 = task._generate_invalid_sequence_simple(5)
    i2 = task._generate_invalid_sequence_simple(5)
    i3 = task._generate_invalid_sequence_simple(5)
    i4 = task._generate_invalid_sequence_based_on_valid()
    i5 = task._generate_invalid_sequence_based_on_valid()
    i6 = task._generate_invalid_sequence_based_on_valid()
    a = task._ends_with_valid_sequence(['M', 'T', 'T', 'V', 'T'])
    b = task._ends_with_valid_sequence(['X', 'R', 'M', 'T', 'T', 'V', 'T'])
    c = task._ends_with_valid_sequence(['M', 'T', 'T', 'V', 'R', 'T'])
    d = task._ends_with_valid_sequence(['T', 'T', 'T', 'V', 'T'])
    for i, seq in enumerate([v1, v2, v3, i1, i2, i3, i4, i5, i6]):
        print(f'{i}: {task._ends_with_valid_sequence(seq)}\t{seq}')
    task._generate_valid_sequence()
    axxx = task.draw_grammar_graph()
    plt.show()
    plt.clf()
    plt.hist(task.sequence_lengths)
    plt.show()
    asdf = 0