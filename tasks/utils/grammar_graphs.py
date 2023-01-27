import networkx as nx


def get_reber_graph():
    reber = nx.DiGraph()
    reber.add_edges_from([
        ('start', '1', {'label': 'M'}),
        ('start', '2', {'label': 'V'}),
        ('1', '1', {'label': 'T'}),
        ('1', '3', {'label': 'V'}),
        ('2', '1', {'label': 'X'}),
        ('2', '4', {'label': 'X'}),
        ('3', '2', {'label': 'R'}),
        ('3', "end", {'label': 'T'}),
        ('4', '4', {'label': 'R'}),
        ('4', "end", {'label': 'M'}),
    ])
    # reber.add_edges_from([
    #     ('start', '1', {'probability': 0.5, 'label': 'M'}),
    #     ('start', '2', {'probability': 0.5, 'label': 'V'}),
    #     ('1', '1', {'probability': 0.5, 'label': 'T'}),
    #     ('1', '3', {'probability': 0.5, 'label': 'V'}),
    #     ('2', '1', {'probability': 0.5, 'label': 'X'}),
    #     ('2', '4', {'probability': 0.5, 'label': 'X'}),
    #     ('3', '2', {'probability': 0.5, 'label': 'R'}),
    #     ('3', "end", {'probability': 0.5, 'label': 'T'}),
    #     ('4', '4', {'probability': 0.5, 'label': 'R'}),
    #     ('4', "end", {'probability': 0.5, 'label': 'M'}),
    # ])

    return reber


def get_graph_A():
    graph = nx.DiGraph()
    graph.add_edges_from([
        ('start', '1', {'label': 'E'}),
        ('start', '5', {'label': 'E'}),
        ('start', '9', {'label': 'E'}),
        ('1', '2', {'label': 'A'}),
        ('2', '3', {'label': 'B'}),
        ('3', '4', {'label': 'C'}),
        ('4', '13', {'label': 'D'}),
        ('5', '6', {'label': 'B'}),
        ('6', '7', {'label': 'C'}),
        ('7', '8', {'label': 'D'}),
        ('8', '14', {'label': 'A'}),
        ('9', '10', {'label': 'C'}),
        ('10', '11', {'label': 'D'}),
        ('11', '12', {'label': 'A'}),
        ('12', '15', {'label': 'B'}),
        ('13', '1', {'label': 'B'}),
        ('13', '14', {'label': 'D'}),
        ('13', '16', {'label': 'A'}),
        ('13', '17', {'label': 'B'}),
        ('13', '18', {'label': 'C'}),
        ('14', '5', {'label': 'B'}),
        ('14', '17', {'label': 'D'}),
        ('14', '18', {'label': 'E'}),
        ('14', '19', {'label': 'A'}),
        ('15', '9', {'label': 'B'}),
        ('15', '14', {'label': 'C'}),
        ('15', '18', {'label': 'B'}),
        ('15', '19', {'label': 'C'}),
        ('15', '20', {'label': 'D'}),
        ('16', 'end', {'label': 'A'}),
        ('17', 'end', {'label': 'B'}),
        ('18', 'end', {'label': 'C'}),
        ('19', 'end', {'label': 'D'}),
        ('20', 'end', {'label': 'E'}),
        # ('20', '19', {'label': 'E'}),
    ])

    return graph


def concatenate_digraphs(graph1, graph2):
    edges = []
    nodenames = list(graph1.nodes) + list(graph2.nodes)
    counter = 0
    while f'{counter}' in nodenames:
        counter += 1
    graph1_noend = nx.relabel_nodes(graph1, {'end': f'{counter}'})
    graph2_nostart = nx.relabel_nodes(graph2, {'start': f'{counter}'})
    new_edges = list(graph1_noend.edges(data=True)) + list(graph2_nostart.edges(data=True))
    concatenated_graph = nx.DiGraph()
    concatenated_graph.add_edges_from(new_edges)

    return concatenated_graph


def add_to_all_node_labels(graph, string_to_add, exclude_start=True, exclude_end=True):
    nodenames = list(graph.nodes)
    if exclude_start:
        nodenames.remove('start')
    if exclude_end:
        nodenames.remove('end')

    rename_map = dict([(n, f"{n}{string_to_add}") for n in nodenames])
    new_graph = nx.relabel_nodes(graph, rename_map)

    return new_graph


def get_multyreber_graph(n_subgraphs=2):
    concatenated_reber = get_reber_graph()
    for i in range(1, n_subgraphs):
        graph_to_add = add_to_all_node_labels(get_reber_graph(), f'_{i}')
        concatenated_reber = concatenate_digraphs(concatenated_reber, graph_to_add)

    return concatenated_reber


def get_multyreber_plus_graphA(n_rebergraphs=2):
    multireber = get_multyreber_graph(n_subgraphs=n_rebergraphs)
    multireber = nx.relabel_nodes(multireber, {'M': 'A', 'V': 'B', 'T': 'C', 'X': 'D', 'R': 'E'})
    graphA = get_graph_A()
    graph = concatenate_digraphs(multireber, graphA)

    return graph





