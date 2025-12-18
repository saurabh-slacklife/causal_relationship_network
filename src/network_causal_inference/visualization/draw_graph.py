import matplotlib.pyplot as plt
import networkx as nx
def visualize_network(model, save_path):
    plt.figure(figsize=(12, 10))
    di_graph = nx.DiGraph()
    di_graph.add_edges_from(model.edges())

    pos = nx.spring_layout(di_graph)

    nx.draw_networkx_nodes(di_graph, pos, node_size=2000,
                           node_color='lightblue',
                           alpha=0.8)

    nx.draw_networkx_edges(di_graph, pos, edge_color='black',
                           arrows=True, arrowsize=20)

    # if feature_names:
    #     labels = {node: feature_names.get(node, node) for node in di_graph.nodes()}
    # else:

    labels = {node: node for node in di_graph.nodes()}

    nx.draw_networkx_labels(di_graph, pos, labels=labels, font_size=10)

    plt.title('Bayesian Network Structure', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
