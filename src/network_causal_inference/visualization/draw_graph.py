import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle
from matplotlib import colormaps
from pgmpy.models import DiscreteBayesianNetwork
import networkx as nx


def visualize_network_daft(dbn_model: DiscreteBayesianNetwork, save_path):
    dbn_model_daft = dbn_model.to_daft()
    dbn_model_daft.render()
    dbn_model_daft.savefig(save_path)

def visualize_network(model: DiscreteBayesianNetwork, name: str, save_path: str):
    plt.figure(figsize=(12, 10))
    di_graph = nx.DiGraph()
    di_graph.add_edges_from(model.edges())

    pos = nx.spring_layout(di_graph)

    # nx.draw_networkx_nodes(di_graph, pos, with_labels=True,
    #                        node_size=2000,
    #                        node_color='lightblue',
    #                        alpha=0.8)
    #
    # nx.draw_networkx_edges(di_graph, pos, edge_color='black',
    #                        arrows=True, arrowsize=20)
    #
    # # if feature_names:
    # #     labels = {node: feature_names.get(node, node) for node in di_graph.nodes()}
    # # else:
    #
    # labels = {node: node for node in di_graph.nodes()}
    #
    # nx.draw_networkx_labels(di_graph, pos, labels=labels, font_size=10)

    arrow_fancy=ArrowStyle.Fancy(head_length=.4, head_width=.4, tail_width=.1)
    nx.draw_networkx(di_graph, pos=pos, with_labels=True,
                     arrows=True, arrowstyle=arrow_fancy,alpha=1.0,
                     # arrowsize=20,
                     node_color='lightblue', cmap=colormaps)
    label = 'Bayesian Network Structure: '+ name
    plt.title(label=label, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
