from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx

def plot_graph(graph_data, title="Graph", feature_index=3):
    """
    Plots a PyTorch Geometric graph using NetworkX with positions and color-coded nodes.

    Parameters:
        graph_data (torch_geometric.data.Data): The graph to plot.
        title (str): The plot title.
        feature_index (int): Index of the node feature to use for color coding. Standard is pdgId
    """
    # Convert to NetworkX
    G = to_networkx(graph_data, to_undirected=True)

    # Extract node positions
    pos_tensor = graph_data.pos  # shape: [num_nodes, 2]
    pos = {i: (pos_tensor[i][0].item(), pos_tensor[i][1].item()) for i in range(len(pos_tensor))}

    # Node color from selected feature
    node_color = graph_data.x[:, feature_index].numpy()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw graph
    nodes = nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=node_color,
                                   cmap='viridis', node_size=500)
    nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color='gray')
    nx.draw_networkx_labels(G, pos=pos, ax=ax)

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    # Axes and colorbar
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$\varphi$')
    ax.set_title(title)
    ax.grid(True)
    # plt.colorbar(nodes, ax=ax, label=f"Node feature at index {feature_index}")
    plt.show()

    return