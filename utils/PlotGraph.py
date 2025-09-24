from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
from ConeGraphProducer import ConeGraphProducer
import numpy as np
import matplotlib.patches as mpatches


category_labels = {
    13:   r"$\mu$",
    -13:  r"$\mu$",
    22:   r"$\gamma$",
    130:  r"$K_L^0$",
    -211: r"$\pi$",
    211:  r"$\pi$",
    -130: r"$\bar{K_L^0}$",
}



def plot_graph(graph_data, title, num ,feature_index=3):
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
    pdg_ids = graph_data.x[:, feature_index].numpy()
    node_color = np.abs(pdg_ids)

    fig, ax = plt.subplots()

    unique_vals_set = set(node_color)
    particle_pairs = [13, -13, 22, 211, -211, 130, -130]
    unique_vals = [val for val in particle_pairs if val in unique_vals_set]
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_vals)))
    color_map = {val: colors[i] for i, val in enumerate(unique_vals)}
    mapped_colors = [color_map[val] for val in node_color]
    # Draw graph
    nodes = nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=mapped_colors,
                               node_size=500)
    nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color='gray')
    # nx.draw_networkx_labels(G, pos=pos, ax=ax)

    legend_patches = [
        mpatches.Patch(color=color_map[val], label=category_labels[val])
        for val in unique_vals
    ]
    ax.legend(handles=legend_patches, title="Particle", loc='best')

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    # Axes and colorbar
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$\phi$')
    ax.set_title(title)
    # plt.colorbar(nodes, ax=ax, label=f"Node feature at index {feature_index}")
    plt.savefig(f"/work/kfuss/SVJ_GNN/npv_histograms/graph_signal{num}.png", dpi=300) 

    return

graph = ConeGraphProducer("/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn/signal_W", "W_decay_signal_skimmed_dzd0_cone_size_0.5_knn_10")
plot_graph(graph[0], r"signal kNN graph $k=10$", 1)
plot_graph(graph[1], r"signal kNN graph $k=10$", 2)
plot_graph(graph[2], r"signal kNN graph $k=10$", 3)
plot_graph(graph[3], r"signal kNN graph $k=10$", 4)
plot_graph(graph[4], r"signal kNN graph $k=10$", 5)
plot_graph(graph[5], r"signal kNN graph $k=10$", 6)
plot_graph(graph[6], r"signal kNN graph $k=10$", 7)
plot_graph(graph[7], r"signal kNN graph $k=10$", 8)
