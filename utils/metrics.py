import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgb
import networkx as nx
import networkx as nx
from GraphicalModels.utils.data_utils import VARIABLE_DEFINITIONS
from matplotlib.gridspec import GridSpec


def PN_centrality(G: nx.DiGraph) -> dict:
    """
    Compute PN-centrality for nodes in a directed graph with signed edge weights.

    This centrality measure distinguishes between positive and negative edges.
    It modifies the adjacency matrix by separating positive and negative weights,
    and uses a linear transformation to derive centrality scores.

    The method is based on the equation:
        PN = (I - (P - 2N)/(2n - 2))^-1 * 1
    where P is the matrix of positive weights, N is the matrix of negative weights,
    and n is the number of nodes.

    Parameters
    ----------
    G : networkx.DiGraph
        A directed graph with potentially positive and negative edge weights.

    Returns
    -------
    dict
        A dictionary mapping each node to its PN-centrality score (float).
    """
    adj_mat = np.array(nx.to_numpy_array(G))
    dims = adj_mat.shape
    p, n = np.zeros(dims), np.zeros(dims)

    for i in range(dims[0]):
        for j in range(dims[1]):
            if adj_mat[i, j] > 0:
                p[i, j] = adj_mat[i, j] 
            elif adj_mat[i, j] < 0:
                n[i, j] = -adj_mat[i, j] 

    temp = np.identity(dims[0]) - (p - 2*n) / (2*dims[0] - 2)
    pn = np.linalg.inv(temp) @ np.ones((dims[0], 1))
    
    return {i: j.item() for i, j in zip(G.nodes, pn)}

def kz_centrality(G: nx.DiGraph, 
                  beta: float =1, 
                  normalize: bool =True, 
                  safety_factor: float =0.9) -> dict:
    """
    Compute Katz centrality scores for a directed graph using dynamic alpha scaling.

    Katz centrality measures the influence of a node in a network by taking into account
    the total number of walks between nodes, exponentially damped by their length.
    This implementation dynamically sets the attenuation factor `alpha` based on the
    largest eigenvalue of the adjacency matrix and a `safety_factor` for numerical stability.

    Parameters
    ----------
    G : networkx.DiGraph
        A directed graph (may contain weighted edges).

    beta : float, optional
        Weight given to the immediate neighbors in the centrality calculation.
        Default is 1.

    normalize : bool, optional
        Whether to normalize the resulting centrality scores to unit norm.
        Default is True.

    safety_factor : float, optional
        A value in (0, 1) used to scale the attenuation factor `alpha` to ensure
        convergence of the matrix inversion. Default is 0.9.

    Returns
    -------
    dict
        A dictionary mapping each node to its Katz centrality score (float).
    """
    adj_mat = np.array(nx.to_numpy_array(G))
    dims = adj_mat.shape
    identity = np.identity(dims[0])
    unit_vector = np.ones((dims[0], 1))

    largest_eig = np.max(np.abs(np.linalg.eigvals(adj_mat)))
    alpha = safety_factor / largest_eig

    katz = beta * np.linalg.inv(identity - alpha * adj_mat.T) @ unit_vector

    if normalize:
        katz = katz / np.linalg.norm(katz)

    return {i: j.item() for i, j in zip(G.nodes, katz)}

def ev_centrality(G: nx.DiGraph, 
                  normalize: bool = True, 
                  imag_tol: float = 1e-10) -> dict:
    """
    Compute the eigenvector centrality of a directed graph.

    The eigenvector centrality of a node measures its influence based on the 
    principal eigenvector of the graph's adjacency matrix. This function calculates 
    the eigenvector corresponding to the largest magnitude eigenvalue and optionally 
    normalizes it. Complex eigenvector values with imaginary parts below a tolerance 
    are converted to their real parts.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed graph for which to compute eigenvector centrality.

    normalize : bool, optional
        Whether to normalize the principal eigenvector to unit norm. Default is True.

    imag_tol : float, optional
        Threshold below which imaginary parts of the eigenvector are ignored (set to zero).
        Default is 1e-10.

    Returns
    -------
    dict
        Dictionary mapping each node to its eigenvector centrality score (float or complex).
    """
    adj_mat = np.array(nx.to_numpy_array(G))
    eigvals, eigvecs = np.linalg.eig(adj_mat)

    idx = np.argmax(np.abs(eigvals))
    principal_eigvec = eigvecs[:, idx]

    if normalize:
        norm = np.linalg.norm(principal_eigvec)
        if norm != 0:
            principal_eigvec = principal_eigvec / norm

    if np.max(np.abs(np.imag(principal_eigvec))) < imag_tol:
        principal_eigvec = np.real(principal_eigvec)  

    return {i: j for i, j in zip(G.nodes, principal_eigvec)}

def compute_node_centralities(G: nx.DiGraph) -> pd.DataFrame:
    """
    Computes Eigenvector Centrality, Katz Centrality, and PN Centrality for a directed, weighted graph.

    Parameters:
    -----------
    G : nx.DiGraph
        A directed, weighted networkx graph. Can have positive and negative weights.

    Returns:
    --------
    centralities_df : pd.DataFrame
        A DataFrame with nodes as index and columns ['node', 'eigenvector', 'katz', 'pn'].
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("Graph must be a networkx.DiGraph.")

    ev_cent = ev_centrality(G)
    k_cent = kz_centrality(G)
    pn_cent = PN_centrality(G)      

    centralities_df = pd.DataFrame({
        'Node': list(G.nodes()),
        'Eigenvector': [ev_cent[node] for node in G.nodes()],
        'Katz': [k_cent[node] for node in G.nodes()],
        'PN': [pn_cent[node] for node in G.nodes()]
    }).reset_index(drop=True)

    return centralities_df

def plot_symptom_network_and_table(G: nx.DiGraph, variable_definitions: dict, condition: str = None, show_edge_labels: bool = False) -> None:
    """
    Plots a symptom network graph alongside a table describing the variables.

    Parameters:
    -----------
    G : networkx.DiGraph
        The network graph representing symptom relationships.

    variable_definitions : dict
        A dictionary containing variable metadata for different conditions.
        Expected structure: {condition_name: {index: variable_name, ...}}

    condition : str, optional
        The key in variable_definitions to use for labeling the table.

    Returns:
    --------
    None
    """
    fig = plt.figure(figsize=(16, 8))

    # Prepare node thresholds (biases h)
    thresholds = nx.get_node_attributes(G, "threshold")
    info_graph = {"thresholds": list(thresholds.values())} if thresholds else None

    # Prepare variable definitions table
    var_defs = variable_definitions[condition]
    df = pd.DataFrame({
        'Index': list(var_defs.keys()),
        'Variable Name': list(var_defs.values()),
        'Threshold': [round(thresholds.get(k, None), 3) for k in var_defs.keys()]
    })

    # Layout
    ax1 = fig.add_subplot(1, 2, 1)
    pos = nx.fruchterman_reingold_layout(G, seed=42)  # spring layout alternative

    # Node colors based on thresholds
    if thresholds:
        threshold_values = [thresholds.get(node, 0.5) for node in G.nodes()]
        min_thresh, max_thresh = min(threshold_values), max(threshold_values)
        cmap = plt.colormaps.get_cmap("YlGnBu")
        node_colors = [cmap((t - min_thresh) / (max_thresh - min_thresh)) for t in threshold_values]
    else:
        node_colors = "blue"

    # Edge colors based on sign, widths based on magnitude
    edge_colors = []
    edge_weights = []
    for u, v, d in G.edges(data=True):
        weight = d.get('weight', 0)
        edge_weights.append(abs(weight))
        if weight >= 0:
            edge_colors.append('blue')
        else:
            edge_colors.append('red')

    # Normalize edge widths
    if edge_weights:
        max_weight = max(edge_weights)
        edge_widths = [3.0 * (w / max_weight) for w in edge_weights]
    else:
        edge_widths = [1 for _ in G.edges()]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax1)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, ax=ax1, arrows=True)

        # Draw node labels with automatic color for readability
    labels = {}
    for node, (x, y) in pos.items():
        labels[node] = node

    label_colors = []
    for color in node_colors:
        r, g, b = to_rgb(color)
        brightness = (r * 0.299 + g * 0.587 + b * 0.114)
        label_colors.append('white' if brightness < 0.5 else 'black')

    for (node, (x, y)), color in zip(pos.items(), label_colors):
        ax1.text(x, y, s=node, bbox=dict(facecolor='none', edgecolor='none', pad=0.5),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=6, color=color)

    if show_edge_labels:
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax1)

    # Right subplot: Table of variable definitions
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')

    table = ax2.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='left'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Adjust table column widths
    col_widths = []
    for col in df.columns:
        max_length = max(df[col].astype(str).apply(len))
        col_widths.append(max(0.2, min(0.5, 0.03 * max_length)))

    for i, width in enumerate(col_widths):
        table.auto_set_column_width(i)
        for key, cell in table.get_celld().items():
            if key[1] == i:
                cell.set_width(width)

    table.scale(1, 1.2)
    fig.text(0.75, 0.85, "Symptom Variable Descriptions", ha='center', fontsize=12)

    # Colorbar for edge strengths
    if edge_weights:
        norm = Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
        sm = ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
        sm.set_array(edge_weights)

        cbar_ax = fig.add_axes([0.12, 0.1, 0.3, 0.02])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Edge Strength (|Weight|)', fontsize=10)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.2)

    fig_name = condition + "_network.png"
    plt.savefig(fig_name, dpi=400, bbox_inches='tight')
    plt.show()

def plot_full_symptom_network(G: nx.DiGraph, variable_definitions: dict, condition: str, save_path: str = None) -> None:
    """
    Plots Positive, Negative, and All edges of a symptom network,
    along with a symptom variable table and a colorbar, in a single figure.

    Parameters:
    -----------
    G : networkx.DiGraph
        Directed graph of the symptom network.
    variable_definitions : dict
        Mapping from variable indices to symptom names.
    condition : str
        Condition name for the title (e.g., 'PTSD', 'Depression').
    save_path : str, optional
        If provided, saves the figure to this path.
    """
    G_pos, G_neg = nx.DiGraph(), nx.DiGraph()

    for u, v, d in G.edges(data=True):
        weight = d.get('weight', 0)
        if weight > 0:
            G_pos.add_edge(u, v, weight=weight)
        elif weight < 0:
            G_neg.add_edge(u, v, weight=weight)

    for node, data in G.nodes(data=True):
        G_pos.add_node(node, **data)
        G_neg.add_node(node, **data)

    pos = nx.fruchterman_reingold_layout(G, seed=42, k=0.4)

    thresholds = nx.get_node_attributes(G, "threshold")
    var_defs = variable_definitions[condition]
    df = pd.DataFrame({
        'Index': list(var_defs.keys()),
        'Variable Name': list(var_defs.values()),
        'Threshold': [round(thresholds.get(k, None), 3) for k in var_defs.keys()]
    })

    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(nrows=2, ncols=4, height_ratios=[12, 1], width_ratios=[1, 1, 1, 1.25])
    fig.suptitle(f"{condition.title()} Network", fontsize=18, weight='bold')

    graphs = [G_pos, G_neg, G]
    titles = ['Positive Weights', 'Negative Weights', 'All Links']
    all_edge_weights = []

    for idx, (graph, title) in enumerate(zip(graphs, titles)):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_title(title, fontsize=14)
        ax.axis('off')

        if thresholds:
            threshold_values = [thresholds.get(node, 0.5) for node in graph.nodes()]
            cmap = plt.colormaps.get_cmap("YlGnBu")
            min_thresh, max_thresh = min(threshold_values), max(threshold_values)
            node_colors = [cmap((t - min_thresh) / (max_thresh - min_thresh)) for t in threshold_values]
        else:
            node_colors = "blue"

        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=400, ax=ax)

        edges = graph.edges(data=True)
        edge_colors, edge_weights = [], []
        for u, v, d in edges:
            weight = d.get('weight', 0)
            edge_weights.append(abs(weight))
            edge_colors.append('blue' if weight >= 0 else 'red')

        widths = [2.5 * (w / max(edge_weights)) for w in edge_weights] if edge_weights else [1 for _ in edges]
        nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=widths, ax=ax, arrows=True)

        label_colors = []
        for color in node_colors:
            r, g, b = to_rgb(color)
            brightness = (r * 0.299 + g * 0.587 + b * 0.114)
            label_colors.append('white' if brightness < 0.6 else 'black')

        for (node, (x, y)), color in zip(pos.items(), label_colors):
            ax.text(x, y, s=node, ha='center', va='center', fontsize=6, color=color)

        all_edge_weights.extend(edge_weights)

    ax_table = fig.add_subplot(gs[0, 3])
    ax_table.axis('off')
    ax_table.set_title("Symptom Variables", fontsize=12)

    table = ax_table.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for i, col in enumerate(df.columns):
        max_length = max(df[col].astype(str).apply(len))
        width = max(0.2, min(0.5, 0.03 * max_length))
        table.auto_set_column_width(i)
        for key, cell in table.get_celld().items():
            if key[1] == i:
                cell.set_width(width)

    table.scale(1, 1.2)

    if all_edge_weights:
        cbar_ax = fig.add_axes([0.30, 0.07, 0.4, 0.03])
        norm = Normalize(vmin=min(all_edge_weights), vmax=max(all_edge_weights))
        sm = ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
        sm.set_array(all_edge_weights)
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label("Edge Strength (|Weight|)", fontsize=10)

    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.10, wspace=0.3)

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()


if __name__ == '__main__':

    # disorders = ['mania', 'anxiety', 'phobia', 'depression', 'schizophrenia', 'PTSD']

    # for dis in disorders:
    #     model_name = dis + ".gml"
    #     df_name = dis + "_centralities.csv"
    #     G = nx.read_gml(model_name)
    #     df = compute_node_centralities(G)       
    #     df.to_csv(df_name)

    #     fig_name = dis + "_network.png"
    #     plot_full_symptom_network(G, VARIABLE_DEFINITIONS, dis, fig_name)
    
    model_name = "comorbidity.gml"
    df_name = "comorbidity_centralities.csv"
    G = nx.read_gml(model_name)
    df = compute_node_centralities(G)       
    df.to_csv(df_name)
    fig_name = "comorbidity_network.png"
    #plot_full_symptom_network(G, VARIABLE_DEFINITIONS, dis, fig_name)
    
