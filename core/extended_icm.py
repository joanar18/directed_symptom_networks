import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Union

def update_adopters(network: Union[nx.Graph, nx.DiGraph],
                    adopters: List[int],
                    external: bool = False,
                    n_external: int = 10) -> Tuple[List[int], int]:
    """
    Updates the list of adopters in a threshold model based on network influence.

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The social network graph where nodes may adopt based on neighbor influence.
        Each node can optionally have a 'threshold' attribute, and each edge a 'weight'.
    adopters : list of int
        The list of currently adopted node IDs.
    external : bool, optional
        If True, randomly selects additional external adopters (default is False).
    n_external : int, optional
        The number of external adopters to introduce if `external` is True (default is 10).

    Returns
    -------
    updated_adopters : list of int
        The updated list of adopter node IDs.
    num_candidates : int
        The number of nodes that were eligible to adopt (i.e., not yet adopters).
    """
    adopters_set = set(adopters)
    candidates = set(network.nodes) - adopters_set
    new_adopters = []
    
    for node in candidates:
        influence = 0.0
        for neighbor in network.predecessors(node) if isinstance(network, nx.DiGraph) else network.neighbors(node):
            if neighbor in adopters_set:
                weight = network[neighbor][node].get('weight', 1.0)
                influence += weight
        threshold = network.nodes[node].get('threshold', 1.0)
        
        if influence >= threshold:
            new_adopters.append(node)
    
    # External adopters
    if external:
        remaining = list(candidates - set(new_adopters))
        if remaining:
            ext_adopters = random.sample(remaining, min(n_external, len(remaining)))
            new_adopters.extend(ext_adopters)
    
    updated_adopters = list(adopters_set.union(new_adopters))
    return updated_adopters, len(candidates)

def simulate_threshold_ICM(network: Union[nx.Graph, nx.DiGraph],
                           source_node: int,
                           external: bool = False,
                           n_external: int = 0) -> List[List[int]]:
    """
    Simulates the Independent Cascade Model (ICM) with threshold-based adoption over a network.

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The social network where diffusion is simulated. Each node may have a 'threshold' attribute,
        and edges may have 'weight' attributes.
    source_node : int
        The initial adopter node from which the cascade starts.
    external : bool, optional
        Whether to introduce external adopters at each time step (default is False).
    n_external : int, optional
        Number of external adopters to randomly introduce per time step (default is 0).

    Returns
    -------
    activated : list of list of int
        A list where each element represents the list of adopters at each time step.
    """
    adopters = [source_node]
    N = network.number_of_nodes()
    L = network.number_of_edges()
    
    time_step = 0
    tried = 1
    activated = [adopters]
    
    while tried < N:
        adopters, ntried = update_adopters(network, adopters, external, n_external)
        activated.append(sorted(adopters))
        tried += ntried
        time_step += 1
        
        if time_step > L:
            break
        if activated[-1] == activated[-2]:
            break

    return activated

def full_influence_matrix(network: Union[nx.Graph, nx.DiGraph],
                          n_runs_per_source: int = 100,
                          external: bool = False,
                          n_external: int = 0) -> pd.DataFrame:
    """
    Computes the full influence matrix by simulating threshold-based diffusion 
    from each node in the network multiple times.

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The network on which to simulate the diffusion. Nodes may have 'threshold' attributes,
        and edges may have 'weight' attributes.
    n_runs_per_source : int, optional
        Number of diffusion simulations to run per source node (default is 100).
    external : bool, optional
        Whether to introduce external adopters during the diffusion (default is False).
    n_external : int, optional
        Number of external adopters to randomly introduce per time step (default is 0).

    Returns
    -------
    pd.DataFrame
        A DataFrame of shape (N, N), where entry (i, j) represents the estimated probability 
        that node j adopts when node i is the initial source.
    """
    node_list = list(network.nodes)
    N = len(node_list)
    node_index = {node: idx for idx, node in enumerate(node_list)}
    
    activation_matrix = np.zeros((N, N))
    
    for i, source in enumerate(node_list):
        activation_counts = np.zeros(N)
        
        for _ in range(n_runs_per_source):
            activated = simulate_threshold_ICM(network, source_node=source, external=external, n_external=n_external)
            final_adopters = set(activated[-1])
            for node in final_adopters:
                activation_counts[node_index[node]] += 1
        
        activation_probs = activation_counts / n_runs_per_source
        activation_matrix[i, :] = activation_probs
    
    df = pd.DataFrame(activation_matrix, index=node_list, columns=node_list)
    return df

if __name__ == '__main__':
    # Application to depression
    G = nx.read_gml('depression.gml')
    G_copy = G.copy()

    # ICM needs positive thresholds
    for n, d in G_copy.nodes(data=True):
        if 'threshold' in d:
            d['threshold'] = float(1 / (1 + np.exp(-float(d['threshold']))))

    positive_edges = 0
    negative_edges = 0

    for u, v, d in G.edges(data=True):
        weight = float(d.get('weight', 0.0))
        if weight > 0:
            positive_edges += 1
        elif weight < 0:
            negative_edges += 1

    print(f"Positive edges: {positive_edges}")
    print(f"Negative edges: {negative_edges}")
    print(f"Total edges: {positive_edges + negative_edges}")

    # activation probabilities
    influence_df = full_influence_matrix(G_copy, n_runs_per_source=100, external=False)
    print(influence_df.round(2).head())

    # Save to CSV
    # influence_df.to_csv('activation_probabilities.csv')

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(influence_df, cmap='viridis')
    plt.title('Activation Probability Heatmap')
    plt.xlabel('Target Node')
    plt.ylabel('Source Node')
    plt.show()
