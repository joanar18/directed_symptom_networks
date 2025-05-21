import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import networkx as nx

import matplotlib
matplotlib.use('TkAgg')

from GraphicalModels.utils.data_utils import load_disorder_subset, to_tensor, get_variable_labels, load_multiple_disorders
from GraphicalModels.core.models import run_model_diagnostics, DirectedInfluenceIsingModel
    

# Utils -----------------------------------------------------------------------------------------

def get_data(df: pd.DataFrame, disorders):
    """Return tensor X and variable names for the selected disorders.

    Parameters
    ----------
    df : pandas.DataFrame
        Full dataset already loaded in memory.
    disorders : str | list[str]
        Disorder name(s) to subset columns.
    """
    if isinstance(disorders, str):
        disorders = [disorders]

    df_subset = load_disorder_subset(df, disorders)
    #df_subset = load_multiple_disorders(df, disorders)    # To import multiple disorders instead TODO: make a single consistent function
    variable_names = list(df_subset.columns)
    X = to_tensor(df_subset)
    return X, variable_names


def compute_relative_threshold(W, fraction=0.1, mode="row"):
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()

    if mode == "global":
        max_weight = np.max(np.abs(W))
        return fraction * max_weight
    elif mode == "row":
        row_maxes = np.max(np.abs(W), axis=1, keepdims=True) 
        thresholds = fraction * row_maxes
        threshold_matrix = thresholds * np.ones_like(W) 
        return threshold_matrix
    else:
        raise ValueError("mode must be 'row' or 'global'")

def convert_attributes_to_python_types(G):
    for n, d in G.nodes(data=True):
        for key in d:
            if isinstance(d[key], np.generic):
                d[key] = d[key].item()
    for u, v, d in G.edges(data=True):
        for key in d:
            if isinstance(d[key], np.generic):
                d[key] = d[key].item()
    return G

def normalize_incoming_weights(G):
    if not isinstance(G, nx.DiGraph):
        raise ValueError("Graph must be directed (DiGraph) for proper incoming normalization.")
    
    for node in G.nodes:
        in_edges = list(G.in_edges(node, data=True))
        total_influence = sum(abs(d.get('weight', 1.0)) for u, v, d in in_edges)
        
        if total_influence > 0:
            for u, v, d in in_edges:
                d['weight'] = d.get('weight', 1.0) / total_influence
    return G

# Single fit ------------------------------------------------------------------------------------
def run_comorbidity_experiment(
        disorders: str | list[str],
        df: pd.DataFrame,
        *,
        threshold_fraction: float = .01,
        auto_lambda   = True,
        lambda_l1: float | list[float] = .0001,
        lambda_l2: float | list[float] = .01,
        epochs: int = 500,
        normalise_rows: bool = True,
        normalise_during_training: bool = False,
        use_l2: bool = True,
        lr: float = .01,
        **extra_kwargs 
):
    """
    Trains one network on the chosen disorder(s) and prints / returns:

        G  : DiGraph  (thresholded)
        info : {'weights', 'thresholds'}
        summary_df : diagnostics table
        labels     : ICD-friendly symptom names
    """
    if df is None:
        raise ValueError("Dataset DataFrame `df` must be provided to `run_comorbidity_experiment`.")
    X, var_names = get_data(disorders)
    p            = X.shape[1]

    X_tensor = X_tensor = X.clone().detach().float()
    prevalence = X_tensor.mean(0).clamp(1e-3, 1-1e-3)  

    if isinstance(lambda_l1, float): lambda_l1 = [lambda_l1] * p
    if isinstance(lambda_l2, float): lambda_l2 = [lambda_l2] * p

    model = DirectedInfluenceIsingModel(
            num_variables               = p,
            lambda_l1                   = lambda_l1,
            lambda_l2                   = lambda_l2,
            normalise_rows              = normalise_rows,
            normalise_during_training   = normalise_during_training,
            use_l2                      = use_l2,
            **extra_kwargs
        )
    if auto_lambda:
            model.set_prevalence(prevalence, scale=1.0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
            opt.zero_grad()
            loss = model(X)
            loss.backward()
            opt.step()

    model.estimate_thresholds_from_data(X)
    W_eff = model.raw_weights.detach().cpu().numpy()
    threshold = compute_relative_threshold(W_eff, fraction=threshold_fraction)
    info = model.get_network_model(threshold_value=threshold)

    W, h = info["weights"], info["thresholds"]
    labels = get_variable_labels(disorders)

    G = nx.DiGraph()
    for i, tgt in enumerate(var_names):
        G.add_node(tgt, threshold=h[i])
        for j, src in enumerate(var_names):
            if abs(W[i, j]) > threshold[i, j]: 
                G.add_edge(src, tgt, weight=W[i, j])

    summary_df = run_model_diagnostics(
                G,                         
                info,                 
                var_names,       
                threshold=threshold  
                )

    print("\n--- Edge Summary ---")
    for u, v, d in G.edges(data=True):
        print(f"{labels.get(u,u)} → {labels.get(v,v)} | w = {d['weight']:.3f}")

    if h is not None:
        print("\n--- Node Thresholds (h) ---")
        for node, data in G.nodes(data=True):
            print(f"{labels.get(node,node)}: h = {data['threshold']:.3f}")

    return G, info, summary_df, labels, model

# Bootstrap -----------------------------------------------------------------------------------
def bootstrap_stability(
    disorders: str | list[str],
    model_class,                     
    *,
    num_bootstraps: int = 100,
    threshold: float = 0.01,
    epochs: int = 500,
    lr: float = 0.01,
    normalise_rows: bool = True,
    normalise_during_training: bool = False,
    use_l2: bool = True,
    auto_lambda: bool = False,
    clip_grad: float | None = 1.0,
    **model_kwargs,
):

    # ------------------------------------------------------------------- data
    X, variable_names = get_data(disorders)
    p = X.shape[1]

    X_tensor = X.clone().detach().float()
    prevalence = X_tensor.mean(0).clamp(1e-3, 1-1e-3)

    for key in ["lambda_l1", "lambda_l2", "h_lambda_l1", "h_lambda_l2"]:
        if isinstance(model_kwargs.get(key), float):
            model_kwargs[key] = torch.tensor([model_kwargs[key]] * p)

    edge_freq   = np.zeros((p, p))
    weight_acc  = np.zeros((p, p))
    th_acc      = np.zeros(p)

    # --------------------------------------------------------------- bootstrap
    for _ in tqdm(range(num_bootstraps), desc=f"Bootstrapping ({disorders})"):
        idx    = torch.randint(0, X.shape[0], (X.shape[0],))
        X_boot = X[idx].clone().detach().float()

        model = model_class(
            num_variables=p,
            normalise_rows=normalise_rows,
            normalise_during_training=normalise_during_training,
            use_l2=use_l2,
            **model_kwargs
        )

        if auto_lambda:
            model.set_prevalence(prevalence, scale=1.0)

        opt = torch.optim.Adam(model.parameters(), lr=lr)

        for _ in range(epochs):
            opt.zero_grad()
            loss = model(X_boot)
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()

        W = model.get_network_model()["weights"]
        W[np.abs(W) < threshold] = 0

        h = model.thresholds.detach().cpu().numpy() if model.use_thresholds else np.zeros(p)

        edge_freq  += (W != 0).astype(int)
        weight_acc += W
        th_acc     += h

    # ---------------------------------------------------------------- return
    edge_freq /= num_bootstraps
    avg_weights = weight_acc / num_bootstraps
    avg_th = th_acc / num_bootstraps
    labels = get_variable_labels(disorders)

    return edge_freq, avg_weights, avg_th, variable_names, labels

def build_stable_graph(edge_freq, avg_weights, disorder, labels, thresholds, freq_threshold=0.7, weight_threshold=0.001):
    _, variable_names = get_data(disorder)
    G_stable = nx.DiGraph()
    n = edge_freq.shape[0]

    for i in range(n):
        label_i = labels.get(variable_names[i], variable_names[i])
        G_stable.add_node(label_i, threshold=thresholds[i])
        for j in range(n):
            if edge_freq[i, j] >= freq_threshold and abs(avg_weights[i, j]) >= weight_threshold:
                label_j = labels.get(variable_names[j], variable_names[j])
                G_stable.add_edge(label_j, label_i,
                                  weight=avg_weights[i, j],
                                  stability=edge_freq[i, j])

    return G_stable


if __name__ == "__main__":
        from pathlib import Path
        DATA_PATH = Path("data") / "symptom_data.tsv"
        df = pd.read_csv(DATA_PATH, sep="\t", low_memory=False)
        disorders = ['mania', 'anxiety', 'phobia', 'depression', 'schizophrenia', 'PTSD']
        G, info, summary_df, labels_map, model = run_comorbidity_experiment(df=df, disorders=disorders,  
                                                                    threshold_fraction=0.0003,
                                                                    lambda_l1=0.01, 
                                                                    lambda_l2=0.0001, 
                                                                    h_lambda_l1   = 0.01,  
                                                                    h_lambda_l2   = 0.00,
                                                                    epochs=500, 
                                                                    auto_lambda=True,
                                                                    normalise_rows=True,
                                                                    use_l2=False,
                                                                    normalise_during_training=True,
                                                                    probabilistic=True)


        # Save model object and network
        model_name = "trained_model_comorbidity.pth"
        torch.save(model, model_name)
        G = convert_attributes_to_python_types(G)

        network_name = "comorbidity.gml"
        nx.write_gml(G, network_name)   


    

