import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Sequence, Union
import networkx as nx


VARIABLE_DEFINITIONS = {
    "mania": {
        "V312": "Elevated mood", 
        "V313": "Irritable mood", 
        "V1608": "Hyperactive Episode", 
        "V1610": "Restless Agitation", 
        "V1611": "Impulsive Spending", 
        "V1612": "Hypersexual Behavior", 
        "V1613": "Pressured speech", 
        "V1614": "Flight of ideas", 
        "V1615": "Grandiosity", 
        "V1617": "Decreased sleep", 
        "V1618": "Distractibility", 
        "V3125": "Increased activity",
        "V3126": "Risk-taking behavior",
    },
    "depression": {
        "V1003": "Frequent Crying",
        "V1004": "Hopelessness",
        "V1005": "Inability to cope",
        "V1006": "Pessimism",
        "V1102": "Loss of Apetite",
        "V1103": "Unintentional Weight Loss",
        "V1105": "Increased Apetite",
        "V1106": "Unintentional Weight Gain",
        "V1108": "Insomnia",
        "V1114": "Excessive Sleeping",
        "V1115": "Chronic fatigue",
        "V1117": "Slowed Speech & Movement",
        "V1124": "Decreased Sex Drive",
        "V1126": "Worthlessness",
        "V1129": "Guilt",
        "V1130": "Feelings of inferiority",
        "V1134": "Loss of interest",
        "V1137": "Indecisiveness",
        "V1142": "Suicidal Thoughts",
        "V1144": "Suicidal Attempt",
    },
    "anxiety": {
        "V302": "Prolonged Generalized Anxiety",
        "V311": "Loss of interest",
        "V823": "Sleep Disturbance", 
        "V803": "Excessive Worry",
        "V810": "Restlessness",
        "V811": "Muscle Tension",
        "V813": "Irritable",
        "V816": "Fatigue",
        "V820": "Difficulty Concentrating",     
        "V823": "Sleep Disturbance"
    },
    "panic":{
        "V301": "Panic Attack",
        "V809": "Trembling",
        "V814": "Heart Palpitations",
        "V815": "Shortness of Breath",
        "V821": "Hot Flashes",
        "V818": "Dry Mouth",
        "V819": "Nausea",
    },
    "phobia": {
        "V321": "Physical anxiety symptoms",
        "V322": "Panic symptoms",
        "V324": "Embarrassed",
        "V337": "Avoidance",
        "V610": "Shortness of Breath",
        "V611": "Heart Palpitations",
        "V613": "Chest & Stomach Discomfort",
        "V614": "Numbness",
        "V615": "Choking Sensation",
        "V616": "Feeling faint",
        "V618": "Trembling",
        "V619": "Hot Flashes",
        "V620": "Derealization",
    },
    "schizophrenia": {
        "V4101": "Paranoid Delusions",
        "V4103": "Persecutory Delusions",
        "V4105": "Delusions of Thought Broadcasting",
        "V4108": "Thought Broadcasting/Insertion",
        "V4112" : "Delusions of Control",
        "V4114" :"Thought Insertion",
        "V4116" : "Referential Delusions",
        "V4118" : "Hypnotic Influence Suspicions",
        "V4120" : "Visual Hallucinations",
        "V4122": "Auditory Hallucinations",
        "V4133": "Olfactory Hallucination", 
        "V4135": "Tactile Hallucination",
        "V4317": "Acting Unusual",
        "V1614": "Flight of Ideas"
    },
    "PTSD": {
        "V6217": "Intrusive Memories",
        "V6218": "Nightmares",
        "V6219": "Flashbacks",
        "V6220": "Trigger Distress",
        "V6222": "Emotional Numbing",
        "V6223": "Avoidant Behavior",
        "V6224": "Thought Suppression",
        "V6225": "Memory Gaps",
        "V6226": "Social Withdrawal",
        "V6227": "Hopeless Outlook",
        "V6228": "Loss of Interest",
        "V6231": "Difficulty Concentrating",
        "V6232": "Irritability",
        "V6233": "Insomnia",
        "V6234": "Excessive Caution",
        "V6235": "Easily Startled",
        "V6236": "Physical Reaction",
        "V808": "Easily Startled",
    }
}

DISORDER_VARIABLES = {disorder: list(symptoms.keys()) for disorder, symptoms in VARIABLE_DEFINITIONS.items()}

def load_disorder_subset(df: pd.DataFrame, 
                         disorder: list[str] | str, 
                         binary: bool = True, 
                         missing_codes: list[int]=[8, 9]
                         ) -> pd.DataFrame:
    """
    Extracts and processes a subset of the DataFrame for one or more specified disorders, 
    replacing missing values, and optionally binarizing the values. The disorder parameter 
    can either be a single disorder or a list of disorders.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing disorder-related data.
    disorder (list[str] | str): Disorder name(s) to extract from `disorder_variables`. Can be a single disorder (str) or a list of disorders.
    binary (bool, optional): If True, converts nonzero values to 1 (default: True).
    missing_codes (list[int], optional): Values to replace with NaN (default: [8, 9]).

    Returns:
    pd.DataFrame: DataFrame containing the extracted and processed subset of variables related to the disorders.

    Raises:
    ValueError: If any disorder in the list is not found in `disorder_variables`.
    """
    if disorder not in DISORDER_VARIABLES:
        raise ValueError(f"Unknown disorder: {disorder}")

    variables = DISORDER_VARIABLES[disorder]
    df_subset = df[variables].copy()
    df_subset.replace(missing_codes, np.nan, inplace=True)

    if binary:
        df_subset = df_subset.apply(lambda col: np.where(col > 0, 1, 0))

    return df_subset

def load_multiple_disorders(df: pd.DataFrame, 
                            disorders: list[str], 
                            binary: bool = True, 
                            missing_codes: list[int]=[8, 9]) -> pd.DataFrame:                         
    """
    Load and concatenate data subsets for multiple disorders.

    This function iterates over a list of disorder names and uses the 
    `load_disorder_subset` function to extract and optionally binarize 
    each subset, handling specified missing value codes. The results are 
    concatenated along the column axis into a single DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the disorder data.
    
    disorders : list of str
        A list of disorder column names to extract from `df`.

    binary : bool, optional
        If True, convert disorder values into binary format. Default is True.
    
    missing_codes : list, optional
        A list of values to be treated as missing and excluded. Default is [8, 9].

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the concatenated disorder subsets.
    """
    combined = []
    for disorder in disorders:
        subset = load_disorder_subset(df, disorder, binary, missing_codes)
        combined.append(subset)
    return pd.concat(combined, axis=1)

def to_tensor(df_binary: pd.DataFrame) -> torch.Tensor:
    """
    Converts a Pandas DataFrame into a PyTorch tensor of type `float32`.

    Parameters:
    df_binary (pd.DataFrame): A Pandas DataFrame containing binary data (0s and 1s).

    Returns:
    torch.Tensor: A PyTorch tensor containing the same data as the input DataFrame, with dtype `float32`.
    """
    return torch.tensor(df_binary.values, dtype=torch.float32)

def get_variable_labels(disorders: str | list[str]) -> dict:
    """
    Retrieve label definitions for one or more disorders.

    This function fetches and combines label mappings from a predefined dictionary
    (`VARIABLE_DEFINITIONS`) for the specified disorder(s). If a disorder is not
    found in the dictionary, it is skipped.

    Parameters
    ----------
    disorders : str or list of str
        A disorder name or a list of disorder names for which to retrieve label definitions.

    Returns
    -------
    dict
        A dictionary containing the combined label definitions for the given disorders.
        If none of the disorders are found, an empty dictionary is returned.
    """
    labels_dict = VARIABLE_DEFINITIONS

    if isinstance(disorders, str):
        disorders = [disorders]

    labels = {}
    for d in disorders:
        if d in labels_dict:
            labels.update(labels_dict[d])

    return labels

def get_all_variable_names():
    """
    Retrieves a list of all unique variable names across all disorders in the `disorder_variables` dictionary.
    
    Returns:
    list: A list of unique variable names associated with disorders.
    """
    return list(set(var for sublist in DISORDER_VARIABLES.values() for var in sublist))

def float_or_seq_to_tensor(x: Union[Sequence[float], float]) -> torch.Tensor:
    """
    Convert a float or a sequence of floats to a 1D PyTorch tensor.

    If the input is a single float or int, it is wrapped into a tensor of shape (1,).
    If the input is a sequence (e.g., list or tuple) of floats, it is converted into 
    a tensor of type `torch.float32`.

    Parameters
    ----------
    x : float or Sequence[float]
        A single float/int or a sequence of floats to be converted into a tensor.

    Returns
    -------
    torch.Tensor
        A 1-dimensional PyTorch tensor containing the input value(s).
    """
    if isinstance(x, (int, float)):
        return torch.tensor([float(x)])  
    return torch.tensor(list(x), dtype=torch.float32)


def run_model_diagnostics(G: nx.DiGraph, 
                          model_info: dict, 
                          variable_names:list[str], 
                          threshold: float =0.1) -> pd.DataFrame:
    """
    Run diagnostics on a graph-based model, summarizing node and edge properties.

    This function analyzes a directed graph (typically representing a network model)
    along with associated model parameters such as weights and node thresholds.
    It computes and displays metrics like node degrees, edge density, and threshold
    statistics. A summary table is returned and a histogram of the node thresholds
    is displayed using Matplotlib.

    Parameters
    ----------
    G : networkx.DiGraph
        A directed graph representing the model, where edges typically correspond
        to non-zero weights between variables.

    model_info : dict
        A dictionary containing model parameters. Expected keys:
            - 'weights' : numpy.ndarray or similar
                Weight matrix between variables.
            - 'thresholds' : numpy.ndarray
                Threshold (bias) values for each variable.

    variable_names : list of str
        List of variable (node) names corresponding to the model dimensions.

    threshold : float, optional
        Threshold used for reporting edge significance in the printout. Default is 0.1.

    Returns
    -------
    pandas.DataFrame
        A DataFrame summarizing each variable's threshold, in-degree, and out-degree.
    """
    weights = model_info['weights']
    thresholds = model_info['thresholds']

    num_vars = len(variable_names)
    num_edges = G.number_of_edges()
    max_edges = num_vars * (num_vars - 1)
    edge_density = num_edges / max_edges

    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    summary = []
    for i, var in enumerate(variable_names):
        summary.append({
            'Variable': var,
            'Threshold': thresholds[i],
            'InDegree': in_degrees.get(var, 0),
            'OutDegree': out_degrees.get(var, 0),
        })

    summary_df = pd.DataFrame(summary)

    print('\n--- MODEL DIAGNOSTICS ---')
    print(f'Number of nodes: {num_vars}')
    print(f'Number of edges (|W_ij| > {threshold}): {num_edges}')
    print(f'Edge density: {edge_density:.3f}')
    print(f'Average threshold (bias): {thresholds.mean():.3f}')
    print(f'Thresholds range: {thresholds.min():.3f} to {thresholds.max():.3f}')
    print('Top 5 most influential symptoms (by out-degree):')
    print(summary_df.sort_values('OutDegree', ascending=False).head(5)[['Variable', 'OutDegree']])

    # Matplotlib version
    plt.figure(figsize=(8, 4))
    plt.hist(thresholds, bins='sturges', color='steelblue', edgecolor='black')
    plt.title('Distribution of Node Thresholds (h)')
    plt.xlabel('Threshold')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return summary_df

