import pandas as pd
import torch

from GraphicalModels.core.models import DirectedInfluenceIsingModel
from GraphicalModels.experiments.experiment import get_data

def run_symptom_cascade_simulations(model, symptom_names, num_simulations=100, burn_in=500):
    """
    For each symptom, simulate probabilistic cascades and record marginals.

    Args:
        model: trained model with .probabilistic_inference(...)
        symptom_names: list of symptom names (ordered like model variables)
        num_simulations: number of Monte Carlo samples per symptom
        burn_in: steps to discard before averaging

    Returns:
        DataFrame: rows = symptoms, cols = average activation of other symptoms
    """
    d = len(symptom_names)
    all_marginals = [] 

    device = next(model.parameters()).device 

    for i in range(d):
        X_init = torch.zeros(num_simulations, d, device=device)
        X_init[:, i] = 1.0

        marginals = model.probabilistic_inference(X_init, target=None, evidence={i}, num_iterations=burn_in + 500, burn_in=burn_in)

        all_marginals.append(marginals.numpy())
 
    cascade_df = pd.DataFrame(all_marginals, columns=symptom_names, index=symptom_names)
    return cascade_df


if __name__ == '__main__':

    disorders = ['mania', 'anxiety', 'phobia', 'depression', 'schizophrenia', 'PTSD']

    for dis in disorders:
        X, var_names = get_data(dis)
        p = X.shape[1]

        model = DirectedInfluenceIsingModel(
            num_variables=p,
            normalise_rows=True,
            normalise_during_training=True,
            probabilistic=True
        )

        model_name = 'trained_model_' + dis + '.pth'
        model = torch.load(model_name)
        model.eval()

        cascades = run_symptom_cascade_simulations(model, var_names)

        name = dis + '_cascades.csv'
        cascades.to_csv(name)
        print('Cascade for', dis, 'saved')
        