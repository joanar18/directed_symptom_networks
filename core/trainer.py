import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from directed_symptom_networks.core.models import DirectedInfluenceIsingModel

class Trainer:
    """
    Wrapper class for the Directed Influence Ising Model.

    This class simplifies model selection, initialization, training, and retrieval of learned structures.
    It supports both deterministic and probabilistic variants and can run on CPU, CUDA, or MPS (Apple Silicon).

    Parameters
    ----------
    num_variables : int
        Number of nodes/variables in the graphical model.

    lasso_lambda : float, optional, default=0.1
        L1 regularization strength for sparsity in weights.

    lambda_l2 : float, optional, default=0.01
        L2 regularization strength for weight shrinkage.

    device : str, optional, default='gpu'
        The computation device. Options: 'gpu' (uses CUDA or MPS if available) or 'cpu'.
    """
    
    def __init__(self, 
                 num_variables: int, 
                 lasso_lambda: float = 0.1, 
                 lambda_l2: float = 0.01, 
                 device: str = "gpu"):
        if device == 'gpu':
            try:
                if torch.cuda.is_available():                                                                  
                    self.device = torch.device('cuda')
                elif torch.backends.mps.is_available():                                         
                    self.device = torch.device('mps')
                else:
                    raise RuntimeError("GPU is not available. Falling back to CPU.")
            except Exception as e:
                print(f"Warning: {e}")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
           
        self.model = DirectedInfluenceIsingModel(num_variables=num_variables,
                                                              lambda_l1=[lasso_lambda] * num_variables, 
                                                              lambda_l2=[lambda_l2] * num_variables,
                                                              normalize=self.normalize).to(self.device)

    def train_model(
            self,
            X: torch.Tensor,
            lr: float = 0.01,
            epochs: int = 500,
            batch_size: int = 32,
            clip_value: float = 1.0) -> None:
        """
        Train the model using the provided input data.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (n_samples, n_features).
        lr : float, optional
            Learning rate for the Adam optimizer (default is 0.01).
        epochs : int, optional
            Number of training epochs (default is 500).
        batch_size : int, optional
            Size of each training batch (default is 32).
        clip_value : float, optional
            Maximum allowed norm for gradient clipping to prevent exploding gradients (default is 1.0).

        Returns
        -------
        None
        """
        X = X.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0

            for batch in dataloader:
                batch_data = batch[0]
                optimizer.zero_grad()
                loss = self.model(batch_data)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Avg Loss = {avg_loss:.7f}")

    def get_network_model(self):
        """
        Retrieves the learned model parameters including weights and (optionally) thresholds.

        Returns:
        -------
        dict :
        A dictionary containing:
            - 'weights': numpy array of learned interaction weights.
            - 'thresholds': numpy array of activation thresholds, if used by the model.
        """
        return self.model.get_network_model()
