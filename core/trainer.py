import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from GraphicalModels.core.models import DirectedInfluenceIsingModel

class GraphicalModels:
    """
    Wrapper class for various graphical models including MGM, IsingFit, IsingFitWithPC, and DirectedIsingProbabilistic.

    This class simplifies model selection, initialization, training, and retrieval of learned structures.
    It supports both deterministic and probabilistic variants and can run on CPU, CUDA, or MPS (Apple Silicon).

    Parameters:
    ----------
    model_type : str
        Type of graphical model to initialize. Options: "MGM", "Ising", "IsingPC", or "DirectedIsingProb".
    
    num_variables : int
        Number of nodes/variables in the graphical model.
    
    data : pd.DataFrame, optional
        Input binary data required for models like IsingFitWithPC which rely on PC Algorithm-based structure learning.
    
    lasso_lambda : float, optional (default=0.01)
        L1 regularization parameter used for weight sparsity.
    
    probabilistic : bool, optional (default=True)
        If True, enables probabilistic inference in models that support it.
    
    device : str, optional (default='gpu')
        The computation device. Options: 'gpu' (uses CUDA or MPS if available) or 'cpu'.
    
    Raises:
    ------
    ValueError:
        If `model_type` is not recognized or data is missing for IsingPC model.
    """
    
    def __init__(self, model_type, num_variables, data=None, lasso_lambda=0.1, lambda_l2=0.01, probabilistic=True, device="gpu", normalize=False):
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
           

        if model_type == "DirectedInfluenceIsingModel":
             self.model = model = DirectedInfluenceIsingModel(num_variables=num_variables,
                                                              lambda_l1=[lasso_lambda] * num_variables, 
                                                              lambda_l2=[lambda_l2] * num_variables,
                                                              normalize=self.normalize).to(self.device)
        else:
            raise ValueError("Invalid model type. Choose 'DirectedInfluenceIsingModel'.")

    def train_model(self, X, lr=0.01, epochs=500, batch_size=32, clip_value=1.0):
        """
    Trains the selected graphical model using Adam optimizer with mini-batch gradient descent and learning rate scheduling.

    Parameters:
    ----------
    X : torch.Tensor
        Input data tensor of shape (n_samples, n_features). Should contain binary values for Ising-based models.
    
    lr : float, optional (default=0.01)
        Initial learning rate for the optimizer.
    
    epochs : int, optional (default=500)
        Number of training epochs.
    
    batch_size : int, optional (default=32)
        Size of each mini-batch for training.
    
    clip_value : float, optional (default=1.0)
        Maximum allowed gradient norm for clipping to avoid exploding gradients.

    Notes:
    -----
    - Uses `ReduceLROnPlateau` scheduler to adapt the learning rate when the loss plateaus.
    - Prints training progress every 50 epochs.
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
