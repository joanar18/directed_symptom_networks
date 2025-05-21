from __future__ import annotations

from typing import Sequence, Optional, Set, Union
from scipy.optimize import minimize_scalar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def signed_row_normalise(W: torch.Tensor) -> torch.Tensor:
    """
    Normalize each row of the input tensor `W` such that the sum of absolute values in each row is 1,
    preserving the sign of the original elements.

    This function performs signed row-wise L1 normalization. A small constant (1e-8) is added to
    the denominator to ensure numerical stability and avoid division by zero.

    Parameters
    ----------
    W : torch.Tensor
        A 2D tensor of shape (n_rows, n_columns) to normalize.

    Returns
    -------
    torch.Tensor
        A tensor of the same shape as `W` where each row is L1-normalized (signed).
    """
    row_sum = torch.sum(torch.abs(W), dim=1, keepdim=True) + 1e-8
    return W / row_sum

def signed_softmax_row(W: torch.Tensor, tau: float = 3.0) -> torch.Tensor:
    # NOT USED NOW, it's mostly a legacy of previous model versions
    """
    Applies a signed softmax-like transformation to each row of the input tensor.

    This function maintains the sign of each weight while scaling the magnitude
    via a softmax-like transformation that sharpens or smooths the distribution
    depending on the temperature parameter `tau`.

    Parameters
    ----------
    W : torch.Tensor
        A 2D tensor of shape (n, m) containing real-valued weights.
    tau : float, optional
        Temperature parameter controlling the softness of the softmax. 
        Higher values produce a smoother distribution (default is 3.0).

    Returns
    -------
    torch.Tensor
        A tensor of the same shape as `W`, where each row has been normalized
        such that the magnitudes sum to 1 while preserving the original signs.
    """
    sign = torch.sign(W)
    mag  = torch.exp(torch.abs(W) / tau)
    return sign * mag / (mag.sum(dim=1, keepdim=True) + 1e-8)

def _to_tensor(x: Union[float, int, Sequence[float]]) -> torch.Tensor:
    """
    Converts a float, int, or a sequence of floats into a 1D PyTorch tensor.

    If a single float or int is provided, it is wrapped in a tensor of shape (1,).
    If a sequence is provided, it is converted into a tensor with dtype float32.

    Parameters
    ----------
    x : Union[float, int, Sequence[float]]
        A single numeric value or a sequence of numeric values to convert.

    Returns
    -------
    torch.Tensor
        A 1D tensor containing the input values.
    """
    return torch.tensor([float(x)]) if isinstance(x, (int, float)) else \
           torch.tensor(list(x), dtype=torch.float32)


class DirectedInfluenceIsingModel(nn.Module):
    """
    Directed Influence Ising Model with optional row normalization and per-node L1/L2 regularization.
    Adds L1/L2 per-node penalties to thresholds h, while preserving the row-wise penalties already applied to W.
    
    Parameters:
    weight penalties  :  lambda_l1 (vector length p or scalar): row-wise
                                 lambda_l2 (vector): row-wise
    threshold penalties: h_lambda_l1 (vector or scalar): element-wise
                                 h_lambda_l2 (vector or scalar): element-wise
    
    """
    
    def __init__(self, num_variables: int, *, lambda_l1=0.01, lambda_l2=0.0, h_lambda_l1=0.0,  h_lambda_l2=0.0, 
        use_thresholds=True, use_l2=True, normalise_rows=True, normalise_during_training=True, probabilistic=True):

        super().__init__()
        self.p = num_variables
        self.use_thresholds = use_thresholds
        self.normalise_rows = bool(normalise_rows)
        self.normalise_during_training   = bool(normalise_during_training) and self.normalise_rows
        self.use_l2 = bool(use_l2)
        self.probabilistic = probabilistic

        # broadcast to tensors 
        self.lambda_l1 = _to_tensor(lambda_l1)
        if self.lambda_l1.ndim == 0:
            self.lambda_l1 = self.lambda_l1.repeat(self.p)

        self.lambda_l2 = _to_tensor(lambda_l2)
        if self.lambda_l2.ndim == 0:
            self.lambda_l2 = self.lambda_l2.repeat(self.p)

        # penalties for thresholds 
        self.h_lambda_l1 = _to_tensor(h_lambda_l1)
        if self.h_lambda_l1.ndim == 0:
            self.h_lambda_l1 = self.h_lambda_l1.repeat(self.p)

        self.h_lambda_l2 = _to_tensor(h_lambda_l2)
        if self.h_lambda_l2.ndim == 0:
            self.h_lambda_l2 = self.h_lambda_l2.repeat(self.p)


        # parameters 
        self.raw_weights = nn.Parameter(torch.randn(self.p, self.p) * 0.1)
        self.thresholds  = nn.Parameter(torch.zeros(self.p))

    # internals
    def _W_eff(self) -> torch.Tensor:
        """
        Compute the effective weight matrix used by the model.

        Applies row-wise signed normalization to the raw weights if both
        `normalise_rows` and `normalise_during_training` are True. Otherwise,
        returns the raw weight matrix.

        Returns
        -------
        torch.Tensor
            The effective weight matrix, optionally normalized.
        """
        if self.normalise_rows and self.normalise_during_training:
            return signed_row_normalise(self.raw_weights)
        return self.raw_weights
    
    def _threshold_weights(self, threshold_value: float | torch.Tensor) -> None:
        """
        Apply thresholding to the raw weight matrix in-place.

        Sets elements in `self.raw_weights` to zero if their absolute value is below
        the given threshold. Supports both scalar and tensor thresholds.

        Parameters
        ----------
        threshold_value : float or torch.Tensor
            Threshold value to apply. If a float, applies the same threshold to all weights.
            If a tensor, applies element-wise thresholds (must be broadcastable to the shape of `raw_weights`).

        Returns
        -------
        None
        """
        with torch.no_grad():
            if isinstance(threshold_value, (float, int)):
                mask = torch.abs(self.raw_weights) < threshold_value
            else:
                threshold_tensor = threshold_value
                if not isinstance(threshold_tensor, torch.Tensor):
                    threshold_tensor = torch.tensor(threshold_tensor, device=self.raw_weights.device)
                mask = torch.abs(self.raw_weights) < threshold_tensor
            self.raw_weights[mask] = 0.0

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss.
        BCE + L1‖W‖(row-wise) + L2‖W‖^2(row-wise) + L1‖h‖(element-wise) + L2‖h‖^2(element-wise)

        Args:
            X_cat (torch.Tensor): Input categorical data tensor.

        Returns:
            torch.Tensor: Computed loss value.
        """
        W = self._W_eff()
        logits = X @ W.T
        logits = logits + self.thresholds

        loss = F.binary_cross_entropy_with_logits(logits, X, reduction="none").mean()

        l1_W = torch.sum(torch.abs(W) * self.lambda_l1[:, None])
        l2_W = torch.sum(W.pow(2)     * self.lambda_l2[:, None]) if self.use_l2 else 0.

        if self.use_thresholds is not None:
            l1_h = torch.sum(torch.abs(self.thresholds) * self.h_lambda_l1)
            l2_h = torch.sum(self.thresholds.pow(2) * self.h_lambda_l2) if self.use_l2 else 0.
        else:
            l1_h, l2_h = 0, 0.

        return loss + l1_W + l2_W + l1_h + l2_h
  
    def get_network_model(self, 
                          threshold_value: Union[float, torch.Tensor, None] = None) -> dict[str, np.ndarray]:
        """
        Retrieve the trained model's weight matrix and node thresholds.

        Optionally applies thresholding to zero out weights below a specified magnitude.

        Parameters
        ----------
        threshold_value : float, torch.Tensor, or None, optional
            A threshold below which weights are set to zero. If None, no thresholding is applied.
            Can be a scalar (applied uniformly) or a tensor of same shape as the weight matrix.

        Returns
        -------
        dict[str, numpy.ndarray]
            A dictionary with two keys:
            - "weights": The (optionally thresholded) weight matrix as a NumPy array.
            - "thresholds": The node thresholds as a NumPy array.
        """
        with torch.no_grad():
            W_raw_copy = self.raw_weights.clone()

            if threshold_value is not None:
                if isinstance(threshold_value, (float, int)):
                    mask = torch.abs(W_raw_copy) < threshold_value
                else:
                    threshold_tensor = threshold_value
                    if not isinstance(threshold_tensor, torch.Tensor):
                        threshold_tensor = torch.tensor(threshold_tensor, device=self.raw_weights.device)
                    mask = torch.abs(W_raw_copy) < threshold_tensor
                W_raw_copy[mask] = 0.0

            W_out = signed_row_normalise(W_raw_copy) \
                if self.normalise_rows and self.normalise_during_training \
                else W_raw_copy

            h_out = self.thresholds.detach()

            return {
                "weights": W_out.detach().cpu().numpy(),
                "thresholds": h_out.cpu().numpy(),
            }
    
    def set_prevalence(self, 
                       p_vec: Sequence[float], 
                       *, 
                       scale: float = 1.0) -> None:
        """
        Rescale the row-wise L1 penalties so that symptoms with lower prevalence receive larger penalties,
        and more common symptoms receive smaller penalties.

        Parameters
        ----------
        p_vec : Sequence[float]
            Prevalence vector indicating the frequency of each symptom.
        scale : float, optional
            A scaling factor to multiply the adjusted L1 penalties, by default 1.0.

        Returns
        -------
        None
        """
        p = p_vec.clone().detach().to(dtype=self.lambda_l1.dtype, device=self.lambda_l1.device)
        self.lambda_l1 = scale * p

    def estimate_thresholds_from_data(self, 
                                      X: torch.Tensor,
                                      *, 
                                      bounds: tuple[float, ...] = (-10, 10), 
                                      l2_th: float=1e-2, 
                                      max_iter: int=500
                                      ) -> None:
        """
        Estimate model thresholds by minimizing binary cross-entropy loss with L2 regularization.

        This method updates the `self.thresholds` tensor by optimizing threshold values for each variable
        in the model. It uses the input data `X` and finds thresholds that minimize a regularized binary
        cross-entropy loss between the model prediction and the actual data.

        Parameters
        ----------
        X : torch.Tensor
            Input data tensor of shape (n_samples, p), where p is the number of variables.

        bounds : tuple of float, optional
            The bounds within which to search for each threshold value during optimization.
            Default is (-10, 10).

        l2_th : float, optional
            L2 regularization coefficient for the threshold values to prevent overfitting.
            Default is 1e-2.

        max_iter : int, optional
            Maximum number of iterations allowed in the scalar minimization procedure.
            Default is 500.

        Returns
        -------
        None
            The method updates `self.thresholds` in-place.
        """
        if self.thresholds is None: return
        W = signed_softmax_row(self.raw_weights) if self.normalise_rows and not self.normalise_during_training else self._W_eff()
        new_h = torch.zeros_like(self.thresholds.data)
        for i in range(self.p):
            z, y = X @ W[i], X[:, i]
            def obj(h):
                bce = F.binary_cross_entropy(torch.sigmoid(z + h), y).item()
                return bce + l2_th * h * h
            res = minimize_scalar(obj, bounds=bounds, method="bounded",
                                  options={"maxiter": max_iter})
            new_h[i] = res.x
        self.thresholds.data = new_h

    def probabilistic_inference(self, 
                                X: torch.Tensor, 
                                target: Optional[int] = None, 
                                evidence: Set[int] = set(), 
                                num_iterations: int = 1000, 
                                burn_in: int = 500
                                ) -> Union[float, torch.Tensor]:
        """
        Perform asynchronous Gibbs-like probabilistic inference.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, num_variables).
        target : int or None, optional
            Index of the target variable to infer. If None, inference is performed for all variables.
            Default is None.
        evidence : set of int, optional
            Set of variable indices to keep fixed during sampling (evidence variables).
            Default is empty set.
        num_iterations : int, optional
            Total number of sampling iterations to perform.
            Default is 1000.
        burn_in : int, optional
            Number of initial iterations to discard to allow the sampler to converge.
            Default is 500.

        Returns
        -------
        float or torch.Tensor
            If `target` is specified, returns the marginal probability for that variable as a float.
            Otherwise, returns a tensor of marginals for all variables.
        """
        if not self.probabilistic:
            raise ValueError("Enable probabilistic inference by setting probabilistic = True")

        samples = X.clone()
        weights = signed_row_normalise(self.raw_weights.detach()).to(X.device)
        bias = self.thresholds if self.use_thresholds else 0.0

        batch_size, d = samples.shape
        accumulator = torch.zeros_like(samples)

        for step in range(num_iterations):
            indices = torch.randperm(d)
            for i in torch.randperm(d).tolist():
                if i not in evidence:
                    logits = torch.sum(samples * weights[i], dim=1)
                    if self.use_thresholds:
                        logits += bias[i]
                    prob = torch.sigmoid(logits)
                    samples[:, i] = (torch.rand(batch_size, device=X.device) < prob).float()
            if step >= burn_in:
                accumulator += samples

        marginals = accumulator / (num_iterations - burn_in)

        if target is not None:
            return marginals[:, target].mean().item()
        else:
            return marginals.mean(dim=0)

