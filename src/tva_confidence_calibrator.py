# title: TVA Confidence Calibrator
# AAI-590 Group 9
# Multi-Class Prediction Confidence Calibration Module Based on Top-Versus-All Method
# Reference: https://arxiv.org/pdf/2411.02988v2
# Base Code: Perplexity (July 2025)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.isotonic import IsotonicRegression
from collections import defaultdict
import warnings
from scipy.special import logsumexp, expit

class TvACalibrator:
    """
    Top-versus-All (TvA) calibration method for multiclass classifiers.
    
    This class implements the TvA approach that reformulates multiclass 
    calibration as a single binary classification problem.
    """
    def __init__(self, method='histogram_binning', n_bins=15, lambda_reg=0.01):
        
        # Initialize TvA calibrator.
        """
        Args:
            method (str): Calibration method ('temperature_scaling', 'vector_scaling', 
                         'histogram_binning', 'isotonic_regression', 'beta_calibration')
            n_bins (int): Number of bins for histogram binning
            lambda_reg (float): Regularization parameter for Vector Scaling/Dirichlet
        """
        self.method = method
        self.n_bins = n_bins
        self.lambda_reg = lambda_reg
        self.calibrator = None
        self.is_fitted = False

    def _tva_max_confidences(self, logits):
        # logits: shape (num_samples, num_classes)
        top_idx = np.argmax(logits, axis=1)
        top_logits = logits[np.arange(logits.shape[0]), top_idx]
    #    
        # log-sum-exp over all *other* classes
        mask = np.ones_like(logits, dtype=bool)
        mask[np.arange(logits.shape[0]), top_idx] = False
        rest_logits = np.where(mask, logits, -np.inf)
        rest_logsumexp = logsumexp(rest_logits, axis=1)
        
        tva_logit = top_logits - rest_logsumexp  # shape (num_samples,)
        tva_confidence = expit(tva_logit)  # turn it into probability (0, 1)
        return tva_confidence, top_idx
        
    def _preprocess_for_tva(self, logits, labels):
        """
        Preprocess multiclass data for TvA approach.
        
        Args:
            logits (torch.Tensor): Raw model outputs, shape (N, C)
            labels (torch.Tensor): True labels, shape (N,)
            
        Returns:
            tuple: (confidence_scores, binary_correctness_labels)
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get confidence scores (max probabilities) and predictions
        confidence_scores, predictions = torch.max(probs, dim=1)
        
        # Create binary correctness labels: 1 if correct, 0 if incorrect
        binary_labels = (predictions == labels).float()

        # get TVA scores from max logits
        tva_scores, _ = self._tva_max_confidences(logits)

        # store confidence and tva probs scores

        
        return confidence_scores, binary_labels, predictions, tva_scores
    

    
    def fit(self, logits, labels):
        """
        Fit the TvA calibrator using validation data.
        
        Args:
            logits (torch.Tensor): Raw model outputs from validation set
            labels (torch.Tensor): True labels for validation set
        """
        confidence_scores, binary_labels, predictions, tva_scores = self._preprocess_for_tva(logits, labels)
        
        if self.method == 'temperature_scaling':
            self.calibrator = TvATemperatureScaling()
            print(f"DEBUG: Temperature Scaler {self.calibrator.temperature}")
            self.calibrator.fit(logits, binary_labels)
            
        elif self.method == 'vector_scaling':
            self.calibrator = TvAVectorScaling(num_classes=logits.size(1), lambda_reg=self.lambda_reg)
            self.calibrator.fit(logits, binary_labels)
            
        elif self.method == 'histogram_binning':
            self.calibrator = TvAHistogramBinning(n_bins=self.n_bins)
            #self.calibrator.fit(confidence_scores.cpu().numpy(), binary_labels.cpu().numpy())
            self.calibrator.fit(tva_scores.cpu().numpy(), binary_labels.cpu().numpy())
        
       
            
        elif self.method == 'isotonic_regression':
            self.calibrator = TvAIsotonicRegression()
            self.calibrator.fit(confidence_scores.cpu().numpy(), binary_labels.cpu().numpy())
            
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")
            
        self.is_fitted = True
        return self
    
    def identify_uncertain_samples(self, confidences: torch.Tensor, 
                                 confidence_threshold: float = 0.8) -> np.ndarray:
        """Identify uncertain samples based on confidence threshold"""
        #max_probs = torch.max(confidences, dim=1)[0]
        uncertain_mask = confidences < confidence_threshold
        return uncertain_mask.cpu().numpy()
    
    def transform(self, logits):
        """
        Apply TvA calibration to get calibrated confidence scores.
        
        Args:
            logits (torch.Tensor): Raw model outputs
            
        Returns:
            torch.Tensor: Calibrated confidence scores
        """
        if not self.is_fitted:
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=1)
        
            # Get confidence scores (max probabilities) and predictions
            max_softmax_probs, _ = torch.max(probs, dim=1)
            return max_softmax_probs
            #raise ValueError("Calibrator must be fitted before making predictions")
            
        if self.method in ['temperature_scaling', 'vector_scaling']:
            return self.calibrator.transform(logits)
        else:
            # For binary methods, we only need confidence scores
            #probs = F.softmax(logits, dim=1)
            #confidence_scores, _ = torch.max(probs, dim=1)
            confidence_scores,_ = self._tva_max_confidences(logits.cpu().numpy())
            #calibrated_confidences = self.calibrator.transform(confidence_scores.cpu().numpy())
            calibrated_confidences = self.calibrator.transform(confidence_scores)
            return calibrated_confidences

class TvATemperatureScaling(nn.Module):
    """
    TvA Temperature Scaling using binary cross-entropy loss.
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        #self.is_fitted = False
        
    def fit(self, logits, binary_labels, max_iter=50, lr=0.01):
        """
        Fit temperature parameter using binary cross-entropy loss.
        
        Args:
            logits (torch.Tensor): Raw model outputs
            binary_labels (torch.Tensor): Binary correctness labels (0/1)
        """
        # Use LBFGS optimizer for temperature scaling
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            
            # Apply temperature scaling to logits
            scaled_logits = logits / self.temperature
            probs = F.softmax(scaled_logits, dim=1)
            confidence_scores, _ = torch.max(probs, dim=1)
            
            # Binary cross-entropy loss for TvA
            # L_BCE = -(y*log(s) + (1-y)*log(1-s))
            epsilon = 1e-12  # For numerical stability
            confidence_scores = torch.clamp(confidence_scores, epsilon, 1-epsilon)
            
            bce_loss = -(binary_labels * torch.log(confidence_scores) + 
                        (1 - binary_labels) * torch.log(1 - confidence_scores))
            
            loss = bce_loss.mean()
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        #self.is_fitted = True
        
    def transform(self, logits):
        """Apply temperature scaling and return calibrated confidence scores."""
        with torch.no_grad():
            scaled_logits = logits / self.temperature
            probs = F.softmax(scaled_logits, dim=1)
            confidence_scores, predictions = torch.max(probs, dim=1)
            #self.cal
            return [confidence_scores, probs, predictions]

class TvAVectorScaling(nn.Module):
    """
    TvA Vector Scaling with L2 regularization.
    """
    
    def __init__(self, num_classes, lambda_reg=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_reg = lambda_reg
        self.temperature_vector = nn.Parameter(torch.ones(num_classes))
        
    def fit(self, logits, binary_labels, max_iter=100, lr=0.01):
        """Fit vector scaling parameters with regularization."""
        optimizer = torch.optim.LBFGS([self.temperature_vector], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            
            # Apply vector scaling
            scaled_logits = logits / self.temperature_vector.unsqueeze(0)
            probs = F.softmax(scaled_logits, dim=1)
            confidence_scores, _ = torch.max(probs, dim=1)
            
            # Binary cross-entropy loss
            epsilon = 1e-12
            confidence_scores = torch.clamp(confidence_scores, epsilon, 1-epsilon)
            
            bce_loss = -(binary_labels * torch.log(confidence_scores) + 
                        (1 - binary_labels) * torch.log(1 - confidence_scores))
            
            # L2 regularization: penalize deviation from 1
            reg_loss = self.lambda_reg * torch.mean((self.temperature_vector - 1) ** 2)
            
            total_loss = bce_loss.mean() + reg_loss
            total_loss.backward()
            return total_loss
        
        optimizer.step(eval_loss)
        
    def transform(self, logits):
        """Apply vector scaling and return calibrated confidence scores."""
        with torch.no_grad():
            scaled_logits = logits / self.temperature_vector.unsqueeze(0)
            probs = F.softmax(scaled_logits, dim=1)
            confidence_scores, _ = torch.max(probs, dim=1)
            return confidence_scores

#class TvAHistogramBinning:
    """
    TvA Histogram Binning for binary calibration.
    """
    
 #   def __init__(self, n_bins=15, strategy='uniform'):
 #       self.n_bins = n_bins
 #       self.strategy = strategy  # 'uniform' or 'quantile'
 #       self.bin_boundaries = None
 #       self.bin_calibrated_probs = None
        
 #   def fit(self, confidence_scores, binary_labels):
 #       """
 #       Fit histogram binning calibrator.
        
 #       Args:
 #           confidence_scores (np.array): Confidence scores from model
 #           binary_labels (np.array): Binary correctness labels
 #       """
    """
        if self.strategy == 'uniform':
            # Equal-width bins
            print("DEBUG: histogram binning uniform")
            self.bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        elif self.strategy == 'quantile':
            # Equal-frequency bins
            print("DEBUG: histogram binning quantile")
            self.bin_boundaries = np.quantile(confidence_scores, 
                                            np.linspace(0, 1, self.n_bins + 1))
            self.bin_boundaries[0] = 0.0  # Ensure first boundary is 0
            self.bin_boundaries[-1] = 1.0  # Ensure last boundary is 1
        
        self.bin_calibrated_probs = np.zeros(self.n_bins)
        
        # Calculate calibrated probability for each bin
        for i in range(self.n_bins):
            # Find samples in this bin
            lower = self.bin_boundaries[i]
            upper = self.bin_boundaries[i + 1]
            
            if i == self.n_bins - 1:  # Last bin includes upper boundary
                in_bin = (confidence_scores >= lower) & (confidence_scores <= upper)
            else:
                in_bin = (confidence_scores >= lower) & (confidence_scores < upper)
            
            if np.sum(in_bin) > 0:
                # Calibrated probability = accuracy in bin
                self.bin_calibrated_probs[i] = np.mean(binary_labels[in_bin])
            else:
                # If no samples in bin, use bin midpoint as default
                self.bin_calibrated_probs[i] = (lower + upper) / 2
    """

    #def predict_proba(self, confidence_scores):
     #   """Apply histogram binning calibration."""
     #   calibrated_probs = np.zeros_like(confidence_scores)
        
     #   for i in range(len(confidence_scores)):
            # Find which bin this confidence score belongs to
     #       bin_idx = np.digitize(confidence_scores[i], self.bin_boundaries) - 1
     #       bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)  # Ensure valid index
            
     #       calibrated_probs[i] = self.bin_calibrated_probs[bin_idx]
            
     #   return calibrated_probs
    
class TvAIsotonicRegression:
    """
    TvA Isotonic Regression for binary calibration.
    """
    
    def __init__(self):
        self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
        
    def fit(self, confidence_scores, binary_labels):
        """Fit isotonic regression calibrator."""
        self.isotonic_regressor.fit(confidence_scores, binary_labels)
        
    def transform(self, confidence_scores):
        """Apply isotonic regression calibration."""
        return self.isotonic_regressor.predict(confidence_scores)

class TvAHistogramBinning:
    """
    Custom TvA Histogram Binning implementation that works properly
    """
    
    def __init__(self, n_bins=15, equal_mass=True):
        self.n_bins = n_bins
        self.equal_mass = equal_mass
        self.bin_boundaries = None
        self.bin_calibrated_probs = None
        self.fitted = False
        
    def fit(self, max_confidences, correctness_labels):
        """Fit the TvA calibrator"""
        max_confidences = np.array(max_confidences).flatten()
        correctness_labels = np.array(correctness_labels).flatten()
        
        n_samples = len(max_confidences)
        
        if self.equal_mass:
            # Equal-mass binning: each bin has approximately same number of samples
            sorted_indices = np.argsort(max_confidences)
            samples_per_bin = n_samples // self.n_bins
            remainder = n_samples % self.n_bins
            
            self.bin_boundaries = [max_confidences.min()]
            current_idx = 0
            
            for bin_idx in range(self.n_bins - 1):
                bin_size = samples_per_bin + (1 if bin_idx < remainder else 0)
                current_idx += bin_size
                if current_idx < len(sorted_indices):
                    boundary = max_confidences[sorted_indices[current_idx - 1]]
                    self.bin_boundaries.append(boundary)
            
            self.bin_boundaries.append(max_confidences.max())
            self.bin_boundaries = np.array(self.bin_boundaries)
            
            # Handle duplicate boundaries
            for i in range(1, len(self.bin_boundaries)):
                if self.bin_boundaries[i] <= self.bin_boundaries[i-1]:
                    self.bin_boundaries[i] = self.bin_boundaries[i-1] + 1e-8
        else:
            # Equal-width binning
            self.bin_boundaries = np.linspace(max_confidences.min(), 
                                            max_confidences.max(), self.n_bins + 1)
        
        # Update n_bins in case we had to adjust
        self.n_bins = len(self.bin_boundaries) - 1
        
        # Calculate calibrated probability for each bin
        self.bin_calibrated_probs = np.zeros(self.n_bins)
        
        for i in range(self.n_bins):
            if i == self.n_bins - 1:
                mask = (max_confidences >= self.bin_boundaries[i]) & \
                       (max_confidences <= self.bin_boundaries[i + 1])
            else:
                mask = (max_confidences >= self.bin_boundaries[i]) & \
                       (max_confidences < self.bin_boundaries[i + 1])
            
            if np.sum(mask) > 0:
                self.bin_calibrated_probs[i] = np.mean(correctness_labels[mask])
            else:
                # Fallback for empty bins
                self.bin_calibrated_probs[i] = (self.bin_boundaries[i] + self.bin_boundaries[i + 1]) / 2
        
        self.fitted = True
        return self
        
    def transform(self, max_confidences):
        """Apply TvA calibration"""
        if not self.fitted:
            raise ValueError("Calibrator must be fitted before transform")
            
        max_confidences = np.array(max_confidences).flatten()
        calibrated_confidences = np.zeros_like(max_confidences)
        
        for i, conf in enumerate(max_confidences):
            if conf <= self.bin_boundaries[0]:
                bin_idx = 0
            elif conf >= self.bin_boundaries[-1]:
                bin_idx = self.n_bins - 1
            else:
                bin_idx = np.searchsorted(self.bin_boundaries[1:], conf, side='right')
                bin_idx = min(bin_idx, self.n_bins - 1)
            
            calibrated_confidences[i] = self.bin_calibrated_probs[bin_idx]
        
        return calibrated_confidences

# Evaluation utilities for TvA calibration
class TvAEvaluator:
    """
    Evaluation utilities for TvA calibrated models.
    """
    
    @staticmethod
    def expected_calibration_error(confidences, accuracies, n_bins=15):
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            confidences (np.array): Calibrated confidence scores
            accuracies (np.array): Binary correctness (1 if correct, 0 if wrong)
            n_bins (int): Number of bins for ECE calculation
            
        Returns:
            float: Expected Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece
    
    @staticmethod
    def reliability_diagram_data(confidences, accuracies, n_bins=15):
        """
        Generate data for reliability diagram plotting.
        
        Returns:
            tuple: (bin_centers, bin_accuracies, bin_confidences, bin_counts)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_accuracies.append(accuracies[in_bin].mean())
                bin_confidences.append(confidences[in_bin].mean())
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(bin_centers[i])
                bin_counts.append(0)
                
        return bin_centers, np.array(bin_accuracies), np.array(bin_confidences), np.array(bin_counts)


class TvAHistogramCalibrator:
    """
    Top-versus-All Histogram Binning Calibrator
    
    Transforms multi-class calibration into single binary problem:
    "Is the prediction correct?" vs "Is it class X?" for each class.
    """
    
    def __init__(self, n_bins=10, equal_intervals=False):
        self.n_bins = n_bins
        self.equal_intervals = equal_intervals
        self.bin_boundaries = None
        self.bin_values = None
        self.is_fitted = False
    
    def fit(self, pred_logits, true_labels):
        """Fit calibrator using validation data"""
        # Convert logits to probabilities
        #if np.any(pred_logits < 0) or not np.allclose(np.sum(pred_logits, axis=1), 1):
        #exp_logits = np.exp(pred_logits - np.max(pred_logits, axis=1, keepdims=True))
        #probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        #else:
        #    probabilities = pred_logits
        probabilities = F.softmax(pred_logits, dim=1).numpy()
        # TvA Transformation: confidence vs correctness
        max_probs = np.max(probabilities, axis=1)
        pred_classes = np.argmax(probabilities, axis=1)
        is_correct = (pred_classes == true_labels).astype(float)
        
        # Create bins for confidence scores
        if self.equal_intervals:
            self.bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        else:
            self.bin_boundaries = np.quantile(max_probs, np.linspace(0, 1, self.n_bins + 1))
            self.bin_boundaries[0] = 0.0
            self.bin_boundaries[-1] = 1.0
        
        self.bin_boundaries = np.unique(self.bin_boundaries)
        n_actual_bins = len(self.bin_boundaries) - 1
        self.bin_values = np.zeros(n_actual_bins)
        
        # Calculate empirical accuracy for each bin
        for i in range(n_actual_bins):
            if i == n_actual_bins - 1:
                in_bin = (max_probs >= self.bin_boundaries[i]) & \
                        (max_probs <= self.bin_boundaries[i + 1])
            else:
                in_bin = (max_probs >= self.bin_boundaries[i]) & \
                        (max_probs < self.bin_boundaries[i + 1])
            
            if np.sum(in_bin) > 0:
                self.bin_values[i] = np.mean(is_correct[in_bin])
            else:
                self.bin_values[i] = (self.bin_boundaries[i] + self.bin_boundaries[i + 1]) / 2
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, pred_logits):
        """Apply TvA calibration to new predictions"""
        if not self.is_fitted:
            raise ValueError("Must call fit() before predict_proba()")
        
        # Convert to probabilities
        #if np.any(pred_logits < 0) or not np.allclose(np.sum(pred_logits, axis=1), 1):
        #    exp_logits = np.exp(pred_logits - np.max(pred_logits, axis=1, keepdims=True))
        #    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        #else:
        #    probabilities = pred_logits
        probabilities = F.softmax(pred_logits, dim=1).numpy()
        max_probs = np.max(probabilities, axis=1)
        
        # Map confidences to calibrated values
        calibrated_confidences = np.zeros_like(max_probs)
        for i in range(len(self.bin_boundaries) - 1):
            if i == len(self.bin_boundaries) - 2:
                in_bin = (max_probs >= self.bin_boundaries[i]) & \
                        (max_probs <= self.bin_boundaries[i + 1])
            else:
                in_bin = (max_probs >= self.bin_boundaries[i]) & \
                        (max_probs < self.bin_boundaries[i + 1])
            calibrated_confidences[in_bin] = self.bin_values[i]
        
        # Scale probabilities while preserving class rankings
        scaling_factors = np.divide(calibrated_confidences, max_probs, 
                                  out=np.ones_like(max_probs), where=(max_probs > 1e-8))
        calibrated_probs = probabilities * scaling_factors[:, np.newaxis]
        
        # Renormalize
        row_sums = np.sum(calibrated_probs, axis=1, keepdims=True)
        calibrated_probs = calibrated_probs / row_sums
        
        return calibrated_probs, calibrated_confidences