import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
import tva_confidence_calibrator as tva_calibrator
import warnings


class ConfidenceEstimator:
    """Integrated confidence monitoring for unlabeled deployment data"""
    
    def __init__(self, num_classes: int, calibration_method='histogram_binning', n_bins = 20):
        super().__init__()
        self.num_classes = num_classes
        self.calibration_method = calibration_method
        self.tva_calibrator = tva_calibrator.TvACalibrator(method=calibration_method, n_bins = n_bins)
        self.is_calibrated = False
        self.reset_statistics()
        
    def reset_statistics(self):
        """Reset accumulated statistics"""
        self.global_stats = {
            'calibrated_confidence': [],
            'original_confidence': [],
            'max_prob': [],
            'margin': [],
            'entropy': []
        }
        self.class_stats = {i: {
            'calibrated_confidence': [],
            'original_confidence': [],
            'max_prob': [],
            'margin': [],
            'entropy': []
        } for i in range(self.num_classes)}
    
    def calibrate(self, validation_logits, validation_labels):
        """
        Calibrate the confidence estimator using validation data.
        
        Args:
            validation_logits (torch.Tensor): Logits from validation set
            validation_labels (torch.Tensor): True labels for validation set
        """
        print(f"Calibrating confidence estimator using TvA {self.calibration_method}...")
        self.tva_calibrator.fit(validation_logits, validation_labels)
        self.is_calibrated = True
        print("Calibration complete!")
        
    @torch.no_grad()
    def compute_confidence_scores(self, logits):
        """Compute the three confidence measures"""
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        
        # 1. Maximum softmax probability
        max_probs, predictions = torch.max(probs, dim=1)
        
        # 2. Margin between top-1 and top-2 probabilities
        top2_probs = torch.topk(probs, 2, dim=1).values
        margins = top2_probs[:, 0] - top2_probs[:, 1]
        
        # 3. Predictive entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)

        # 4. TvA calibrated confidence
        if self.is_calibrated:
            calibrated_confidence = self.tva_calibrator.predict_proba(logits)
        else:
            calibrated_confidence = max_probs  # Fallback to original if not calibrated
            warnings.warn("Confidence estimator not calibrated. Using original confidence scores.")
        
        
        return max_probs, margins, entropy, predictions, calibrated_confidence
    
    def update_statistics(self, logits):
        """Update running statistics for confidence monitoring"""
        max_probs, margins, entropy, predictions, calibrated_confidence = self.compute_confidence_scores(logits)
        
        # Update global statistics
        self.global_stats['calibrated_confidence'].extend(calibrated_confidence.cpu().numpy())
        self.global_stats['original_confidence'].extend(max_probs.cpu().numpy())
        self.global_stats['max_prob'].extend(max_probs.cpu().numpy())
        self.global_stats['margin'].extend(margins.cpu().numpy())
        self.global_stats['entropy'].extend(entropy.cpu().numpy())
        
        # Update per-class statistics
        for class_idx in range(self.num_classes):
            mask = predictions == class_idx
            if mask.any():
                self.class_stats[class_idx]['calibrated_confidence'].extend(
                    calibrated_confidence[mask].cpu().numpy())
                self.class_stats[class_idx]['original_confidence'].extend(
                    max_probs[mask].cpu().numpy())
                self.class_stats[class_idx]['max_prob'].extend(
                    max_probs[mask].cpu().numpy())
                self.class_stats[class_idx]['margin'].extend(
                    margins[mask].cpu().numpy())
                self.class_stats[class_idx]['entropy'].extend(
                    entropy[mask].cpu().numpy())
    
    def get_confidence_report(self):
        """Generate comprehensive confidence statistics report"""
        def compute_stats(values):
            if not values:
                return {'count': 0, 'mean': 0.0, 'std': 0.0, 'p95': 0.0}
                
            arr = np.array(values)
            return {
                'count': len(arr),
                'mean': np.mean(arr),
                'std': np.std(arr),
                'p95': np.percentile(arr, 95)
            }
        
        report = {
            'global': {
                
                metric: compute_stats(values) 
                for metric, values in self.global_stats.items()
            },
            'per_class': {
                f'class_{i}': {
                    
                    metric: compute_stats(values)
                    for metric, values in class_data.items()
                }
                for i, class_data in self.class_stats.items()
            }
        }
        return report