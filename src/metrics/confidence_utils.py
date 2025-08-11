# Wild-Scan: A tool for wildlife monitoring using AI
# contains helper functions for evaluating confidence metrics
# AAI-590 Group 9


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Union, Tuple, Dict

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
)
logger = logging.getLogger(__name__)


def reliability_diagram_data(y_true, y_pred, confidences, n_bins=15):
    """Generate data for reliability diagram (calibration plot)."""
    # Calculate correctness (1 if prediction correct, 0 otherwise)
    correctness = (y_pred == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if in_bin.sum() > 0:
            bin_center = (bin_lower + bin_upper) / 2
            bin_accuracy = correctness[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()
            bin_count = in_bin.sum()
            
            bin_centers.append(bin_center)
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_count)
    
    return {
        'bin_centers': np.array(bin_centers),
        'bin_accuracies': np.array(bin_accuracies),
        'bin_confidences': np.array(bin_confidences),
        'bin_counts': np.array(bin_counts)
    }

def plot_reliability_diagram_exact_reproduction(y_true, y_pred_raw, y_pred_cal, 
                                               raw_confidences, cal_confidences,
                                               raw_ece, cal_ece, n_bins=15, figsize=(10, 8)):
    """Plot the exact reliability diagram as shown in the ECE example above."""
    
    # Generate reliability diagram data
    raw_data = reliability_diagram_data(y_true, y_pred_raw, raw_confidences, n_bins=n_bins)
    cal_data = reliability_diagram_data(y_true, y_pred_cal, cal_confidences, n_bins=n_bins)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Perfect calibration diagonal line (black dashed)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=2, label='Perfect Calibration')
    
    # Raw classifier (red circles with size proportional to count)
    scatter_raw = ax.scatter(raw_data['bin_confidences'], raw_data['bin_accuracies'], 
                            s=raw_data['bin_counts'] * 3,  # Scale factor for visibility
                            alpha=0.7, color='red', marker='o', 
                            edgecolors='darkred', linewidth=1,
                            label='Raw Softmax')
    
    # Calibrated classifier (blue squares with size proportional to count)  
    scatter_cal = ax.scatter(cal_data['bin_confidences'], cal_data['bin_accuracies'],
                            s=cal_data['bin_counts'] * 3,  # Scale factor for visibility
                            alpha=0.7, color='blue', marker='s',
                            edgecolors='darkblue', linewidth=1,
                            label='Calibrated')
    
    # Add ECE text annotations
    ax.text(0.05, 0.95, f'Raw ECE: {raw_ece:.3f}', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.1),
            fontsize=12, color='darkred', fontweight='bold')
    
    ax.text(0.05, 0.85, f'Calibrated ECE: {cal_ece:.3f}', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='blue', alpha=0.1), 
            fontsize=12, color='darkblue', fontweight='bold')
    
    # Formatting to match the example
    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Reliability Diagram: Raw vs Calibrated Confidence (Multiclass)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits and ticks
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Legend
    ax.legend(loc='lower right', fontsize=12)
    
    # Make the plot look professional
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.tight_layout()
    
    return fig


class ConfidenceEvaluator:
    
    # End-to-end helper for evaluation of confidence metrics in multiclass classification.
    
    def __init__(self, y_true, y_pred, orig_confidences, cal_confidences, n_bins=15):
        
        self.y_true = np.array(y_true) # true class labels if available
        self.y_pred = np.array(y_pred) # predicted class labels (must have)

        self.orig_confidences = np.array(orig_confidences)
        self.cal_confidences = np.array(cal_confidences) if cal_confidences is not None else None

        self.n_bins = n_bins
        # Validate inputs
        self._validate_inputs()

        self.correct = (self.y_true == self.y_pred).astype(int)
    
    def _validate_inputs(self):
        """Validate input arrays for consistency."""
        if len(self.y_true) != len(self.y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        if len(self.y_true) != len(self.orig_confidences):
            raise ValueError("y_true and confidences must have the same length")
        if self.cal_confidences is not None and len(self.y_true) != len(self.cal_confidences):
            raise ValueError("y_true and calibrated_confidences must have the same length")
        
        if not (0 <= self.orig_confidences.min() and self.orig_confidences.max() <= 1):
            logger.error("Confidences should be in range [0, 1]")
        
        if self.cal_confidences is not None:
            if not (0 <= self.cal_confidences.min() and self.cal_confidences.max() <= 1):
                logger.error("Calibrated confidences should be in range [0, 1]")
    
    # ---------- Confidence Metrics to Evaluate Given Ground Truth Labels ----------
    def expected_calibration_error(self, confidences: np.ndarray = None, n_bins: int = None) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            confidences: Confidence scores to evaluate (uses self.orig_confidences if None)
            
        Returns:
            ECE value
        """
        if confidences is None:
            confidences = self.orig_confidences
        
        if n_bins is None:
            n_bins = self.n_bins
            
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = self.correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece

    

    
    def overconfidence_error(self, confidences: np.ndarray = None) -> float:
        """
        Compute Overconfidence Error (positive when model is overconfident).
        
        Args:
            confidences: Confidence scores to evaluate (uses self.confidences if None)
            
        Returns:
            Overconfidence error value
        """
        if confidences is None:
            confidences = self.orig_confidences
            
        return confidences.mean() - self.correct.mean()

    def brier_score(self, confidences: np.ndarray = None) -> float:
        """
        Compute Brier Score (lower is better).
        
        Args:
            confidences: Confidence scores to evaluate (uses self.confidences if None)
            
        Returns:
            Brier score
        """
        if confidences is None:
            confidences = self.orig_confidences
            
        return np.mean((confidences - self.correct) ** 2)    

    def reliability_diagram_data(self, confidences: np.ndarray = None, n_bins = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute data for reliability diagram.
        
        Args:
            confidences: Confidence scores to evaluate (uses self.confidences if None)
            
        Returns:
            Tuple of (bin_means, accuracies, counts)
        """
        if confidences is None:
            confidences = self.orig_confidences
        if n_bins is None:
            n_bins = self.n_bins
            
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_means = []
        accuracies = []
        counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            count = in_bin.sum()
            
            if count > 0:
                bin_means.append(confidences[in_bin].mean())
                accuracies.append(self.correct[in_bin].mean())
                counts.append(count)
            #else:
            #    bin_means.append((bin_lower + bin_upper) / 2)
            #    accuracies.append(0)
            #    counts.append(0)
                
        return np.array(bin_means), np.array(accuracies), np.array(counts)
    

    def get_all_metrics(self, confidences: np.ndarray = None) -> Dict[str, float]:
        """
        Compute all calibration metrics.
        
        Args:
            confidences: Confidence scores to evaluate (uses self.confidences if None)
            
        Returns:
            Dictionary containing all computed metrics
        """
        if confidences is None:
            confidences = self.orig_confidences
            
        metrics = {
            'ECE': self.expected_calibration_error(confidences),
            #'MCE': self.maximum_calibration_error(confidences),
            #'ACE': self.average_calibration_error(confidences),
            'Overconfidence': self.overconfidence_error(confidences),
            'Brier_Score': self.brier_score(confidences),
            #'Log_Loss': self.log_loss(confidences),
            #'Sharpness': self.sharpness(confidences),
            #'Resolution': self.resolution(confidences),
            #'Reliability': self.reliability(confidences),
            'Accuracy': self.correct.mean()
        }
        
        return metrics
    
    
    # ---------- create annotation reduction comparison data and plots -------
    def _analyze_annotation_reduction(self, thresholds=np.linspace(0.5, 1.0, 50)):
        """
        Compare annotation reduction between raw and calibrated confidence scores 
        """

        # Storage for results
        results = {
            'thresholds': thresholds,
            'raw': {'annotation_reduction': [], 'accuracy': [], 'coverage': [], 'num_samples': []},
            'calibrated': {'annotation_reduction': [], 'accuracy': [], 'coverage': [], 'num_samples': []}
        }
        
        for tau in thresholds:
            # Raw classifier analysis - samples with confidence >= threshold are auto-labeled
            raw_auto_mask = self.orig_confidences >= tau
            raw_coverage = raw_auto_mask.mean()  # proportion of samples auto-labeled
            raw_num_auto = raw_auto_mask.sum()
            
            if raw_num_auto > 0:
                # Accuracy on auto-labeled samples (using correctness)
                raw_accuracy = self.correct[raw_auto_mask].mean()
            else:
                raw_accuracy = np.nan
                
            results['raw']['annotation_reduction'].append(raw_coverage)
            results['raw']['accuracy'].append(raw_accuracy)
            results['raw']['coverage'].append(raw_coverage)
            results['raw']['num_samples'].append(raw_num_auto)
            
            # Calibrated classifier analysis
            cal_auto_mask = self.cal_confidences >= tau
            cal_coverage = cal_auto_mask.mean()
            cal_num_auto = cal_auto_mask.sum()
            
            if cal_num_auto > 0:
                # Accuracy on auto-labeled samples (using correctness)
                cal_accuracy = self.correct[cal_auto_mask].mean()
            else:
                cal_accuracy = np.nan
                
            results['calibrated']['annotation_reduction'].append(cal_coverage)
            results['calibrated']['accuracy'].append(cal_accuracy)
            results['calibrated']['coverage'].append(cal_coverage)
            results['calibrated']['num_samples'].append(cal_num_auto)
        
        return results

    def plot_annotation_reduction_stats(self, thresholds=np.linspace(0.5, 1.0, 50)):
        """Create comprehensive comparison plots"""
        
        results = self._analyze_annotation_reduction(thresholds)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Annotation Reduction
        ax1.plot(thresholds, np.array(results['raw']['annotation_reduction']) * 100, 
                label='Raw Softmax', color='red', linewidth=2, marker='o', markersize=3)
        ax1.plot(thresholds, np.array(results['calibrated']['annotation_reduction']) * 100, 
                label='Calibrated', color='blue', linewidth=2, marker='s', markersize=3)
        ax1.set_xlabel('Confidence Threshold')
        ax1.set_ylabel('Annotation Reduction (%)')
        ax1.set_title('Annotation Reduction vs Confidence Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy on Auto-labeled
        ax2.plot(thresholds, results['raw']['accuracy'], 
                label='Raw Softmax', color='red', linewidth=2, marker='o', markersize=3)
        ax2.plot(thresholds, results['calibrated']['accuracy'], 
                label='Calibrated', color='blue', linewidth=2, marker='s', markersize=3)
        ax2.set_xlabel('Confidence Threshold')
        ax2.set_ylabel('Accuracy on Auto-labeled Samples')
        ax2.set_title('Accuracy vs Confidence Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Accuracy vs Coverage Trade-off
        ax3.plot(results['raw']['coverage'], results['raw']['accuracy'], 
                label='Raw Softmax', color='red', linewidth=2, marker='o', markersize=4)
        ax3.plot(results['calibrated']['coverage'], results['calibrated']['accuracy'], 
                label='Calibrated', color='blue', linewidth=2, marker='s', markersize=4)
        ax3.set_xlabel('Coverage (Proportion Auto-labeled)')
        ax3.set_ylabel('Accuracy on Auto-labeled Samples')
        ax3.set_title('Accuracy vs Coverage Trade-off')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Improvement
        improvement = (np.array(results['calibrated']['annotation_reduction']) - 
                    np.array(results['raw']['annotation_reduction'])) * 100
        ax4.plot(thresholds, improvement, color='green', linewidth=2, marker='d', markersize=4)
        ax4.set_xlabel('Confidence Threshold')
        ax4.set_ylabel('Improvement (%)')
        ax4.set_title('Calibration Improvement in Annotation Reduction')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        return fig

    def plot_reliability_with_histogram(self, confidences: np.ndarray = None, n_bins=15):
        """
        Reliability diagram with confidence distribution histogram
        """
        if confidences is None:
            confidences = self.orig_confidences
        correct = self.correct
        # Get reliability data
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        #bin_boundaries = np.quantile(confidences, np.linspace(0, 1, n_bins + 1))
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_means, accuracies, counts = [], [], []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            count = in_bin.sum()
            
            if count > 0:
                bin_means.append(confidences[in_bin].mean())
                accuracies.append(correct[in_bin].mean())
                counts.append(count)
            #else:
            #    bin_means.append((bin_lower + bin_upper) / 2)
            #    accuracies.append(0)
            #    counts.append(0)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
        
        # Main reliability plot
        ax1 = fig.add_subplot(gs[0])
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax1.plot(bin_means, accuracies, 'o-', markersize=8, linewidth=2, 
                label='Model Calibration', color='red')
        ax1.fill_between(bin_means, accuracies, bin_means, alpha=0.2, color='red')
        
        ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax1.set_ylabel('Fraction of Positives', fontsize=12)
        ax1.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Add ECE
        ece = np.sum(np.abs(np.array(bin_means) - np.array(accuracies)) * np.array(counts)) / np.sum(counts)
        ax1.text(0.05, 0.95, f'ECE: {ece:.4f}', transform=ax1.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Histogram
        ax2 = fig.add_subplot(gs[1])
        ax2.hist(confidences, bins=n_bins, range=(0, 1), alpha=0.7, 
                color='skyblue', edgecolor='black')
        ax2.set_xlabel('Predicted Probability', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Confidence Distribution', fontsize=12)
        ax2.grid(True, alpha=0.3)


        
        return fig, (ax1, ax2)

    def compare_reliability_diagrams(self, n_bins=15):
        """
        Reliability diagram with confidence distribution histogram
        """
        # get reliability data for original and calibrated confidences
        orig_mean_conf, orig_acc, orig_counts = self.reliability_diagram_data(self.orig_confidences, n_bins=n_bins)
        cal_mean_conf, cal_acc, cal_counts = self.reliability_diagram_data(self.cal_confidences, n_bins=n_bins)
        
        fig, ax = plt.subplots(figsize=(9,7))
        #ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', lw=2)
        
        #ax.plot(orig_mean_conf, orig_acc, 'o-', color='red', lw=2, label="Original Confidence")# (ECE={ece_orig:.3f})')
        #ax.plot(cal_mean_conf, cal_acc, 's-', color='blue', lw=2, label="Calibrated Confidence") # (ECE={ece_calib:.3f})')
        
        #ax.set_xlabel('Confidence', fontsize=13)
        #ax.set_ylabel('Empirical Accuracy', fontsize=13)
        #ax.set_title('Reliability Diagram: Original vs Calibrated', fontsize=15)
        #ax.grid(True)
        #ax.legend()
        #plt.tight_layout()
        #plt.show()


        # Perfect calibration
        ax.plot([0,1], [0,1], 'k--', lw=2, label='Perfect Calibration', alpha=0.5)

        # Calibrated curve/shaded
        ax.plot(cal_mean_conf, cal_acc, 's-', color='royalblue', lw=2, label="TvA Histogram Binning")# (ECE={ece_calib:.3f})')
        #ax.fill_between(cal_mean_conf, cal_acc, cal_mean_conf, color='royalblue', alpha=0.15, label='Calibrated Miscalibration')

        # Original curve/shaded
        ax.plot(orig_mean_conf, orig_acc, 'o-', color='crimson', lw=2, label="Original Confidence")# (ECE={ece_orig:.3f})')
        #ax.fill_between(orig_mean_conf, orig_acc, cal_mean_conf, color='crimson', alpha=0.18, label='Original Miscalibration')

        
        
        # Highlight improvement for each bin (arrow)
        #for x0, y0, x1, y1 in zip(orig_mean, orig_acc, calib_mean, calib_acc):
        #    ax.annotate(
        #        '',
        #        xy=(x1, y1), xytext=(x0, y0),
        #        arrowprops=dict(arrowstyle="->", color='green', lw=1.4, alpha=0.7)
        #    )

        # Overlay confidence histogram under plot
        #ax2 = ax.twinx()
        #hist_orig = ax2.bar(bin_centers-0.015, orig_count, width=0.025, color='crimson', alpha=0.22, label='Original Count')
        #hist_cal = ax2.bar(bin_centers+0.015, calib_count, width=0.025, color='royalblue', alpha=0.22, label='Calibrated Count')
        #ax2.set_ylabel('Bin Count')
        #ax2.set_ylim(0, max(max(orig_count), max(calib_count)) * 1.1)

        ax.set_xlabel('Confidence', fontsize=14)
        ax.set_ylabel('Accuracy of Correct Predictions', fontsize=14)
        ax.set_title('Reliability Diagram: Original vs Calibrated', fontsize=15)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right', fontsize=10)
        #ax2.legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.show()
        return fig, ax

        
        
        # Main reliability plot
        #ax1 = fig.add_subplot(gs[0])
        #ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        #ax1.plot(bin_means, accuracies, 'o-', markersize=8, linewidth=2, 
        #        label='Model Calibration', color='red')
        #ax1.fill_between(bin_means, accuracies, bin_means, alpha=0.2, color='red')
        
        #ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
        #ax1.set_ylabel('Fraction of Positives', fontsize=12)
        #ax1.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
        #ax1.legend()
        #ax1.grid(True, alpha=0.3)
        #ax1.set_xlim([0, 1])
        #ax1.set_ylim([0, 1])
        
        # Add ECE
        #ece = np.sum(np.abs(np.array(bin_means) - np.array(accuracies)) * np.array(counts)) / np.sum(counts)
        #ax1.text(0.05, 0.95, f'ECE: {ece:.4f}', transform=ax1.transAxes, 
        #        fontsize=12, verticalalignment='top',
        #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Histogram
        #ax2 = fig.add_subplot(gs[1])
        #ax2.hist(confidences, bins=n_bins, range=(0, 1), alpha=0.7, 
        #        color='skyblue', edgecolor='black')
        #ax2.set_xlabel('Predicted Probability', fontsize=12)
        #ax2.set_ylabel('Count', fontsize=12)
        #ax2.set_title('Confidence Distribution', fontsize=12)
        #ax2.grid(True, alpha=0.3)


        
        #return fig, (ax1, ax2)
    #@staticmethod
    def compare_metrics_orig_vs_calibrated(self, n_bins=15):
    # Compare Confidence Metrics between original and calibrated classifiers for multiclass.

        # check if calibrated confidences are available
        if self.cal_confidences is None:
            logger.error("No calibrated confidences available for comparison.")
            return
    
        # ECE Metric
        ece_orig = self.expected_calibration_error(self.orig_confidences, n_bins=n_bins)
        ece_cal = self.expected_calibration_error(self.cal_confidences, n_bins=n_bins) 
        ece_improvement = (ece_orig - ece_cal) / ece_orig * 100 if ece_orig > 0 else 0
        
        # Overconfidence Metric
        overconfidence_orig = self.overconfidence_error(self.orig_confidences)
        overconfidence_cal = self.overconfidence_error(self.cal_confidences)

        # Brier Score Metric
        brier_orig = self.brier_score(self.orig_confidences)
        brier_cal = self.brier_score(self.cal_confidences)

        # create metrics table to display
        metrics_table = pd.DataFrame({
            'Metric': ['ECE', 'Overconfidence Error', 'Brier Score'],
            'Original': [ece_orig, overconfidence_orig, brier_orig],
            'Calibrated': [ece_cal, overconfidence_cal, brier_cal],
            'Improvement (%)': [
                ece_improvement, 0.0, 0.0
                #(overconfidence_orig - overconfidence_cal) / overconfidence_orig * 100 if overconfidence_orig > 0 else 0,
                #(brier_orig - brier_cal) / brier_orig * 100 if brier_orig > 0 else 0
            ]
        })
        
        print(f"Dataset  {len(self.y_true)} test samples, {len(np.unique(self.y_true))} classes")

        display(metrics_table)

        #print(f"Original ECE: {ece_orig:.4f} ")
        #print(f"Calibrated ECE: {ece_cal:.4f} ")
        #print(f"improvement: {improvement:.1f}%")
    #    return {
    #        'raw_ece': ece_raw,
    #        'calibrated_ece': ece_cal,
    #        'ece_improvement': improvement,
    #        'raw_bins': bins_raw,
    #        'calibrated_bins': bins_cal
    #    }
    
