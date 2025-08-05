# AAI-590 Group 9
# Evaluator module use in wildscan ml system and pretraining pipeline

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, classification_report
)
from scipy.stats import entropy
from torch.utils.data import DataLoader
import logging
import torch
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
)
logger = logging.getLogger(__name__)

class WildScanEvaluator:
    def __init__(self, model, test_loader:DataLoader=None, device='cpu', label_mapping:dict=None):
        """
        model: torch model (must implement .eval())
        test_loader: PyTorch DataLoader yielding (images, labels)
        device: torch.device ('cpu' or 'cuda or 'mps')
        label_binarizer: optional sklearn LabelBinarizer for multiclass ROC-AUC
        """
        
        self.test_loader = test_loader
        self.device = device
        self.label2idx = label_mapping
        # Create a list where index matches int label, value is label_name
        self.idx2label = {v: k for k, v in self.label2idx.items()}

        
        self.model = model
        self.model = self.model.to(self.device)
        logger.info(f"Model initialized on device: {self.device}")

        # Storage for later analysis
        self.metrics = {}
        self.pred_probs = None
        self.pred_labels = None
        self.true_labels = None


    def _predict_all(self):
        # predict all data in the test_loader
        self.model.eval()
        all_probs, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for images, features, labels in self.test_loader:
                images, features, labels = images.to(self.device), features.to(self.device), labels.to(self.device)
                #logger.info(f"test batch loaded to: {self.device}")
                if(self.model.use_temporal_features):
                # If the model uses temporal features, pass them along
                    outputs = self.model(images, features)
                else:
                    outputs = self.model(images)
                

               
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        return (
            np.concatenate(all_preds),
            np.concatenate(all_probs),
            np.concatenate(all_labels)
        )

    def evaluate(self, test_loader:DataLoader=None):
        # update test loader if specified
        self.test_loader = test_loader if test_loader is not None else self.test_loader

        # make predictions
        y_pred, y_prob, y_true = self._predict_all()
        
        # Store predictions for later use
        self.pred_probs = y_prob 
        self.pred_labels = y_pred # scalar
        self.true_labels = y_true # scaler

        print(f"\n=== MODEL {self.model.name} ===")
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')

        full_class_indices = list(self.idx2label.keys())  # all class indices in the model
        unique_classes = np.unique(y_true) # unique classes in evaluation set

        self.metrics = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "roc_auc": roc_auc,
            "classification_report": classification_report(
                y_true, y_pred, 
                zero_division=0, output_dict=True,
                labels=full_class_indices, target_names=[self.idx2label[i] for i in full_class_indices]
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=full_class_indices)
        }

        print("Accuracy:", acc)
        print("F1-score (macro):", f1_macro)
        print("ROC-AUC:", roc_auc)
        print("Classification Report:\n", classification_report(y_true, y_pred,zero_division=0,labels=full_class_indices, target_names=[self.idx2label[i] for i in full_class_indices]))
        print("Confusion Matrix:\n", self.metrics["confusion_matrix"])
        
        # plot ROC curve
        plt.figure(figsize=(8, 6))
        for i in full_class_indices:
            if i not in unique_classes:
                continue
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
            auc = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
            plt.plot(fpr, tpr, label=f"{self.idx2label[i]} (AUC={auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--', label='Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    
    def check_robustness(self):
        _, y_prob, _ = self._predict_all()
        pred_entropy = entropy(y_prob, axis=1)
        print(f"\n{self.name} Prediction Entropy Stats:")
        print("Mean entropy:", np.mean(pred_entropy))
        print("Std entropy:", np.std(pred_entropy))
        y_pred = np.argmax(y_prob, axis=1)
        values, counts = np.unique(y_pred, return_counts=True)
        print("Predicted class distribution:", dict(zip(values, counts)))