# AAI-590 Group 9
# traininig module using Custom Classifier and Cyclical Temporal Features
# to be updated later
import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import json
from custom_models import AnimalTemporalClassifier
from custom_models import AnimalClassifier
from custom_datasets import S3ImageWithTimeFeatureDataset
from custom_losses import CrossEntropyMarginLoss
import time

from typing import Optional
from utils.utils import parse_s3_uri
import torch.optim.lr_scheduler as lr_scheduler
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
)
logger = logging.getLogger(__name__)




class WildScanTrainer:
    
    def __init__(self, model, 
                optimizer: optim.Optimizer = None,
                criterion: nn.Module = nn.CrossEntropyLoss(), # default is CrossEntropyLoss
                scheduler : Optional[optim.lr_scheduler.ReduceLROnPlateau] = None, # optim.lr_scheduler._LRScheduler = None,
                               
                device='cpu'):
        
        # check optimizer, criterion, scheduler format
        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError("optimizer must be a PyTorch optimizer (torch.optim.Optimizer)")
        
        # check if scheduler is pytorch lr_scheduler 
        if scheduler is not None and not isinstance(scheduler, (optim.lr_scheduler._LRScheduler, optim.lr_scheduler.ReduceLROnPlateau)):
        #if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler._LRScheduler):
            # check if scheduler is a ReduceLROnPlateau scheduler
            raise TypeError("scheduler must be a PyTorch scheduler (torch.optim.lr_scheduler._LRScheduler or optim.lr_scheduler.ReduceLROnPlateau) or None")
        
        # check criterion
        if not isinstance(criterion, nn.Module):
            raise TypeError("criterion must be a PyTorch loss function (torch.nn.Module)")

        self.model = model.to(device)

        # load optimizer 
        if(optimizer is None):
            # Default Adam optimizer used with learning rate of 1e-4 suitable for transfer learning
            logger.info("No optimizer provided, using default Adam with lr=1e-4")
            logger.warning("Default optimizer is Adam with lr=1e-4, consider using a different optimizer for better performance")
                       
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        else:
            self.optimizer = optimizer

        # log the loss function to be used
        logger.info(f"Using loss function: {criterion.__class__.__name__}")
        self.criterion = criterion

        # log the scheduler to be used if any
        if scheduler is not None:
            logger.info(f"Using learning rate scheduler: {scheduler.__class__.__name__} with params: {scheduler.__dict__}")
            self.scheduler = scheduler
        else:
            logger.info("No learning rate scheduler provided, training will use the optimizer's default behavior")
            
        

        # initialize training history loss and accuracy metrics
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'best_epoch': None,  # to store the best epoch based on validation loss
        }

        # container for best model weights during training
        self.best_model_weights = None
        
        self.device = device
        self.model = self.model.to(self.device)
        logger.info(f"Model initialized on device: {self.device}")
    
    def _update_scheduler(self, val_loss):
        """Helper method to handle different scheduler types"""
        if self.scheduler is None:
            return
            
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _calculate_accuracy(self, outputs, targets):
        """Calculate accuracy and return both accuracy and number of correct predictions."""
        # For multiclass classification (specific for AnimalClassifier/WildScan Project)
            
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum()
        return correct.item(), targets.size(0)
        

    def train_step(self, images, features, labels):
        self.model.train()
        # Move data to the specified device
        images, features, labels = images.to(self.device), features.to(self.device), labels.to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        if(self.model.use_temporal_features):
            # If the model uses temporal features, pass them along
            outputs = self.model(images, features)
        else:
            outputs = self.model(images)
        
        # Backward pass and optimization
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        # Calculate accuracy - returns (correct_count, total_count)
        correct_count, total_count = self._calculate_accuracy(outputs, labels) 

        return loss.item(), correct_count, total_count

    def validation_step(self, images, features, labels):
        """
        Performs a single validation step on one batch of data.
        Returns the loss for this batch.
        """
        images, features, labels = images.to(self.device), features.to(self.device), labels.to(self.device)
        
        with torch.no_grad():
            if(self.model.use_temporal_features):
                # If the model uses temporal features, pass them along
                outputs = self.model(images, features)
            else:
                outputs = self.model(images)
            
            loss = self.criterion(outputs, labels)
        
        # Calculate accuracy - returns (correct_count, total_count)
        correct_count, total_count = self._calculate_accuracy(outputs, labels) 
        
        return loss.item(), correct_count, total_count
    
    def fit(self, train_loader, val_loader=None, epochs=10):
        """
        Trains the model for a specified number of epochs.
        If a validation loader is provided, it will also validate after each epoch.
        """
        for epoch in range(epochs):
            logger.info(f"{'='*30}")
            logger.info(f"Training epoch {epoch + 1}/{epochs}")
            logger.info(f"{'='*30}")
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # store train and val loss per epoch to plot later

            
            # Iterate over the training data
            for batch_idx, (images, features, labels) in enumerate(train_loader):
                batch_loss, batch_correct_count, batch_count = 0, 0, 0
                batch_loss, batch_correct_count, batch_count = self.train_step(images, features, labels)
                train_loss += batch_loss * batch_count  # accumulate loss weighted by batch size
                train_correct += batch_correct_count
                train_total += batch_count
                #
                #logger.debug(f"Batch {batch_idx + 1}/{len(train_loader)}: Loss = {batch_loss:.4f}")
            
            train_loss /= train_total  # average loss over the entire training set
            train_accuracy = train_correct / train_total
            logger.info(f"Epoch {epoch + 1}/{epochs} - Training Loss: {train_loss:.4f} - Training Accuracy: {train_accuracy:.4f}")
            
            # Validation step
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for images, features, labels in val_loader:
                        batch_loss, batch_correct_count, batch_count = 0, 0, 0
                        batch_loss, batch_correct_count, batch_count = self.validation_step(images, features, labels)
                        val_loss += batch_loss * batch_count
                        val_correct += batch_correct_count
                        val_total += batch_count
                val_loss /= val_total
                val_accuracy = val_correct / val_total
                logger.info(f"Epoch {epoch + 1}/{epochs} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
            
            # Step the scheduler if provided
            if self.scheduler is not None:
                self._update_scheduler(val_loss)
                logger.info(f"Updated learning rate to {self.optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping when validation loss does not improve

            # Append loss and accuracy to training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            if val_loader is not None:
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_accuracy)
            else:
                self.training_history['val_loss'].append(None)
                self.training_history['val_accuracy'].append(None)
            
            # torch save model if validation loss this epoch is less than the best validation loss so far
            if val_loader is not None:
                if epoch == 0 or val_loss < min(self.training_history['val_loss'][:-1]):
                    self.training_history['best_epoch'] = epoch + 1
                    logger.info(f"Validation loss improved, saving model checkpoint for epoch {epoch + 1}")

                    self.best_model_weights = self.model.state_dict()
                    

        logger.info("Training complete.")

    def plot_metrics(self):
        plt.figure(figsize=(14, 6))
        # Loss curves
        plt.subplot(1, 2, 1)
        
        plt.plot(self.training_history["train_loss"], label=f"Train Loss")
        plt.plot(self.training_history["val_loss"], '--', label=f"Val Loss")
        plt.title("Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        # Accuracy curves
        plt.subplot(1, 2, 2)
        
        plt.plot(self.training_history["train_accuracy"], label=f"Train Acc")
        plt.plot(self.training_history["val_accuracy"], '--', label=f"Val Acc")
        plt.title("Accuracy Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()



