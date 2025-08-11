# ===============================================================================
# WildScan Machine Learning Deployment System
# ===============================================================================
# Custom Machine Learning System for Camera Trap Image Classification with Production Data Simulation, Active Learning via Confidence Estimation, and Model Retraining
# Tyler Clinscales, Edwin Merchan, Geoffrey Fadera
# MS-AAI University of San Diego, 2025

# import necessary libraries
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import torch
import json
import logging
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
from dataclasses import dataclass, field

# import custom modules
from custom_models import AnimalClassifier, AnimalTemporalClassifier, ScratchResNet, WrapperModel
from custom_datasets import S3ImageWithTimeFeatureDataset
from custom_losses import CrossEntropyMarginLoss
from trainer import WildScanTrainer as Trainer
from evaluator import WildScanEvaluator as Evaluator
import tva_confidence_calibrator

from tva_confidence_calibrator import TvACalibrator
from tva_confidence_calibrator import TvAHistogramCalibrator
from tva_confidence_calibrator import TvAHistogramBinning as TvAHistogramBinning
import metrics.confidence_utils as confidence_utils

from metrics.confidence_utils import ConfidenceEvaluator as ConfidenceEvaluator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score

import logging
# Set up logging
#logging.basicConfig(
#    level=logging.DEBUG,
#    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
#)
logger = logging.getLogger(__name__)

MODELS_DICT = {
        0: {
            'name' : 'ScratchResNet', 
            'class' : ScratchResNet, 
            #'image_shape' : (3, 224, 224),  # input image shape
            'num_classes' : 14,
            #'aux_dim' : None,  # no auxiliary input
            #'final_activation' : None,  # returns logits due to nn.CrossEntropyLoss
            },
        1: {
            'name' : 'AnimalClassifier_ResNet18', 
            'class' : AnimalClassifier,
            'num_classes' : 14,
            #'image_shape' : (3, 224, 224),  # input image shape
            #'aux_dim' : None,  # no auxiliary input
            #'final_activation' : None,  # returns logits due to nn.CrossEntropyLoss 
            
            },
        2: {
            'name' : 'AnimalTemporalClassifier_ResNet18', 
            'class' : AnimalTemporalClassifier, 
            #'image_shape' : (3, 224, 224),  # input image shape
            #'aux_dim' : (1, 4),  # cyclical time features (e.g., day of year, hour of day, etc.)
            'num_classes' : 14,  # number of classes for classification
            'proj_dim' : 128,  # projection dimension for each of the image vector and temporal features for fusion
            #'fc_hidden_layer': 128,  # hidden layer size for final classification
            #'final_activation' : None,  # returns logits due to use of nn.CrossEntropyLoss 
            },
        3: {
            'name' : 'AnimalClassifier_Resnet50', 
            'class' : WrapperModel, 
            'backbone' : 'resnet50',  # backbone model
            #'image_shape' : (3, 224, 224),  # input image shape
            #'aux_dim' : (1, 4),  # cyclical time features (e.g., day of year, hour of day, etc.)
            'num_classes' : 14,  # number of classes for classification
            
            #'fc_hidden_layer': 128,  # hidden layer size for final classification
            #'final_activation' : None,  # returns logits due to use of nn.CrossEntropyLoss 
            },
        4: {
            'name' : 'AnimalClassifier_Vgg16', 
            'class' : WrapperModel, 
            'backbone' : 'vgg16',  # backbone model
            #'image_shape' : (3, 224, 224),  # input image shape
            #'aux_dim' : (1, 4),  # cyclical time features (e.g., day of year, hour of day, etc.)
            'num_classes' : 14,  # number of classes for classification
            
            #'fc_hidden_layer': 128,  # hidden layer size for final classification
            #'final_activation' : None,  # returns logits due to use of nn.CrossEntropyLoss 
            },
        
    }

@dataclass
class SystemConfig:
    """Master configuration for the entire ML system"""
    # General settings
    project_name: str = "WildScan_MLSystem"
    experiment_name: str = "default_experiment"
    # get device whether it's cuda, cpu, or mps
    if torch.cuda.is_available():
        device: str = "cuda"
    elif torch.backends.mps.is_available():
        device: str = "mps"
    else:
        device: str = "cpu"
    
    seed: int = 42
    
    # Meta_Data paths
    train_data_path: str = ""
    val_data_path: str = ""
    test_data_path: str = ""
    ood_data_path: str = ""
    
    production_data_path: str = ""
    production_ood_data_path: str = ""
    calibration_data_path: str = ""

    production_predictions_path: str ="",

    # Preprocessed images path
    preprocessed_images_path: str = "./preprocessed_images"

    # Models Configurations
    model_configs: Dict[int, Dict] = field(default_factory=lambda:MODELS_DICT)

    # label mapping path
    label2idx_path: str = "./label2idx.json"

    
    #update_num_classes_from_label2idx()
    models_to_use: List[int]  = field(default_factory=lambda:[0, 1, 2])  # which models to use from model_configs

    # model training output dir
    training_output_dir: str = "./pretrained_models"
    
    # TrainerClass Settings Default
    
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    patience: int = 3
    epochs: int = 10
    criterion: nn.Module = CrossEntropyMarginLoss(reduction='mean', margin_lambda=0.1, margin_type="logits")
    
    # Criterion Settings# Batch Inference Batch Size (Depends on GPU memory)
    inference_batch_size: int =128

    # confidence estimation settings
    confidence_threshold: float = 0.8 # by default
    # Pipeline settings
    enable_pretraining_pipeline: bool = True
    enable_production_pipeline: bool = True

    
    # Logging and output
    output_dir: str = "./outputs"
    log_level: str = "INFO"
    

@dataclass
class TrainerConfig:
    optimizer_class: type = optim.Adam
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {"lr": 1e-3, "weight_decay": 1e-4})
    scheduler_class: Optional[type] = optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {"patience": 5, "factor": 0.5})
    loss_function_class: type = nn.CrossEntropyLoss
    loss_function_params: Dict[str, Any] = field(default_factory=dict)
    epochs: int = 50
    batch_size: int = 32
    device: str = "cpu"

    def get_loss_function(self):
        return self.loss_function_class(**self.loss_function_params)

@dataclass
class ProductionConfig:
    """Configuration Data class for Production Pipeline Use only"""
    # General settings
    project_name: str = "WildScan_MLSystem"
    experiment_name: str = "Production Simulation"
    
    # get device whether it's cuda, cpu, or mps
    device: str = '' # updated to what is available during initialization 
    
    
    
    # Meta_Data paths
    train_data_path: str = ""
    val_data_path: str = ""
    test_data_path: str = ""
    ood_data_path: str = ""
    
    production_data_path: str = ""
    production_ood_data_path: str = ""
    calibration_data_path: str = ""

    production_predictions_path: str ="",

    # Preprocessed images path
    preprocessed_images_path: str = "./preprocessed_images"

    # Models Configurations
    model_configs: Dict[int, Dict] = field(default_factory=lambda:MODELS_DICT)

    # label mapping path
    label2idx_path: str = "./label2idx.json"

    
    #update_num_classes_from_label2idx()
    models_to_use: List[int]  = field(default_factory=lambda:[0, 1, 2])  # which models to use from model_configs

    # model training output dir
    training_output_dir: str = "./pretrained_models"
    
    # TrainerClass Settings Default
    
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    patience: int = 3
    epochs: int = 10
    criterion: nn.Module = CrossEntropyMarginLoss(reduction='mean', margin_lambda=0.1, margin_type="logits")
    
    # Criterion Settings# Batch Inference Batch Size (Depends on GPU memory)
    inference_batch_size: int =128

    # confidence estimation settings
    confidence_threshold: float = 0.8 # by default
    # Pipeline settings
    enable_pretraining_pipeline: bool = True
    enable_production_pipeline: bool = True

    
    # Logging and output
    output_dir: str = "./outputs"
    log_level: str = "INFO"
    


class PretrainingPipeline:
    """
    Pipeline for research, experimentation, and model comparison.
    Focuses on model development and evaluation.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".ProductionPipeline")
        
        # load label2idx mapping
        try:
            with open(self.config.label2idx_path, 'r') as f:
                self.label2idx = json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Label mapping file not found: {self.config.label2idx_path}")
    
        # Components
        self.datasets = {}
        self.dataloaders = {}
        self.models = {}
        self.trainers = {}
        self.evaluators = {}
        self.confidence_calibrators = {}
        self.results = {}
        self.default_trainer_config = self.get_default_trainer_config()
        
    
    def get_default_trainer_config(self) -> TrainerConfig:
        return TrainerConfig(
            device=self.config.device if hasattr(self.config, 'device') else 'cpu'
        )
        
    def setup_datasets(self):
        """Setup datasets for pretraining pipeline"""
        self.logger.info("Setting up datasets for pretraining...")
        
        self.datasets['train'] = S3ImageWithTimeFeatureDataset(
            csv_path = self.config.train_data_path, 
            #label2idx_path = self.config.label2idx_path,
            label2idx = self.label2idx,  # use the loaded label2idx mapping
            images_path = self.config.preprocessed_images_path,
        )
        self.datasets['val'] = S3ImageWithTimeFeatureDataset(
            csv_path = self.config.val_data_path, 
            #label2idx_path = self.config.label2idx_path,
            label2idx = self.label2idx,  # use the loaded label2idx mapping
            images_path = self.config.preprocessed_images_path,
        )
        self.datasets['test'] = S3ImageWithTimeFeatureDataset(
            csv_path = self.config.test_data_path, 
            #label2idx_path = self.config.label2idx_path,
            label2idx = self.label2idx,  # use the loaded label2idx mapping
            images_path = self.config.preprocessed_images_path,
        )
        self.datasets['ood'] = S3ImageWithTimeFeatureDataset(
            csv_path = self.config.ood_data_path, 
            #label2idx_path = self.config.label2idx_path,
            label2idx = self.label2idx,  # use the loaded label2idx mapping
            images_path = self.config.preprocessed_images_path,
        )
        self.logger.info(f"Train dataset size: {len(self.datasets['train'])}")
        self.logger.info(f"Validation dataset size: {len(self.datasets['val'])}")
        self.logger.info(f"Test (in-distribution) dataset size: {len(self.datasets['test'])}")
        self.logger.info(f"Test (out-of-distribution) dataset size: {len(self.datasets['ood'])}")
        
        # Create data loaders
        self.dataloaders = {
            'train': torch.utils.data.DataLoader(
                self.datasets['train'], 
                batch_size=self.config.batch_size, 
                shuffle=True
            ),
            'val': torch.utils.data.DataLoader(
                self.datasets['val'], 
                batch_size=self.config.batch_size
            ),
            'test': torch.utils.data.DataLoader(
                self.datasets['test'], 
                batch_size=self.config.batch_size
            ),
            'ood': torch.utils.data.DataLoader(
                self.datasets['ood'], 
                batch_size=self.config.batch_size
            )
        }
        
    def setup_models(self, model_configs: Dict[int, Dict], models_to_use: Optional[List[int]] = None):
        """Setup multiple models for comparison"""
        if models_to_use is None:
            models_to_use = self.config.models_to_use
        
        # log number of models to setup based on models_to_use
        model_configs = {k: v for k, v in model_configs.items() if k in models_to_use}
        self.logger.info(f"Setting up {len(model_configs)} models...")
        
        for model_id, model_config in model_configs.items():
            self.logger.info(f"Setting up {len(model_configs)} models: {model_config['name']} (ID: {model_id})")
            model_class = model_config.pop('class')
            self.logger.debug(f"Initializing model {model_config['name']} with class {model_class.__name__}")
            self.models[model_id] = model_class(**model_config)
            #model_name = model_config['name']
            #self.models[model_name] = self.models[model_id]  # map model_id to model_name
            self.logger.info(f"Initialized Model No {model_id}: {model_class.__name__}")
            
    def train_models(self):
        """Train the specified models"""
        self.logger.info(f"Found {len(self.models.items())} model(s) to train.")
        self.logger.info("Starting model training...")
        
        for model_id, model in self.models.items():
            self.logger.info(f"Training Model No {model_id} : {model.name}...")
            
            # Setup trainer
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay= self.config.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.1, patience = self.config.patience)
            criterion = self.config.criterion
            
            trainer = Trainer(model, optimizer, criterion, scheduler, device=self.config.device)
            self.trainers[model_id] = trainer
            
            # Train
            trainer.fit(
                self.dataloaders['train'], 
                self.dataloaders['val'], 
                epochs=self.config.epochs
            )
            self.logger.info(f"Model No {model_id} : {model.name} trained successfully.")
            # Save the historical learning curves and the best model weights of the trained model
            # Dump training results to config training_output_dir, using os and makedirs
            # create subdirectory for each model using model_id
            output_dir = Path(self.config.training_output_dir) / f"model_{model_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            # Save model state
            model_save_path = output_dir / f"best_model.pth"
            torch.save(trainer.best_model_weights, model_save_path)
            # dump learning history to json
            history_save_path = output_dir / f"training_history.json"
            with open(history_save_path, 'w') as f:
                json.dump(trainer.training_history, f, indent=4)
            self.logger.info(f"Model No {model_id} : {model.name} saved to {model_save_path}")
            self.logger.info(f"Training history saved to {history_save_path}")

            # plot loss and accuracy curves
            trainer.plot_metrics()

            # update model with the best weights
            model.load_state_dict(trainer.best_model_weights)
            self.models[model_id] = model  # update the model in the dictionary
            self.logger.info(f"Model No {model_id} : {model.name} loaded with best weights.")

        
    def evaluate_models(self):
        """Evaluate all trained models"""
        logger.info("Evaluating models...")

        if not self.models:
            raise ValueError("No models available for evaluation. Run setup_models and train_models first.")
        if 'test' not in self.dataloaders:
            raise ValueError("Test dataloader not found. Ensure datasets are set up correctly.")
        logger.info(f"Found {len(self.models.items())} model(s) to evaluate.")

        # pass all models and dataset to WildScanEvaluator
        #evaluator = Evaluator(self.models, device=self.config.device)


        for model_id, model in self.models.items():
            self.logger.info(f"Evaluating Model No {model_id} : {model.name}...")
            self.evaluators[model_id] = {}

            # get evaluations for in-distribution test set
            evaluator = Evaluator(model, device=self.config.device,
                      test_loader=self.dataloaders['test'],
                      label_mapping=self.label2idx)
            
            self.evaluators[model_id]['id'] = evaluator.evaluate()

            # get evaluations for out-of-distribution test set
            evaluator2 = Evaluator(model, device=self.config.device,
                      test_loader=self.dataloaders['ood'],
                      label_mapping=self.label2idx)
            self.evaluators[model_id]['ood'] = evaluator2.evaluate()

            #evaluator.evaluate()
            
            # Evaluate on test set
            #results = evaluator.comprehensive_evaluation(self.dataloaders['test'])
            #self.results[model_name] = results
            
            #logger.info(f"{model_id} - Test Accuracy: {results['accuracy']:.4f}")

    def evaluate_confidences(self):
        """Evaluate confidence estimation for all models"""
        
        for model_id, model in self.models.items():
            self.logger.info(f"Evaluating Confidence Metrics for Model No {model_id} : {model.name}...")
            self.logger.info(f"Setup calibration data from validation set for this model...")
            # Setup calibration data for this model
            cal_evaluator = Evaluator(model, device=self.config.device,
                      test_loader=self.dataloaders['val'],
                      label_mapping=self.label2idx)
            
            cal_pred_labels, cal_pred_probs, cal_true_labels, cal_pred_logits = cal_evaluator._predict_all()

            # try TvAHistogramBinning
            n_bins = 15
            tva_hb_calibrator = []
            tva_hb_calibrator = TvAHistogramBinning(n_bins=n_bins, equal_mass=False)

            
            # Convert to numpy for calibration
            cal_orig_confidences = np.max(cal_pred_probs, axis=1)
            
            cal_correctness = (cal_pred_labels == cal_true_labels).astype(int)

            tva_hb_calibrator = tva_hb_calibrator.fit(cal_orig_confidences, cal_correctness)
            
            self.confidence_calibrators[model_id] = tva_hb_calibrator
            self.logger.info(f"Calibration complete for Model No {model_id} : {model.name}")
            # Evaluate confidence metrics on in-distribution test set
            self.logger.info(f"Evaluating confidence metrics on in-distribution test set...")
            iid_evaluator = Evaluator(model, device=self.config.device,
                      test_loader=self.dataloaders['test'],
                      label_mapping=self.label2idx)
            iid_pred_labels, iid_pred_probs, iid_true_labels, iid_pred_logits = iid_evaluator._predict_all()
            iid_orig_confidences = np.max(iid_pred_probs, axis=1)
            iid_calibrated_confidences = tva_hb_calibrator.transform(iid_orig_confidences)

            confidence_evaluator = []
            confidence_evaluator = ConfidenceEvaluator(
                y_true = iid_true_labels,
                y_pred = iid_pred_labels,
                orig_confidences = iid_orig_confidences,
                cal_confidences = iid_calibrated_confidences,
                n_bins = n_bins,
                
            )
            confidence_evaluator.compare_metrics_orig_vs_calibrated()
            confidence_evaluator.compare_reliability_diagrams(n_bins=n_bins)
            _ = confidence_evaluator.plot_annotation_reduction_stats()

            # Evaluate confidence metrics on out-of-distribution test set
            ood_evaluator = Evaluator(model, device=self.config.device,
                        test_loader=self.dataloaders['ood'],
                        label_mapping=self.label2idx)
            ood_pred_labels, ood_pred_probs, ood_true_labels, ood_pred_logits = ood_evaluator._predict_all()
            
            ood_orig_confidences = np.max(ood_pred_probs, axis=1)
        
            ood_calibrated_confidences = tva_hb_calibrator.transform(ood_orig_confidences)
            self.logger.info(f"Evaluating confidence metrics on out-of-distribution test set...")
            # Evaluate confidence metrics
            confidence_evaluator2 = []
            confidence_evaluator2 = ConfidenceEvaluator(
                y_true = ood_true_labels,
                y_pred = ood_pred_labels,
                orig_confidences = ood_orig_confidences,
                cal_confidences = ood_calibrated_confidences,
                n_bins = n_bins,
                
            )
            confidence_evaluator2.compare_metrics_orig_vs_calibrated()
            confidence_evaluator2.compare_reliability_diagrams(n_bins=n_bins)
            _ = confidence_evaluator2.plot_annotation_reduction_stats()

            #self.confidence_evaluators[model_id] = {}
            #self.confidence_evaluators[model_id]['iid'] = {
            #    'orig_confidences': iid_orig_confidences,
            #    'calibrated_confidences': iid_calibrated_confidences,
            #    'metrics': confidence_evaluator.metrics
            #}
            
            
            #self.evaluators[model_id]['id'] = evaluator.evaluate()

            
            
    def select_best_model(self):
        """Select the best performing model"""
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_models first.")
            
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_model = self.models[best_model_name]
        
        logger.info(f"Best model: {best_model_name} with accuracy: {self.results[best_model_name]['accuracy']:.4f}")
        
        return best_model_name, best_model
        
    def run_pipeline(self, model_configs: Dict[str, Dict]=MODELS_DICT, models_to_use: Optional[List[int]] = None):
        """Run the complete pretraining pipeline"""
        logger.info("🔬 Starting Pretraining Pipeline...")
        
        try:
            self.setup_datasets()
            self.setup_models(model_configs, models_to_use)
            self.train_models()
            
            # plot learning curves for all trained models


            self.evaluate_models()
            self.evaluate_confidences()
            #best_model_name, best_model = self.select_best_model()
        
            return {
            #    'best_model_name': best_model_name,
            #    'best_model': best_model,
            #    'all_results': self.results,
                'models': self.models
            }
        except Exception as e:
            logger.error(f"Pretraining pipeline failed: {e}")
            raise

class ProductionPipeline:
    """
    Pipeline for production simulation, deployment testing, and monitoring.
    Focuses on production readiness and performance.
    """
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".ProductionPipeline")

        self.logger.info(f"{'='*30}")
        self.logger.info(f"Initializing Production Pipeline execution with the followning configurations")
        
        if(self.config.device == ''):
            if torch.cuda.is_available():
                self.config.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.config.device = 'mps'
            else:
                self.config.device = 'cpu'
        self.logger.info(f"Device: {self.config.device}")        
    
        # load label2idx mapping
        try:
            with open(self.config.label2idx_path, 'r') as f:
                self.label2idx = json.load(f)
                self.num_classes = len(self.label2idx)
                self.idx2label = {v: k for k, v in self.label2idx.items()}
                self.logger.info(f"Loaded label mapping with {self.num_classes} classes")

        except FileNotFoundError:
            self.logger.error(f"Label mapping file not found: {self.config.label2idx_path}")
            self.logger.info(f"{'='*30}")
            exit(1)
        
        # Components
        self.production_datasets = {} # in-distribution and out-of-distribution datasets within a year
        self.production_dataloaders = {}

        self.uncertain_datasets = {}
        self.uncertain_dataloaders = {}
        
        self.production_model = None
        self.confidence_estimator = None
        self.production_evaluator = None
        self.fine_tuner = None
        self.monitoring_results = {}
        self.confidence_threshold = self.config.confidence_threshold  # default confidence threshold for monitoring
        self.performance_history = []
        self.retraining_history = []
        self.retraining_interval = self.config.retraining_interval  # in months

        self.current_dataset = []
        self.current_dataloader = None
        self.logger.info(f"{'='*30}")


    def setup_production_data(self):
        """Setup production-like dataset"""
        

        self.logger.info("Setting up datasets for production...")
        
        # in_loc is the in-distribution dataset, out_loc is the out-of-distribution dataset
        # in_loc will be used for inference, confidence estimation, monitoring, and fine-tuning
        # out_loc will be used entirely for model evaluation and monitoring
        self.production_datasets['in_loc'] = S3ImageWithTimeFeatureDataset(
            csv_path = self.config.production_data_path, 
            #label2idx_path = self.config.label2idx_path,
            label2idx = self.label2idx,  # use the loaded label2idx mapping
            images_path = self.config.preprocessed_images_path,
        )
        self.production_datasets['out_loc'] = S3ImageWithTimeFeatureDataset(
            csv_path = self.config.production_ood_data_path, 
            #label2idx_path = self.config.label2idx_path,
            label2idx = self.label2idx,  # use the loaded label2idx mapping
            images_path = self.config.preprocessed_images_path,
        )
        self.production_datasets['cal'] = S3ImageWithTimeFeatureDataset(
            csv_path = self.config.calibration_data_path, 
            #label2idx_path = self.config.label2idx_path,
            label2idx = self.label2idx,  # use the loaded label2idx mapping
            images_path = self.config.preprocessed_images_path,
        )

        self.logger.info("Setting up dataloaders for production...")

        self.production_dataloaders = {
            'in_loc': torch.utils.data.DataLoader(
                self.production_datasets['in_loc'], 
                batch_size=self.config.inference_batch_size, 
            ),
            'out_loc': torch.utils.data.DataLoader(
                self.production_datasets['out_loc'], 
                batch_size=self.config.inference_batch_size
            ),
            'cal': torch.utils.data.DataLoader(
                self.production_datasets['cal'], 
                batch_size=self.config.inference_batch_size
            ),
        }
        
            
        self.logger.info(f"Production (in loc) dataset loaded: {len(self.production_datasets['in_loc'])} samples")
        self.logger.info(f"Production (out loc) dataset loaded: {len(self.production_datasets['out_loc'])} samples")
        self.logger.info(f"Calibration dataset loaded: {len(self.production_datasets['cal'])} samples")
            
        
        
    def deploy_model(self, model: nn.Module, model_name: str):
        """Deploy model for production simulation"""
        self.logger.info(f"Deploying {model_name} for production simulation...")
        
        try:
            # Wrap model for production (add any production-specific logic)
            #self.production_model = ProductionModel(model, model_name)
            self.production_model = model
            self.production_model.to(self.config.device)
            self.production_model.eval()
            
            self.logger.info(f"Model {model_name} deployed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to deploy model {model_name}: {e}")
            raise
        
    def setup_confidence_estimation(self):
        """Setup confidence estimation"""
        self.logger.info("Setting up confidence estimation...")
        
        if self.production_model is None:
            raise ValueError("Deploy a model first before setting up confidence estimation")
            
        try:
            #self.confidence_estimator = ConfidenceEstimator(
            #    self.production_model.base_model,
            #    method=self.config.confidence_estimation_method,
            #    n_samples=self.config.confidence_n_samples
            #)
            self.logger.info(f"Setup calibration data from validation set for this model...")
            # Setup calibration data for this model
            cal_evaluator = Evaluator(self.production_model, device=self.config.device,
                      test_loader=self.production_dataloaders['cal'],
                      label_mapping=self.label2idx)
            
            cal_pred_labels, cal_pred_probs, cal_true_labels, cal_pred_logits = cal_evaluator._predict_all()

            # try TvAHistogramBinning
            n_bins = 15
            tva_hb_calibrator = []
            tva_hb_calibrator = TvAHistogramBinning(n_bins=n_bins, equal_mass=False)

            # Convert to numpy for calibration
            cal_orig_confidences = np.max(cal_pred_probs, axis=1)
            cal_correctness = (cal_pred_labels == cal_true_labels).astype(int)
            tva_hb_calibrator = tva_hb_calibrator.fit(cal_orig_confidences, cal_correctness)
            self.confidence_estimator = tva_hb_calibrator
            self.logger.info(f"Confidence estimation setup complete - Method: TvAHistogramBinning with {n_bins} bins")
            
        except Exception as e:
            self.logger.error(f"Failed to setup confidence estimation: {e}")
            raise

    
        
    def run_production_simulation(self) -> Dict[str, Any]:
        """Simulate production environment"""
        self.logger.info("Running production simulation...")
        
        try:
            # Production evaluation
            self.production_evaluator = Evaluator(self.production_model, device=self.config.device,
                      test_loader=self.production_dataloaders['in_loc'],
                      label_mapping=self.label2idx)

            # Run batch inference on production data using Evaluator
            self.logger.info("Runnning Batch Inference on Production Data")
            pred_labels, pred_probs, true_labels, pred_logits = self.production_evaluator._predict_all()
            
            # Calculate performance metrics for evaluation purpose only
            accuracy = accuracy_score(
                true_labels,
                pred_labels
            )

            self.production_evaluator2 = Evaluator(self.production_model, device=self.config.device,
                      test_loader=self.production_dataloaders['out_loc'],
                      label_mapping=self.label2idx)

            # Run batch inference on production data using Evaluator
            self.logger.info("Runnning Batch Inference on OOD Production Data")
            pred_labels2, pred_probs2, true_labels2, pred_logits2 = self.production_evaluator2._predict_all()
            
            # Calculate performance metrics for evaluation purpose only
            accuracy2 = accuracy_score(
                true_labels2,
                pred_labels2
            )
            
            self.logger.info(f"Current model accuracy on ID: {accuracy:.4f}")
            self.logger.info(f"Current model accuracy on OOD: {accuracy2:.4f}")

            # Run TvA Confidence Estimation using calibrated confidence estimator
            self.logger.info("Estimating confidence scores for production data...")
            if(self.confidence_estimator is None):
                self.setup_confidence_estimation()

            orig_confidences = np.max(pred_probs, axis=1)
            calibrated_confidences = self.confidence_estimator.transform(orig_confidences)
            uncertainty_mask = calibrated_confidences < self.confidence_threshold

            # add results to the dataset df for futher analysis
            idx2label = {v: k for k, v in self.label2idx.items()}
            self.production_datasets['in_loc'].df['pred_label'] = [idx2label[label] for label in pred_labels]
            self.production_datasets['in_loc'].df['orig_confidence'] = orig_confidences
            self.production_datasets['in_loc'].df['calibrated_confidence'] = calibrated_confidences
            self.production_datasets['in_loc'].df['uncertainty_mask'] = uncertainty_mask

            # save the results to a csv file, use os make dirs if necessary
            self.production_datasets['in_loc'].df.to_csv(
                self.config.production_predictions_path,
                index=False
            )
            self.logger.info(f"Production meta-data with predictions, confidences, and uncertainty saved to {self.config.production_predictions_path}")
            # show some statistics
            self.logger.info(f"Found {uncertainty_mask.sum()} uncertain samples ({uncertainty_mask.sum()/len(self.production_datasets['in_loc'].df)*100:.1f}%)")
            #self.logger.info(f"Evaluating confidence metrics on out-of-distribution test set...")
            #confidence_results = self.confidence_estimator.estimate_batch_confidence(
            #    self.production_dataloader
            #)
            self.logger.info("Preparing Dataset for Human Annotation...")
            self._prepare_dataset_for_fine_tuning()

            self.logger.info("Fine Tune Model on Newly Annotated Uncertain Samples...")
            self._fine_tune_model()
            # Monitoring simulation
            #monitoring_results = self._simulate_monitoring()
            pred_labels, pred_probs, true_labels, pred_logits = self.production_evaluator._predict_all()
            # Calculate performance metrics
            fine_tuned_accuracy = accuracy_score(
                true_labels,
                pred_labels
            )
            pred_labels2, pred_probs2, true_labels2, pred_logits2 = self.production_evaluator2._predict_all()
            # Calculate performance metrics
            fine_tuned_accuracy2 = accuracy_score(
                true_labels2,
                pred_labels2
            )
            
            self.logger.info(f"Fine-Tuned accuracy on ID: {fine_tuned_accuracy:.4f}")
            self.logger.info(f"Fine-Tuned accuracy on OOD: {fine_tuned_accuracy2:.4f}")
            # Log results
            improvement = fine_tuned_accuracy - accuracy
            improvement2 = fine_tuned_accuracy2 - accuracy2
            #logger.info(f"Retrained model accuracy: {retrained_accuracy:.4f}")
            self.logger.info(f"Improvement on ID: {improvement:.4f}")
            self.logger.info(f"Improvement on OOD: {improvement2:.4f}")

            
            # Store performance history
            #self.performance_history.append({
            #    'year_month': current_year_month,-
            #    'rolling_window_start': window_data['year_month'].min(),
            #    'rolling_window_end': current_year_month,
            #    'original_accuracy': accuracy,
            #    'retrained_accuracy': retrained_accuracy,
            #    'improvement': improvement,
            #    'uncertain_samples_count': len(uncertain_samples),
            #    'retraining_samples_count': len(retraining_data),
            #    'train_samples_count': len(retrain_train),
            #    'val_samples_count': len(retrain_val),
            #    'split_method': 'stratified',
            #    'retrain_train_samples': len(retrain_train),
            #    'retrain_val_samples': len(retrain_val)
            #})
            
            # Save checkpoint
            #checkpoint_path = f"model_{current_year_month.replace('-', '_')}.pth"
            #torch.save({
            #    'model_state_dict': self.model.state_dict(),
            #    'class_mapping': self.class_mapping,
            #    #'calibrator_temperature': self.calibrator.temperature,
            #    'year_month': current_year_month,
            #    'accuracy': retrained_accuracy
            #}, checkpoint_path)
            
            
            return {
             #   'production_metrics': prod_results,
             #   'confidence_scores': confidence_results,
              #  'monitoring_results': monitoring_results
            }
            
        except Exception as e:
            self.logger.error(f"Production simulation failed: {e}")
            raise

    def _fine_tune_model(self):
        """Fine tune the deployed model """

        self.logger.info("Freezing Early Layers...")
        # Freeze early layers for fine-tuning
        for name, param in self.production_model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
            else:
                self.logger.info(f"Unfreezing layer: {name}")
                param.requires_grad = True
        
        
        self.logger.info("Model Fine-Tuning Started...")


        # Setup trainer
        #optimizer = torch.optim.Adam(self.production_model.parameters(), lr=self.config.learning_rate, weight_decay= self.config.weight_decay)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.production_model.parameters()), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.1, patience = self.config.patience)
        criterion = self.config.criterion
        
        trainer = Trainer(self.production_model, optimizer, criterion, scheduler, device=self.config.device)
        self.fine_tuner = trainer
        
        # Train
        trainer.fit(
            self.uncertain_dataloaders['train'], 
            self.uncertain_dataloaders['val'], 
            epochs=self.config.epochs
        )
        self.logger.info(f"Production Model {self.production_model.name} Fine Tuning Completed successfully.")
        # Save the historical learning curves and the best model weights of the trained model
        # Dump training results to config training_output_dir, using os and makedirs
        # create subdirectory for each model using model_id
        # get model_id from the model name
        output_dir = Path(self.config.training_output_dir) / f"model_{self.production_model.name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save model state
        model_save_path = output_dir / f"best_model.pth"
        torch.save(trainer.best_model_weights, model_save_path)
        # dump learning history to json
        history_save_path = output_dir / f"training_history.json"
        with open(history_save_path, 'w') as f:
            json.dump(trainer.training_history, f, indent=4)
        #self.logger.info(f"Model No {model_id} : {model.name} saved to {model_save_path}")
        self.logger.info(f"Training history saved to {history_save_path}")

        # plot loss and accuracy curves
        trainer.plot_metrics()

        # update model with the best weights
        self.production_model.load_state_dict(trainer.best_model_weights)
        #self.models[model_id] = model  # update the model in the dictionary
        self.logger.info(f"Deployed Model {self.production_model.name} loaded with best weights.")

    def _prepare_dataset_for_fine_tuning(self):
        """Prepare dataset for fine-tuning based on uncertainty"""
        self.logger.info("Preparing dataset for fine-tuning...")
        
        # Filter uncertain samples
        uncertain_data = self.production_datasets['in_loc'].df[self.production_datasets['in_loc'].df['uncertainty_mask']]
        
        if len(uncertain_data) == 0:
            self.logger.warning("No uncertain samples found for fine-tuning.")
            return pd.DataFrame(), pd.DataFrame()
        self.logger.info(f"Found {len(uncertain_data)} uncertain samples for fine-tuning.")
        
        # Split uncertain samples into training and validation sets
        train_uncertain, val_uncertain = self._stratified_uncertain_samples_split(uncertain_data)
        
        self.logger.info(f"Prepared {len(train_uncertain)} training and {len(val_uncertain)} validation samples for fine-tuning.")

        # load using CustomDataset class and dataloaders
        self.uncertain_datasets['train'] = S3ImageWithTimeFeatureDataset(
            meta_df = train_uncertain, 
            label2idx= self.label2idx,
            images_path = self.config.preprocessed_images_path)

        self.uncertain_datasets['val'] = S3ImageWithTimeFeatureDataset(
            meta_df = val_uncertain, 
            label2idx=self.label2idx,
            images_path = self.config.preprocessed_images_path)
        
        self.uncertain_dataloaders = {
            'train': torch.utils.data.DataLoader(
                self.uncertain_datasets['train'], 
                batch_size=self.config.batch_size, 
                shuffle=True
            ),
            'val': torch.utils.data.DataLoader(
                self.uncertain_datasets['val'], 
                batch_size=self.config.batch_size
            )
        }
        
        return train_uncertain, val_uncertain
    
    
    def _stratified_uncertain_samples_split(self, uncertain_data: pd.DataFrame, 
                                            split_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        from sklearn.model_selection import train_test_split
        """Split uncertain samples using stratification"""
        
        if len(uncertain_data) == 0:
            return pd.DataFrame(), pd.DataFrame()
        
        # Minimum samples needed for stratified split
        min_samples_needed = max(10, int(2 / split_ratio))
        
        if len(uncertain_data) < min_samples_needed:
            self.logger.warning(f"Too few uncertain samples ({len(uncertain_data)}) for stratified split, using all for training")
            return uncertain_data, uncertain_data.sample(min(5, len(uncertain_data)), random_state=42)
        
        # Check class distribution
        class_counts = uncertain_data['label'].value_counts()
        min_samples_per_class = max(2, int(1 / split_ratio) + 1)
        
        classes_with_insufficient_samples = class_counts[class_counts < min_samples_per_class].index.tolist()
        
        if len(classes_with_insufficient_samples) > 0:
            self.logger.warning(f"Some classes have insufficient samples for stratification: {classes_with_insufficient_samples}")
            self.logger.warning("Using random split instead")
            
            # Random split fallback
            train_uncertain, val_uncertain = train_test_split(
                uncertain_data, 
                test_size=split_ratio, 
                random_state=42
            )
        else:
            try:
                # Stratified split
                train_uncertain, val_uncertain = train_test_split(
                    uncertain_data,
                    test_size=split_ratio,
                    stratify=uncertain_data['label'],
                    random_state=42
                )
                self.logger.info("Successfully performed stratified split on uncertain samples")
            except ValueError as e:
                self.logger.warning(f"Stratified split on uncertain samples failed: {e}")
                # Random split fallback
                train_uncertain, val_uncertain = train_test_split(
                    uncertain_data, 
                    test_size=split_ratio, 
                    random_state=42
                )
        
        return train_uncertain, val_uncertain
        
    def _simulate_monitoring(self) -> Dict[str, Any]:
        """Simulate production monitoring"""
        self.logger.info("Simulating production monitoring...")
        
        try:
            # Simulate various production scenarios
            monitoring_results = {
                'latency_tests': self._test_inference_latency(),
                'memory_usage': self._test_memory_usage(),
                'throughput': self._test_throughput(),
                'drift_detection': self._test_data_drift()
            }
            
            return monitoring_results
            
        except Exception as e:
            self.logger.error(f"Monitoring simulation failed: {e}")
            raise
        
    def run_pipeline(self, model: nn.Module, model_name: str) -> Dict[str, Any]:
        """Run the complete production pipeline"""
        self.logger.info("🚀 Starting Production Pipeline...")
        
        try:
            # Setup production environment
            self.setup_production_data()
            self.deploy_model(model, model_name)
            self.setup_confidence_estimation()
            
            # Run simulation
            results = self.run_production_simulation()
            
            #results = self.periodic_retraining_simulation()
            # Assess deployment readiness
            #deployment_assessment = self.assess_deployment_readiness(results)
            #results['deployment_assessment'] = deployment_assessment
            
            #self.logger.info(f"Production pipeline completed - "
            #               f"Deployment Ready: {deployment_assessment['deployment_ready']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Production pipeline failed: {e}")
            raise

    #### import from original production_pipeline implementation
    def periodic_retraining_simulation(self):
        """Simulate periodic retraining with rolling windows and stratified splitting"""
        
        self.logger.info(f"Starting periodic retraining simulation:")
        # get minimum year_month from self.production_datasets['in_loc']
        df = self.production_datasets['in_loc'].df
        if df.empty:
            self.logger.error("No data available for periodic retraining simulation")
            return
        start_year_month = df['year_month'].min()
        self.logger.info(f"  Start Year Month: {start_year_month} ")
        self.logger.info(f"  Retrain period: every {self.retraining_interval} month(s)")
        #self.logger.info(f"  Rolling window: {self.rolling_window_months} month(s)")
        #self.logger.info(f"  Validation split: {self.validation_split_ratio} (stratified)")
        
        # Start simulation to start_year_month + self.retraining_interval months
        from datetime import datetime, timedelta
        from dateutil.relativedelta import relativedelta
        start_year_month = datetime.strptime(start_year_month, '%Y-%m')
        start_year_month += relativedelta(months=self.retraining_interval)

        current_year_month = start_year_month
        end_year_month = df['year_month'].max()
        while current_year_month < end_year_month:
            logger.info(f"{'='*30}")
            logger.info(f"RETRAINING SESSION: {current_year_month} to {current_year_month + relativedelta(months=self.retraining_interval)}")
            logger.info(f"{'='*30}")
            
            # Step 1: Get rolling window data with stratified split 
            logger.info("STEP 1: Filter production...")
            #train_data, val_data = self.get_rolling_window_data_with_stratification(df, current_year_month)
            window_data = self.get_rolling_window_data(df, current_year_month)
            
            
            # Step 2: Perform Batch inference on most recent window data using latest model
            logger.info("STEP 2: Perform Batch Transform on Window Data using latest model...")
            uncertain_samples, predictions_df = self.identify_uncertain_samples(window_data)
            
            # Calculate performance metrics
            accuracy = accuracy_score(
                predictions_df['label'], 
                predictions_df['pred_label']
            )
            
            logger.debug(f"Current model accuracy on {current_year_month}: {accuracy:.4f}")
            logger.info(f"Uncertain samples: {len(uncertain_samples)} ({len(uncertain_samples)/len(window_data)*100:.1f}%)")
            
            # Step 3: Request for annotation of uncertain samples
            # Note: In a real scenario, this would involve human annotation or external service
            # For simulation, we assume uncertain samples are annotated and added to training set
            logger.info("STEP 3: Request for annotation of uncertain samples...")
            #uncertain_train_samples, _ = self.identify_uncertain_samples(train_data)
            
            # Combine uncertain samples for retraining
            #retraining_data = pd.concat([uncertain_train_samples, uncertain_samples], ignore_index=True)
            retraining_data = uncertain_samples.copy()

            if len(retraining_data) < 10:  # Minimum samples for retraining
                logger.warning("Too few uncertain samples for retraining")
                
                # Store results even without retraining
                self.performance_history.append({
                    'year_month': current_year_month,
                    'rolling_window_start': retraining_data['year_month'].min() if len(retraining_data) > 0 else None,
                    'rolling_window_end': current_year_month,
                    'original_accuracy': accuracy,
                    'retrained_accuracy': accuracy,
                    'improvement': 0.0,
                    'uncertain_samples_count': len(uncertain_samples),
                    'retraining_samples_count': 0,
                    #'train_samples_count': len(train_data),
                    #'val_samples_count': len(val_data),
                    #'split_method': 'stratified'
                })
                continue
            
            
            
            # Step 4: Stratified split of retraining data
            logger.info(f"STEP 4: Retraining on {len(retraining_data)} annotated uncertain samples within the rolling window")
            retrain_train, retrain_val = self.stratified_uncertain_samples_split(
                retraining_data, split_ratio=self.validation_split_ratio
            )

            retrain_train_dataset = S3ImageWithTimeFeatureDataset(
                meta_df = retrain_train, 
                label2idx=self.class_mapping, 
                session = self.session)
            
            retrain_val_dataset = S3ImageWithTimeFeatureDataset(
                meta_df = retrain_val, 
                label2idx=self.class_mapping, 
                session = self.session)
            
            
            
            retrain_train_loader = DataLoader(
                retrain_train_dataset, 
                batch_size=min(self.config['batch_size'], len(retrain_train_dataset)),
                shuffle=True, num_workers=10
            )
              
            retrain_val_loader = DataLoader(
                retrain_val_dataset,
                batch_size=min(self.config['batch_size'], len(retrain_val_dataset)), 
                shuffle=False, num_workers=8
            )
            
            # Step 5: Fine-tune model
            logger.info("STEP 5: Fine-tuning model...")
            self.fine_tune_model(
                retrain_train_loader, 
                retrain_val_loader,
                epochs=self.config['epochs']
            )
            
            # Step 6: Re-calibrate confidence estimator
            #logger.info("STEP 6: Re-calibrating confidence estimator...")
            #self.calibrate_confidence_estimator(retrain_val_loader)
            
            # Step 6: RE-EVALUATE the retrained model on the entire rolling window to check for improvement
            logger.info("STEP 6: Re-evaluating retrained model on rolling window data...")
            _, retrained_predictions_df = self.identify_uncertain_samples(window_data)
            
            # Calculate performance metrics
            retrained_accuracy = accuracy_score(
                retrained_predictions_df['label'], 
                retrained_predictions_df['pred_label']
            )
            
            logger.debug(f"retrained model accuracy on {current_year_month}: {retrained_accuracy:.4f}")

            
            # Log results
            improvement = retrained_accuracy - accuracy
            logger.info(f"Retrained model accuracy: {retrained_accuracy:.4f}")
            logger.info(f"Improvement: {improvement:.4f}")
            
            # Store performance history
            self.performance_history.append({
                'year_month': current_year_month,
                'rolling_window_start': window_data['year_month'].min(),
                'rolling_window_end': current_year_month,
                'original_accuracy': accuracy,
                'retrained_accuracy': retrained_accuracy,
                'improvement': improvement,
                'uncertain_samples_count': len(uncertain_samples),
                'retraining_samples_count': len(retraining_data),
                'train_samples_count': len(retrain_train),
                'val_samples_count': len(retrain_val),
                'split_method': 'stratified',
                'retrain_train_samples': len(retrain_train),
                'retrain_val_samples': len(retrain_val)
            })
            
            # Save checkpoint
            checkpoint_path = f"model_{current_year_month.replace('-', '_')}.pth"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'class_mapping': self.class_mapping,
                #'calibrator_temperature': self.calibrator.temperature,
                'year_month': current_year_month,
                'accuracy': retrained_accuracy
            }, checkpoint_path)
            
            logger.info(f"Retraining session {current_year_month} completed. Model saved to {checkpoint_path}")
        
        logger.info("Periodic retraining simulation completed!")

class MasterMLSystem:
    """
    Ultimate ML System that orchestrates both pretraining and production pipelines.
    This is the main class that combines all components and pipelines.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
        # Setup logging first
        self.config.setup_system_logging()
        self.logger = logging.getLogger(__name__ + ".MasterMLSystem")
        
        # Set random seeds
        self.config.set_random_seeds()
        
        # Initialize pipelines
        self.pretraining_pipeline = PretrainingPipeline(config) if config.enable_pretraining_pipeline else None
        self.production_pipeline = ProductionPipeline(config) if config.enable_production_pipeline else None
        
        # Results storage
        self.system_results = {}
        
        self.logger.info("=" * 100)
        self.logger.info("MASTER ML SYSTEM INITIALIZED")
        self.logger.info("=" * 100)
        self.logger.info(f"Project: {config.project_name}")
        self.logger.info(f"Experiment: {config.experiment_name}")
        self.logger.info(f"Device: {config.device}")
        self.logger.info(f"Pretraining Pipeline: {'Enabled' if self.pretraining_pipeline else 'Disabled'}")
        self.logger.info(f"Production Pipeline: {'Enabled' if self.production_pipeline else 'Disabled'}")
        self.logger.info(f"Output Directory: {config.output_dir}")
        
    def run_pretraining_phase(self, model_configs: Dict[str, Dict], **kwargs) -> Optional[Dict[str, Any]]:
        """Run the pretraining/research phase"""
        if not self.pretraining_pipeline:
            self.logger.warning("Pretraining pipeline is disabled")
            return None
            
        self.logger.info("=" * 80)
        self.logger.info("STARTING PRETRAINING PHASE")
        self.logger.info("=" * 80)
        
        try:
            pretraining_results = self.pretraining_pipeline.run_pipeline(model_configs, **kwargs)
            self.system_results['pretraining'] = pretraining_results
            
            self.logger.info("✅ Pretraining phase completed successfully!")
            self.logger.info(f"Best model: {pretraining_results['best_model_name']} - "
                           f"Accuracy: {pretraining_results['all_results'][pretraining_results['best_model_name']]['accuracy']:.4f}")
            
            return pretraining_results
            
        except Exception as e:
            self.logger.error(f"❌ Pretraining phase failed: {e}")
            raise
        
    def run_production_phase(self, model: Optional[nn.Module] = None, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Run the production simulation phase"""
        if not self.production_pipeline:
            self.logger.warning("Production pipeline is disabled")
            return None
            
        self.logger.info("=" * 80)
        self.logger.info("STARTING PRODUCTION PHASE")
        self.logger.info("=" * 80)
        
        # Use best model from pretraining if not provided
        if model is None and 'pretraining' in self.system_results:
            model_name = self.system_results['pretraining']['best_model_name']
            model = self.system_results['pretraining']['best_model']
            self.logger.info(f"Using best model from pretraining: {model_name}")
        elif model is None:
            raise ValueError("No model provided and no pretraining results available")
            
        try:
            production_results = self.production_pipeline.run_pipeline(model, model_name)
            self.system_results['production'] = production_results
            
            deployment_ready = production_results['deployment_assessment']['deployment_ready']
            self.logger.info(f"✅ Production phase completed successfully!")
            self.logger.info(f"Deployment ready: {'Yes' if deployment_ready else 'No'}")
            
            return production_results
            
        except Exception as e:
            self.logger.error(f"❌ Production phase failed: {e}")
            raise
        
    def run_complete_system(self, model_configs: Dict[str, Dict], **kwargs) -> Dict[str, Any]:
        """Run both pretraining and production phases"""
        self.logger.info("🌟 STARTING COMPLETE ML SYSTEM")
        self.logger.info("=" * 100)
        
        try:
            # Phase 1: Pretraining/Research
            pretraining_results = self.run_pretraining_phase(model_configs, **kwargs)
            
            # Phase 2: Production Simulation
            production_results = self.run_production_phase()
            
            # Generate final report
            final_report = self._generate_final_report()
            
            self.logger.info("🎉 COMPLETE ML SYSTEM FINISHED SUCCESSFULLY!")
            return final_report
            
        except Exception as e:
            self.logger.error(f"❌ Complete system execution failed: {e}")
            raise
        
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        self.logger.info("Generating final system report...")
        
        report = {
            'system_config': {
                'project_name': self.config.project_name,
                'experiment_name': self.config.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'device': self.config.device
            },
            'pretraining_summary': self._summarize_pretraining(),
            'production_summary': self._summarize_production(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_path = f"{self.config.output_dir}/final_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Save system results
        results_path = f"{self.config.output_dir}/system_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(self.system_results, f)
            
        self.logger.info(f"Final report saved to: {report_path}")
        self.logger.info(f"System results saved to: {results_path}")
        
        return report
        
    def _summarize_pretraining(self) -> Dict[str, Any]:
        """Summarize pretraining results"""
        if 'pretraining' not in self.system_results:
            return {'status': 'skipped'}
            
        results = self.system_results['pretraining']
        return {
            'status': 'completed',
            'best_model': results['best_model_name'],
            'best_accuracy': results['all_results'][results['best_model_name']]['accuracy'],
            'models_compared': len(results['models']),
            'all_accuracies': {name: res['accuracy'] for name, res in results['all_results'].items()}
        }
        
    def _summarize_production(self) -> Dict[str, Any]:
        """Summarize production results"""
        if 'production' not in self.system_results:
            return {'status': 'skipped'}
            
        results = self.system_results['production']
        monitoring = results['monitoring_results']
        
        return {
            'status': 'completed',
            'production_accuracy': results['production_metrics']['accuracy'],
            'mean_confidence': results['confidence_scores']['mean_confidence'],
            'mean_latency_ms': monitoring['latency_tests']['mean_latency'] * 1000,
            'throughput_batches_per_sec': monitoring['throughput']['batches_per_second'],
            'deployment_ready': results['deployment_assessment']['deployment_ready'],
            'checks_passed': results['deployment_assessment']['passed_checks'],
            'total_checks': results['deployment_assessment']['total_checks']
        }
        
    def _generate_recommendations(self) -> List[str]:
        """Generate system recommendations"""
        recommendations = []
        
        if 'pretraining' in self.system_results and 'production' in self.system_results:
            pretrain_acc = self.system_results['pretraining']['all_results'][
                self.system_results['pretraining']['best_model_name']]['accuracy']
            prod_acc = self.system_results['production']['production_metrics']['accuracy']
            
            acc_drop = pretrain_acc - prod_acc
            if acc_drop > 0.05:
                recommendations.append(f"Significant accuracy drop in production ({acc_drop:.3f}). Check for data drift or distribution shift.")
            elif acc_drop > 0.02:
                recommendations.append(f"Minor accuracy drop in production ({acc_drop:.3f}). Monitor closely.")
            else:
                recommendations.append("Accuracy maintained well in production environment.")
                
            # Performance recommendations
            latency = self.system_results['production']['monitoring_results']['latency_tests']['mean_latency']
            if latency > 0.1:  # 100ms threshold
                recommendations.append("High inference latency detected. Consider model optimization or hardware upgrade.")
            else:
                recommendations.append("Inference latency is acceptable for real-time applications.")
                
            # Deployment readiness
            deployment_ready = self.system_results['production']['deployment_assessment']['deployment_ready']
            if deployment_ready:
                recommendations.append("✅ System performance looks good! Ready for deployment.")
            else:
                failed_checks = []
                checks = self.system_results['production']['deployment_assessment']['checks']
                for check_name, check_result in checks.items():
                    if not check_result['passed']:
                        failed_checks.append(check_name)
                recommendations.append(f"❌ Failed deployment checks: {', '.join(failed_checks)}. Address these issues before deployment.")
                
        elif 'pretraining' in self.system_results:
            recommendations.append("Pretraining completed successfully. Consider running production simulation.")
        elif 'production' in self.system_results:
            recommendations.append("Production simulation completed. Results are based on provided model.")
        else:
            recommendations.append("No pipeline results available.")
            
        return recommendations

    def save_model(self, model_name: str, save_path: Optional[str] = None) -> str:
        """Save a trained model"""
        if 'pretraining' not in self.system_results:
            raise ValueError("No pretraining results available")
            
        if model_name not in self.system_results['pretraining']['detailed_results']:
            raise ValueError(f"Model {model_name} not found in results")
            
        model_result = self.system_results['pretraining']['detailed_results'][model_name]
        model = model_result['model']
        
        if save_path is None:
            save_path = f"{self.config.output_dir}/saved_models/{model_name}.pth"
            
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model with additional metadata
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'model_class': model.__class__.__name__,
            'accuracy': model_result['test_results']['accuracy'],
            'trainer_config': model_result['trainer_config'],
            'timestamp': datetime.now().isoformat()
        }, save_path)
        
        self.logger.info(f"Model {model_name} saved to: {save_path}")
        return save_path

    def load_model(self, model_path: str) -> tuple:
        """Load a saved model"""
        checkpoint = torch.load(model_path, map_location=self.config.device)
        
        model_name = checkpoint.get('model_name', 'unknown')
        model_class_name = checkpoint.get('model_class', 'unknown')
        accuracy = checkpoint.get('accuracy', 0.0)
        
        self.logger.info(f"Loaded model: {model_name} ({model_class_name}) - Accuracy: {accuracy:.4f}")
        
        return checkpoint, model_name

if __name__ == "__main__":
    # Example usage
    config = SystemConfig(
        project_name="example_ml_system",
        experiment_name="test_run",
        enable_pretraining_pipeline=True,
        enable_production_pipeline=True,
        epochs=10,  # Quick test
        output_dir="./test_outputs"
    )
    
    # Example model configurations
    model_configs: Dict[int, Dict] = MODELS_DICT





    
    # Initialize and run system
    system = MasterMLSystem(config)
    
    try:
        results = system.run_complete_system(model_configs)
        print("System execution completed successfully!")
    except Exception as e:
        print(f"System execution failed: {e}")