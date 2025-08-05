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
from custom_models import AnimalClassifier, AnimalTemporalClassifier, CustomClassifier, WrapperModel
from custom_datasets import S3ImageWithTimeFeatureDataset
from custom_losses import CrossEntropyMarginLoss
from trainer import WildScanTrainer as Trainer
from evaluator import WildScanEvaluator as Evaluator

import matplotlib.pyplot as plt

import logging
# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
)
logger = logging.getLogger(__name__)

MODELS_DICT = {
        0: {
            'name' : 'BuiltFromScratch', 
            'class' : CustomClassifier, 
            'image_shape' : (3, 224, 224),  # input image shape
            'aux_dim' : None,  # no auxiliary input
            'final_activation' : None,  # returns logits due to nn.CrossEntropyLoss
            },
        1: {
            'name' : 'AnimalClassifier_ResNet18', 
            'class' : AnimalClassifier,
            'num_classes' : 15,
            #'image_shape' : (3, 224, 224),  # input image shape
            #'aux_dim' : None,  # no auxiliary input
            #'final_activation' : None,  # returns logits due to nn.CrossEntropyLoss 
            
            },
        2: {
            'name' : 'AnimalTemporalClassifier_ResNet18', 
            'class' : AnimalTemporalClassifier, 
            #'image_shape' : (3, 224, 224),  # input image shape
            #'aux_dim' : (1, 4),  # cyclical time features (e.g., day of year, hour of day, etc.)
            'num_classes' : 15,  # number of classes for classification
            'proj_dim' : 256,  # projection dimension for each of the image vector and temporal features for fusion
            #'fc_hidden_layer': 128,  # hidden layer size for final classification
            #'final_activation' : None,  # returns logits due to use of nn.CrossEntropyLoss 
            },
        3: {
            'name' : 'AnimalClassifier_Resnet50', 
            'class' : WrapperModel, 
            'backbone' : 'resnet50',  # backbone model
            #'image_shape' : (3, 224, 224),  # input image shape
            #'aux_dim' : (1, 4),  # cyclical time features (e.g., day of year, hour of day, etc.)
            'num_classes' : 15,  # number of classes for classification
            
            #'fc_hidden_layer': 128,  # hidden layer size for final classification
            #'final_activation' : None,  # returns logits due to use of nn.CrossEntropyLoss 
            },
        4: {
            'name' : 'AnimalClassifier_Vgg16', 
            'class' : WrapperModel, 
            'backbone' : 'vgg16',  # backbone model
            #'image_shape' : (3, 224, 224),  # input image shape
            #'aux_dim' : (1, 4),  # cyclical time features (e.g., day of year, hour of day, etc.)
            'num_classes' : 15,  # number of classes for classification
            
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
    production_data_path: str = ""

    # Preprocessed images path
    preprocessed_images_path: str = "./preprocessed_images"

    # label mapping path
    label2idx_path: str = "./label2idx.json"
    
    # Models Configurations
    model_configs: Dict[int, Dict] = field(default_factory=lambda:MODELS_DICT)
    
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
        
class PretrainingPipeline:
    """
    Pipeline for research, experimentation, and model comparison.
    Focuses on model development and evaluation.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        

    
        # Components
        self.datasets = {}
        self.dataloaders = {}
        self.models = {}
        self.trainers = {}
        self.evaluators = {}
        self.results = {}
        self.default_trainer_config = self.get_default_trainer_config()
        
    
    def get_default_trainer_config(self) -> TrainerConfig:
        return TrainerConfig(
            device=self.config.device if hasattr(self.config, 'device') else 'cpu'
        )
        
    def setup_datasets(self):
        """Setup datasets for pretraining pipeline"""
        logger.info("Setting up datasets for pretraining...")
        
        self.datasets['train'] = S3ImageWithTimeFeatureDataset(
            csv_path = self.config.train_data_path, 
            label2idx_path = self.config.label2idx_path,
            images_path = self.config.preprocessed_images_path,
        )
        self.datasets['val'] = S3ImageWithTimeFeatureDataset(
            csv_path = self.config.val_data_path, 
            label2idx_path = self.config.label2idx_path,
            images_path = self.config.preprocessed_images_path,
        )
        self.datasets['test'] = S3ImageWithTimeFeatureDataset(
            csv_path = self.config.test_data_path, 
            label2idx_path = self.config.label2idx_path,
            images_path = self.config.preprocessed_images_path,
        )
        logger.info(f"Train dataset size: {len(self.datasets['train'])}")
        logger.info(f"Validation dataset size: {len(self.datasets['val'])}")
        logger.info(f"Test dataset size: {len(self.datasets['test'])}")
        
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
            )
        }
        
    def setup_models(self, model_configs: Dict[int, Dict], models_to_use: Optional[List[int]] = None):
        """Setup multiple models for comparison"""
        if models_to_use is None:
            models_to_use = self.config.models_to_use
        
        # log number of models to setup based on models_to_use
        model_configs = {k: v for k, v in model_configs.items() if k in models_to_use}
        logger.info(f"Setting up {len(model_configs)} models...")
        
        for model_id, model_config in model_configs.items():
            logger.info(f"Setting up {len(model_configs)} models: {model_config['name']} (ID: {model_id})")
            model_class = model_config.pop('class')
            logger.debug(f"Initializing model {model_config['name']} with class {model_class.__name__}")
            self.models[model_id] = model_class(**model_config)
            #model_name = model_config['name']
            #self.models[model_name] = self.models[model_id]  # map model_id to model_name
            logger.info(f"Initialized Model No {model_id}: {model_class.__name__}")
            
    def train_models(self):
        """Train the specified models"""
        logger.info(f"Found {len(self.models.items())} model(s) to train.")
        logger.info("Starting model training...")
        
        for model_id, model in self.models.items():
            logger.info(f"Training Model No {model_id} : {model.name}...")
            
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
            logger.info(f"Model No {model_id} : {model.name} trained successfully.")
            # Save the historical learning curves and the best model weights of the trained model
            # Dump training results to config training_output_dir, using os and makedirs
            # create subdirectory for each model using model_id
            output_dir = Path(self.config.training_output_dir) / f"model_{model_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            # Save model state
            model_save_path = output_dir / f"best_model.pth"
            torch.save(model.state_dict(), model_save_path)
            # dump learning history to json
            history_save_path = output_dir / f"training_history.json"
            with open(history_save_path, 'w') as f:
                json.dump(trainer.training_history, f, indent=4)
            logger.info(f"Model No {model_id} : {model.name} saved to {model_save_path}")
            logger.info(f"Training history saved to {history_save_path}")

            # plot loss and accuracy curves
            trainer.plot_metrics()

        
    def evaluate_models(self):
        """Evaluate all trained models"""
        logger.info("Evaluating models...")

        if not self.models:
            raise ValueError("No models available for evaluation. Run setup_models and train_models first.")
        if 'test' not in self.dataloaders:
            raise ValueError("Test dataloader not found. Ensure datasets are set up correctly.")
        logger.info(f"Found {len(self.models.items())} model(s) to evaluate.")

        # pass all models and dataset to WildScanEvaluator
        evaluator = Evaluator(self.models, device=self.config.device)


        for model_name, model in self.models.items():
            evaluator = WildScanEvaluator(model, device=self.config.device)
            self.evaluators[model_name] = evaluator
            
            # Evaluate on test set
            results = evaluator.comprehensive_evaluation(self.dataloaders['test'])
            self.results[model_name] = results
            
            logger.info(f"{model_name} - Test Accuracy: {results['accuracy']:.4f}")
            
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


            #self.evaluate_models()
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
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".ProductionPipeline")
        
        # Components
        self.production_dataset = None
        self.production_dataloader = None
        self.production_model = None
        self.confidence_estimator = None
        self.production_evaluator = None
        self.monitoring_results = {}
        
    def setup_production_data(self):
        """Setup production-like dataset"""
        self.logger.info("Setting up production dataset...")
        
        try:
            self.production_dataset = ProductionDataset(
                self.config.production_data_path,
                batch_size=self.config.production_batch_size
            )
            
            self.production_dataloader = torch.utils.data.DataLoader(
                self.production_dataset,
                batch_size=self.config.production_batch_size,
                shuffle=False  # Production data should maintain order
            )
            
            self.logger.info(f"Production dataset loaded: {len(self.production_dataset)} samples")
            
        except Exception as e:
            self.logger.error(f"Failed to setup production data: {e}")
            raise
        
    def deploy_model(self, model: nn.Module, model_name: str):
        """Deploy model for production simulation"""
        self.logger.info(f"Deploying {model_name} for production simulation...")
        
        try:
            # Wrap model for production (add any production-specific logic)
            self.production_model = ProductionModel(model, model_name)
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
            self.confidence_estimator = ConfidenceEstimator(
                self.production_model.base_model,
                method=self.config.confidence_estimation_method,
                n_samples=self.config.confidence_n_samples
            )
            
            self.logger.info(f"Confidence estimation setup complete - Method: {self.config.confidence_estimation_method}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup confidence estimation: {e}")
            raise
        
    def run_production_simulation(self) -> Dict[str, Any]:
        """Simulate production environment"""
        self.logger.info("Running production simulation...")
        
        try:
            # Production evaluation
            self.production_evaluator = ProductionEvaluator(
                self.production_model,
                device=self.config.device
            )
            
            # Run comprehensive production tests
            prod_results = self.production_evaluator.evaluate_production_readiness(
                self.production_dataloader
            )
            
            # Confidence estimation
            confidence_results = self.confidence_estimator.estimate_batch_confidence(
                self.production_dataloader
            )
            
            # Monitoring simulation
            monitoring_results = self._simulate_monitoring()
            
            return {
                'production_metrics': prod_results,
                'confidence_scores': confidence_results,
                'monitoring_results': monitoring_results
            }
            
        except Exception as e:
            self.logger.error(f"Production simulation failed: {e}")
            raise
        
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
        
    def _test_inference_latency(self) -> Dict[str, float]:
        """Test inference latency"""
        import time
        
        self.logger.debug("Testing inference latency...")
        
        latencies = []
        sample_batch = next(iter(self.production_dataloader))
        
        # Warm up
        for _ in range(10):
            with torch.no_grad():
                _ = self.production_model(sample_batch[0].to(self.config.device))
        
        # Actual timing
        for _ in range(100):  # Run 100 inference tests
            start_time = time.time()
            with torch.no_grad():
                _ = self.production_model(sample_batch[0].to(self.config.device))
            end_time = time.time()
            latencies.append(end_time - start_time)
            
        return {
            'mean_latency': sum(latencies) / len(latencies),
            'max_latency': max(latencies),
            'min_latency': min(latencies),
            'std_latency': torch.std(torch.tensor(latencies)).item()
        }
        
    def _test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage"""
        self.logger.debug("Testing memory usage...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # Run inference
            sample_batch = next(iter(self.production_dataloader))
            with torch.no_grad():
                _ = self.production_model(sample_batch[0].to(self.config.device))
                
            peak_memory = torch.cuda.max_memory_allocated()
            
            return {
                'initial_memory_mb': initial_memory / (1024**2),
                'peak_memory_mb': peak_memory / (1024**2),
                'memory_increase_mb': (peak_memory - initial_memory) / (1024**2)
            }
        else:
            return {'message': 'CUDA not available, memory testing skipped'}
            
    def _test_throughput(self) -> Dict[str, float]:
        """Test throughput"""
        import time
        
        self.logger.debug("Testing throughput...")
        
        start_time = time.time()
        batch_count = 0
        
        for batch in self.production_dataloader:
            with torch.no_grad():
                _ = self.production_model(batch[0].to(self.config.device))
            batch_count += 1
            if batch_count >= 20:  # Test with 20 batches
                break
                
        end_time = time.time()
        total_time = end_time - start_time
        
        return {
            'batches_processed': batch_count,
            'total_time_seconds': total_time,
            'batches_per_second': batch_count / total_time if total_time > 0 else 0,
            'samples_per_second': (batch_count * self.config.production_batch_size) / total_time if total_time > 0 else 0
        }
        
    def _test_data_drift(self) -> Dict[str, Any]:
        """Simulate data drift detection"""
        self.logger.debug("Testing data drift detection...")
        
        # This is a simplified simulation
        # In practice, you'd compare with training data statistics
        import random
        
        drift_score = random.uniform(0.0, 0.3)  # Simulate drift score
        drift_detected = drift_score > 0.2
        
        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'recommendation': 'Consider retraining' if drift_detected else 'No action needed'
        }
    
    def assess_deployment_readiness(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess if model is ready for production deployment"""
        
        metrics = results['production_metrics']
        monitoring = results['monitoring_results']
        confidence = results['confidence_scores']
        
        # Check criteria against thresholds
        accuracy_ok = metrics['accuracy'] >= self.config.min_accuracy_threshold
        latency_ok = monitoring['latency_tests']['mean_latency'] * 1000 <= self.config.max_latency_threshold_ms
        confidence_ok = confidence['mean_confidence'] >= self.config.min_confidence_threshold
        throughput_ok = monitoring['throughput']['batches_per_second'] >= self.config.min_throughput_threshold
        drift_ok = not monitoring['drift_detection']['drift_detected']
        
        all_checks = [accuracy_ok, latency_ok, confidence_ok, throughput_ok, drift_ok]
        deployment_ready = all(all_checks)
        
        assessment = {
            'deployment_ready': deployment_ready,
            'checks': {
                'accuracy': {'passed': accuracy_ok, 'value': metrics['accuracy'], 'threshold': self.config.min_accuracy_threshold},
                'latency': {'passed': latency_ok, 'value': monitoring['latency_tests']['mean_latency'] * 1000, 'threshold': self.config.max_latency_threshold_ms},
                'confidence': {'passed': confidence_ok, 'value': confidence['mean_confidence'], 'threshold': self.config.min_confidence_threshold},
                'throughput': {'passed': throughput_ok, 'value': monitoring['throughput']['batches_per_second'], 'threshold': self.config.min_throughput_threshold},
                'drift': {'passed': drift_ok, 'detected': monitoring['drift_detection']['drift_detected']}
            },
            'passed_checks': sum(all_checks),
            'total_checks': len(all_checks)
        }
        
        return assessment
        
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
            
            # Assess deployment readiness
            deployment_assessment = self.assess_deployment_readiness(results)
            results['deployment_assessment'] = deployment_assessment
            
            self.logger.info(f"Production pipeline completed - "
                           f"Deployment Ready: {deployment_assessment['deployment_ready']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Production pipeline failed: {e}")
            raise

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