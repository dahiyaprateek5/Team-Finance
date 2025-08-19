"""
Neural Networks Configuration Manager
Optimized configurations for different financial scenarios
"""

import numpy as np 
from typing import Dict, Any, List, Optional, Tuple
import json

class NeuralNetworkConfig:
    """
    Configuration manager for Neural Network models
    Pre-optimized settings for different financial analysis scenarios
    """
    
    @staticmethod
    def get_config_by_scenario(scenario: str) -> Dict[str, Any]:
        """
        Get pre-optimized configuration for specific scenario
        
        Args:
            scenario: Configuration scenario name
            
        Returns:
            Dictionary with optimized parameters
        """
        config_map = {
            # Financial Health Prediction
            'financial_health': {
                'hidden_layers': [128, 64, 32],
                'activation': 'relu',
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'early_stopping_patience': 15,
                'l2_regularization': 0.01,
                'optimizer': 'adam',
                'loss_function': 'mse',
                'validation_split': 0.2,
                'use_batch_normalization': True,
                'description': 'Optimized for financial health score prediction'
            },
            
            # Risk Classification
            'risk_classification': {
                'hidden_layers': [256, 128, 64],
                'activation': 'relu',
                'dropout_rate': 0.4,
                'learning_rate': 0.0005,
                'batch_size': 64,
                'epochs': 150,
                'early_stopping_patience': 20,
                'l2_regularization': 0.005,
                'optimizer': 'adam',
                'loss_function': 'categorical_crossentropy',
                'validation_split': 0.25,
                'use_batch_normalization': True,
                'class_weight': 'balanced',
                'description': 'Optimized for multi-class risk prediction'
            },
            
            # Bankruptcy Prediction
            'bankruptcy_prediction': {
                'hidden_layers': [512, 256, 128, 64],
                'activation': 'relu',
                'dropout_rate': 0.5,
                'learning_rate': 0.0001,
                'batch_size': 16,
                'epochs': 200,
                'early_stopping_patience': 25,
                'l2_regularization': 0.02,
                'optimizer': 'adam',
                'loss_function': 'binary_crossentropy',
                'validation_split': 0.3,
                'use_batch_normalization': True,
                'class_weight': {0: 1.0, 1: 3.0},  # Higher weight for bankruptcy class
                'description': 'Deep model for bankruptcy prediction'
            },
            
            # Cash Flow Forecasting
            'cash_flow_forecasting': {
                'hidden_layers': [64, 32],
                'activation': 'relu',
                'dropout_rate': 0.2,
                'learning_rate': 0.002,
                'batch_size': 64,
                'epochs': 80,
                'early_stopping_patience': 10,
                'l2_regularization': 0.001,
                'optimizer': 'adam',
                'loss_function': 'mae',
                'validation_split': 0.2,
                'use_batch_normalization': False,
                'description': 'Lightweight model for cash flow prediction'
            },
            
            # High Performance
            'high_performance': {
                'hidden_layers': [1024, 512, 256, 128],
                'activation': 'relu',
                'dropout_rate': 0.3,
                'learning_rate': 0.0005,
                'batch_size': 128,
                'epochs': 300,
                'early_stopping_patience': 30,
                'l2_regularization': 0.01,
                'optimizer': 'adam',
                'loss_function': 'mse',
                'validation_split': 0.2,
                'use_batch_normalization': True,
                'description': 'High-capacity model for complex patterns'
            },
            
            # Fast Training
            'fast': {
                'hidden_layers': [32, 16],
                'activation': 'relu',
                'dropout_rate': 0.1,
                'learning_rate': 0.01,
                'batch_size': 128,
                'epochs': 50,
                'early_stopping_patience': 5,
                'l2_regularization': 0.001,
                'optimizer': 'adam',
                'loss_function': 'mse',
                'validation_split': 0.1,
                'use_batch_normalization': False,
                'description': 'Fast training for quick prototyping'
            },
            
            # Memory Efficient
            'memory_efficient': {
                'hidden_layers': [64, 32],
                'activation': 'relu',
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 16,
                'epochs': 100,
                'early_stopping_patience': 15,
                'l2_regularization': 0.01,
                'optimizer': 'adam',
                'loss_function': 'mse',
                'validation_split': 0.2,
                'use_batch_normalization': False,
                'description': 'Optimized for low memory usage'
            },
            
            # Robust (less overfitting)
            'robust': {
                'hidden_layers': [128, 64],
                'activation': 'relu',
                'dropout_rate': 0.6,
                'learning_rate': 0.0005,
                'batch_size': 32,
                'epochs': 200,
                'early_stopping_patience': 25,
                'l2_regularization': 0.05,
                'optimizer': 'adam',
                'loss_function': 'mse',
                'validation_split': 0.3,
                'use_batch_normalization': True,
                'description': 'High regularization to prevent overfitting'
            },
            
            # Time Series
            'time_series': {
                'hidden_layers': [128, 64, 32],
                'activation': 'tanh',
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 150,
                'early_stopping_patience': 20,
                'l2_regularization': 0.01,
                'optimizer': 'adam',
                'loss_function': 'mse',
                'validation_split': 0.2,
                'use_batch_normalization': True,
                'use_lstm': True,
                'description': 'Optimized for time series financial data'
            },
            
            # Custom Financial
            'custom_financial': {
                'hidden_layers': [256, 128, 64, 32],
                'activation': 'relu',
                'dropout_rate': 0.4,
                'learning_rate': 0.0008,
                'batch_size': 64,
                'epochs': 120,
                'early_stopping_patience': 18,
                'l2_regularization': 0.015,
                'optimizer': 'adam',
                'loss_function': 'mse',
                'validation_split': 0.25,
                'use_batch_normalization': True,
                'description': 'Custom configuration for financial analysis'
            }
        }
        
        if scenario not in config_map:
            available_scenarios = ', '.join(config_map.keys())
            raise ValueError(f"Unknown scenario: '{scenario}'. Available scenarios: {available_scenarios}")
        
        return config_map[scenario].copy()
    
    @staticmethod
    def get_custom_config(base_scenario: str = 'financial_health', **custom_params) -> Dict[str, Any]:
        """
        Get custom configuration based on base scenario with modifications
        
        Args:
            base_scenario: Base configuration to start with
            **custom_params: Parameters to override
            
        Returns:
            Custom configuration dictionary
        """
        try:
            config = NeuralNetworkConfig.get_config_by_scenario(base_scenario)
            config.update(custom_params)
            return NeuralNetworkConfig.validate_config(config)
        except Exception as e:
            print(f"⚠️ Custom config creation failed: {e}")
            return NeuralNetworkConfig.get_config_by_scenario('financial_health')
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix configuration parameters
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Validated configuration
        """
        validated_config = config.copy()
        
        try:
            # Validate hidden layers
            if 'hidden_layers' in validated_config:
                layers = validated_config['hidden_layers']
                if not isinstance(layers, list) or len(layers) == 0:
                    validated_config['hidden_layers'] = [128, 64, 32]
                else:
                    # Ensure all layer sizes are positive integers
                    validated_config['hidden_layers'] = [max(1, int(layer)) for layer in layers]
            
            # Validate learning rate
            if 'learning_rate' in validated_config:
                lr = validated_config['learning_rate']
                validated_config['learning_rate'] = max(0.0001, min(1.0, float(lr)))
            
            # Validate dropout rate
            if 'dropout_rate' in validated_config:
                dropout = validated_config['dropout_rate']
                validated_config['dropout_rate'] = max(0.0, min(0.9, float(dropout)))
            
            # Validate batch size
            if 'batch_size' in validated_config:
                batch_size = validated_config['batch_size']
                validated_config['batch_size'] = max(1, int(batch_size))
            
            # Validate epochs
            if 'epochs' in validated_config:
                epochs = validated_config['epochs']
                validated_config['epochs'] = max(1, int(epochs))
            
            # Validate early stopping patience
            if 'early_stopping_patience' in validated_config:
                patience = validated_config['early_stopping_patience']
                validated_config['early_stopping_patience'] = max(1, int(patience))
            
            # Validate L2 regularization
            if 'l2_regularization' in validated_config:
                l2_reg = validated_config['l2_regularization']
                validated_config['l2_regularization'] = max(0.0, min(1.0, float(l2_reg)))
            
            # Validate validation split
            if 'validation_split' in validated_config:
                val_split = validated_config['validation_split']
                validated_config['validation_split'] = max(0.0, min(0.5, float(val_split)))
            
            # Validate activation function
            valid_activations = ['relu', 'tanh', 'sigmoid', 'linear', 'elu', 'selu']
            if 'activation' in validated_config:
                if validated_config['activation'] not in valid_activations:
                    validated_config['activation'] = 'relu'
            
            # Validate optimizer
            valid_optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad']
            if 'optimizer' in validated_config:
                if validated_config['optimizer'] not in valid_optimizers:
                    validated_config['optimizer'] = 'adam'
            
            # Validate loss function
            valid_losses = ['mse', 'mae', 'binary_crossentropy', 'categorical_crossentropy', 'sparse_categorical_crossentropy']
            if 'loss_function' in validated_config:
                if validated_config['loss_function'] not in valid_losses:
                    validated_config['loss_function'] = 'mse'
            
            return validated_config
            
        except Exception as e:
            print(f"⚠️ Config validation failed: {e}")
            return NeuralNetworkConfig.get_config_by_scenario('financial_health')
    
    @staticmethod
    def get_architecture_recommendations(n_features: int, n_samples: int, 
                                       task_type: str = 'regression') -> Dict[str, Any]:
        """
        Get architecture recommendations based on data characteristics
        
        Args:
            n_features: Number of input features
            n_samples: Number of training samples
            task_type: 'regression' or 'classification'
            
        Returns:
            Recommended configuration
        """
        try:
            # Base recommendations
            if n_samples < 1000:
                scenario = 'fast'
            elif n_samples < 5000:
                scenario = 'memory_efficient'
            elif n_samples < 20000:
                scenario = 'financial_health' if task_type == 'regression' else 'risk_classification'
            else:
                scenario = 'high_performance'
            
            config = NeuralNetworkConfig.get_config_by_scenario(scenario)
            
            # Adjust based on feature count
            if n_features < 10:
                # Reduce model complexity for few features
                config['hidden_layers'] = [max(32, n_features * 4), max(16, n_features * 2)]
                config['dropout_rate'] = min(config['dropout_rate'], 0.2)
            elif n_features > 100:
                # Increase model capacity for many features
                config['hidden_layers'] = [min(512, n_features * 2), min(256, n_features), 128, 64]
                config['dropout_rate'] = max(config['dropout_rate'], 0.3)
            
            # Adjust based on sample size
            if n_samples < 500:
                config['epochs'] = min(config['epochs'], 50)
                config['batch_size'] = min(config['batch_size'], 16)
                config['l2_regularization'] = max(config['l2_regularization'], 0.01)
            elif n_samples > 50000:
                config['epochs'] = max(config['epochs'], 100)
                config['batch_size'] = max(config['batch_size'], 64)
            
            config['recommendation_reason'] = f"Optimized for {n_features} features and {n_samples} samples"
            
            return config
            
        except Exception as e:
            print(f"⚠️ Architecture recommendation failed: {e}")
            return NeuralNetworkConfig.get_config_by_scenario('financial_health')
    
    @staticmethod
    def optimize_for_memory(config: Dict[str, Any], memory_limit_gb: float = 2.0) -> Dict[str, Any]:
        """
        Optimize configuration for memory constraints
        
        Args:
            config: Base configuration
            memory_limit_gb: Memory limit in GB
            
        Returns:
            Memory-optimized configuration
        """
        try:
            optimized_config = config.copy()
            
            if memory_limit_gb < 1.0:
                # Very limited memory
                optimized_config['hidden_layers'] = [32, 16]
                optimized_config['batch_size'] = 8
                optimized_config['use_batch_normalization'] = False
            elif memory_limit_gb < 2.0:
                # Limited memory
                optimized_config['hidden_layers'] = [64, 32]
                optimized_config['batch_size'] = 16
            elif memory_limit_gb < 4.0:
                # Moderate memory
                optimized_config['batch_size'] = min(optimized_config.get('batch_size', 32), 32)
            
            # Reduce layers if still too large
            max_layer_size = int(memory_limit_gb * 128)
            optimized_config['hidden_layers'] = [
                min(layer, max_layer_size) for layer in optimized_config['hidden_layers']
            ]
            
            optimized_config['memory_optimized'] = True
            optimized_config['memory_limit_gb'] = memory_limit_gb
            
            return optimized_config
            
        except Exception as e:
            print(f"⚠️ Memory optimization failed: {e}")
            return config
    
    @staticmethod
    def create_hyperparameter_grid(scenario: str = 'financial_health') -> Dict[str, List]:
        """
        Create hyperparameter grid for tuning
        
        Args:
            scenario: Base scenario for grid creation
            
        Returns:
            Grid of hyperparameters for tuning
        """
        try:
            base_config = NeuralNetworkConfig.get_config_by_scenario(scenario)
            
            grid = {
                'hidden_layers': [
                    [64, 32],
                    [128, 64],
                    [128, 64, 32],
                    [256, 128, 64],
                    [256, 128, 64, 32]
                ],
                'learning_rate': [0.01, 0.005, 0.001, 0.0005, 0.0001],
                'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
                'l2_regularization': [0.001, 0.005, 0.01, 0.02, 0.05],
                'batch_size': [16, 32, 64, 128],
                'activation': ['relu', 'tanh', 'elu'],
                'optimizer': ['adam', 'rmsprop']
            }
            
            return grid
            
        except Exception as e:
            print(f"⚠️ Hyperparameter grid creation failed: {e}")
            return {}
    
    @staticmethod
    def get_all_scenarios() -> List[str]:
        """Get list of all available scenarios"""
        return [
            'financial_health', 'risk_classification', 'bankruptcy_prediction',
            'cash_flow_forecasting', 'high_performance', 'fast', 'memory_efficient',
            'robust', 'time_series', 'custom_financial'
        ]
    
    @staticmethod
    def print_scenario_info(scenario: str = None):
        """Print information about scenarios"""
        if scenario:
            try:
                config = NeuralNetworkConfig.get_config_by_scenario(scenario)
                print(f"\n🧠 Neural Network Configuration: {scenario}")
                print(f"Description: {config.get('description', 'No description')}")
                print(f"Hidden Layers: {config['hidden_layers']}")
                print(f"Learning Rate: {config['learning_rate']}")
                print(f"Dropout Rate: {config['dropout_rate']}")
                print(f"Batch Size: {config['batch_size']}")
                print(f"Epochs: {config['epochs']}")
                print(f"Regularization: {config['l2_regularization']}")
            except Exception as e:
                print(f"❌ Error getting scenario info: {e}")
        else:
            print("\n🧠 Available Neural Network Scenarios:")
            scenarios = NeuralNetworkConfig.get_all_scenarios()
            for i, scenario_name in enumerate(scenarios, 1):
                try:
                    config = NeuralNetworkConfig.get_config_by_scenario(scenario_name)
                    print(f"{i}. {scenario_name}: {config.get('description', 'No description')}")
                except:
                    print(f"{i}. {scenario_name}: Configuration error")

# Test the configuration system
if __name__ == "__main__":
    print("🧠 Testing Neural Network Configuration System...")
    
    # Test different scenarios
    scenarios_to_test = ['financial_health', 'risk_classification', 'bankruptcy_prediction', 'fast']
    
    for scenario in scenarios_to_test:
        try:
            config = NeuralNetworkConfig.get_config_by_scenario(scenario)
            print(f"✅ {scenario}: {len(config)} parameters")
        except Exception as e:
            print(f"❌ {scenario}: {e}")
    
    # Test custom configuration
    try:
        custom_config = NeuralNetworkConfig.get_custom_config(
            'financial_health',
            learning_rate=0.01,
            hidden_layers=[256, 128],
            epochs=50
        )
        print(f"✅ Custom config created with learning_rate: {custom_config['learning_rate']}")
    except Exception as e:
        print(f"❌ Custom config failed: {e}")
    
    # Test architecture recommendations
    try:
        recommendations = NeuralNetworkConfig.get_architecture_recommendations(
            n_features=20, n_samples=5000, task_type='classification'
        )
        print(f"✅ Architecture recommendations: {recommendations['recommendation_reason']}")
    except Exception as e:
        print(f"❌ Architecture recommendations failed: {e}")
    
    # Test memory optimization
    try:
        base_config = NeuralNetworkConfig.get_config_by_scenario('high_performance')
        optimized = NeuralNetworkConfig.optimize_for_memory(base_config, memory_limit_gb=1.0)
        print(f"✅ Memory optimized: {optimized['hidden_layers']}")
    except Exception as e:
        print(f"❌ Memory optimization failed: {e}")
    
    print("\n🎉 Neural Network Configuration System Test Completed!")