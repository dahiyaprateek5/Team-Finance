"""
LightGBM Configuration Manager
Optimized configurations for different financial analysis scenarios
"""

import sys
import os
from typing import Dict, Any, Optional, List

class LGBConfig:
    """
    Configuration manager for LightGBM models
    Provides optimized settings for different financial analysis scenarios
    """
    
    @staticmethod
    def get_default_regressor_config() -> Dict[str, Any]:
        """Default configuration for financial regression tasks"""
        return {
            # Core parameters
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            
            # Tree parameters
            'num_leaves': 31,
            'max_depth': 8,
            'min_data_in_leaf': 10,
            'min_sum_hessian_in_leaf': 1e-3,
            
            # Learning parameters
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            
            # Regularization
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            
            # Performance
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1,
            
            # Early stopping
            'early_stopping_rounds': 20,
            
            # Advanced
            'extra_trees': False,
            'feature_fraction_bynode': 0.8,
            'min_gain_to_split': 0.02,
            'max_bin': 255
        }
    
    @staticmethod
    def get_default_classifier_config() -> Dict[str, Any]:
        """Default configuration for financial classification tasks"""
        return {
            # Core parameters
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            
            # Tree parameters
            'num_leaves': 31,
            'max_depth': 8,
            'min_data_in_leaf': 10,
            'min_sum_hessian_in_leaf': 1e-3,
            
            # Learning parameters
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            
            # Regularization
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            
            # Class imbalance handling
            'class_weight': 'balanced',
            'is_unbalance': True,
            
            # Performance
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1,
            
            # Early stopping
            'early_stopping_rounds': 20,
            
            # Advanced
            'extra_trees': False,
            'feature_fraction_bynode': 0.8,
            'min_gain_to_split': 0.02,
            'max_bin': 255
        }
    
    @staticmethod
    def get_fast_config() -> Dict[str, Any]:
        """Fast training configuration for quick prototyping"""
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 15,
            'max_depth': 5,
            'learning_rate': 0.2,
            'n_estimators': 50,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1,
            'early_stopping_rounds': 10
        }
    
    @staticmethod
    def get_accurate_config() -> Dict[str, Any]:
        """High accuracy configuration for production models"""
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'max_depth': 12,
            'min_data_in_leaf': 5,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.85,
            'subsample_freq': 1,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.05,
            'reg_lambda': 0.05,
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1,
            'early_stopping_rounds': 50,
            'feature_fraction_bynode': 0.9,
            'min_gain_to_split': 0.01,
            'max_bin': 512
        }
    
    @staticmethod
    def get_small_dataset_config() -> Dict[str, Any]:
        """Configuration optimized for small datasets (<1000 samples)"""
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 15,
            'max_depth': 6,
            'min_data_in_leaf': 3,
            'learning_rate': 0.15,
            'n_estimators': 100,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.2,
            'reg_lambda': 0.2,
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1,
            'early_stopping_rounds': 15,
            'min_gain_to_split': 0.05
        }
    
    @staticmethod
    def get_large_dataset_config() -> Dict[str, Any]:
        """Configuration optimized for large datasets (>10,000 samples)"""
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'max_depth': 15,
            'min_data_in_leaf': 20,
            'learning_rate': 0.08,
            'n_estimators': 300,
            'subsample': 0.8,
            'subsample_freq': 5,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.05,
            'reg_lambda': 0.05,
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1,
            'early_stopping_rounds': 30,
            'feature_fraction_bynode': 0.7,
            'max_bin': 1024
        }
    
    @staticmethod
    def get_imbalanced_classification_config() -> Dict[str, Any]:
        """Configuration for imbalanced classification problems"""
        return {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 8,
            'min_data_in_leaf': 5,
            'learning_rate': 0.08,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            
            # Imbalanced data handling
            'class_weight': 'balanced',
            'is_unbalance': True,
            'scale_pos_weight': 1.0,
            
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1,
            'early_stopping_rounds': 25,
            'min_gain_to_split': 0.01
        }
    
    @staticmethod
    def get_financial_health_config() -> Dict[str, Any]:
        """Specialized configuration for financial health prediction"""
        return {
            'objective': 'regression',
            'metric': ['rmse', 'mae'],
            'boosting_type': 'gbdt',
            
            # Optimized for financial ratios and metrics
            'num_leaves': 40,
            'max_depth': 9,
            'min_data_in_leaf': 8,
            'min_sum_hessian_in_leaf': 1e-3,
            
            # Conservative learning for financial stability
            'learning_rate': 0.09,
            'n_estimators': 250,
            'subsample': 0.82,
            'subsample_freq': 2,
            'colsample_bytree': 0.85,
            
            # Regularization for financial data
            'reg_alpha': 0.12,
            'reg_lambda': 0.12,
            
            # Performance
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1,
            
            # Early stopping
            'early_stopping_rounds': 25,
            
            # Financial data specific
            'feature_fraction_bynode': 0.85,
            'min_gain_to_split': 0.015,
            'max_bin': 300,
            'extra_trees': False,
            
            # Handle financial outliers
            'min_child_weight': 1e-3,
            'bagging_fraction': 0.82,
            'bagging_freq': 2
        }
    
    @staticmethod
    def get_risk_classification_config() -> Dict[str, Any]:
        """Specialized configuration for financial risk classification"""
        return {
            'objective': 'multiclass',
            'metric': ['multi_logloss', 'multi_error'],
            'boosting_type': 'gbdt',
            
            # Risk assessment optimized
            'num_leaves': 35,
            'max_depth': 10,
            'min_data_in_leaf': 6,
            'min_sum_hessian_in_leaf': 1e-3,
            
            # Learning parameters
            'learning_rate': 0.08,
            'n_estimators': 280,
            'subsample': 0.80,
            'subsample_freq': 3,
            'colsample_bytree': 0.83,
            
            # Regularization for risk stability
            'reg_alpha': 0.15,
            'reg_lambda': 0.15,
            
            # Class handling for risk categories
            'class_weight': 'balanced',
            'is_unbalance': True,
            
            # Performance
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1,
            
            # Early stopping
            'early_stopping_rounds': 30,
            
            # Risk-specific parameters
            'feature_fraction_bynode': 0.80,
            'min_gain_to_split': 0.02,
            'max_bin': 400,
            'min_child_weight': 2e-3
        }
    
    @staticmethod
    def get_time_series_config() -> Dict[str, Any]:
        """Configuration for time series financial data"""
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            
            # Time series optimized
            'num_leaves': 25,
            'max_depth': 7,
            'min_data_in_leaf': 12,
            'learning_rate': 0.12,
            'n_estimators': 150,
            
            # Prevent overfitting in time series
            'subsample': 0.85,
            'colsample_bytree': 0.88,
            'reg_alpha': 0.2,
            'reg_lambda': 0.2,
            
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1,
            'early_stopping_rounds': 20,
            
            # Time series specific
            'min_gain_to_split': 0.03,
            'extra_trees': True,
            'feature_fraction_bynode': 0.9
        }
    
    @classmethod
    def get_config_by_scenario(cls, scenario: str, **kwargs) -> Dict[str, Any]:
        """
        Get configuration by scenario name
        
        Args:
            scenario: Configuration scenario name
            **kwargs: Additional parameters to override
        
        Available scenarios:
        - 'default_regressor'
        - 'default_classifier'
        - 'fast'
        - 'accurate'
        - 'small_dataset'
        - 'large_dataset'
        - 'imbalanced_classification'
        - 'financial_health'
        - 'risk_classification'
        - 'time_series'
        """
        
        config_map = {
            'default_regressor': cls.get_default_regressor_config,
            'default_classifier': cls.get_default_classifier_config,
            'fast': cls.get_fast_config,
            'accurate': cls.get_accurate_config,
            'small_dataset': cls.get_small_dataset_config,
            'large_dataset': cls.get_large_dataset_config,
            'imbalanced_classification': cls.get_imbalanced_classification_config,
            'financial_health': cls.get_financial_health_config,
            'risk_classification': cls.get_risk_classification_config,
            'time_series': cls.get_time_series_config
        }
        
        if scenario not in config_map:
            available_scenarios = ', '.join(config_map.keys())
            raise ValueError(f"Unknown scenario: '{scenario}'. Available scenarios: {available_scenarios}")
        
        config = config_map[scenario]()
        
        # Override with custom parameters
        if kwargs:
            config.update(kwargs)
        
        return config
    
    @staticmethod
    def get_custom_config(base_scenario: str = 'default_regressor', **overrides) -> Dict[str, Any]:
        """
        Create custom configuration based on base scenario
        
        Args:
            base_scenario: Base configuration to start with
            **overrides: Parameters to override
        """
        config = LGBConfig.get_config_by_scenario(base_scenario)
        config.update(overrides)
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean configuration parameters
        
        Args:
            config: Configuration dictionary to validate
        
        Returns:
            Validated and cleaned configuration
        """
        validated_config = config.copy()
        
        # Ensure required parameters
        if 'objective' not in validated_config:
            validated_config['objective'] = 'regression'
        
        if 'random_state' not in validated_config:
            validated_config['random_state'] = 42
        
        if 'verbose' not in validated_config:
            validated_config['verbose'] = -1
        
        # Validate ranges
        if 'learning_rate' in validated_config:
            lr = validated_config['learning_rate']
            validated_config['learning_rate'] = max(0.01, min(1.0, lr))
        
        if 'max_depth' in validated_config:
            depth = validated_config['max_depth']
            validated_config['max_depth'] = max(3, min(20, depth))
        
        if 'num_leaves' in validated_config:
            leaves = validated_config['num_leaves']
            validated_config['num_leaves'] = max(10, min(300, leaves))
        
        if 'n_estimators' in validated_config:
            estimators = validated_config['n_estimators']
            validated_config['n_estimators'] = max(10, min(2000, estimators))
        
        # Validate subsample parameters
        for param in ['subsample', 'colsample_bytree', 'feature_fraction_bynode']:
            if param in validated_config:
                value = validated_config[param]
                validated_config[param] = max(0.1, min(1.0, value))
        
        return validated_config
    
    @staticmethod
    def print_config_comparison(configs: Dict[str, Dict[str, Any]]) -> None:
        """
        Print comparison of multiple configurations
        
        Args:
            configs: Dictionary of configuration name -> configuration dict
        """
        if not configs:
            print("No configurations to compare")
            return
        
        # Get all parameter names
        all_params = set()
        for config in configs.values():
            all_params.update(config.keys())
        
        all_params = sorted(list(all_params))
        
        print("📊 LightGBM Configuration Comparison")
        print("=" * 80)
        
        # Print header
        header = f"{'Parameter':<25}"
        for config_name in configs.keys():
            header += f"{config_name[:15]:<16}"
        print(header)
        print("-" * 80)
        
        # Print parameters
        for param in all_params:
            row = f"{param:<25}"
            for config in configs.values():
                value = config.get(param, 'N/A')
                if isinstance(value, float):
                    value_str = f"{value:.3f}"
                elif isinstance(value, list):
                    value_str = str(value)[:12] + "..."
                else:
                    value_str = str(value)
                row += f"{value_str[:15]:<16}"
            print(row)
        
        print("=" * 80)
    
    @staticmethod
    def get_scenario_recommendations(dataset_size: int, n_features: int, 
                                   n_classes: Optional[int] = None,
                                   is_imbalanced: bool = False,
                                   training_time_priority: str = 'balanced') -> str:
        """
        Get scenario recommendation based on dataset characteristics
        
        Args:
            dataset_size: Number of samples
            n_features: Number of features
            n_classes: Number of classes (for classification)
            is_imbalanced: Whether classes are imbalanced
            training_time_priority: 'fast', 'balanced', or 'accurate'
        
        Returns:
            Recommended scenario name
        """
        
        # Classification vs Regression
        if n_classes is not None:
            # Classification task
            if is_imbalanced:
                return 'imbalanced_classification'
            elif training_time_priority == 'fast':
                return 'fast'
            else:
                return 'risk_classification'
        else:
            # Regression task
            if dataset_size < 500:
                return 'small_dataset'
            elif dataset_size > 10000:
                return 'large_dataset'
            elif training_time_priority == 'fast':
                return 'fast'
            elif training_time_priority == 'accurate':
                return 'accurate'
            else:
                return 'financial_health'
    
    @staticmethod
    def optimize_for_memory(config: Dict[str, Any], memory_limit_gb: float = 4.0) -> Dict[str, Any]:
        """
        Optimize configuration for memory constraints
        
        Args:
            config: Base configuration
            memory_limit_gb: Memory limit in GB
        
        Returns:
            Memory-optimized configuration
        """
        optimized_config = config.copy()
        
        if memory_limit_gb < 2.0:
            # Very low memory
            optimized_config.update({
                'num_leaves': min(optimized_config.get('num_leaves', 31), 15),
                'max_depth': min(optimized_config.get('max_depth', 8), 5),
                'max_bin': min(optimized_config.get('max_bin', 255), 127),
                'subsample': max(optimized_config.get('subsample', 0.8), 0.9),
                'colsample_bytree': max(optimized_config.get('colsample_bytree', 0.8), 0.9)
            })
        elif memory_limit_gb < 4.0:
            # Low memory
            optimized_config.update({
                'num_leaves': min(optimized_config.get('num_leaves', 31), 25),
                'max_depth': min(optimized_config.get('max_depth', 8), 7),
                'max_bin': min(optimized_config.get('max_bin', 255), 200)
            })
        
        return optimized_config
    
    @staticmethod
    def get_all_scenarios() -> Dict[str, str]:
        """
        Get all available scenarios with descriptions
        
        Returns:
            Dictionary of scenario name -> description
        """
        return {
            'default_regressor': 'Standard regression configuration',
            'default_classifier': 'Standard classification configuration',
            'fast': 'Quick training for prototyping',
            'accurate': 'High accuracy for production',
            'small_dataset': 'Optimized for <1000 samples',
            'large_dataset': 'Optimized for >10,000 samples',
            'imbalanced_classification': 'Handles class imbalance',
            'financial_health': 'Specialized for financial health prediction',
            'risk_classification': 'Specialized for risk categorization',
            'time_series': 'Optimized for time series data'
        }
    
    @staticmethod
    def create_hyperparameter_grid(scenario: str = 'default_regressor') -> Dict[str, List]:
        """
        Create hyperparameter grid for tuning based on scenario
        
        Args:
            scenario: Base scenario for grid creation
        
        Returns:
            Dictionary with parameter names and value lists
        """
        base_grids = {
            'fast': {
                'num_leaves': [10, 15, 20],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.1, 0.2, 0.3],
                'n_estimators': [30, 50, 100]
            },
            'accurate': {
                'num_leaves': [31, 50, 100, 150],
                'max_depth': [8, 10, 12, 15],
                'learning_rate': [0.03, 0.05, 0.08, 0.1],
                'n_estimators': [200, 300, 500, 800],
                'reg_alpha': [0.01, 0.05, 0.1, 0.2],
                'reg_lambda': [0.01, 0.05, 0.1, 0.2]
            },
            'default': {
                'num_leaves': [20, 31, 50],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'n_estimators': [100, 200, 300],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        }
        
        # Select appropriate grid
        if scenario in ['fast']:
            return base_grids['fast']
        elif scenario in ['accurate']:
            return base_grids['accurate']
        else:
            return base_grids['default']

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing LightGBM Configuration Manager...")
    
    try:
        # Test different configurations
        scenarios = ['default_regressor', 'financial_health', 'fast', 'accurate']
        configs = {}
        
        print("\n📋 Loading configurations...")
        for scenario in scenarios:
            configs[scenario] = LGBConfig.get_config_by_scenario(scenario)
            print(f"✅ {scenario} config loaded")
        
        print("\n📊 Configuration Comparison:")
        LGBConfig.print_config_comparison(configs)
        
        print("\n🎯 Scenario Recommendations:")
        rec1 = LGBConfig.get_scenario_recommendations(500, 10)
        print(f"Small dataset (500 samples): {rec1}")
        
        rec2 = LGBConfig.get_scenario_recommendations(50000, 20)
        print(f"Large dataset (50000 samples): {rec2}")
        
        rec3 = LGBConfig.get_scenario_recommendations(2000, 15, n_classes=3, is_imbalanced=True)
        print(f"Imbalanced classification: {rec3}")
        
        print("\n🔧 Custom Configuration Test:")
        custom_config = LGBConfig.get_custom_config(
            'financial_health',
            learning_rate=0.05,
            n_estimators=300
        )
        print(f"Custom learning rate: {custom_config['learning_rate']}")
        print(f"Custom n_estimators: {custom_config['n_estimators']}")
        
        print("\n✅ Validation Test:")
        invalid_config = {
            'learning_rate': 2.0,  # Invalid (>1.0)
            'max_depth': 25,       # Invalid (>20)
            'num_leaves': 5        # Invalid (<10)
        }
        validated = LGBConfig.validate_config(invalid_config)
        print(f"Original learning_rate: {invalid_config['learning_rate']} -> Validated: {validated['learning_rate']}")
        print(f"Original max_depth: {invalid_config['max_depth']} -> Validated: {validated['max_depth']}")
        print(f"Original num_leaves: {invalid_config['num_leaves']} -> Validated: {validated['num_leaves']}")
        
        print("\n📈 Available Scenarios:")
        scenarios_info = LGBConfig.get_all_scenarios()
        for name, description in scenarios_info.items():
            print(f"  {name}: {description}")
        
        print("\n🎛️ Hyperparameter Grid Test:")
        grid = LGBConfig.create_hyperparameter_grid('fast')
        print(f"Fast grid keys: {list(grid.keys())}")
        print(f"Learning rates: {grid['learning_rate']}")
        
        print("\n💾 Memory Optimization Test:")
        base_config = LGBConfig.get_config_by_scenario('accurate')
        memory_optimized = LGBConfig.optimize_for_memory(base_config, memory_limit_gb=1.5)
        print(f"Original num_leaves: {base_config['num_leaves']} -> Optimized: {memory_optimized['num_leaves']}")
        print(f"Original max_depth: {base_config['max_depth']} -> Optimized: {memory_optimized['max_depth']}")
        
        print("\n🎉 All configuration tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()