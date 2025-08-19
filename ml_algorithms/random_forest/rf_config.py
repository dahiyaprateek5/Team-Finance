# =====================================
# File: ml_algorithms/random_forest/rf_config.py
# Random Forest Configuration Manager
# =====================================

"""
Random Forest Configuration Manager
Optimized configurations for different financial scenarios
"""

import numpy as np # type: ignore
from typing import Dict, Any, List, Optional, Tuple
import json

class RandomForestConfig:
    """
    Configuration manager for Random Forest models
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
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1,
                'oob_score': True,
                'max_samples': 0.8,
                'ccp_alpha': 0.0,
                'description': 'Optimized for financial health prediction with balanced accuracy and interpretability'
            },
            
            # Risk Classification
            'risk_classification': {
                'n_estimators': 300,
                'max_depth': 20,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1,
                'oob_score': True,
                'class_weight': 'balanced',
                'max_samples': 0.7,
                'ccp_alpha': 0.001,
                'description': 'Optimized for multi-class risk classification with class imbalance handling'
            },
            
            # Bankruptcy Prediction
            'bankruptcy_prediction': {
                'n_estimators': 500,
                'max_depth': 25,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'log2',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1,
                'oob_score': True,
                'class_weight': {0: 1.0, 1: 5.0},  # Higher weight for bankruptcy class
                'max_samples': 0.8,
                'ccp_alpha': 0.0,
                'description': 'Deep forest for bankruptcy prediction with high recall for positive cases'
            },
            
            # Cash Flow Forecasting
            'cash_flow_forecasting': {
                'n_estimators': 150,
                'max_depth': 12,
                'min_samples_split': 8,
                'min_samples_leaf': 3,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1,
                'oob_score': True,
                'max_samples': 0.9,
                'ccp_alpha': 0.002,
                'description': 'Stable model for cash flow prediction with reduced overfitting'
            },
            
            # High Performance
            'high_performance': {
                'n_estimators': 1000,
                'max_depth': 30,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': None,  # Use all features
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1,
                'oob_score': True,
                'max_samples': 0.8,
                'ccp_alpha': 0.0,
                'description': 'Maximum performance configuration for complex patterns'
            },
            
            # Fast Training
            'fast': {
                'n_estimators': 50,
                'max_depth': 8,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1,
                'oob_score': False,
                'max_samples': 0.6,
                'ccp_alpha': 0.01,
                'description': 'Fast training for quick prototyping and testing'
            },
            
            # Memory Efficient
            'memory_efficient': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': 1,  # Single thread to save memory
                'oob_score': False,
                'max_samples': 0.5,
                'ccp_alpha': 0.005,
                'description': 'Memory-optimized configuration for limited resources'
            },
            
            # Robust (less overfitting)
            'robust': {
                'n_estimators': 200,
                'max_depth': 8,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1,
                'oob_score': True,
                'max_samples': 0.7,
                'ccp_alpha': 0.01,
                'description': 'Robust configuration with high regularization to prevent overfitting'
            },
            
            # Interpretable
            'interpretable': {
                'n_estimators': 100,
                'max_depth': 6,
                'min_samples_split': 15,
                'min_samples_leaf': 8,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1,
                'oob_score': True,
                'max_samples': 0.8,
                'ccp_alpha': 0.005,
                'description': 'Interpretable model with shallow trees for better understanding'
            },
            
            # Custom Financial
            'custom_financial': {
                'n_estimators': 250,
                'max_depth': 18,
                'min_samples_split': 8,
                'min_samples_leaf': 3,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1,
                'oob_score': True,
                'max_samples': 0.8,
                'ccp_alpha': 0.001,
                'description': 'Custom balanced configuration for general financial analysis'
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
            config = RandomForestConfig.get_config_by_scenario(base_scenario)
            config.update(custom_params)
            return RandomForestConfig.validate_config(config)
        except Exception as e:
            print(f"⚠️ Custom config creation failed: {e}")
            return RandomForestConfig.get_config_by_scenario('financial_health')
    
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
            # Validate n_estimators
            if 'n_estimators' in validated_config:
                n_est = validated_config['n_estimators']
                validated_config['n_estimators'] = max(1, min(10000, int(n_est)))
            
            # Validate max_depth
            if 'max_depth' in validated_config:
                max_d = validated_config['max_depth']
                if max_d is not None:
                    validated_config['max_depth'] = max(1, min(50, int(max_d)))
            
            # Validate min_samples_split
            if 'min_samples_split' in validated_config:
                min_split = validated_config['min_samples_split']
                validated_config['min_samples_split'] = max(2, int(min_split))
            
            # Validate min_samples_leaf
            if 'min_samples_leaf' in validated_config:
                min_leaf = validated_config['min_samples_leaf']
                validated_config['min_samples_leaf'] = max(1, int(min_leaf))
            
            # Validate max_features
            if 'max_features' in validated_config:
                max_feat = validated_config['max_features']
                valid_max_features = ['sqrt', 'log2', None, 'auto']
                if isinstance(max_feat, str) and max_feat not in valid_max_features:
                    validated_config['max_features'] = 'sqrt'
                elif isinstance(max_feat, (int, float)):
                    validated_config['max_features'] = max(1, int(max_feat))
            
            # Validate max_samples
            if 'max_samples' in validated_config:
                max_samp = validated_config['max_samples']
                if max_samp is not None:
                    validated_config['max_samples'] = max(0.1, min(1.0, float(max_samp)))
            
            # Validate ccp_alpha
            if 'ccp_alpha' in validated_config:
                ccp = validated_config['ccp_alpha']
                validated_config['ccp_alpha'] = max(0.0, float(ccp))
            
            # Validate n_jobs
            if 'n_jobs' in validated_config:
                n_j = validated_config['n_jobs']
                if n_j is not None and n_j != -1:
                    validated_config['n_jobs'] = max(1, int(n_j))
            
            # Validate boolean parameters
            bool_params = ['bootstrap', 'oob_score']
            for param in bool_params:
                if param in validated_config:
                    validated_config[param] = bool(validated_config[param])
            
            # Validate random_state
            if 'random_state' in validated_config:
                rs = validated_config['random_state']
                if rs is not None:
                    validated_config['random_state'] = int(rs)
            
            return validated_config
            
        except Exception as e:
            print(f"⚠️ Config validation failed: {e}")
            return RandomForestConfig.get_config_by_scenario('financial_health')
    
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
            # Base scenario selection
            if n_samples < 1000:
                scenario = 'fast'
            elif n_samples < 5000:
                scenario = 'memory_efficient'
            elif n_samples < 20000:
                scenario = 'financial_health' if task_type == 'regression' else 'risk_classification'
            else:
                scenario = 'high_performance'
            
            config = RandomForestConfig.get_config_by_scenario(scenario)
            
            # Adjust based on feature count
            if n_features < 10:
                # Few features - use all or most
                config['max_features'] = None if n_features <= 5 else 'sqrt'
                config['n_estimators'] = min(config['n_estimators'], 100)
            elif n_features > 100:
                # Many features - use feature selection
                config['max_features'] = 'log2'
                config['n_estimators'] = max(config['n_estimators'], 200)
            
            # Adjust based on sample size
            if n_samples < 500:
                config['n_estimators'] = min(config['n_estimators'], 50)
                config['max_depth'] = min(config.get('max_depth', 10), 8)
                config['min_samples_split'] = max(config['min_samples_split'], 10)
                config['min_samples_leaf'] = max(config['min_samples_leaf'], 5)
            elif n_samples > 50000:
                config['n_estimators'] = max(config['n_estimators'], 300)
                config['max_samples'] = min(config.get('max_samples', 0.8), 0.6)  # Use subsampling
            
            # Task-specific adjustments
            if task_type == 'classification':
                config['class_weight'] = 'balanced'
                config['oob_score'] = True
            
            config['recommendation_reason'] = f"Optimized for {n_features} features and {n_samples} samples"
            
            return config
            
        except Exception as e:
            print(f"⚠️ Architecture recommendation failed: {e}")
            return RandomForestConfig.get_config_by_scenario('financial_health')
    
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
                optimized_config['n_estimators'] = min(optimized_config.get('n_estimators', 100), 50)
                optimized_config['max_depth'] = min(optimized_config.get('max_depth', 10), 6)
                optimized_config['n_jobs'] = 1
                optimized_config['oob_score'] = False
                optimized_config['max_samples'] = 0.3
            elif memory_limit_gb < 2.0:
                # Limited memory
                optimized_config['n_estimators'] = min(optimized_config.get('n_estimators', 200), 100)
                optimized_config['max_depth'] = min(optimized_config.get('max_depth', 15), 10)
                optimized_config['n_jobs'] = min(optimized_config.get('n_jobs', -1), 2)
                optimized_config['max_samples'] = min(optimized_config.get('max_samples', 0.8), 0.5)
            elif memory_limit_gb < 4.0:
                # Moderate memory
                optimized_config['n_estimators'] = min(optimized_config.get('n_estimators', 300), 200)
                optimized_config['max_samples'] = min(optimized_config.get('max_samples', 0.8), 0.7)
            
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
            base_config = RandomForestConfig.get_config_by_scenario(scenario)
            
            grid = {
                'n_estimators': [50, 100, 200, 300, 500],
                'max_depth': [5, 10, 15, 20, 25, None],
                'min_samples_split': [2, 5, 10, 15, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'max_features': ['sqrt', 'log2', None],
                'max_samples': [0.6, 0.7, 0.8, 0.9, None],
                'ccp_alpha': [0.0, 0.001, 0.005, 0.01, 0.02]
            }
            
            # Add classification-specific parameters
            if 'class_weight' in base_config:
                grid['class_weight'] = ['balanced', 'balanced_subsample', None]
            
            return grid
            
        except Exception as e:
            print(f"⚠️ Hyperparameter grid creation failed: {e}")
            return {}
    
    @staticmethod
    def get_feature_importance_config() -> Dict[str, Any]:
        """Get configuration optimized for feature importance analysis"""
        return {
            'n_estimators': 500,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': None,
            'bootstrap': False,  # Use all samples for stable importance
            'random_state': 42,
            'n_jobs': -1,
            'oob_score': False,
            'description': 'Configuration optimized for reliable feature importance calculation'
        }
    
    @staticmethod
    def get_all_scenarios() -> List[str]:
        """Get list of all available scenarios"""
        return [
            'financial_health', 'risk_classification', 'bankruptcy_prediction',
            'cash_flow_forecasting', 'high_performance', 'fast', 'memory_efficient',
            'robust', 'interpretable', 'custom_financial'
        ]
    
    @staticmethod
    def print_scenario_info(scenario: str = None):
        """Print information about scenarios"""
        if scenario:
            try:
                config = RandomForestConfig.get_config_by_scenario(scenario)
                print(f"\n🌲 Random Forest Configuration: {scenario}")
                print(f"Description: {config.get('description', 'No description')}")
                print(f"Trees: {config['n_estimators']}")
                print(f"Max Depth: {config.get('max_depth', 'None')}")
                print(f"Min Samples Split: {config['min_samples_split']}")
                print(f"Max Features: {config['max_features']}")
                print(f"Bootstrap: {config['bootstrap']}")
                print(f"OOB Score: {config['oob_score']}")
            except Exception as e:
                print(f"❌ Error getting scenario info: {e}")
        else:
            print("\n🌲 Available Random Forest Scenarios:")
            scenarios = RandomForestConfig.get_all_scenarios()
            for i, scenario_name in enumerate(scenarios, 1):
                try:
                    config = RandomForestConfig.get_config_by_scenario(scenario_name)
                    print(f"{i}. {scenario_name}: {config.get('description', 'No description')}")
                except:
                    print(f"{i}. {scenario_name}: Configuration error")
    
    @staticmethod
    def compare_configs(scenarios: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple configurations side by side
        
        Args:
            scenarios: List of scenario names to compare
            
        Returns:
            Dictionary with configuration comparison
        """
        try:
            comparison = {}
            
            for scenario in scenarios:
                try:
                    config = RandomForestConfig.get_config_by_scenario(scenario)
                    comparison[scenario] = {
                        'n_estimators': config['n_estimators'],
                        'max_depth': config.get('max_depth'),
                        'min_samples_split': config['min_samples_split'],
                        'min_samples_leaf': config['min_samples_leaf'],
                        'max_features': config['max_features'],
                        'description': config.get('description', 'No description')
                    }
                except Exception as e:
                    comparison[scenario] = {'error': str(e)}
            
            return comparison
            
        except Exception as e:
            print(f"⚠️ Config comparison failed: {e}")
            return {}

# Test the configuration system
if __name__ == "__main__":
    print("🌲 Testing Random Forest Configuration System...")
    
    # Test different scenarios
    scenarios_to_test = ['financial_health', 'risk_classification', 'bankruptcy_prediction', 'fast']
    
    for scenario in scenarios_to_test:
        try:
            config = RandomForestConfig.get_config_by_scenario(scenario)
            print(f"✅ {scenario}: {len(config)} parameters")
        except Exception as e:
            print(f"❌ {scenario}: {e}")
    
    # Test custom configuration
    try:
        custom_config = RandomForestConfig.get_custom_config(
            'financial_health',
            n_estimators=300,
            max_depth=20,
            min_samples_split=10
        )
        print(f"✅ Custom config created with n_estimators: {custom_config['n_estimators']}")
    except Exception as e:
        print(f"❌ Custom config failed: {e}")
    
    # Test architecture recommendations
    try:
        recommendations = RandomForestConfig.get_architecture_recommendations(
            n_features=25, n_samples=5000, task_type='classification'
        )
        print(f"✅ Architecture recommendations: {recommendations['recommendation_reason']}")
    except Exception as e:
        print(f"❌ Architecture recommendations failed: {e}")
    
    # Test memory optimization
    try:
        base_config = RandomForestConfig.get_config_by_scenario('high_performance')
        optimized = RandomForestConfig.optimize_for_memory(base_config, memory_limit_gb=1.0)
        print(f"✅ Memory optimized: {optimized['n_estimators']} trees")
    except Exception as e:
        print(f"❌ Memory optimization failed: {e}")
    
    # Test hyperparameter grid
    try:
        grid = RandomForestConfig.create_hyperparameter_grid('financial_health')
        print(f"✅ Hyperparameter grid: {len(grid)} parameters")
    except Exception as e:
        print(f"❌ Hyperparameter grid failed: {e}")
    
    # Test scenario comparison
    try:
        comparison = RandomForestConfig.compare_configs(['fast', 'high_performance'])
        print(f"✅ Config comparison: {len(comparison)} scenarios")
    except Exception as e:
        print(f"❌ Config comparison failed: {e}")
    
    print("\n🎉 Random Forest Configuration System Test Completed!")