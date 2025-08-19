# =====================================
# File: ml_algorithms/xgboost/xgb_config.py
# XGBoost Configuration Manager
# =====================================

"""
XGBoost Configuration Manager
Optimized configurations for different financial analysis scenarios
"""

import sys
import os
from typing import Dict, Any, Optional, List

class XGBConfig:
    """
    Configuration manager for XGBoost models
    Provides optimized hyperparameters for different financial scenarios
    """
    
    @staticmethod
    def get_config_by_scenario(scenario: str) -> Dict[str, Any]:
        """
        Get optimized configuration for specific scenario
        
        Args:
            scenario: Scenario name
            
        Returns:
            Dictionary with XGBoost parameters
        """
        
        config_map = {
            'financial_health': XGBConfig.get_financial_health_config(),
            'risk_classification': XGBConfig.get_risk_classification_config(),
            'bankruptcy_prediction': XGBConfig.get_bankruptcy_prediction_config(),
            'cash_flow_forecasting': XGBConfig.get_cash_flow_forecasting_config(),
            'high_performance': XGBConfig.get_high_performance_config(),
            'fast': XGBConfig.get_fast_config(),
            'memory_efficient': XGBConfig.get_memory_efficient_config(),
            'robust': XGBConfig.get_robust_config(),
            'interpretable': XGBConfig.get_interpretable_config(),
            'custom_financial': XGBConfig.get_custom_financial_config()
        }
        
        if scenario not in config_map:
            available_scenarios = ', '.join(config_map.keys())
            raise ValueError(f"Unknown scenario: '{scenario}'. Available scenarios: {available_scenarios}")
        
        return config_map[scenario]
    
    @staticmethod
    def get_financial_health_config() -> Dict[str, Any]:
        """Optimized for financial health score prediction"""
        return {
            'objective': 'reg:squarederror',
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
            'eval_metric': 'rmse'
        }
    
    @staticmethod
    def get_risk_classification_config() -> Dict[str, Any]:
        """Optimized for financial risk classification"""
        return {
            'objective': 'multi:softprob',
            'n_estimators': 250,
            'max_depth': 6,
            'learning_rate': 0.08,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 2,
            'gamma': 0.05,
            'reg_alpha': 0.05,
            'reg_lambda': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 30,
            'eval_metric': 'mlogloss'
        }
    
    @staticmethod
    def get_bankruptcy_prediction_config() -> Dict[str, Any]:
        """Optimized for bankruptcy prediction (binary classification)"""
        return {
            'objective': 'binary:logistic',
            'n_estimators': 400,
            'max_depth': 10,
            'learning_rate': 0.03,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'min_child_weight': 5,
            'gamma': 0.2,
            'reg_alpha': 0.2,
            'reg_lambda': 1.5,
            'scale_pos_weight': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 75,
            'eval_metric': 'auc'
        }
    
    @staticmethod
    def get_cash_flow_forecasting_config() -> Dict[str, Any]:
        """Optimized for cash flow forecasting"""
        return {
            'objective': 'reg:squarederror',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.08,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 2,
            'gamma': 0.05,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 40,
            'eval_metric': 'rmse'
        }
    
    @staticmethod
    def get_high_performance_config() -> Dict[str, Any]:
        """High performance configuration for maximum accuracy"""
        return {
            'objective': 'reg:squarederror',
            'n_estimators': 500,
            'max_depth': 12,
            'learning_rate': 0.02,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 1,
            'gamma': 0.3,
            'reg_alpha': 0.3,
            'reg_lambda': 2.0,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 100,
            'eval_metric': 'rmse'
        }
    
    @staticmethod
    def get_fast_config() -> Dict[str, Any]:
        """Fast training configuration for quick prototyping"""
        return {
            'objective': 'reg:squarederror',
            'n_estimators': 50,
            'max_depth': 4,
            'learning_rate': 0.2,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 1,
            'gamma': 0.0,
            'reg_alpha': 0.0,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 10,
            'eval_metric': 'rmse'
        }
    
    @staticmethod
    def get_memory_efficient_config() -> Dict[str, Any]:
        """Memory efficient configuration for large datasets"""
        return {
            'objective': 'reg:squarederror',
            'n_estimators': 150,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'min_child_weight': 5,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.5,
            'random_state': 42,
            'n_jobs': 2,
            'early_stopping_rounds': 25,
            'eval_metric': 'rmse'
        }
    
    @staticmethod
    def get_robust_config() -> Dict[str, Any]:
        """Robust configuration with high regularization"""
        return {
            'objective': 'reg:squarederror',
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 10,
            'gamma': 0.5,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
            'eval_metric': 'rmse'
        }
    
    @staticmethod
    def get_interpretable_config() -> Dict[str, Any]:
        """Interpretable configuration with shallow trees"""
        return {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.15,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.5,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 20,
            'eval_metric': 'rmse'
        }
    
    @staticmethod
    def get_custom_financial_config() -> Dict[str, Any]:
        """Custom balanced configuration for financial analysis"""
        return {
            'objective': 'reg:squarederror',
            'n_estimators': 250,
            'max_depth': 7,
            'learning_rate': 0.06,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.15,
            'reg_alpha': 0.15,
            'reg_lambda': 1.2,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 40,
            'eval_metric': 'rmse'
        }
    
    @staticmethod
    def get_custom_config(base_scenario: str, **kwargs) -> Dict[str, Any]:
        """
        Create custom configuration based on base scenario
        
        Args:
            base_scenario: Base scenario name
            **kwargs: Parameters to override
            
        Returns:
            Custom configuration dictionary
        """
        base_config = XGBConfig.get_config_by_scenario(base_scenario)
        base_config.update(kwargs)
        return base_config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix configuration parameters
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration
        """
        validated_config = config.copy()
        
        # Validate learning rate
        if 'learning_rate' in validated_config:
            lr = validated_config['learning_rate']
            validated_config['learning_rate'] = max(0.001, min(1.0, lr))
        
        # Validate max_depth
        if 'max_depth' in validated_config:
            depth = validated_config['max_depth']
            validated_config['max_depth'] = max(1, min(15, depth))
        
        # Validate n_estimators
        if 'n_estimators' in validated_config:
            n_est = validated_config['n_estimators']
            validated_config['n_estimators'] = max(10, min(1000, n_est))
        
        # Validate subsample
        if 'subsample' in validated_config:
            subsample = validated_config['subsample']
            validated_config['subsample'] = max(0.1, min(1.0, subsample))
        
        # Validate colsample_bytree
        if 'colsample_bytree' in validated_config:
            colsample = validated_config['colsample_bytree']
            validated_config['colsample_bytree'] = max(0.1, min(1.0, colsample))
        
        return validated_config
    
    @staticmethod
    def get_scenario_recommendations(n_features: int, n_samples: int, task_type: str = 'regression') -> List[str]:
        """
        Get recommended scenarios based on data characteristics
        
        Args:
            n_features: Number of features
            n_samples: Number of samples
            task_type: 'regression' or 'classification'
            
        Returns:
            List of recommended scenario names
        """
        recommendations = []
        
        # Based on sample size
        if n_samples < 1000:
            recommendations.append('fast')
            recommendations.append('interpretable')
        elif n_samples < 5000:
            recommendations.append('memory_efficient')
            recommendations.append('custom_financial')
        else:
            recommendations.append('high_performance')
            recommendations.append('financial_health')
        
        # Based on feature count
        if n_features > 50:
            recommendations.append('robust')
        
        # Based on task type
        if task_type == 'classification':
            recommendations.append('risk_classification')
        else:
            recommendations.append('cash_flow_forecasting')
        
        return list(set(recommendations))
    
    @staticmethod
    def create_hyperparameter_grid(scenario: str = 'financial_health') -> Dict[str, List]:
        """
        Create hyperparameter grid for tuning
        
        Args:
            scenario: Base scenario
            
        Returns:
            Parameter grid for GridSearchCV
        """
        base_grids = {
            'fast': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.2]
            },
            'financial_health': {
                'n_estimators': [200, 300, 400],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.03, 0.05, 0.08],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            },
            'risk_classification': {
                'n_estimators': [150, 250, 350],
                'max_depth': [5, 6, 8],
                'learning_rate': [0.05, 0.08, 0.1],
                'min_child_weight': [1, 2, 3]
            }
        }
        
        return base_grids.get(scenario, base_grids['financial_health'])
    
    @staticmethod
    def optimize_for_memory(config: Dict[str, Any], memory_limit_gb: float = 4.0) -> Dict[str, Any]:
        """
        Optimize configuration for memory constraints
        
        Args:
            config: Original configuration
            memory_limit_gb: Memory limit in GB
            
        Returns:
            Memory-optimized configuration
        """
        optimized_config = config.copy()
        
        if memory_limit_gb < 2.0:
            # Very limited memory
            optimized_config.update({
                'n_estimators': min(100, optimized_config.get('n_estimators', 100)),
                'max_depth': min(4, optimized_config.get('max_depth', 6)),
                'subsample': 0.6,
                'colsample_bytree': 0.6,
                'n_jobs': 1
            })
        elif memory_limit_gb < 4.0:
            # Limited memory
            optimized_config.update({
                'n_estimators': min(200, optimized_config.get('n_estimators', 200)),
                'max_depth': min(6, optimized_config.get('max_depth', 8)),
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'n_jobs': 2
            })
        
        return optimized_config

# Test the configuration manager
if __name__ == "__main__":
    print("🔧 Testing XGBoost Configuration Manager...")
    
    # Test all scenarios
    scenarios = [
        'financial_health', 'risk_classification', 'bankruptcy_prediction',
        'cash_flow_forecasting', 'high_performance', 'fast',
        'memory_efficient', 'robust', 'interpretable', 'custom_financial'
    ]
    
    configs = {}
    for scenario in scenarios:
        try:
            configs[scenario] = XGBConfig.get_config_by_scenario(scenario)
            print(f"✅ {scenario} config loaded")
        except Exception as e:
            print(f"❌ {scenario} failed: {e}")
    
    # Test custom configuration
    try:
        custom_config = XGBConfig.get_custom_config(
            'financial_health',
            learning_rate=0.1,
            n_estimators=150
        )
        print(f"✅ Custom config created")
    except Exception as e:
        print(f"❌ Custom config failed: {e}")
    
    # Test validation
    try:
        invalid_config = {'learning_rate': 2.0, 'max_depth': 20, 'n_estimators': -10}
        validated_config = XGBConfig.validate_config(invalid_config)
        print(f"✅ Config validation working")
    except Exception as e:
        print(f"❌ Config validation failed: {e}")
    
    # Test recommendations
    try:
        recommendations = XGBConfig.get_scenario_recommendations(
            n_features=20, n_samples=5000, task_type='regression'
        )
        print(f"✅ Scenario recommendations: {recommendations}")
    except Exception as e:
        print(f"❌ Recommendations failed: {e}")
    
    # Test hyperparameter grid
    try:
        grid = XGBConfig.create_hyperparameter_grid('fast')
        print(f"✅ Hyperparameter grid created with {len(grid)} parameters")
    except Exception as e:
        print(f"❌ Hyperparameter grid failed: {e}")
    
    print("\n🎉 XGBoost Configuration Manager Test Completed!")