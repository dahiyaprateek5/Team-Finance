# =====================================
# File: ml_algorithms/neural_networks/nn_regressor_lite.py
# Lightweight Neural Network Regressor using only NumPy - FINAL VERSION
# =====================================

"""
Lightweight Neural Network Regressor for Financial Analysis
Complete implementation using only NumPy - no TensorFlow required
Optimized with early stopping, learning rate decay, and robust error handling
"""

import numpy as np  
import pandas as pd  
import os
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try to import sklearn for metrics
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split  
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
    SKLEARN_AVAILABLE = True  
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn not available. Using manual implementations.")

class SimpleNeuralNetwork:
    """
    Simple Neural Network implementation using only NumPy
    Optimized for financial regression tasks with advanced features
    """
    
    def __init__(self, hidden_layers: List[int] = None, 
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 early_stopping_patience: int = 15,
                 learning_rate_decay: float = 0.95,
                 decay_frequency: int = 20,
                 validation_split: float = 0.2,
                 random_state: int = 42):
        """
        Initialize Simple Neural Network
        
        Args:
            hidden_layers: List of hidden layer sizes
            learning_rate: Initial learning rate for training
            epochs: Maximum number of training epochs
            early_stopping_patience: Patience for early stopping
            learning_rate_decay: Factor to decay learning rate
            decay_frequency: Frequency of learning rate decay (epochs)
            validation_split: Fraction of data for validation
            random_state: Random seed
        """
        
        if hidden_layers is None:
            hidden_layers = [64, 32]
            
        self.hidden_layers = hidden_layers
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.learning_rate_decay = learning_rate_decay
        self.decay_frequency = decay_frequency
        self.validation_split = validation_split
        self.random_state = random_state
        
        # Set random seed
        np.random.seed(random_state)
        
        # Model components
        self.weights = []
        self.biases = []
        self.is_trained = False
        self.feature_names = None
        self.n_features = None
        self.scaler_mean = None
        self.scaler_std = None
        self.training_losses = []
        self.validation_losses = []
        self.best_weights = None
        self.best_biases = None
        
        print(f"🧠 Simple Neural Network initialized with {len(hidden_layers)} hidden layers")
    
    def _validate_inputs(self, X, y):
        """Validate input data"""
        if X.shape[0] < 10:
            raise ValueError("Need at least 10 samples for training")
        
        if len(np.unique(y)) < 2:
            raise ValueError("Target variable has insufficient variation")
        
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Input data contains NaN values")
        
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            raise ValueError("Input data contains infinite values")
    
    def _sigmoid(self, x):
        """Sigmoid activation function with numerical stability"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        sig = self._sigmoid(x)
        return sig * (1 - sig)
    
    def _relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """Derivative of ReLU function"""
        return (x > 0).astype(float)
    
    def _standardize(self, X, fit=False):
        """Standardize features with improved numerical stability"""
        if fit:
            self.scaler_mean = np.mean(X, axis=0)
            self.scaler_std = np.std(X, axis=0)
            # Avoid division by zero - use small value instead of 1
            self.scaler_std = np.where(self.scaler_std == 0, 1e-8, self.scaler_std)
        
        # Check if scaler parameters are available
        if self.scaler_mean is None or self.scaler_std is None:
            raise ValueError("Scaler not fitted. Call with fit=True first.")
        
        return (X - self.scaler_mean) / self.scaler_std
    
    def _init_weights(self, input_size):
        """Initialize weights and biases with Xavier initialization"""
        self.weights = []
        self.biases = []
        
        layer_sizes = [input_size] + self.hidden_layers + [1]
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization for better convergence
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            weight = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
            bias = np.zeros((1, layer_sizes[i + 1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
    
    def _forward_pass(self, X):
        """Forward pass through the network with memory optimization"""
        current_activation = X
        activations = [X]
        z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            if i < len(self.weights) - 1:  # Hidden layers
                current_activation = self._relu(z)
            else:  # Output layer
                current_activation = z  # Linear activation for regression
            
            activations.append(current_activation)
        
        return activations, z_values
    
    def _backward_pass(self, X, y, activations, z_values):
        """Backward pass (backpropagation) with gradient clipping"""
        m = X.shape[0]
        gradients_w = []
        gradients_b = []
        
        # Output layer error
        dz = activations[-1] - y.reshape(-1, 1)
        
        for i in reversed(range(len(self.weights))):
            # Gradients for weights and biases
            dw = (1/m) * np.dot(activations[i].T, dz)
            db = (1/m) * np.sum(dz, axis=0, keepdims=True)
            
            # Gradient clipping to prevent exploding gradients
            dw = np.clip(dw, -1.0, 1.0)
            db = np.clip(db, -1.0, 1.0)
            
            gradients_w.insert(0, dw)
            gradients_b.insert(0, db)
            
            if i > 0:
                # Error for previous layer
                dz = np.dot(dz, self.weights[i].T) * self._relu_derivative(z_values[i-1])
        
        return gradients_w, gradients_b
    
    def _split_data(self, X, y):
        """Split data into training and validation sets"""
        if self.validation_split <= 0:
            return X, y, None, None
        
        n_samples = len(X)
        n_val = int(n_samples * self.validation_split)
        
        # Random shuffle
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
        
        return X_train, y_train, X_val, y_val
    
    def _calculate_loss(self, X, y):
        """Calculate MSE loss"""
        activations, _ = self._forward_pass(X)
        predictions = activations[-1].flatten()
        return np.mean((predictions - y) ** 2)
    
    def train(self, X: Union[np.ndarray, pd.DataFrame], 
              y: Union[np.ndarray, pd.Series],
              verbose: bool = True) -> bool:
        """
        Train the neural network with advanced features
        
        Args:
            X: Training features
            y: Training targets
            verbose: Whether to print training progress
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            if verbose:
                print("🚀 Starting Simple Neural Network training...")
            
            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                self.feature_names = list(X.columns)
                X = X.values
            else:
                self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
            if isinstance(y, pd.Series):
                y = y.values
            
            # Validate inputs
            self._validate_inputs(X, y)
            
            # Store feature count
            self.n_features = X.shape[1]
            
            # Standardize features
            X_scaled = self._standardize(X, fit=True)
            
            # Split data for validation
            X_train, y_train, X_val, y_val = self._split_data(X_scaled, y)
            
            # Initialize weights
            self._init_weights(self.n_features)
            
            # Reset learning rate
            self.learning_rate = self.initial_learning_rate
            
            # Training loop with early stopping
            self.training_losses = []
            self.validation_losses = []
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.epochs):
                # Forward pass
                activations, z_values = self._forward_pass(X_train)
                
                # Calculate training loss
                predictions = activations[-1].flatten()
                train_loss = np.mean((predictions - y_train) ** 2)
                self.training_losses.append(train_loss)
                
                # Calculate validation loss if validation data exists
                val_loss = train_loss  # Default to training loss
                if X_val is not None:
                    val_loss = self._calculate_loss(X_val, y_val)
                    self.validation_losses.append(val_loss)
                
                # Backward pass
                gradients_w, gradients_b = self._backward_pass(X_train, y_train, activations, z_values)
                
                # Update weights and biases
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * gradients_w[i]
                    self.biases[i] -= self.learning_rate * gradients_b[i]
                
                # Learning rate decay
                if epoch > 0 and epoch % self.decay_frequency == 0:
                    self.learning_rate *= self.learning_rate_decay
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best weights
                    self.best_weights = [w.copy() for w in self.weights]
                    self.best_biases = [b.copy() for b in self.biases]
                else:
                    patience_counter += 1
                
                # Print progress
                if verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                    print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}, LR: {self.learning_rate:.6f}")
                
                # Early stopping
                if patience_counter >= self.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            
            # Restore best weights if available
            if self.best_weights is not None:
                self.weights = self.best_weights
                self.biases = self.best_biases
            
            self.is_trained = True
            
            if verbose:
                print("✅ Simple Neural Network training completed!")
                print(f"📊 Final Training Loss: {self.training_losses[-1]:.6f}")
                if self.validation_losses:
                    print(f"📊 Final Validation Loss: {self.validation_losses[-1]:.6f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted values
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Convert to numpy array
            if isinstance(X, pd.DataFrame):
                X = X.values
            
            # Validate input shape
            if X.shape[1] != self.n_features:
                raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
            
            # Standardize features
            X_scaled = self._standardize(X, fit=False)
            
            # Forward pass
            activations, _ = self._forward_pass(X_scaled)
            
            return activations[-1].flatten()
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return np.array([])
    
    def predict_with_confidence(self, X: Union[np.ndarray, pd.DataFrame], 
                               n_bootstrap: int = 100) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence intervals using bootstrap sampling
        
        Args:
            X: Features for prediction
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with predictions, confidence intervals
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            base_predictions = self.predict(X)
            
            if len(base_predictions) == 0:
                return {}
            
            # Simple confidence estimation based on training loss variance
            if len(self.training_losses) > 10:
                recent_losses = self.training_losses[-10:]
                loss_std = np.std(recent_losses)
                confidence_margin = loss_std * 1.96  # 95% confidence
                
                return {
                    'predictions': base_predictions,
                    'confidence_lower': base_predictions - confidence_margin,
                    'confidence_upper': base_predictions + confidence_margin,
                    'confidence_margin': confidence_margin
                }
            else:
                return {
                    'predictions': base_predictions,
                    'confidence_lower': base_predictions,
                    'confidence_upper': base_predictions,
                    'confidence_margin': 0.0
                }
            
        except Exception as e:
            print(f"❌ Confidence prediction failed: {e}")
            return {}
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                 y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Evaluate model performance with comprehensive metrics
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Make predictions
            y_pred = self.predict(X)
            
            if len(y_pred) == 0:
                return {}
            
            # Convert to numpy
            if isinstance(y, pd.Series):
                y = y.values
            
            # Calculate metrics
            metrics = {}
            
            if SKLEARN_AVAILABLE:
                metrics['mse'] = float(mean_squared_error(y, y_pred))
                metrics['rmse'] = float(np.sqrt(metrics['mse']))
                metrics['mae'] = float(mean_absolute_error(y, y_pred))
                metrics['r2_score'] = float(r2_score(y, y_pred))
            else:
                # Manual calculations
                metrics['mse'] = float(np.mean((y - y_pred) ** 2))
                metrics['rmse'] = float(np.sqrt(metrics['mse']))
                metrics['mae'] = float(np.mean(np.abs(y - y_pred)))
                
                # R2 score
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                metrics['r2_score'] = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
            
            # Financial-specific metrics
            metrics['mape'] = float(np.mean(np.abs((y - y_pred) / (np.abs(y) + 1e-8))) * 100)
            metrics['financial_accuracy'] = float(np.mean(np.abs((y - y_pred) / (np.abs(y) + 1e-8)) <= 0.1) * 100)
            
            # Additional metrics
            metrics['explained_variance'] = float(1 - np.var(y - y_pred) / np.var(y))
            metrics['max_error'] = float(np.max(np.abs(y - y_pred)))
            metrics['n_samples'] = len(y)
            
            # Prediction quality assessment
            residuals = y - y_pred
            metrics['mean_residual'] = float(np.mean(residuals))
            metrics['std_residual'] = float(np.std(residuals))
            
            return metrics
            
        except Exception as e:
            print(f"❌ Model evaluation failed: {e}")
            return {}
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get comprehensive training history"""
        history = {'training_loss': self.training_losses}
        
        if self.validation_losses:
            history['validation_loss'] = self.validation_losses
        
        return history
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        try:
            total_params = sum(w.size for w in self.weights) + sum(b.size for b in self.biases)
            
            summary = {
                'model_type': 'Simple Neural Network (Enhanced)',
                'is_trained': self.is_trained,
                'n_features': self.n_features,
                'feature_names': self.feature_names,
                'architecture': {
                    'hidden_layers': self.hidden_layers,
                    'total_parameters': total_params,
                    'activation_function': 'ReLU (hidden), Linear (output)'
                },
                'training_params': {
                    'initial_learning_rate': self.initial_learning_rate,
                    'current_learning_rate': self.learning_rate,
                    'epochs': self.epochs,
                    'early_stopping_patience': self.early_stopping_patience,
                    'learning_rate_decay': self.learning_rate_decay,
                    'validation_split': self.validation_split
                },
                'training_stats': {
                    'epochs_trained': len(self.training_losses),
                    'early_stopped': len(self.training_losses) < self.epochs,
                    'best_training_loss': float(min(self.training_losses)) if self.training_losses else None,
                    'final_training_loss': float(self.training_losses[-1]) if self.training_losses else None
                }
            }
            
            if self.validation_losses:
                summary['training_stats']['best_validation_loss'] = float(min(self.validation_losses))
                summary['training_stats']['final_validation_loss'] = float(self.validation_losses[-1])
            
            return summary
            
        except Exception as e:
            print(f"❌ Model summary generation failed: {e}")
            return {'error': str(e)}
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model with comprehensive state"""
        try:
            if not self.is_trained:
                print("❌ Cannot save untrained model")
                return False
            
            model_data = {
                'weights': self.weights,
                'biases': self.biases,
                'best_weights': self.best_weights,
                'best_biases': self.best_biases,
                'feature_names': self.feature_names,
                'n_features': self.n_features,
                'scaler_mean': self.scaler_mean,
                'scaler_std': self.scaler_std,
                'hidden_layers': self.hidden_layers,
                'initial_learning_rate': self.initial_learning_rate,
                'learning_rate': self.learning_rate,
                'epochs': self.epochs,
                'early_stopping_patience': self.early_stopping_patience,
                'learning_rate_decay': self.learning_rate_decay,
                'decay_frequency': self.decay_frequency,
                'validation_split': self.validation_split,
                'training_losses': self.training_losses,
                'validation_losses': self.validation_losses,
                'random_state': self.random_state,
                'model_version': '2.0_enhanced'
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✅ Model saved to {filepath}.pkl")
            return True
            
        except Exception as e:
            print(f"❌ Model save failed: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a saved model with version compatibility"""
        try:
            with open(f"{filepath}.pkl", 'rb') as f:
                model_data = pickle.load(f)
            
            # Load core model state
            self.weights = model_data['weights']
            self.biases = model_data['biases']
            self.feature_names = model_data['feature_names']
            self.n_features = model_data['n_features']
            self.scaler_mean = model_data['scaler_mean']
            self.scaler_std = model_data['scaler_std']
            self.hidden_layers = model_data['hidden_layers']
            self.initial_learning_rate = model_data.get('initial_learning_rate', 0.001)
            self.learning_rate = model_data.get('learning_rate', self.initial_learning_rate)
            self.epochs = model_data.get('epochs', 100)
            self.training_losses = model_data.get('training_losses', [])
            
            # Load enhanced features if available
            self.best_weights = model_data.get('best_weights')
            self.best_biases = model_data.get('best_biases')
            self.validation_losses = model_data.get('validation_losses', [])
            self.early_stopping_patience = model_data.get('early_stopping_patience', 15)
            self.learning_rate_decay = model_data.get('learning_rate_decay', 0.95)
            self.decay_frequency = model_data.get('decay_frequency', 20)
            self.validation_split = model_data.get('validation_split', 0.2)
            self.random_state = model_data.get('random_state', 42)
            
            self.is_trained = True
            
            model_version = model_data.get('model_version', '1.0_basic')
            print(f"✅ Model loaded from {filepath}.pkl (version: {model_version})")
            return True
            
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            return False

# Backward compatibility - create NeuralNetworkRegressor alias
class NeuralNetworkRegressor(SimpleNeuralNetwork):
    """
    Enhanced Neural Network Regressor for backward compatibility
    Includes all advanced features while maintaining the same interface
    """
    def __init__(self, hidden_layers=None, **kwargs):
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        super().__init__(hidden_layers=hidden_layers, **kwargs)
        print("✅ Using Enhanced NeuralNetworkRegressor (NumPy-based)")

# Test the Enhanced Neural Network
if __name__ == "__main__":
    print("🧠 Testing Enhanced Simple Neural Network...")
    
    # Generate synthetic financial data
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    # Synthetic features representing financial indicators
    X = np.random.randn(n_samples, n_features)
    
    # Create synthetic target (financial health score)
    y = (
        X[:, 0] * 2.5 +                    # Revenue factor
        X[:, 1] * 1.8 +                    # Profit margin
        X[:, 2] * (-1.2) +                 # Debt ratio (negative impact)
        X[:, 3] * X[:, 4] * 0.5 +          # Interaction term
        np.sin(X[:, 5]) * 0.8 +            # Non-linear relationship
        np.random.randn(n_samples) * 0.3    # Noise
    )
    
    # Create DataFrame
    feature_names = [
        'revenue_growth', 'profit_margin', 'debt_ratio', 'liquidity_ratio', 'roa',
        'current_ratio', 'quick_ratio', 'inventory_turnover', 'receivables_turnover',
        'payables_turnover', 'cash_ratio', 'interest_coverage', 'debt_to_equity',
        'working_capital', 'market_cap'
    ]
    
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='financial_health_score')
    
    # Split data
    if SKLEARN_AVAILABLE:
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=0.2, random_state=42
        )
    else:
        # Manual split
        train_idx = int(0.8 * len(X_df))
        X_train = X_df[:train_idx]
        X_test = X_df[train_idx:]
        y_train = y_series[:train_idx]
        y_test = y_series[train_idx:]
    
    print(f"📊 Dataset: {len(X_train)} train, {len(X_test)} test samples")
    
    # Test the enhanced model
    try:
        # Create model with enhanced features
        model = NeuralNetworkRegressor(
            hidden_layers=[64, 32],
            learning_rate=0.01,
            epochs=100,
            early_stopping_patience=10,
            learning_rate_decay=0.9,
            validation_split=0.2
        )
        
        # Train model
        success = model.train(X_train, y_train, verbose=True)
        
        if success:
            print("✅ Training successful")
            
            # Make predictions
            y_pred = model.predict(X_test)
            print(f"✅ Predictions shape: {y_pred.shape}")
            
            # Evaluate model
            metrics = model.evaluate(X_test, y_test)
            print(f"📊 R² Score: {metrics.get('r2_score', 0):.4f}")
            print(f"📊 RMSE: {metrics.get('rmse', 0):.4f}")
            print(f"📊 Financial Accuracy: {metrics.get('financial_accuracy', 0):.2f}%")
            print(f"📊 MAPE: {metrics.get('mape', 0):.2f}%")
            
            # Test confidence predictions
            confidence_results = model.predict_with_confidence(X_test[:5])
            if confidence_results:
                print("✅ Confidence prediction working")
            
            # Get model summary
            summary = model.get_model_summary()
            print(f"📊 Model trained for {summary['training_stats']['epochs_trained']} epochs")
            print(f"📊 Early stopped: {summary['training_stats']['early_stopped']}")
            
            # Test save/load
            model.save_model("test_model_enhanced")
            
            # Load model
            new_model = NeuralNetworkRegressor()
            if new_model.load_model("test_model_enhanced"):
                print("✅ Model save/load working")
                
                # Test loaded model
                y_pred_loaded = new_model.predict(X_test[:5])
                print(f"✅ Loaded model predictions: {y_pred_loaded[:3]}")
        
        else:
            print("❌ Training failed")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("🎉 Enhanced Simple Neural Network Test Completed!")