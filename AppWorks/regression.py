"""
Regression module for AppWorks.

This module handles training of regression models to predict Fourier coefficients
from gear parameters using both linear and neural network approaches.
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime
import warnings

# Machine learning imports
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

from .config import MODELS_PATH, REGRESSION_MODELS


class GearCoefficientPredictor:
    """
    A complete predictor for Fourier coefficients from gear parameters.
    
    This class handles both linear and neural network regression models
    with proper preprocessing and model management.
    """
    
    def __init__(self, model_type: str = 'linear'):
        """
        Initialize the predictor.
        
        Args:
            model_type: Type of model ('linear' or 'neural')
        """
        if model_type not in ['linear', 'neural']:
            raise ValueError("model_type must be 'linear' or 'neural'")
        
        self.model_type = model_type
        self.models = {}  # One model per coefficient
        self.scalers = {}  # One scaler per coefficient
        self.feature_scaler = StandardScaler()  # For input features
        self.is_fitted = False
        self.feature_names = None
        self.n_coefficients = None
        
    def _create_model(self) -> Union[LinearRegression, MLPRegressor]:
        """Create a single model instance based on model_type."""
        if self.model_type == 'linear':
            config = REGRESSION_MODELS['linear']
            return LinearRegression(
                fit_intercept=config['fit_intercept']
            )
        else:  # neural
            config = REGRESSION_MODELS['neural']
            return MLPRegressor(
                hidden_layer_sizes=config['hidden_layer_sizes'],
                activation=config['activation'],
                solver=config['solver'],
                max_iter=config['max_iter'],
                random_state=config['random_state']
            )
    
    def fit(self, X: np.ndarray, Y: np.ndarray, 
            feature_names: Optional[List[str]] = None,
            validation_split: float = 0.2,
            verbose: bool = True) -> Dict[str, Any]:
        """
        Train regression models to predict each coefficient.
        
        Args:
            X: Feature matrix (n_samples x n_features)
            Y: Target matrix (n_samples x n_coefficients)
            feature_names: Names of input features
            validation_split: Fraction for validation
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training metrics
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples")
        
        if validation_split < 0 or validation_split >= 1:
            raise ValueError("validation_split must be in [0, 1)")
        
        n_samples, n_features = X.shape
        n_coefficients = Y.shape[1]
        
        if verbose:
            print(f"Training {self.model_type} models:")
            print(f"  Samples: {n_samples}")
            print(f"  Features: {n_features}")
            print(f"  Coefficients: {n_coefficients}")
        
        # Store metadata
        self.feature_names = feature_names or [f'feature_{i}' for i in range(n_features)]
        self.n_coefficients = n_coefficients
        
        # Split data
        if validation_split > 0:
            X_train, X_val, Y_train, Y_val = train_test_split(
                X, Y, test_size=validation_split, random_state=42
            )
        else:
            X_train, Y_train = X, Y
            X_val, Y_val = None, None
        
        # Fit feature scaler
        self.feature_scaler.fit(X_train)
        X_train_scaled = self.feature_scaler.transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.feature_scaler.transform(X_val)
        
        # Train one model per coefficient
        training_metrics = {
            'train_scores': [],
            'val_scores': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_mae': [],
            'val_mae': []
        }
        
        for coeff_idx in range(n_coefficients):
            if verbose and coeff_idx % 20 == 0:
                print(f"  Training coefficient {coeff_idx + 1}/{n_coefficients}")
            
            # Get target for this coefficient
            y_train = Y_train[:, coeff_idx]
            y_val = Y_val[:, coeff_idx] if Y_val is not None else None
            
            # Create and train model
            model = self._create_model()
            
            # Fit target scaler
            scaler = StandardScaler()
            y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            
            # Train model
            model.fit(X_train_scaled, y_train_scaled)
            
            # Store model and scaler
            self.models[coeff_idx] = model
            self.scalers[coeff_idx] = scaler
            
            # Compute metrics
            # Training metrics
            y_train_pred_scaled = model.predict(X_train_scaled)
            y_train_pred = scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
            
            train_r2 = r2_score(y_train, y_train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            
            training_metrics['train_scores'].append(train_r2)
            training_metrics['train_rmse'].append(train_rmse)
            training_metrics['train_mae'].append(train_mae)
            
            # Validation metrics
            if y_val is not None:
                y_val_pred_scaled = model.predict(X_val_scaled)
                y_val_pred = scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()
                
                val_r2 = r2_score(y_val, y_val_pred)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                val_mae = mean_absolute_error(y_val, y_val_pred)
                
                training_metrics['val_scores'].append(val_r2)
                training_metrics['val_rmse'].append(val_rmse)
                training_metrics['val_mae'].append(val_mae)
        
        self.is_fitted = True
        
        # Compute summary statistics
        summary = {
            'model_type': self.model_type,
            'n_coefficients': n_coefficients,
            'mean_train_r2': np.mean(training_metrics['train_scores']),
            'mean_train_rmse': np.mean(training_metrics['train_rmse']),
            'mean_train_mae': np.mean(training_metrics['train_mae']),
        }
        
        if validation_split > 0:
            summary.update({
                'mean_val_r2': np.mean(training_metrics['val_scores']),
                'mean_val_rmse': np.mean(training_metrics['val_rmse']),
                'mean_val_mae': np.mean(training_metrics['val_mae']),
            })
        
        if verbose:
            print(f"Training complete:")
            print(f"  Mean train R²: {summary['mean_train_r2']:.4f}")
            print(f"  Mean train RMSE: {summary['mean_train_rmse']:.6f}")
            if validation_split > 0:
                print(f"  Mean val R²: {summary['mean_val_r2']:.4f}")
                print(f"  Mean val RMSE: {summary['mean_val_rmse']:.6f}")
        
        return summary
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict Fourier coefficients for given parameters.
        
        Args:
            X: Feature matrix (n_samples x n_features)
            
        Returns:
            Predicted coefficients (n_samples x n_coefficients)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        n_samples = X.shape[0]
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Predict each coefficient
        predictions = np.zeros((n_samples, self.n_coefficients))
        
        for coeff_idx in range(self.n_coefficients):
            model = self.models[coeff_idx]
            scaler = self.scalers[coeff_idx]
            
            # Predict scaled values
            y_pred_scaled = model.predict(X_scaled)
            
            # Inverse transform
            y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            predictions[:, coeff_idx] = y_pred
        
        return predictions
    
    def save(self, filename: Optional[str] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_type}_predictor_{timestamp}.pkl"
        
        filepath = os.path.join(MODELS_PATH, filename)
        
        # Save all components
        save_dict = {
            'model_type': self.model_type,
            'models': self.models,
            'scalers': self.scalers,
            'feature_scaler': self.feature_scaler,
            'feature_names': self.feature_names,
            'n_coefficients': self.n_coefficients,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"Model saved to: {filepath}")
        return filepath
    
    def load(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to saved model file
        """
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Restore all components
        self.model_type = save_dict['model_type']
        self.models = save_dict['models']
        self.scalers = save_dict['scalers']
        self.feature_scaler = save_dict['feature_scaler']
        self.feature_names = save_dict['feature_names']
        self.n_coefficients = save_dict['n_coefficients']
        self.is_fitted = save_dict['is_fitted']
        
        print(f"Model loaded from: {filepath}")


def prepare_features(parameters: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    """
    Convert parameter dictionaries to feature matrix.
    
    Args:
        parameters: List of parameter dictionaries
        
    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    if len(parameters) == 0:
        raise ValueError("Parameters list cannot be empty")
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(parameters)
    
    # Extract feature names and values
    feature_names = list(df.columns)
    feature_matrix = df.values.astype(float)
    
    return feature_matrix, feature_names


def train_linear_models(X: np.ndarray, Y: np.ndarray, 
                       feature_names: Optional[List[str]] = None,
                       **kwargs) -> GearCoefficientPredictor:
    """
    Train linear regression models for coefficient prediction.
    
    Args:
        X: Feature matrix
        Y: Target coefficients matrix
        feature_names: Names of features
        **kwargs: Additional arguments for training
        
    Returns:
        Trained predictor
    """
    predictor = GearCoefficientPredictor(model_type='linear')
    predictor.fit(X, Y, feature_names=feature_names, **kwargs)
    return predictor


def train_neural_models(X: np.ndarray, Y: np.ndarray,
                       feature_names: Optional[List[str]] = None,
                       **kwargs) -> GearCoefficientPredictor:
    """
    Train neural network models for coefficient prediction.
    
    Args:
        X: Feature matrix
        Y: Target coefficients matrix
        feature_names: Names of features
        **kwargs: Additional arguments for training
        
    Returns:
        Trained predictor
    """
    predictor = GearCoefficientPredictor(model_type='neural')
    predictor.fit(X, Y, feature_names=feature_names, **kwargs)
    return predictor


def save_models(models: Dict[str, GearCoefficientPredictor], 
               base_filename: Optional[str] = None) -> Dict[str, str]:
    """
    Save multiple trained models.
    
    Args:
        models: Dictionary of models to save
        base_filename: Base filename for saved models
        
    Returns:
        Dictionary mapping model names to saved file paths
    """
    if base_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"gear_models_{timestamp}"
    
    saved_paths = {}
    
    for model_name, model in models.items():
        filename = f"{base_filename}_{model_name}.pkl"
        filepath = model.save(filename)
        saved_paths[model_name] = filepath
    
    return saved_paths


def load_models(model_paths: Dict[str, str]) -> Dict[str, GearCoefficientPredictor]:
    """
    Load multiple trained models.
    
    Args:
        model_paths: Dictionary mapping model names to file paths
        
    Returns:
        Dictionary of loaded models
    """
    models = {}
    
    for model_name, filepath in model_paths.items():
        predictor = GearCoefficientPredictor()
        predictor.load(filepath)
        models[model_name] = predictor
    
    return models


def evaluate_model(predictor: GearCoefficientPredictor, 
                  X_test: np.ndarray, Y_test: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Args:
        predictor: Trained predictor
        X_test: Test features
        Y_test: Test targets
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    Y_pred = predictor.predict(X_test)
    
    # Compute metrics for each coefficient
    coefficient_metrics = []
    for i in range(Y_test.shape[1]):
        y_true = Y_test[:, i]
        y_pred = Y_pred[:, i]
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        coefficient_metrics.append({
            'coefficient': i,
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        })
    
    # Overall metrics
    overall_r2 = np.mean([m['r2'] for m in coefficient_metrics])
    overall_rmse = np.mean([m['rmse'] for m in coefficient_metrics])
    overall_mae = np.mean([m['mae'] for m in coefficient_metrics])
    
    return {
        'overall_r2': overall_r2,
        'overall_rmse': overall_rmse,
        'overall_mae': overall_mae,
        'coefficient_metrics': coefficient_metrics,
        'n_coefficients': len(coefficient_metrics),
        'n_samples': Y_test.shape[0]
    }


if __name__ == "__main__":
    # Test the regression module
    print("Testing regression module...")
    
    # Create synthetic test data
    n_samples = 500
    n_features = 9  # Number of gear parameters
    n_coefficients = 20  # Number of Fourier coefficients
    
    np.random.seed(42)
    X_test = np.random.randn(n_samples, n_features)
    
    # Create synthetic relationships
    Y_test = np.zeros((n_samples, n_coefficients))
    for i in range(n_coefficients):
        # Each coefficient depends on a few parameters
        weights = np.random.randn(n_features) * 0.1
        Y_test[:, i] = X_test @ weights + np.random.randn(n_samples) * 0.01
    
    feature_names = [f'param_{i}' for i in range(n_features)]
    
    # Test linear model
    print("Testing linear regression...")
    linear_predictor = train_linear_models(X_test, Y_test, feature_names, verbose=True)
    
    # Test neural model
    print("\nTesting neural network...")
    neural_predictor = train_neural_models(X_test, Y_test, feature_names, verbose=True)
    
    # Test predictions
    Y_pred_linear = linear_predictor.predict(X_test[:10])
    Y_pred_neural = neural_predictor.predict(X_test[:10])
    
    print(f"\nLinear predictions shape: {Y_pred_linear.shape}")
    print(f"Neural predictions shape: {Y_pred_neural.shape}")
    
    # Test save/load
    models = {'linear': linear_predictor, 'neural': neural_predictor}
    saved_paths = save_models(models)
    loaded_models = load_models(saved_paths)
    
    print(f"\nSaved and loaded {len(loaded_models)} models")
    print("Regression module test completed successfully!")
