"""
Symbolic regression module for AppWorks.

This module provides optional symbolic regression capabilities to discover
closed-form mathematical expressions for Fourier coefficients as functions
of gear parameters using genetic programming.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import warnings
import os
from datetime import datetime

# Optional imports for symbolic regression
try:
    from gplearn.genetic import SymbolicRegressor
    GPLEARN_AVAILABLE = True
except ImportError:
    GPLEARN_AVAILABLE = False
    warnings.warn("gplearn not available, symbolic regression disabled")

try:
    import pysr
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    warnings.warn("PySR not available, advanced symbolic regression disabled")

from .config import SYMBOLIC_CONFIG, DATA_PATH


class SymbolicCoefficientRegressor:
    """
    Symbolic regression for discovering mathematical relationships between
    gear parameters and Fourier coefficients.
    """
    
    def __init__(self, method: str = 'gplearn', **kwargs):
        """
        Initialize symbolic regressor.
        
        Args:
            method: Method to use ('gplearn' or 'pysr')
            **kwargs: Additional configuration parameters
        """
        self.method = method
        self.config = {**SYMBOLIC_CONFIG, **kwargs}
        self.regressors = {}
        self.expressions = {}
        self.feature_names = None
        self.is_fitted = False
        
        # Check availability
        if method == 'gplearn' and not GPLEARN_AVAILABLE:
            raise ImportError("gplearn is required for symbolic regression")
        elif method == 'pysr' and not PYSR_AVAILABLE:
            raise ImportError("PySR is required for advanced symbolic regression")
    
    def _create_gplearn_regressor(self) -> SymbolicRegressor:
        """Create a gplearn SymbolicRegressor instance."""
        return SymbolicRegressor(
            population_size=self.config['population_size'],
            generations=self.config['generations'],
            stopping_criteria=self.config['stopping_criteria'],
            p_crossover=self.config['p_crossover'],
            p_subtree_mutation=self.config['p_subtree_mutation'],
            p_hoist_mutation=self.config['p_hoist_mutation'],
            p_point_mutation=self.config['p_point_mutation'],
            max_samples=self.config['max_samples'],
            verbose=self.config['verbose'],
            parsimony_coefficient=self.config['parsimony_coefficient'],
            random_state=self.config['random_state']
        )
    
    def _create_pysr_regressor(self) -> 'pysr.PySRRegressor':
        """Create a PySR regressor instance."""
        return pysr.PySRRegressor(
            niterations=self.config['generations'],
            populations=self.config['population_size'] // 100,
            procs=4,
            random_state=self.config['random_state'],
            verbosity=self.config['verbose']
        )
    
    def fit(self, X: np.ndarray, Y: np.ndarray,
            feature_names: Optional[List[str]] = None,
            max_coefficients: Optional[int] = None,
            verbose: bool = True) -> Dict[str, Any]:
        """
        Fit symbolic regression models to discover coefficient relationships.
        
        Args:
            X: Feature matrix (gear parameters)
            Y: Target matrix (Fourier coefficients)
            feature_names: Names of input features
            max_coefficients: Maximum number of coefficients to process
            verbose: Whether to print progress
            
        Returns:
            Dictionary with fitting results
        """
        if not self.config['enabled']:
            raise RuntimeError("Symbolic regression is disabled in configuration")
        
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples")
        
        n_samples, n_features = X.shape
        n_coefficients = Y.shape[1]
        
        # Limit coefficients if specified
        if max_coefficients is not None:
            n_coefficients = min(n_coefficients, max_coefficients)
            Y = Y[:, :n_coefficients]
        
        if verbose:
            print(f"Symbolic regression ({self.method}):")
            print(f"  Samples: {n_samples}")
            print(f"  Features: {n_features}")
            print(f"  Coefficients: {n_coefficients}")
            print(f"  Generations: {self.config['generations']}")
        
        # Store metadata
        self.feature_names = feature_names or [f'x{i}' for i in range(n_features)]
        
        # Results tracking
        results = {
            'method': self.method,
            'n_coefficients': n_coefficients,
            'successful_fits': 0,
            'failed_fits': 0,
            'best_scores': [],
            'expressions': {}
        }
        
        # Fit model for each coefficient
        for coeff_idx in range(n_coefficients):
            if verbose:
                print(f"  Fitting coefficient {coeff_idx + 1}/{n_coefficients}")
            
            y_target = Y[:, coeff_idx]
            
            # Skip if coefficient is constant (no variation)
            if np.std(y_target) < 1e-10:
                if verbose:
                    print(f"    Skipping constant coefficient {coeff_idx}")
                results['failed_fits'] += 1
                continue
            
            try:
                if self.method == 'gplearn':
                    regressor = self._create_gplearn_regressor()
                    regressor.fit(X, y_target)
                    
                    # Store results
                    self.regressors[coeff_idx] = regressor
                    self.expressions[coeff_idx] = str(regressor._program)
                    
                    # Get score
                    score = regressor.score(X, y_target)
                    results['best_scores'].append(score)
                    
                    if verbose:
                        print(f"    Score: {score:.4f}")
                        print(f"    Expression: {self.expressions[coeff_idx]}")
                
                elif self.method == 'pysr':
                    regressor = self._create_pysr_regressor()
                    regressor.fit(X, y_target, variable_names=self.feature_names)
                    
                    # Store results
                    self.regressors[coeff_idx] = regressor
                    
                    # Get best equation
                    if hasattr(regressor, 'equations_'):
                        best_eq = regressor.equations_.iloc[-1]
                        self.expressions[coeff_idx] = best_eq['equation']
                        score = best_eq['score']
                        results['best_scores'].append(score)
                        
                        if verbose:
                            print(f"    Score: {score:.4f}")
                            print(f"    Expression: {self.expressions[coeff_idx]}")
                    else:
                        results['failed_fits'] += 1
                        continue
                
                results['successful_fits'] += 1
                
            except Exception as e:
                if verbose:
                    print(f"    Failed: {e}")
                results['failed_fits'] += 1
                continue
        
        self.is_fitted = True
        
        # Summary statistics
        if results['best_scores']:
            results['mean_score'] = np.mean(results['best_scores'])
            results['std_score'] = np.std(results['best_scores'])
            results['max_score'] = np.max(results['best_scores'])
        
        results['expressions'] = self.expressions
        
        if verbose:
            print(f"Symbolic regression complete:")
            print(f"  Successful: {results['successful_fits']}")
            print(f"  Failed: {results['failed_fits']}")
            if results['best_scores']:
                print(f"  Mean score: {results['mean_score']:.4f}")
                print(f"  Best score: {results['max_score']:.4f}")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict coefficients using discovered symbolic expressions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted coefficients
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        n_samples = X.shape[0]
        n_coefficients = len(self.regressors)
        
        predictions = np.zeros((n_samples, n_coefficients))
        
        for coeff_idx, regressor in self.regressors.items():
            try:
                pred = regressor.predict(X)
                predictions[:, coeff_idx] = pred
            except Exception as e:
                warnings.warn(f"Prediction failed for coefficient {coeff_idx}: {e}")
                predictions[:, coeff_idx] = 0
        
        return predictions
    
    def get_expressions(self) -> Dict[int, str]:
        """
        Get the discovered mathematical expressions.
        
        Returns:
            Dictionary mapping coefficient indices to expression strings
        """
        return self.expressions.copy()
    
    def save_expressions(self, filename: Optional[str] = None) -> str:
        """
        Save discovered expressions to a text file.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"symbolic_expressions_{timestamp}.txt"
        
        filepath = os.path.join(DATA_PATH, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"Symbolic Regression Results\n")
            f.write(f"Method: {self.method}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Features: {', '.join(self.feature_names)}\n")
            f.write(f"\n{'='*50}\n\n")
            
            for coeff_idx in sorted(self.expressions.keys()):
                expression = self.expressions[coeff_idx]
                f.write(f"Coefficient {coeff_idx}:\n")
                f.write(f"  {expression}\n\n")
        
        print(f"Expressions saved to: {filepath}")
        return filepath


def symbolic_regression(X: np.ndarray, Y: np.ndarray,
                       feature_names: Optional[List[str]] = None,
                       method: str = 'gplearn',
                       max_coefficients: Optional[int] = 20,
                       **kwargs) -> SymbolicCoefficientRegressor:
    """
    Perform symbolic regression to discover coefficient relationships.
    
    Args:
        X: Feature matrix (gear parameters)
        Y: Target matrix (Fourier coefficients) 
        feature_names: Names of input features
        method: Symbolic regression method ('gplearn' or 'pysr')
        max_coefficients: Maximum coefficients to process
        **kwargs: Additional configuration parameters
        
    Returns:
        Fitted symbolic regressor
        
    Raises:
        RuntimeError: If symbolic regression is disabled or unavailable
    """
    if not SYMBOLIC_CONFIG['enabled']:
        raise RuntimeError("Symbolic regression is disabled in configuration")
    
    regressor = SymbolicCoefficientRegressor(method=method, **kwargs)
    regressor.fit(X, Y, feature_names=feature_names, 
                 max_coefficients=max_coefficients)
    
    return regressor


def analyze_expressions(expressions: Dict[int, str],
                       feature_names: List[str]) -> Dict[str, Any]:
    """
    Analyze the complexity and patterns in discovered expressions.
    
    Args:
        expressions: Dictionary of expressions
        feature_names: Names of input features
        
    Returns:
        Analysis results
    """
    if not expressions:
        return {'n_expressions': 0}
    
    # Basic statistics
    expression_lengths = [len(expr) for expr in expressions.values()]
    
    # Count feature usage
    feature_usage = {name: 0 for name in feature_names}
    for expr in expressions.values():
        for feature in feature_names:
            if feature in expr:
                feature_usage[feature] += 1
    
    # Count operators
    operators = ['+', '-', '*', '/', '**', 'sin', 'cos', 'exp', 'log']
    operator_usage = {op: 0 for op in operators}
    for expr in expressions.values():
        for op in operators:
            operator_usage[op] += expr.count(op)
    
    analysis = {
        'n_expressions': len(expressions),
        'mean_length': np.mean(expression_lengths),
        'std_length': np.std(expression_lengths),
        'max_length': np.max(expression_lengths),
        'min_length': np.min(expression_lengths),
        'feature_usage': feature_usage,
        'operator_usage': operator_usage,
        'most_used_feature': max(feature_usage, key=feature_usage.get),
        'most_used_operator': max(operator_usage, key=operator_usage.get)
    }
    
    return analysis


if __name__ == "__main__":
    # Test symbolic regression (if available)
    print("Testing symbolic regression...")
    
    if not GPLEARN_AVAILABLE:
        print("gplearn not available, skipping test")
        exit(0)
    
    # Create test data with known relationships
    np.random.seed(42)
    n_samples = 200
    n_features = 4
    
    X_test = np.random.uniform(-2, 2, (n_samples, n_features))
    
    # Create synthetic relationships
    Y_test = np.zeros((n_samples, 3))
    Y_test[:, 0] = X_test[:, 0] + 2 * X_test[:, 1]  # Linear
    Y_test[:, 1] = X_test[:, 0] * X_test[:, 1]      # Product
    Y_test[:, 2] = X_test[:, 0]**2 + X_test[:, 2]   # Quadratic
    
    # Add noise
    Y_test += np.random.normal(0, 0.1, Y_test.shape)
    
    feature_names = ['x0', 'x1', 'x2', 'x3']
    
    # Test symbolic regression
    try:
        # Enable symbolic regression for testing
        config_override = {'enabled': True, 'generations': 10, 'population_size': 100}
        regressor = SymbolicCoefficientRegressor(method='gplearn', **config_override)
        results = regressor.fit(X_test, Y_test, feature_names=feature_names)
        
        print(f"Fitted {results['successful_fits']} expressions")
        
        # Test predictions
        Y_pred = regressor.predict(X_test[:10])
        print(f"Predictions shape: {Y_pred.shape}")
        
        # Get expressions
        expressions = regressor.get_expressions()
        for i, expr in expressions.items():
            print(f"Coefficient {i}: {expr}")
        
        # Save expressions
        saved_file = regressor.save_expressions()
        
        print("Symbolic regression test completed successfully!")
        
    except Exception as e:
        print(f"Symbolic regression test failed: {e}")
