"""
AppWorks - Hybrid Fourier-based Gear Modeling Package

A complete Python package for converting analytical gear profiles into tunable
Fourier coefficient representations for digital twin applications.

This package integrates with PythonCore to:
- Sample gear parameters and generate analytical profiles
- Convert profiles to polar coordinates with uniform resampling
- Extract Fourier coefficients using high-performance FFT
- Train regression models to predict coefficients from parameters
- Discover symbolic expressions for coefficient relationships
- Inject synthetic anomalies for wear simulation
"""

from .config import *
from .data_generation import sample_parameters, get_profile
from .fft_extraction import polar_sample, compute_fft
from .regression import train_linear_models, train_neural_models, save_models, load_models
from .symbolic import symbolic_regression
from .anomalies import add_local_dent, add_wear_pattern
from .main import run_pipeline

__version__ = "1.0.0"
__author__ = "AppWorks Team"
__description__ = "Hybrid Fourier-based gear modeling for digital twins"

__all__ = [
    # Configuration
    'PYTHONCORE_PATH', 'DATA_PATH', 'FFT_POINTS', 'HARMONICS',
    # Data generation
    'sample_parameters', 'get_profile',
    # FFT extraction
    'polar_sample', 'compute_fft',
    # Regression
    'train_linear_models', 'train_neural_models', 'save_models', 'load_models',
    # Symbolic regression
    'symbolic_regression',
    # Anomalies
    'add_local_dent', 'add_wear_pattern',
    # Main pipeline
    'run_pipeline'
]
