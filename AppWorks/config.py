"""
Configuration constants for AppWorks package.

This module defines all configuration parameters including paths,
FFT settings, and other constants used throughout the package.
"""

import os
from typing import Dict, Any

# Get the directory containing this file
HERE = os.path.dirname(os.path.abspath(__file__))

# Core paths - Update this to match your PythonCore location
# Try multiple possible locations for PythonCore
possible_paths = [
    os.path.join(HERE, "..", "attached_assets"),
    os.path.join(HERE, "..", "PythonCore"),
    r"C:\Users\Labor\Desktop\GearEngineering\PythonCore",
    "./PythonCore",
    "../PythonCore"
]

PYTHONCORE_PATH = None
for path in possible_paths:
    if os.path.exists(path):
        PYTHONCORE_PATH = path
        break

if PYTHONCORE_PATH is None:
    PYTHONCORE_PATH = possible_paths[0]  # Default fallback
DATA_PATH = os.path.join(HERE, "..", "data")
MODELS_PATH = os.path.join(DATA_PATH, "models")

# FFT Configuration
FFT_POINTS = 2048      # Number of FFT points for high resolution
HARMONICS = 200        # Number of Fourier harmonics to extract

# Sampling Configuration
DEFAULT_SAMPLE_SIZE = 1000           # Default number of gear parameter samples
POLAR_RESAMPLE_POINTS = FFT_POINTS   # Points for uniform polar resampling

# Gear Parameter Ranges for Sampling
PARAMETER_RANGES: Dict[str, Dict[str, Any]] = {
    'module': {'min': 0.5, 'max': 10.0, 'type': 'uniform'},
    'teeth': {'min': 5, 'max': 200, 'type': 'randint'},
    'pressure_angle': {'min': 10.0, 'max': 35.0, 'type': 'uniform'},
    'profile_shift': {'min': -0.8, 'max': 0.8, 'type': 'uniform'},
    'addendum_factor': {'min': 0.5, 'max': 2.0, 'type': 'uniform'},
    'dedendum_factor': {'min': 0.8, 'max': 2.0, 'type': 'uniform'},
    'backlash_factor': {'min': 0.0, 'max': 0.5, 'type': 'uniform'},
    'edge_round_factor': {'min': 0.0, 'max': 0.3, 'type': 'uniform'},
    'root_round_factor': {'min': 0.0, 'max': 0.5, 'type': 'uniform'}
}

# Model Configuration
REGRESSION_MODELS = {
    'linear': {
        'enabled': True,
        'fit_intercept': True,
        'normalize': False
    },
    'neural': {
        'enabled': True,
        'hidden_layer_sizes': (100, 50),
        'activation': 'relu',
        'solver': 'adam',
        'max_iter': 1000,
        'random_state': 42
    }
}

# Symbolic Regression Configuration
SYMBOLIC_CONFIG = {
    'enabled': False,  # Set to True to enable symbolic regression
    'population_size': 5000,
    'generations': 20,
    'stopping_criteria': 0.01,
    'p_crossover': 0.7,
    'p_subtree_mutation': 0.1,
    'p_hoist_mutation': 0.05,
    'p_point_mutation': 0.1,
    'max_samples': 0.9,
    'verbose': 1,
    'parsimony_coefficient': 0.01,
    'random_state': 42
}

# Anomaly Configuration
ANOMALY_CONFIG = {
    'gaussian_dent': {
        'default_depth': 0.1,    # Default depth as fraction of radius
        'default_width': 0.2,    # Default width in radians
        'min_depth': 0.01,
        'max_depth': 0.5,
        'min_width': 0.05,
        'max_width': 1.0
    },
    'wear_pattern': {
        'default_severity': 0.05,  # Default wear severity
        'default_extent': 0.3,     # Default extent in radians
        'min_severity': 0.001,
        'max_severity': 0.2,
        'min_extent': 0.1,
        'max_extent': 2.0
    }
}

# File naming conventions
FILE_PATTERNS = {
    'profiles_csv': 'gear_profiles_{timestamp}.csv',
    'coefficients_csv': 'fourier_coefficients_{timestamp}.csv',
    'models_pkl': 'regression_models_{timestamp}.pkl',
    'symbolic_txt': 'symbolic_expressions_{timestamp}.txt'
}

# Ensure data directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)

# Initialize directories on import
ensure_directories()
