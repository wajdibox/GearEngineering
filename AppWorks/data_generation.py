"""
Data generation module for AppWorks.

This module handles sampling of gear parameters and generation of gear profiles
using the existing PythonCore analytical engine.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import warnings
from datetime import datetime

# Add PythonCore to path
from .config import PYTHONCORE_PATH, PARAMETER_RANGES, DEFAULT_SAMPLE_SIZE

# Ensure PythonCore is available
if PYTHONCORE_PATH not in sys.path:
    sys.path.insert(0, PYTHONCORE_PATH)

try:
    from PythonCore.gear_parameters import GearParameters
    from PythonCore.geometry_generator import ScientificGearGeometry
except ImportError as e:
    raise ImportError(
        f"Failed to import PythonCore modules. Please ensure PythonCore is available at: {PYTHONCORE_PATH}\n"
        f"Original error: {e}"
    )


def sample_parameters(n_samples: int = DEFAULT_SAMPLE_SIZE, 
                     random_state: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Generate random samples of gear parameters within valid ranges.
    
    Args:
        n_samples: Number of parameter sets to generate
        random_state: Random seed for reproducibility
        
    Returns:
        List of dictionaries containing gear parameters
        
    Raises:
        ValueError: If n_samples is not positive
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    samples = []
    
    for _ in range(n_samples):
        sample = {}
        
        for param_name, config in PARAMETER_RANGES.items():
            if config['type'] == 'uniform':
                sample[param_name] = np.random.uniform(config['min'], config['max'])
            elif config['type'] == 'randint':
                sample[param_name] = np.random.randint(config['min'], config['max'] + 1)
            else:
                raise ValueError(f"Unknown parameter type: {config['type']}")
        
        samples.append(sample)
    
    return samples


def validate_parameters(params: Dict[str, Any]) -> bool:
    """
    Validate gear parameters against known constraints.
    
    Args:
        params: Dictionary of gear parameters
        
    Returns:
        True if parameters are valid, False otherwise
    """
    try:
        # Create GearParameters object to leverage existing validation
        gear_params = GearParameters(
            module=params['module'],
            teeth=params['teeth'],
            pressure_angle=params['pressure_angle'],
            profile_shift=params['profile_shift'],
            addendum_factor=params['addendum_factor'],
            dedendum_factor=params['dedendum_factor'],
            backlash_factor=params['backlash_factor'],
            edge_round_factor=params['edge_round_factor'],
            root_round_factor=params['root_round_factor']
        )
        
        # Use existing validation
        errors = gear_params.validate()
        return len(errors) == 0
        
    except Exception:
        return False


def get_profile(params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate gear profile using PythonCore analytical engine.
    
    Args:
        params: Dictionary containing gear parameters
        
    Returns:
        Tuple of (x_coords, y_coords) as numpy arrays
        
    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If profile generation fails
    """
    # Validate parameters first
    if not validate_parameters(params):
        raise ValueError(f"Invalid gear parameters: {params}")
    
    try:
        # Create GearParameters object
        gear_params = GearParameters(
            module=params['module'],
            teeth=params['teeth'],
            pressure_angle=params['pressure_angle'],
            profile_shift=params['profile_shift'],
            addendum_factor=params['addendum_factor'],
            dedendum_factor=params['dedendum_factor'],
            backlash_factor=params['backlash_factor'],
            edge_round_factor=params['edge_round_factor'],
            root_round_factor=params['root_round_factor']
        )
        
        # Generate geometry
        geometry = ScientificGearGeometry(gear_params)
        geometry.generate_single_tooth()
        geometry.generate_full_gear()
        
        # Extract coordinates
        x_coords = np.array(geometry.full_gear_x)
        y_coords = np.array(geometry.full_gear_y)
        
        # Validate output
        if len(x_coords) == 0 or len(y_coords) == 0:
            raise RuntimeError("Generated profile is empty")
        
        if len(x_coords) != len(y_coords):
            raise RuntimeError("Coordinate arrays have different lengths")
        
        if np.any(~np.isfinite(x_coords)) or np.any(~np.isfinite(y_coords)):
            raise RuntimeError("Generated profile contains invalid values")
        
        return x_coords, y_coords
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate gear profile: {e}")


def generate_dataset(n_samples: int = DEFAULT_SAMPLE_SIZE,
                    random_state: Optional[int] = None,
                    validation_split: float = 0.2,
                    verbose: bool = True) -> Tuple[List[Dict[str, Any]], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Generate a complete dataset of gear parameters and corresponding profiles.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        validation_split: Fraction of data to reserve for validation
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (parameter_list, profile_list)
        
    Raises:
        ValueError: If validation_split is not in [0, 1)
        RuntimeError: If too many samples fail to generate
    """
    if not 0 <= validation_split < 1:
        raise ValueError("validation_split must be in [0, 1)")
    
    if verbose:
        print(f"Generating {n_samples} gear samples...")
        print(f"PythonCore path: {PYTHONCORE_PATH}")
    
    # Sample parameters
    parameter_samples = sample_parameters(n_samples, random_state)
    
    successful_params = []
    successful_profiles = []
    failed_count = 0
    
    for i, params in enumerate(parameter_samples):
        try:
            # Generate profile
            x_coords, y_coords = get_profile(params)
            
            successful_params.append(params)
            successful_profiles.append((x_coords, y_coords))
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_samples} samples ({failed_count} failures)")
                
        except Exception as e:
            failed_count += 1
            if verbose and failed_count <= 10:  # Show first 10 failures
                print(f"Warning: Failed to generate sample {i + 1}: {e}")
    
    success_rate = len(successful_params) / n_samples
    
    if verbose:
        print(f"Dataset generation complete:")
        print(f"  Successful: {len(successful_params)}")
        print(f"  Failed: {failed_count}")
        print(f"  Success rate: {success_rate:.1%}")
    
    if success_rate < 0.5:
        raise RuntimeError(f"Too many failures ({success_rate:.1%} success rate)")
    
    return successful_params, successful_profiles


def save_dataset(params: List[Dict[str, Any]], 
                profiles: List[Tuple[np.ndarray, np.ndarray]],
                base_filename: Optional[str] = None) -> str:
    """
    Save dataset to CSV files.
    
    Args:
        params: List of parameter dictionaries
        profiles: List of (x, y) coordinate tuples
        base_filename: Base filename (timestamp will be added if None)
        
    Returns:
        Path to saved parameter file
    """
    from .config import DATA_PATH, FILE_PATTERNS
    
    if base_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"gear_data_{timestamp}"
    
    # Save parameters
    params_df = pd.DataFrame(params)
    params_file = os.path.join(DATA_PATH, f"{base_filename}_params.csv")
    params_df.to_csv(params_file, index=False)
    
    # Save profiles (flattened format)
    profile_data = []
    for i, (x_coords, y_coords) in enumerate(profiles):
        for j, (x, y) in enumerate(zip(x_coords, y_coords)):
            profile_data.append({
                'sample_id': i,
                'point_id': j,
                'x': x,
                'y': y
            })
    
    profiles_df = pd.DataFrame(profile_data)
    profiles_file = os.path.join(DATA_PATH, f"{base_filename}_profiles.csv")
    profiles_df.to_csv(profiles_file, index=False)
    
    print(f"Dataset saved:")
    print(f"  Parameters: {params_file}")
    print(f"  Profiles: {profiles_file}")
    
    return params_file


def load_dataset(params_file: str) -> Tuple[List[Dict[str, Any]], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Load dataset from CSV files.
    
    Args:
        params_file: Path to parameters CSV file
        
    Returns:
        Tuple of (parameter_list, profile_list)
    """
    # Load parameters
    params_df = pd.read_csv(params_file)
    params = params_df.to_dict('records')
    
    # Load profiles
    profiles_file = params_file.replace('_params.csv', '_profiles.csv')
    profiles_df = pd.read_csv(profiles_file)
    
    # Reconstruct profiles
    profiles = []
    for sample_id in profiles_df['sample_id'].unique():
        sample_data = profiles_df[profiles_df['sample_id'] == sample_id].sort_values('point_id')
        x_coords = sample_data['x'].values
        y_coords = sample_data['y'].values
        profiles.append((x_coords, y_coords))
    
    return params, profiles


if __name__ == "__main__":
    # Test the data generation
    print("Testing data generation...")
    
    # Generate small test dataset
    params, profiles = generate_dataset(n_samples=10, verbose=True)
    
    # Save and reload
    saved_file = save_dataset(params, profiles)
    loaded_params, loaded_profiles = load_dataset(saved_file)
    
    print(f"Successfully generated, saved, and loaded {len(loaded_params)} samples")
