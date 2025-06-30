"""
Anomalies module for AppWorks.

This module provides utilities for injecting synthetic anomalies into gear profiles
to simulate wear, damage, and other real-world effects on the Fourier coefficients.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import warnings
from scipy.signal.windows import gaussian
from scipy.interpolate import interp1d

from .config import ANOMALY_CONFIG


def add_local_dent(r: np.ndarray, theta: np.ndarray, 
                   center: float, width: float, depth: float) -> np.ndarray:
    """
    Add a localized Gaussian dent to the radius profile.
    
    This simulates localized wear, pitting, or mechanical damage.
    
    Args:
        r: Original radius values
        theta: Corresponding angle values
        center: Angular position of dent center (radians)
        width: Angular width of dent (radians)
        depth: Depth of dent as fraction of local radius
        
    Returns:
        Modified radius array with dent
        
    Raises:
        ValueError: If parameters are invalid
    """
    if len(r) != len(theta):
        raise ValueError("r and theta arrays must have the same length")
    
    if len(r) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    # Validate parameters
    config = ANOMALY_CONFIG['gaussian_dent']
    if not config['min_depth'] <= depth <= config['max_depth']:
        raise ValueError(f"Depth must be between {config['min_depth']} and {config['max_depth']}")
    
    if not config['min_width'] <= width <= config['max_width']:
        raise ValueError(f"Width must be between {config['min_width']} and {config['max_width']}")
    
    # Normalize center to [0, 2π)
    center = np.mod(center, 2 * np.pi)
    
    # Handle periodic boundary conditions
    theta_normalized = np.mod(theta, 2 * np.pi)
    
    # Calculate angular distances (handling wrap-around)
    angular_distances = np.minimum(
        np.abs(theta_normalized - center),
        2 * np.pi - np.abs(theta_normalized - center)
    )
    
    # Create Gaussian dent profile
    sigma = width / 4  # Standard deviation for Gaussian
    dent_profile = depth * np.exp(-0.5 * (angular_distances / sigma) ** 2)
    
    # Apply dent (subtract from radius)
    r_modified = r - dent_profile * r  # Depth is relative to local radius
    
    # Ensure radius remains positive
    r_modified = np.maximum(r_modified, 0.01 * np.mean(r))
    
    return r_modified


def add_wear_pattern(r: np.ndarray, theta: np.ndarray,
                    start_angle: float, end_angle: float,
                    severity: float, pattern: str = 'linear') -> np.ndarray:
    """
    Add a wear pattern over a range of angles.
    
    This simulates gradual wear over a section of the gear.
    
    Args:
        r: Original radius values
        theta: Corresponding angle values  
        start_angle: Start of wear region (radians)
        end_angle: End of wear region (radians)
        severity: Wear severity as fraction of radius
        pattern: Wear pattern type ('linear', 'quadratic', 'exponential')
        
    Returns:
        Modified radius array with wear pattern
        
    Raises:
        ValueError: If parameters are invalid
    """
    if len(r) != len(theta):
        raise ValueError("r and theta arrays must have the same length")
    
    if len(r) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    # Validate severity
    config = ANOMALY_CONFIG['wear_pattern']
    if not config['min_severity'] <= severity <= config['max_severity']:
        raise ValueError(f"Severity must be between {config['min_severity']} and {config['max_severity']}")
    
    # Normalize angles to [0, 2π)
    start_angle = np.mod(start_angle, 2 * np.pi)
    end_angle = np.mod(end_angle, 2 * np.pi)
    theta_normalized = np.mod(theta, 2 * np.pi)
    
    # Handle wrap-around case
    if start_angle > end_angle:
        # Wear region wraps around 0
        in_wear_region = (theta_normalized >= start_angle) | (theta_normalized <= end_angle)
        # Calculate relative position within wear region
        relative_pos = np.zeros_like(theta_normalized)
        total_span = (2 * np.pi - start_angle) + end_angle
        
        mask1 = theta_normalized >= start_angle
        relative_pos[mask1] = (theta_normalized[mask1] - start_angle) / total_span
        
        mask2 = theta_normalized <= end_angle
        relative_pos[mask2] = (2 * np.pi - start_angle + theta_normalized[mask2]) / total_span
    else:
        # Normal case - no wrap around
        in_wear_region = (theta_normalized >= start_angle) & (theta_normalized <= end_angle)
        relative_pos = np.zeros_like(theta_normalized)
        span = end_angle - start_angle
        
        relative_pos[in_wear_region] = (theta_normalized[in_wear_region] - start_angle) / span
    
    # Create wear profile based on pattern
    wear_profile = np.zeros_like(r)
    
    if pattern == 'linear':
        wear_profile[in_wear_region] = severity * relative_pos[in_wear_region]
    elif pattern == 'quadratic':
        wear_profile[in_wear_region] = severity * (relative_pos[in_wear_region] ** 2)
    elif pattern == 'exponential':
        wear_profile[in_wear_region] = severity * (np.exp(relative_pos[in_wear_region]) - 1) / (np.e - 1)
    else:
        raise ValueError(f"Unknown wear pattern: {pattern}")
    
    # Apply wear (reduce radius)
    r_modified = r - wear_profile * r
    
    # Ensure radius remains positive
    r_modified = np.maximum(r_modified, 0.01 * np.mean(r))
    
    return r_modified


def add_multiple_dents(r: np.ndarray, theta: np.ndarray,
                      dent_specs: List[Dict[str, float]]) -> np.ndarray:
    """
    Add multiple localized dents to the profile.
    
    Args:
        r: Original radius values
        theta: Corresponding angle values
        dent_specs: List of dent specifications, each containing:
                   {'center': angle, 'width': width, 'depth': depth}
                   
    Returns:
        Modified radius array with all dents
    """
    r_modified = r.copy()
    
    for spec in dent_specs:
        r_modified = add_local_dent(
            r_modified, theta,
            spec['center'], spec['width'], spec['depth']
        )
    
    return r_modified


def add_random_noise(r: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    """
    Add random noise to radius profile.
    
    Args:
        r: Original radius values
        noise_level: Noise level as fraction of mean radius
        
    Returns:
        Noisy radius array
    """
    if noise_level <= 0:
        return r.copy()
    
    mean_radius = np.mean(r)
    noise = np.random.normal(0, noise_level * mean_radius, len(r))
    
    r_noisy = r + noise
    
    # Ensure radius remains positive
    r_noisy = np.maximum(r_noisy, 0.01 * mean_radius)
    
    return r_noisy


def add_periodic_distortion(r: np.ndarray, theta: np.ndarray,
                           frequency: int, amplitude: float,
                           phase: float = 0.0) -> np.ndarray:
    """
    Add periodic distortion to the profile.
    
    This simulates manufacturing errors or systematic wear patterns.
    
    Args:
        r: Original radius values
        theta: Corresponding angle values
        frequency: Distortion frequency (cycles per revolution)
        amplitude: Distortion amplitude as fraction of radius
        phase: Phase shift (radians)
        
    Returns:
        Modified radius array with periodic distortion
    """
    if amplitude <= 0:
        return r.copy()
    
    # Create periodic distortion
    distortion = amplitude * np.sin(frequency * theta + phase)
    
    # Apply distortion
    r_modified = r * (1 + distortion)
    
    # Ensure radius remains positive
    r_modified = np.maximum(r_modified, 0.01 * np.mean(r))
    
    return r_modified


def simulate_gear_aging(r: np.ndarray, theta: np.ndarray,
                       age_factor: float = 0.1,
                       n_dents: int = 3,
                       wear_severity: float = 0.05) -> np.ndarray:
    """
    Simulate comprehensive gear aging with multiple types of wear.
    
    Args:
        r: Original radius values
        theta: Corresponding angle values
        age_factor: Overall aging factor (0-1)
        n_dents: Number of random dents to add
        wear_severity: Severity of wear patterns
        
    Returns:
        Aged radius profile
    """
    r_aged = r.copy()
    
    if age_factor <= 0:
        return r_aged
    
    # Add random dents
    for _ in range(n_dents):
        center = np.random.uniform(0, 2 * np.pi)
        width = np.random.uniform(0.1, 0.3) * age_factor
        depth = np.random.uniform(0.01, 0.1) * age_factor
        
        r_aged = add_local_dent(r_aged, theta, center, width, depth)
    
    # Add wear patterns
    n_wear_regions = np.random.randint(1, 4)
    for _ in range(n_wear_regions):
        start = np.random.uniform(0, 2 * np.pi)
        span = np.random.uniform(0.5, 1.5)
        end = start + span
        
        severity = wear_severity * age_factor
        pattern = np.random.choice(['linear', 'quadratic', 'exponential'])
        
        r_aged = add_wear_pattern(r_aged, theta, start, end, severity, pattern)
    
    # Add noise
    noise_level = 0.005 * age_factor
    r_aged = add_random_noise(r_aged, noise_level)
    
    # Add periodic distortions
    n_distortions = np.random.randint(1, 3)
    for _ in range(n_distortions):
        frequency = np.random.randint(2, 10)
        amplitude = np.random.uniform(0.001, 0.01) * age_factor
        phase = np.random.uniform(0, 2 * np.pi)
        
        r_aged = add_periodic_distortion(r_aged, theta, frequency, amplitude, phase)
    
    return r_aged


def analyze_anomaly_effects(r_original: np.ndarray, r_modified: np.ndarray,
                           theta: np.ndarray) -> Dict[str, Any]:
    """
    Analyze the effects of anomalies on the gear profile.
    
    Args:
        r_original: Original radius values
        r_modified: Modified radius values
        theta: Angle values
        
    Returns:
        Dictionary with analysis results
    """
    # Calculate differences
    radius_diff = r_modified - r_original
    relative_diff = radius_diff / r_original
    
    # Statistics
    analysis = {
        'max_absolute_change': np.max(np.abs(radius_diff)),
        'mean_absolute_change': np.mean(np.abs(radius_diff)),
        'std_absolute_change': np.std(np.abs(radius_diff)),
        'max_relative_change': np.max(np.abs(relative_diff)),
        'mean_relative_change': np.mean(np.abs(relative_diff)),
        'std_relative_change': np.std(np.abs(relative_diff)),
        'rms_error': np.sqrt(np.mean(radius_diff**2)),
        'affected_fraction': np.sum(np.abs(relative_diff) > 0.001) / len(r_original),
        'max_change_location': theta[np.argmax(np.abs(radius_diff))],
        'volume_change': np.trapz(r_modified**2, theta) - np.trapz(r_original**2, theta)
    }
    
    return analysis


if __name__ == "__main__":
    # Test anomaly injection
    print("Testing anomaly injection...")
    
    # Create test profile (simple circle)
    theta_test = np.linspace(0, 2*np.pi, 200, endpoint=False)
    r_test = 10 * np.ones_like(theta_test)  # Perfect circle
    
    # Test single dent
    r_dent = add_local_dent(r_test, theta_test, 
                           center=np.pi/2, width=0.3, depth=0.1)
    
    dent_analysis = analyze_anomaly_effects(r_test, r_dent, theta_test)
    print(f"Single dent - Max change: {dent_analysis['max_absolute_change']:.3f}")
    
    # Test wear pattern
    r_wear = add_wear_pattern(r_test, theta_test,
                             start_angle=0, end_angle=np.pi/2,
                             severity=0.05, pattern='linear')
    
    wear_analysis = analyze_anomaly_effects(r_test, r_wear, theta_test)
    print(f"Wear pattern - Max change: {wear_analysis['max_absolute_change']:.3f}")
    
    # Test comprehensive aging
    r_aged = simulate_gear_aging(r_test, theta_test, age_factor=0.2)
    
    aging_analysis = analyze_anomaly_effects(r_test, r_aged, theta_test)
    print(f"Aging simulation - Max change: {aging_analysis['max_absolute_change']:.3f}")
    print(f"Affected fraction: {aging_analysis['affected_fraction']:.1%}")
    
    print("Anomaly injection test completed successfully!")
