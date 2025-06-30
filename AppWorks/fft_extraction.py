"""
FFT extraction module for AppWorks.

This module handles conversion of gear profiles to polar coordinates,
uniform resampling, and high-performance FFT computation using pyFFTW.
"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from typing import Tuple, Optional, List, Dict, Any
import warnings

# High-performance FFT
try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as fft
    PYFFTW_AVAILABLE = True
    
    # Enable multi-threading for better performance
    pyfftw.config.NUM_THREADS = 4
    pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
    
except ImportError:
    import numpy.fft as fft
    PYFFTW_AVAILABLE = False
    warnings.warn("pyFFTW not available, falling back to numpy.fft (slower)")

from .config import FFT_POINTS, HARMONICS, POLAR_RESAMPLE_POINTS


def cartesian_to_polar(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to polar coordinates.
    
    Args:
        x: X coordinates
        y: Y coordinates
        
    Returns:
        Tuple of (radius, theta) arrays
        
    Raises:
        ValueError: If input arrays are invalid
    """
    if len(x) != len(y):
        raise ValueError("x and y arrays must have the same length")
    
    if len(x) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    # Convert to polar
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Ensure theta is in [0, 2π)
    theta = np.mod(theta, 2 * np.pi)
    
    # Check for invalid values
    if np.any(~np.isfinite(radius)) or np.any(~np.isfinite(theta)):
        raise ValueError("Generated polar coordinates contain invalid values")
    
    return radius, theta


def sort_by_angle(radius: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort polar coordinates by angle.
    
    Args:
        radius: Radius values
        theta: Angle values
        
    Returns:
        Tuple of sorted (radius, theta) arrays
    """
    # Sort by theta
    sort_indices = np.argsort(theta)
    radius_sorted = radius[sort_indices]
    theta_sorted = theta[sort_indices]
    
    return radius_sorted, theta_sorted


def ensure_periodicity(radius: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure the profile is periodic by adding the first point at the end if needed.
    
    Args:
        radius: Radius values
        theta: Angle values
        
    Returns:
        Tuple of periodic (radius, theta) arrays
    """
    # Check if we need to close the loop
    if abs(theta[-1] - theta[0]) < (2 * np.pi - 0.1):
        # Add first point at 2π to ensure periodicity
        radius_periodic = np.append(radius, radius[0])
        theta_periodic = np.append(theta, theta[0] + 2 * np.pi)
    else:
        radius_periodic = radius
        theta_periodic = theta
    
    return radius_periodic, theta_periodic


def polar_sample(x: np.ndarray, y: np.ndarray, 
                num_points: int = POLAR_RESAMPLE_POINTS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to polar and uniformly resample.
    
    This function:
    1. Converts (x, y) to polar coordinates (r, θ)
    2. Sorts by angle
    3. Performs periodic linear interpolation to uniform angular grid
    4. Returns uniformly sampled (theta_uniform, r_uniform)
    
    Args:
        x: X coordinates
        y: Y coordinates
        num_points: Number of points for uniform resampling
        
    Returns:
        Tuple of (theta_uniform, r_uniform) arrays
        
    Raises:
        ValueError: If input is invalid or interpolation fails
    """
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    
    # Convert to polar coordinates
    radius, theta = cartesian_to_polar(x, y)
    
    # Sort by angle
    radius_sorted, theta_sorted = sort_by_angle(radius, theta)
    
    # Ensure periodicity
    radius_periodic, theta_periodic = ensure_periodicity(radius_sorted, theta_sorted)
    
    try:
        # Create interpolation function
        # Use 'linear' interpolation with periodic boundary conditions
        interp_func = interp1d(
            theta_periodic, 
            radius_periodic, 
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate',
            assume_sorted=True
        )
        
        # Create uniform angular grid
        theta_uniform = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        
        # Interpolate to uniform grid
        r_uniform = interp_func(theta_uniform)
        
        # Handle any NaN values that might have appeared
        if np.any(~np.isfinite(r_uniform)):
            # Find valid indices
            valid_mask = np.isfinite(r_uniform)
            if np.sum(valid_mask) == 0:
                raise ValueError("All interpolated values are invalid")
            
            # Use nearest neighbor for invalid points
            for i in range(len(r_uniform)):
                if not np.isfinite(r_uniform[i]):
                    # Find nearest valid point
                    valid_indices = np.where(valid_mask)[0]
                    distances = np.abs(valid_indices - i)
                    nearest_idx = valid_indices[np.argmin(distances)]
                    r_uniform[i] = r_uniform[nearest_idx]
        
        return theta_uniform, r_uniform
        
    except Exception as e:
        raise ValueError(f"Polar resampling failed: {e}")


def compute_fft(r_uniform: np.ndarray, harmonics: int = HARMONICS) -> np.ndarray:
    """
    Compute FFT of uniformly sampled radius function and extract harmonics.
    
    Args:
        r_uniform: Uniformly sampled radius values
        harmonics: Number of harmonics to extract
        
    Returns:
        Flattened array of [real_0, real_1, ..., real_H, imag_0, imag_1, ..., imag_H]
        
    Raises:
        ValueError: If input is invalid or FFT computation fails
    """
    if len(r_uniform) == 0:
        raise ValueError("Input array cannot be empty")
    
    if harmonics <= 0:
        raise ValueError("Number of harmonics must be positive")
    
    if harmonics > len(r_uniform) // 2:
        raise ValueError(f"Number of harmonics ({harmonics}) cannot exceed half the input length ({len(r_uniform) // 2})")
    
    try:
        # Ensure input is finite
        if np.any(~np.isfinite(r_uniform)):
            raise ValueError("Input contains invalid values")
        
        # Pad or truncate to desired FFT size
        if len(r_uniform) < FFT_POINTS:
            # Zero-pad
            r_padded = np.zeros(FFT_POINTS)
            r_padded[:len(r_uniform)] = r_uniform
        elif len(r_uniform) > FFT_POINTS:
            # Truncate
            r_padded = r_uniform[:FFT_POINTS]
        else:
            r_padded = r_uniform.copy()
        
        # Apply window function to reduce spectral leakage
        window = np.hanning(len(r_padded))
        r_windowed = r_padded * window
        
        # Compute FFT
        fft_result = fft.fft(r_windowed)
        
        # Extract the desired number of harmonics
        fft_truncated = fft_result[:harmonics]
        
        # Split into real and imaginary parts
        real_parts = np.real(fft_truncated)
        imag_parts = np.imag(fft_truncated)
        
        # Concatenate real and imaginary parts
        coefficients = np.concatenate([real_parts, imag_parts])
        
        # Validate output
        if np.any(~np.isfinite(coefficients)):
            raise ValueError("FFT computation produced invalid results")
        
        return coefficients
        
    except Exception as e:
        raise ValueError(f"FFT computation failed: {e}")


def extract_fourier_descriptors(x: np.ndarray, y: np.ndarray, 
                               harmonics: int = HARMONICS) -> np.ndarray:
    """
    Complete pipeline: Cartesian → Polar → Uniform sampling → FFT → Coefficients.
    
    Args:
        x: X coordinates
        y: Y coordinates
        harmonics: Number of harmonics to extract
        
    Returns:
        Fourier coefficients array
        
    Raises:
        ValueError: If extraction fails at any step
    """
    try:
        # Convert to polar and resample uniformly
        theta_uniform, r_uniform = polar_sample(x, y)
        
        # Compute FFT and extract coefficients
        coefficients = compute_fft(r_uniform, harmonics)
        
        return coefficients
        
    except Exception as e:
        raise ValueError(f"Fourier descriptor extraction failed: {e}")


def process_dataset(profiles: List[Tuple[np.ndarray, np.ndarray]], 
                   harmonics: int = HARMONICS,
                   verbose: bool = True) -> np.ndarray:
    """
    Process a complete dataset of profiles to extract Fourier coefficients.
    
    Args:
        profiles: List of (x, y) coordinate tuples
        harmonics: Number of harmonics to extract
        verbose: Whether to print progress
        
    Returns:
        2D array where each row contains coefficients for one profile
        
    Raises:
        ValueError: If processing fails
    """
    if len(profiles) == 0:
        raise ValueError("Profile list cannot be empty")
    
    n_profiles = len(profiles)
    n_coefficients = 2 * harmonics  # Real + imaginary parts
    
    # Initialize result array
    coefficients_matrix = np.zeros((n_profiles, n_coefficients))
    
    failed_count = 0
    
    for i, (x, y) in enumerate(profiles):
        try:
            coefficients = extract_fourier_descriptors(x, y, harmonics)
            coefficients_matrix[i] = coefficients
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{n_profiles} profiles ({failed_count} failures)")
                
        except Exception as e:
            failed_count += 1
            if verbose and failed_count <= 10:
                print(f"Warning: Failed to process profile {i}: {e}")
            
            # Fill with zeros for failed profiles
            coefficients_matrix[i] = 0
    
    success_rate = (n_profiles - failed_count) / n_profiles
    
    if verbose:
        print(f"FFT processing complete:")
        print(f"  Successful: {n_profiles - failed_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Success rate: {success_rate:.1%}")
    
    if success_rate < 0.8:
        warnings.warn(f"High failure rate in FFT processing: {success_rate:.1%}")
    
    return coefficients_matrix


def save_coefficients(coefficients: np.ndarray, 
                     parameters: List[Dict[str, Any]],
                     filename: Optional[str] = None) -> str:
    """
    Save Fourier coefficients along with corresponding parameters.
    
    Args:
        coefficients: Coefficients matrix (n_samples x n_coefficients)
        parameters: List of parameter dictionaries
        filename: Output filename (auto-generated if None)
        
    Returns:
        Path to saved file
    """
    from .config import DATA_PATH
    from datetime import datetime
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fourier_coefficients_{timestamp}.csv"
    
    # Create combined dataframe
    params_df = pd.DataFrame(parameters)
    
    # Add coefficient columns
    n_harmonics = coefficients.shape[1] // 2
    coeff_columns = []
    
    # Real parts
    for i in range(n_harmonics):
        coeff_columns.append(f'real_{i}')
    
    # Imaginary parts
    for i in range(n_harmonics):
        coeff_columns.append(f'imag_{i}')
    
    coeffs_df = pd.DataFrame(coefficients, columns=coeff_columns)
    
    # Combine
    combined_df = pd.concat([params_df, coeffs_df], axis=1)
    
    # Save
    filepath = os.path.join(DATA_PATH, filename)
    combined_df.to_csv(filepath, index=False)
    
    print(f"Coefficients saved to: {filepath}")
    return filepath


def analyze_frequency_content(coefficients: np.ndarray, 
                            harmonics: int = HARMONICS) -> Dict[str, Any]:
    """
    Analyze the frequency content of the extracted coefficients.
    
    Args:
        coefficients: Coefficients matrix
        harmonics: Number of harmonics
        
    Returns:
        Dictionary with analysis results
    """
    n_harmonics = harmonics
    
    # Split into real and imaginary parts
    real_parts = coefficients[:, :n_harmonics]
    imag_parts = coefficients[:, n_harmonics:]
    
    # Compute magnitudes
    magnitudes = np.sqrt(real_parts**2 + imag_parts**2)
    
    # Compute statistics
    analysis = {
        'n_samples': coefficients.shape[0],
        'n_harmonics': n_harmonics,
        'magnitude_mean': np.mean(magnitudes, axis=0),
        'magnitude_std': np.std(magnitudes, axis=0),
        'magnitude_max': np.max(magnitudes, axis=0),
        'dominant_frequencies': np.argmax(magnitudes, axis=1),
        'total_power': np.sum(magnitudes**2, axis=1),
        'dc_component': real_parts[:, 0],
        'fundamental_magnitude': magnitudes[:, 1] if n_harmonics > 1 else np.zeros(coefficients.shape[0])
    }
    
    return analysis


if __name__ == "__main__":
    # Test the FFT extraction
    print("Testing FFT extraction...")
    
    # Create test profile (simple circle)
    theta_test = np.linspace(0, 2*np.pi, 100, endpoint=False)
    r_test = 10 + 0.5 * np.sin(5 * theta_test)  # Circle with 5th harmonic
    x_test = r_test * np.cos(theta_test)
    y_test = r_test * np.sin(theta_test)
    
    # Test polar sampling
    theta_uniform, r_uniform = polar_sample(x_test, y_test)
    print(f"Polar sampling: {len(theta_uniform)} points")
    
    # Test FFT
    coefficients = compute_fft(r_uniform, harmonics=20)
    print(f"FFT coefficients: {len(coefficients)} values")
    
    # Test complete extraction
    coeffs_full = extract_fourier_descriptors(x_test, y_test, harmonics=20)
    print(f"Full extraction: {len(coeffs_full)} coefficients")
    
    print("FFT extraction test completed successfully!")
