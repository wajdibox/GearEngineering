#!/usr/bin/env python3
"""
Simple AppWorks Example - Get Started in 5 Minutes

This shows the easiest way to use your AppWorks package.
"""

import sys
sys.path.insert(0, '.')

def simple_example():
    """Basic example - generate and analyze a few gears"""
    print("AppWorks Simple Example")
    print("=" * 40)
    
    # 1. Generate some gear parameters
    from AppWorks.data_generation import sample_parameters
    
    print("1. Generating gear parameters...")
    params = sample_parameters(n_samples=3, random_state=42)
    
    for i, p in enumerate(params):
        print(f"   Gear {i+1}: {p['teeth']} teeth, module {p['module']:.2f}")
    
    # 2. Create one gear profile
    from AppWorks.data_generation import get_profile
    
    print("\n2. Creating gear profile...")
    x_coords, y_coords = get_profile(params[0])
    print(f"   Generated {len(x_coords)} coordinate points")
    
    # 3. Extract Fourier coefficients
    from AppWorks.fft_extraction import extract_fourier_descriptors
    
    print("\n3. Extracting Fourier coefficients...")
    coefficients = extract_fourier_descriptors(x_coords, y_coords, harmonics=20)
    print(f"   Extracted {len(coefficients)} coefficients")
    print(f"   First 5 coefficients: {coefficients[:5]}")
    
    # 4. Add some wear to simulate damage
    from AppWorks.fft_extraction import cartesian_to_polar
    from AppWorks.anomalies import add_local_dent
    
    print("\n4. Simulating gear wear...")
    r_values, theta_values = cartesian_to_polar(x_coords, y_coords)
    
    # Add a small dent
    damaged_r = add_local_dent(r_values, theta_values, 
                              center=1.0, width=0.2, depth=0.02)
    
    print("   Added localized dent to gear profile")
    
    # 5. Compare original vs damaged coefficients
    import numpy as np
    
    # Convert back to cartesian for coefficient extraction
    damaged_x = damaged_r * np.cos(theta_values)
    damaged_y = damaged_r * np.sin(theta_values)
    
    damaged_coeffs = extract_fourier_descriptors(damaged_x, damaged_y, harmonics=20)
    
    print("\n5. Comparing original vs damaged gear...")
    difference = np.abs(coefficients - damaged_coeffs)
    print(f"   Maximum coefficient change: {np.max(difference):.4f}")
    print(f"   Average coefficient change: {np.mean(difference):.4f}")
    
    print("\n✓ Example completed successfully!")
    print("Your AppWorks package is working correctly.")
    
    return {
        'original_coefficients': coefficients,
        'damaged_coefficients': damaged_coeffs,
        'difference': difference
    }

if __name__ == "__main__":
    try:
        results = simple_example()
        print(f"\nCoefficient analysis complete. Max change: {max(results['difference']):.6f}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the AppWorks package is properly installed.")