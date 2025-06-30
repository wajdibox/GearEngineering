"""
Main orchestrator module for AppWorks.

This module provides the complete workflow pipeline:
1. Sample gear parameters
2. Generate analytical profiles using PythonCore
3. Extract Fourier coefficients via FFT
4. Train regression models
5. Optional symbolic regression
6. Save all results

Run with: python -m AppWorks.main
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
import warnings
from datetime import datetime
import argparse

# Internal imports
from .config import ensure_directories, DATA_PATH, MODELS_PATH
from .data_generation import generate_dataset, save_dataset
from .fft_extraction import process_dataset, save_coefficients
from .regression import prepare_features, train_linear_models, train_neural_models, save_models
from .symbolic import symbolic_regression, SYMBOLIC_CONFIG
from .anomalies import simulate_gear_aging, analyze_anomaly_effects


def run_pipeline(n_samples: int = 1000,
                random_state: Optional[int] = None,
                include_symbolic: bool = False,
                include_anomalies: bool = False,
                verbose: bool = True,
                output_prefix: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the complete AppWorks pipeline.
    
    Args:
        n_samples: Number of gear samples to generate
        random_state: Random seed for reproducibility
        include_symbolic: Whether to run symbolic regression
        include_anomalies: Whether to include anomaly injection
        verbose: Whether to print progress information
        output_prefix: Prefix for output files
        
    Returns:
        Dictionary with pipeline results and file paths
    """
    if verbose:
        print("="*60)
        print("AppWorks - Hybrid Fourier Gear Modeling Pipeline")
        print("="*60)
        print(f"Samples: {n_samples}")
        print(f"Random seed: {random_state}")
        print(f"Include symbolic regression: {include_symbolic}")
        print(f"Include anomalies: {include_anomalies}")
        print()
    
    # Ensure output directories exist
    ensure_directories()
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_prefix is None:
        output_prefix = f"appworks_run_{timestamp}"
    
    results = {
        'timestamp': timestamp,
        'n_samples': n_samples,
        'pipeline_steps': [],
        'file_paths': {},
        'metrics': {}
    }
    
    try:
        # Step 1: Generate gear parameters and profiles
        if verbose:
            print("Step 1: Generating gear dataset...")
        
        parameters, profiles = generate_dataset(
            n_samples=n_samples,
            random_state=random_state,
            verbose=verbose
        )
        
        results['pipeline_steps'].append('data_generation')
        results['n_successful_samples'] = len(parameters)
        
        # Save raw dataset
        params_file = save_dataset(parameters, profiles, f"{output_prefix}_raw")
        results['file_paths']['raw_data'] = params_file
        
        if verbose:
            print(f"Generated {len(parameters)} successful samples")
            print()
        
        # Step 2: Extract Fourier coefficients
        if verbose:
            print("Step 2: Extracting Fourier coefficients...")
        
        coefficients_matrix = process_dataset(profiles, verbose=verbose)
        results['pipeline_steps'].append('fft_extraction')
        results['n_coefficients'] = coefficients_matrix.shape[1]
        
        # Save coefficients
        coeffs_file = save_coefficients(coefficients_matrix, parameters, 
                                      f"{output_prefix}_coefficients.csv")
        results['file_paths']['coefficients'] = coeffs_file
        
        if verbose:
            print(f"Extracted {coefficients_matrix.shape[1]} coefficients per sample")
            print()
        
        # Step 3: Prepare features for regression
        if verbose:
            print("Step 3: Preparing features for regression...")
        
        X, feature_names = prepare_features(parameters)
        Y = coefficients_matrix
        
        if verbose:
            print(f"Feature matrix shape: {X.shape}")
            print(f"Target matrix shape: {Y.shape}")
            print(f"Features: {', '.join(feature_names)}")
            print()
        
        # Step 4: Train regression models
        if verbose:
            print("Step 4: Training regression models...")
        
        models = {}
        
        # Linear regression
        if verbose:
            print("  Training linear regression...")
        linear_model = train_linear_models(X, Y, feature_names=feature_names, verbose=verbose)
        models['linear'] = linear_model
        results['pipeline_steps'].append('linear_regression')
        
        # Neural network regression
        if verbose:
            print("\n  Training neural network...")
        neural_model = train_neural_models(X, Y, feature_names=feature_names, verbose=verbose)
        models['neural'] = neural_model
        results['pipeline_steps'].append('neural_regression')
        
        # Save models
        model_paths = save_models(models, f"{output_prefix}_models")
        results['file_paths']['models'] = model_paths
        
        if verbose:
            print(f"\nTrained {len(models)} regression models")
            print()
        
        # Step 5: Optional symbolic regression
        if include_symbolic and SYMBOLIC_CONFIG['enabled']:
            if verbose:
                print("Step 5: Running symbolic regression...")
            
            try:
                symbolic_model = symbolic_regression(
                    X, Y, feature_names=feature_names,
                    max_coefficients=20, verbose=verbose
                )
                
                # Save expressions
                expr_file = symbolic_model.save_expressions(f"{output_prefix}_expressions.txt")
                results['file_paths']['expressions'] = expr_file
                results['pipeline_steps'].append('symbolic_regression')
                
                # Get expressions summary
                expressions = symbolic_model.get_expressions()
                results['n_symbolic_expressions'] = len(expressions)
                
                if verbose:
                    print(f"Discovered {len(expressions)} symbolic expressions")
                    print()
                    
            except Exception as e:
                if verbose:
                    print(f"Symbolic regression failed: {e}")
                warnings.warn(f"Symbolic regression failed: {e}")
        
        # Step 6: Optional anomaly analysis
        if include_anomalies:
            if verbose:
                print("Step 6: Analyzing anomaly effects...")
            
            # Generate anomalous profiles for a subset of samples
            n_anomaly_samples = min(100, len(profiles))
            anomaly_effects = []
            
            for i in range(n_anomaly_samples):
                x_coords, y_coords = profiles[i]
                
                # Convert to polar
                from .fft_extraction import cartesian_to_polar
                r_original, theta = cartesian_to_polar(x_coords, y_coords)
                
                # Add aging effects
                r_aged = simulate_gear_aging(r_original, theta, age_factor=0.1)
                
                # Analyze effects
                effects = analyze_anomaly_effects(r_original, r_aged, theta)
                anomaly_effects.append(effects)
            
            # Summarize anomaly effects
            if anomaly_effects:
                results['anomaly_analysis'] = {
                    'n_samples': len(anomaly_effects),
                    'mean_max_change': np.mean([e['max_relative_change'] for e in anomaly_effects]),
                    'mean_affected_fraction': np.mean([e['affected_fraction'] for e in anomaly_effects])
                }
                results['pipeline_steps'].append('anomaly_analysis')
                
                if verbose:
                    print(f"Analyzed anomaly effects on {len(anomaly_effects)} samples")
                    print(f"Mean maximum change: {results['anomaly_analysis']['mean_max_change']:.1%}")
                    print(f"Mean affected fraction: {results['anomaly_analysis']['mean_affected_fraction']:.1%}")
                    print()
        
        # Pipeline summary
        if verbose:
            print("="*60)
            print("Pipeline Complete!")
            print("="*60)
            print(f"Total samples processed: {results['n_successful_samples']}")
            print(f"Coefficients extracted: {results['n_coefficients']}")
            print(f"Steps completed: {', '.join(results['pipeline_steps'])}")
            print(f"Files saved in: {DATA_PATH}")
            print()
            print("Output files:")
            for name, path in results['file_paths'].items():
                if isinstance(path, dict):
                    for model_name, model_path in path.items():
                        print(f"  {name}_{model_name}: {os.path.basename(model_path)}")
                else:
                    print(f"  {name}: {os.path.basename(path)}")
        
        results['status'] = 'success'
        return results
        
    except Exception as e:
        results['status'] = 'failed'
        results['error'] = str(e)
        if verbose:
            print(f"Pipeline failed: {e}")
        raise


def main():
    """Command-line interface for AppWorks pipeline."""
    parser = argparse.ArgumentParser(
        description="AppWorks - Hybrid Fourier-based Gear Modeling Pipeline"
    )
    
    parser.add_argument(
        '--samples', '-n', type=int, default=1000,
        help='Number of gear samples to generate (default: 1000)'
    )
    
    parser.add_argument(
        '--seed', '-s', type=int, default=None,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--symbolic', action='store_true',
        help='Include symbolic regression (requires gplearn/PySR)'
    )
    
    parser.add_argument(
        '--anomalies', action='store_true',
        help='Include anomaly analysis'
    )
    
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Quiet mode (minimal output)'
    )
    
    parser.add_argument(
        '--output-prefix', '-o', type=str, default=None,
        help='Prefix for output files'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    try:
        results = run_pipeline(
            n_samples=args.samples,
            random_state=args.seed,
            include_symbolic=args.symbolic,
            include_anomalies=args.anomalies,
            verbose=not args.quiet,
            output_prefix=args.output_prefix
        )
        
        print("\nPipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
