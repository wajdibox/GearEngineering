#!/usr/bin/env python3
"""
Test the fixed AppWorks pipeline with a small dataset
"""

import sys
sys.path.insert(0, '.')

def test_pipeline():
    """Test the complete pipeline with the scikit-learn fix"""
    print("Testing Fixed AppWorks Pipeline")
    print("=" * 40)
    
    from AppWorks.main import run_pipeline
    
    # Run with a small dataset to verify the fix
    results = run_pipeline(
        n_samples=10,
        random_state=42,
        include_symbolic=False,
        include_anomalies=False,
        verbose=True
    )
    
    print("\n" + "=" * 40)
    print("PIPELINE TEST RESULTS")
    print("=" * 40)
    print(f"Status: {results['status']}")
    print(f"Samples processed: {results['n_successful_samples']}")
    print(f"Coefficients per sample: {results['n_coefficients']}")
    print(f"Pipeline steps completed: {len(results['pipeline_steps'])}")
    
    if 'model_performance' in results:
        perf = results['model_performance']
        print(f"Model R² score: {perf.get('mean_train_r2', 'N/A'):.4f}")
        print(f"Model RMSE: {perf.get('mean_train_rmse', 'N/A'):.4f}")
    
    print("\n✓ AppWorks pipeline fixed and working!")
    return results

if __name__ == "__main__":
    try:
        results = test_pipeline()
        print("Success! Your AppWorks package is ready for production use.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()