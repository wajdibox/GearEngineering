# AppWorks Usage Guide

## Quick Start

The AppWorks package provides three main ways to use it:

### 1. Complete Pipeline (Recommended for beginners)
Run the entire workflow from gear parameters to trained models:

```python
from AppWorks.main import run_pipeline

# Run complete pipeline with 100 gear samples
results = run_pipeline(
    n_samples=100,           # Number of gear designs to generate
    random_state=42,         # For reproducible results
    include_symbolic=True,   # Find mathematical relationships
    include_anomalies=True,  # Include wear simulation
    verbose=True            # Show progress
)

print(f"Generated {results['n_successful_samples']} gear samples")
print(f"Extracted {results['n_coefficients']} Fourier coefficients per gear")
```

### 2. Step-by-Step Usage
For more control over each stage:

```python
# Step 1: Generate gear parameters and profiles
from AppWorks.data_generation import sample_parameters, get_profile

params = sample_parameters(n_samples=50, random_state=42)
profiles = []
for p in params:
    x, y = get_profile(p)
    profiles.append((x, y))

# Step 2: Extract Fourier coefficients
from AppWorks.fft_extraction import extract_fourier_descriptors

coefficients = []
for x, y in profiles:
    coeffs = extract_fourier_descriptors(x, y, harmonics=50)
    coefficients.append(coeffs)

# Step 3: Train prediction models
from AppWorks.regression import GearCoefficientPredictor

predictor = GearCoefficientPredictor()
predictor.fit(params, coefficients)

# Step 4: Make predictions for new gear designs
new_params = sample_parameters(n_samples=5)
predicted_coeffs = predictor.predict(new_params)
```

### 3. Individual Module Usage
Use specific components for targeted tasks:

```python
# Just generate gear profiles
from AppWorks.data_generation import sample_parameters, get_profile

params = sample_parameters(n_samples=10)
x_coords, y_coords = get_profile(params[0])

# Just extract Fourier coefficients from existing profiles
from AppWorks.fft_extraction import extract_fourier_descriptors

coefficients = extract_fourier_descriptors(x_coords, y_coords)

# Just add wear patterns to existing profiles
from AppWorks.anomalies import add_local_dent, simulate_gear_aging

# Add a dent at 45 degrees with 10-degree width and 5% depth
damaged_profile = add_local_dent(r_values, theta_values, 
                                center=np.pi/4, width=np.pi/18, depth=0.05)
```

## Command Line Usage

Run directly from command line:

```bash
# Run complete pipeline
python -m AppWorks.main

# Or with custom parameters
python -c "
from AppWorks.main import run_pipeline
results = run_pipeline(n_samples=200, include_symbolic=True)
print('Pipeline completed successfully!')
"
```

## Understanding the Output

### Generated Files
The pipeline creates several output files in the `data/` directory:

- `*_raw_params.csv` - Original gear parameters
- `*_raw_profiles.csv` - Generated gear coordinates
- `*_coefficients.csv` - Extracted Fourier coefficients
- `models/*_predictor.pkl` - Trained prediction models

### Results Dictionary
The `run_pipeline()` function returns a results dictionary with:

```python
{
    'status': 'completed',
    'n_successful_samples': 100,
    'n_coefficients': 400,
    'pipeline_steps': ['data_generation', 'fft_extraction', 'regression', ...],
    'file_paths': {'params': '...', 'coefficients': '...', 'models': '...'},
    'model_performance': {'mse': 0.001, 'r2': 0.99},
    'anomaly_analysis': {...}  # If include_anomalies=True
}
```

## Common Use Cases

### 1. Design Space Exploration
Generate many gear variations and analyze their frequency characteristics:

```python
# Generate 500 different gear designs
results = run_pipeline(n_samples=500, random_state=42)

# Analyze which parameters most affect the frequency content
from AppWorks.regression import analyze_feature_importance
importance = analyze_feature_importance(results['model'])
```

### 2. Wear Pattern Analysis
Study how different types of damage affect gear performance:

```python
from AppWorks.anomalies import simulate_gear_aging, analyze_anomaly_effects

# Create aged gear profile
aged_profile = simulate_gear_aging(r_original, theta, 
                                  age_factor=0.3, n_dents=5)

# Compare original vs aged
analysis = analyze_anomaly_effects(r_original, aged_profile, theta)
print(f"Maximum deviation: {analysis['max_deviation']:.3f}")
```

### 3. Custom Gear Analysis
Analyze your own gear designs:

```python
# If you have your own gear coordinates
my_x_coords = [...]  # Your gear X coordinates
my_y_coords = [...]  # Your gear Y coordinates

# Extract Fourier representation
my_coefficients = extract_fourier_descriptors(my_x_coords, my_y_coords)

# Compare with database of known gears
from AppWorks.regression import load_trained_model
predictor = load_trained_model('path/to/saved/model.pkl')
similar_params = predictor.inverse_predict(my_coefficients)  # If available
```

## Advanced Configuration

### Customizing FFT Parameters
```python
from AppWorks.fft_extraction import extract_fourier_descriptors

# Extract more harmonics for higher precision
high_res_coeffs = extract_fourier_descriptors(x, y, harmonics=500)

# Or fewer for faster processing
fast_coeffs = extract_fourier_descriptors(x, y, harmonics=20)
```

### Custom Parameter Ranges
```python
from AppWorks.data_generation import sample_parameters

# Override default parameter ranges
custom_params = sample_parameters(
    n_samples=100,
    custom_ranges={
        'module': (1.0, 5.0),      # Smaller gears only
        'teeth': (10, 50),         # Fewer teeth
        'pressure_angle': (20, 25) # Narrow pressure angle range
    }
)
```

### Model Training Options
```python
from AppWorks.regression import GearCoefficientPredictor

predictor = GearCoefficientPredictor(
    model_type='neural_network',  # or 'linear'
    neural_config={
        'hidden_layers': (200, 100, 50),
        'max_iter': 2000,
        'learning_rate': 'adaptive'
    }
)
```

## Troubleshooting

### Common Issues

1. **"PythonCore not found"** - Ensure the PythonCore directory is in the correct location
2. **"FFT extraction failed"** - Check that gear profiles are closed curves
3. **"Low model accuracy"** - Try increasing the number of samples or harmonics

### Performance Tips

- Start with small sample sizes (10-50) for testing
- Use `verbose=False` for faster execution
- Skip symbolic regression (`include_symbolic=False`) for speed
- Consider using fewer harmonics for initial exploration

### Memory Usage
For large datasets:
- Process in batches using the step-by-step approach
- Save intermediate results frequently
- Monitor memory usage with large harmonic counts

## Next Steps

1. **Experiment** with different parameter combinations
2. **Visualize** results using the built-in plotting functions
3. **Integrate** with your existing gear design workflow
4. **Extend** the package with custom anomaly patterns or analysis methods

The AppWorks package provides a solid foundation for gear analysis that you can build upon for your specific applications.