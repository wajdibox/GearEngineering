# gear_parametric_unified.jl
using PyCall
using FFTW
using LinearAlgebra
using Symbolics
using Plots

# Configure Python environment
if !haskey(ENV, "PYTHONCORE_ADDED")
    sys = pyimport("sys")
    pushfirst!(sys.path, raw"C:\Users\DELL\Desktop\GearEngineering\PythonCore")
    ENV["PYTHONCORE_ADDED"] = "true"
end

# Import PythonCore modules
gp = pyimport("PythonCore.gear_parameters")
geom = pyimport("PythonCore.geometry_generator")

# Symbolic Fourier System ====================================================
@variables Z M X α a d b c e θ n

# Base radius function (analytical foundation)
R₀ = (M * Z / 2) + (M * X) * cos(θ)  # Pitch circle + profile shift effect

# Fourier coefficient models (symbolic functions of gear parameters)
aₙ(n) = (0.1*M/Z) * exp(-0.2*n) + 0.03*X * (1 - exp(-0.5*n))
bₙ(n) = (0.05*α/20) * sin(π*n/5) + 0.02*X * n/(n+1)

# Parametric radius function (unified model)
r(θ) = R₀ + sum(aₙ(n)*cos(n*θ) + bₙ(n)*sin(n*θ) for n in 1:10)

# Convert to Cartesian coordinates
x(θ) = r(θ) * cos(θ)
y(θ) = r(θ) * sin(θ)

# 1. Analytical Gear Generation ==============================================
function generate_gear(params)
    py_params = gp.GearParameters(
        params.m, params.teeth, params.pressure_angle,
        params.profile_shift, params.addendum_factor,
        params.dedendum_factor, params.backlash_factor,
        params.edge_round_factor, params.root_round_factor
    )
    py_geom = geom.ScientificGearGeometry(py_params)
    py_geom.generate()
    return (
        x = convert(Vector{Float64}, py_geom.tooth_profile_x),
        y = convert(Vector{Float64}, py_geom.tooth_profile_y)
    )
end

# 2. Symbolic Coefficient Extraction ========================================
function compute_symbolic_coeffs(params, harmonics=10)
    # Convert symbolic functions to concrete values
    coeff_funcs = [
        (n -> substitute(aₙ(n), (Z=>params.teeth, M=>params.m, X=>params.profile_shift, α=>params.pressure_angle))),
        (n -> substitute(bₙ(n), (Z=>params.teeth, M=>params.m, X=>params.profile_shift, α=>params.pressure_angle)))
    ]
    
    # Compute coefficients for each harmonic
    coeffs = Vector{Float64}(undef, 2*harmonics)
    for n in 1:harmonics
        coeffs[2n-1] = coeff_funcs[1](n)
        coeffs[2n] = coeff_funcs[2](n)
    end
    return coeffs
end

# 3. Hybrid Reconstruction ==================================================
function reconstruct_gear(params, harmonics=10)
    # Get actual gear for comparison
    actual_gear = generate_gear(params)
    
    # Compute symbolic coefficients
    coeffs = compute_symbolic_coeffs(params, harmonics)
    
    # Reconstruct using Fourier series
    θ_range = LinRange(0, 2π, 1000)
    r_rec = zeros(length(θ_range))
    
    # Base radius component
    R₀_val = (params.m * params.teeth / 2) + (params.m * params.profile_shift)
    
    for (i, θ) in enumerate(θ_range)
        r_sum = 0.0
        for n in 1:harmonics
            aₙ = coeffs[2n-1]
            bₙ = coeffs[2n]
            r_sum += aₙ * cos(n*θ) + bₙ * sin(n*θ)
        end
        r_rec[i] = R₀_val + r_sum
    end
    
    # Convert to Cartesian
    x_rec = @. r_rec * cos(θ_range)
    y_rec = @. r_rec * sin(θ_range)
    
    return actual_gear.x, actual_gear.y, x_rec, y_rec
end

# 4. Symbolic Regression Core ===============================================
function build_symbolic_model(gear_specs, harmonics=5)
    # Build dataset: gear params → Fourier coefficients
    X = Matrix{Float64}(undef, length(gear_specs), 4)  # m, Z, X, α
    Y = Matrix{Float64}(undef, length(gear_specs), 2*harmonics)
    
    for (i, params) in enumerate(gear_specs)
        # Generate gear and convert to polar
        gear = generate_gear(params)
        r_vals = hypot.(gear.x, gear.y)
        θ_vals = atan.(gear.y, gear.x)
        
        # Compute FFT coefficients
        interp = linear_interpolation(θ_vals, r_vals, extrapolation_bc=Periodic())
        θ_uniform = LinRange(0, 2π, 1024)
        r_uniform = interp(θ_uniform)
        fft_coeffs = fft(r_uniform)[1:harmonics]
        
        # Store data
        X[i, :] = [params.m, params.teeth, params.profile_shift, params.pressure_angle]
        Y[i, :] = [real.(fft_coeffs); imag.(fft_coeffs)]
    end
    
    # Symbolic regression for each coefficient
    @variables m z x α
    coeff_models = []
    
    for k in 1:size(Y, 2)
        # Find best symbolic expression
        expr = build_symbolic_expression(X, Y[:, k], [m, z, x, α])
        push!(coeff_models, expr)
    end
    
    return coeff_models
end

# 5. Visualization and Validation ===========================================
function visualize_hybrid(actual_x, actual_y, rec_x, rec_y)
    p = plot(
        actual_x, actual_y, 
        label="Analytical Model", linewidth=2,
        title="Unified Parametric Reconstruction",
        xlabel="X (mm)", ylabel="Y (mm)",
        aspect_ratio=:equal
    )
    plot!(
        rec_x, rec_y, 
        label="Fourier Parametric", linewidth=2, linestyle=:dash
    )
    return p
end

# MAIN EXECUTION ============================================================
if abspath(PROGRAM_FILE) == @__FILE__
    # Test parameters
    test_params = (m=2.0, teeth=20, pressure_angle=20.0, profile_shift=0.1,
                   addendum_factor=1.0, dedendum_factor=1.25, backlash_factor=0.0,
                   edge_round_factor=0.1, root_round_factor=0.2)
    
    # Reconstruct using unified model
    actual_x, actual_y, rec_x, rec_y = reconstruct_gear(test_params)
    
    # Visualize comparison
    plt = visualize_hybrid(actual_x, actual_y, rec_x, rec_y)
    savefig(plt, "unified_gear_reconstruction.png")
    
    # Build symbolic model from data
    gear_specs = [
        (m=1.5, teeth=25, pressure_angle=20.0, profile_shift=0.0),
        (m=2.0, teeth=30, pressure_angle=25.0, profile_shift=0.2),
        (m=2.5, teeth=18, pressure_angle=18.0, profile_shift=-0.1)
    ]
    symbolic_models = build_symbolic_model(gear_specs)
    
    # Print discovered symbolic relationships
    println("Discovered Fourier coefficient models:")
    for (i, model) in enumerate(symbolic_models)
        println("Coefficient $i: ", model)
    end
end