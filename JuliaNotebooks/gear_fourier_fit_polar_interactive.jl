#!/usr/bin/env julia
# gear_fourier_fit_polar_interactive_matplotlib.jl

using PyCall
using FFTW
using Interpolations: LinearInterpolation, Periodic

# Ensure Matplotlib uses Qt5Agg backend
pyimport("matplotlib")["use"]("qt5agg")
plt = pyimport("matplotlib.pyplot")
widgets = pyimport("matplotlib.widgets")

# 1️⃣ Push PythonCore folder onto sys.path
@info "Adding PythonCore to sys.path"
if !haskey(ENV, "PYTHONCORE_ADDED")
    _sys = pyimport("sys")
    push!(_sys["path"], raw"C:\Users\Labor\Desktop\GearEngineering\PythonCore")
    ENV["PYTHONCORE_ADDED"] = "true"
end

# 2️⃣ Import PythonCore modules
@info "Importing PythonCore modules…"
gear_params_mod = pyimport("PythonCore.gear_parameters")
geom_mod = pyimport("PythonCore.geometry_generator")

# 3️⃣ Function to generate gear data
function generate_gear_data(m, z, α, x, a, d, b, e, r, k, N_points=1000)
    # Build Python GearParameters
    params_py = gear_params_mod.GearParameters(m, z, α, x, a, d, b, e, r)
    # Instantiate & generate everything
    geom = geom_mod.ScientificGearGeometry(params_py)
    geom[:generate]()  # Runs calculate + single tooth + full gear + circles

    # Pull out full_gear_x/y as Julia floats
    xs = convert(Vector{Float64}, geom[:full_gear_x])
    ys = convert(Vector{Float64}, geom[:full_gear_y])
    N = length(xs)

    # To complex, then polar & sort by θ
    zc = ComplexF64.(xs, ys)
    θ0 = angle.(zc)  # in [-π,π]
    rs0 = abs.(zc)
    θs = mod.(θ0, 2π)  # [0,2π)
    ord = sortperm(θs)
    θs = θs[ord]
    rs = rs0[ord]

    # Resample onto a uniform θ-grid via a periodic linear interpolator
    θ_uni = range(0, 2π; length=N_points)
    itp = LinearInterpolation(θs, rs; extrapolation_bc=Periodic())
    ru = itp.(θ_uni)

    # Perform FFT
    R = fft(ru)
    k = min(Int(k), length(R))  # Ensure k is within bounds
    Rtr = vcat(R[1:k], zeros(ComplexF64, length(R)-k))
    ru_rec = real(ifft(Rtr))
    # Reconstruct in XY from θ_uni
    zc_rec = ru_rec .* exp.(im .* θ_uni)

    return real(zc), imag(zc), real(zc_rec), imag(zc_rec), k
end

# 4️⃣ Main GUI function
function create_interactive_gui()
    # Initialize figure and axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_title("Gear Polar-Fourier Fit")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(true)

    # Initial parameters
    initial_params = Dict(
        :m => 2.0,    # Module
        :z => 20,     # Number of teeth
        :α => 20.0,   # Pressure angle (°)
        :x => 0.0,    # Profile shift
        :a => 1.0,    # Addendum factor
        :d => 1.25,   # Dedendum factor
        :b => 0.0,    # Backlash factor
        :e => 0.1,    # Edge-round factor
        :r => 0.2,    # Root-round factor
        :k => 500     # Fourier terms
    )

    # Plot initial data
    orig_x, orig_y, rec_x, rec_y, k = generate_gear_data(
        initial_params[:m], initial_params[:z], initial_params[:α],
        initial_params[:x], initial_params[:a], initial_params[:d],
        initial_params[:b], initial_params[:e], initial_params[:r],
        initial_params[:k]
    )
    orig_line, = ax.plot(orig_x, orig_y, "b-", linewidth=2, label="Original")
    rec_line, = ax.plot(rec_x, rec_y, "r--", linewidth=2, label="Fourier Recon (k=$k)")
    ax.legend()

    # Adjust plot layout to make room for sliders
    fig.subplots_adjust(bottom=0.38)

    # Define slider axes
    slider_axes = [
        fig.add_axes([0.15, 0.33, 0.65, 0.03]),  # m
        fig.add_axes([0.15, 0.30, 0.65, 0.03]),  # z
        fig.add_axes([0.15, 0.27, 0.65, 0.03]),  # α
        fig.add_axes([0.15, 0.24, 0.65, 0.03]),  # x
        fig.add_axes([0.15, 0.21, 0.65, 0.03]),  # a
        fig.add_axes([0.15, 0.18, 0.65, 0.03]),  # d
        fig.add_axes([0.15, 0.15, 0.65, 0.03]),  # b
        fig.add_axes([0.15, 0.12, 0.65, 0.03]),  # e
        fig.add_axes([0.15, 0.09, 0.65, 0.03]),  # r
        fig.add_axes([0.15, 0.06, 0.65, 0.03])   # k
    ]

    # Create sliders
    sliders = [
        widgets.Slider(slider_axes[1], "Module (m)", 0.5, 5.0, valinit=initial_params[:m], valstep=0.1),
        widgets.Slider(slider_axes[2], "Teeth (z)", 5, 50, valinit=initial_params[:z], valstep=1),
        widgets.Slider(slider_axes[3], "Pressure angle (°)", 10.0, 30.0, valinit=initial_params[:α], valstep=0.5),
        widgets.Slider(slider_axes[4], "Profile shift (x)", -1.0, 1.0, valinit=initial_params[:x], valstep=0.05),
        widgets.Slider(slider_axes[5], "Addendum (a)", 0.5, 2.0, valinit=initial_params[:a], valstep=0.05),
        widgets.Slider(slider_axes[6], "Dedendum (d)", 0.5, 2.0, valinit=initial_params[:d], valstep=0.05),
        widgets.Slider(slider_axes[7], "Backlash (b)", 0.0, 0.5, valinit=initial_params[:b], valstep=0.01),
        widgets.Slider(slider_axes[8], "Edge-round (e)", 0.0, 0.5, valinit=initial_params[:e], valstep=0.01),
        widgets.Slider(slider_axes[9], "Root-round (r)", 0.0, 0.5, valinit=initial_params[:r], valstep=0.01),
        widgets.Slider(slider_axes[10], "Fourier terms (k)", 1, 3000, valinit=initial_params[:k], valstep=1)
    ]

    # Update function for sliders
    function update(val)
        try
            params = Dict(
                :m => sliders[1].val,
                :z => Int(sliders[2].val),
                :α => sliders[3].val,
                :x => sliders[4].val,
                :a => sliders[5].val,
                :d => sliders[6].val,
                :b => sliders[7].val,
                :e => sliders[8].val,
                :r => sliders[9].val,
                :k => Int(sliders[10].val)
            )
            orig_x, orig_y, rec_x, rec_y, k = generate_gear_data(
                params[:m], params[:z], params[:α],
                params[:x], params[:a], params[:d],
                params[:b], params[:e], params[:r],
                params[:k]
            )
            orig_line.set_data(orig_x, orig_y)
            rec_line.set_data(rec_x, rec_y)
            rec_line.set_label("Fourier Recon (k=$k)")
            ax.legend()
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
        catch err
            @warn "Error updating plot: $err"
        end
    end

    # Connect sliders to update function
    for slider in sliders
        slider.on_changed(update)
    end

    # Display the plot
    plt.show()
end

# 5️⃣ Run the GUI
if isinteractive()
    create_interactive_gui()
end