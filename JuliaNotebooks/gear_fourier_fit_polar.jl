#!/usr/bin/env julia
# gear_fourier_fit_polar.jl

using PyCall
# force an interactive Matplotlib backend
pyimport("matplotlib")["use"]("qt5agg")   # or "tkagg"

using FFTW
using Interpolations: LinearInterpolation, Periodic
using PyPlot

# 1️⃣ Push your PythonCore folder onto sys.path
@info "Adding PythonCore to sys.path"
_sys = pyimport("sys")
push!(_sys["path"], raw"C:\Users\Labor\Desktop\GearEngineering\PythonCore")

# 2️⃣ Import your modular PythonCore packages
@info "Importing PythonCore modules…"
gear_params_mod = pyimport("PythonCore.gear_parameters")
geom_mod        = pyimport("PythonCore.geometry_generator")

# 3️⃣ Simple prompt‐with‐default helper
function prompt_with_default(prompt::String, default::T) where T
    print("$prompt [$default]: ")
    s = readline()
    isempty(s) && return default
    try
        return parse(T, strip(s))
    catch
        println("  ↳ invalid, using default $default")
        return default
    end
end

# 4️⃣ Main routine
function main()
    println("\n=== Gear Polar–Fourier Fit ===\n")
    m = prompt_with_default("Module (m)",               2.0)
    z = prompt_with_default("Number of teeth (z)",     20)
    α = prompt_with_default("Pressure angle (°)",      20.0)
    x = prompt_with_default("Profile shift",           0.0)
    a = prompt_with_default("Addendum factor",         1.0)
    d = prompt_with_default("Dedendum factor",         1.25)
    b = prompt_with_default("Backlash factor",         0.0)
    e = prompt_with_default("Edge-round factor",       0.1)
    r = prompt_with_default("Root-round factor",       0.2)
    println()

    @info "🧠 Generating gear in PythonCore…"
    # build Python GearParameters
    params_py = gear_params_mod.GearParameters(m, z, α, x, a, d, b, e, r)
    # instantiate & generate everything
    geom = geom_mod.ScientificGearGeometry(params_py)
    geom[:generate]()   # runs calculate + single tooth + full gear + circles

    # pull out full_gear_x/y as Julia floats
    xs = convert(Vector{Float64}, geom[:full_gear_x])
    ys = convert(Vector{Float64}, geom[:full_gear_y])
    N  = length(xs)
    @info "✅ Retrieved $N points"

    # to complex, then polar & sort by θ
    zc  = ComplexF64.(xs, ys)
    θ0  = angle.(zc)               # in [-π,π]
    rs0 = abs.(zc)
    θs  = mod.(θ0, 2π)             # [0,2π)
    ord = sortperm(θs)
    θs  = θs[ord]
    rs  = rs0[ord]

    # resample onto a uniform θ-grid via a periodic linear interpolator
    θ_uni = range(0, 2π; length=N)
    itp   = LinearInterpolation(θs, rs; extrapolation_bc=Periodic())
    ru    = itp.(θ_uni)

    @info "⚙️ Performing FFT…"
    R  = fft(ru)
    k  = min(500, length(R))
    Rtr= vcat(R[1:k], zeros(ComplexF64, length(R)-k))
    ru_rec = real(ifft(Rtr))
    # reconstruct in XY from θ_uni
    zc_rec = ru_rec .* exp.(im .* θ_uni)

    # 5️⃣ Plot
    figure(figsize=(6,6))
    plot(real(zc),  imag(zc),  "b-", linewidth=2, label="Original")
    plot(real(zc_rec), imag(zc_rec), "r--", linewidth=2,
         label="Fourier Recon (k=$k)")
    axis("equal")
    title("Gear Polar–Fourier Fit")
    xlabel("x"); ylabel("y")
    legend()
    grid(true)
    tight_layout()
    show()
end

# auto–run when you do `include("gear_fourier_fit_polar.jl")` in the REPL
if isinteractive()
    main()
end