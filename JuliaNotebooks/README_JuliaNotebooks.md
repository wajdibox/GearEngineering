# Julia Notebooks for Gear Polar–Fourier Analysis

This directory contains two Julia scripts (and soon more) for computing, visualizing and interactively exploring polar‐Fourier fits of involute-style gear profiles. They rely on our PythonCore modules (written in Python) to generate the geometry, then use Julia’s FFT and interpolation libraries (and Matplotlib via PyCall) for the signal-processing and plotting.

---

## Contents

- **gear_fourier_fit_polar.jl**  
  A non-interactive REPL script that:
  1. Prompts you (with defaults) for gear parameters (module, # teeth, pressure angle, profile shift, addendum/dedendum, relief, rounding, …).  
  2. Calls into PythonCore to generate the full 2D outline.  
  3. Converts to polar radius – θ, interpolates onto a uniform grid, runs an FFT, truncates after *k* terms (default 500), inverts back to the spatial domain, and overlays the reconstruction on the original outline in a Matplotlib window.

- **gear_fourier_fit_polar_interactive.jl**  
  A fully GUI‐driven Matplotlib+Slider version.  Move the sliders for module, # teeth, pressure angle, profile shift, addendum/dedendum factors, tip/root rounding, and *k* (number of Fourier modes) and watch the reconstructed outline update live against the original.

- **STFT_Gear-fit.jl** _(coming soon)_  
  A script to apply a short-time Fourier transform along the tooth flank, visualize local spectral content.

- **STFT_Gear-fit_interactive.jl** _(coming soon)_  
  Interactive version of the STFT approach, with sliders for window size, overlap, etc.

---

## Challenges & Lessons Learned

- **Python GUI backends**  
  Matplotlib’s default Agg backend is non-interactive.  On Windows, forcing `qt5agg` or `tkagg` in the embedded Conda environment (via `Conda.add("pyqt")` or `Conda.add("tk")`) is necessary to get real popup windows.  We ended up simply letting `PyPlot` choose a working GUI backend and calling `ion()`.

- **Data wrangling across languages**  
  Converting Python lists (via PyCall) to Julia `Vector{Float64}` and back again was straightforward with `pyconvert`.  Be careful to push the correct parent path onto `sys.path` so that `pyimport("PythonCore.gear_parameters")` actually finds your local modules.

- **Interpolation warnings**  
  `Interpolations.jl` will warn about “duplicated knots” if your θ‐vector has repeated endpoints at 0 and 2π—these get deduplicated automatically, but you can silence the warning with `Interpolations.deduplicate_knots!` if you wish.

- **Keeping state in sliders**  
  Matplotlib’s widget sliders are stateful; make sure the initial plot draws *before* connecting the callbacks, and always call `fig.canvas.draw()` after updating.

---

## Environment Setup

The easiest approach is to let Julia manage its own Python via Conda.jl:

1. **Clone the repo** (so that you have `JuliaNotebooks/` and `PythonCore/` side by side):

   ```bash
   git clone https://github.com/your-org/GearEngineering.git
   cd GearEngineering/JuliaNotebooks

2. **Start Julia and activate this folder as a project**

   ```bash
   julia> import Pkg
   julia> Pkg.activate(".")

3. **Install the Julia dependencies**

   ```bash
   julia> Pkg.instantiate()         # reads Project.toml and Manifest.toml
   julia> Pkg.add("FFTW")           # fast Fourier transforms
   julia> Pkg.add("Interpolations") # for polar→uniform interpolation
   julia> Pkg.add("PyPlot")         # plot via Matplotlib

4. **Ensure your PythonCore modules are on the path**

   ```bash
   sys = pyimport("sys")
   pushfirst!(sys["path"], raw"C:\Users\Labor\Desktop\GearEngineering\PythonCore")

Adjust that raw path to wherever you checked out your repo.

5. **Install any missing Python packages into the built-in Conda environment**

   ```bash
   julia> using Conda
   julia> Conda.add("pyqt")       # for Qt5Agg backend
   julia> Conda.add("tk")         # for TkAgg backend

6. **Run the scripts**

   ```bash
   julia> include("gear_fourier_fit_polar.jl")
   julia> include("gear_fourier_fit_polar_interactive.jl")

– or call their exported functions from the REPL.


What’s Next
STFT_Gear-fit.jl: apply a sliding window Fourier transform to extract local shape harmonics along each flank.

Interactive STFT version: sliders for window length, hop size, number of harmonics, etc.

Additional reconstruction models: wavelets, parametric spline fits, modal decomposition, machine-learning based shape approximators.

Export options: save reconstructed profiles to DXF/CSV for downstream CAD/CAM.

Pull requests, suggestions and issues are very welcome!



## 📑 `JuliaNotebooks/README_JuliaNotebooks.md`

```markdown
# JuliaNotebooks

Julia scripts to interface with PythonCore, do Fourier fitting of gear profiles, and explore interactively.

## Scripts

- `gear_fourier_fit_polar.jl`  
  Static, prompt-driven script:
  1. Pull gear `(x,y)` from PythonCore
  2. Convert to polar `r(θ)`, sort, interpolate
  3. FFT→truncate→IFFT → reconstruct
  4. Plot original vs. reconstruction

- `gear_fourier_fit_polar_interactive.jl`  
  Matplotlib widget sliders (via PyCall) for live tuning of:
  - Module, teeth, shifts, radii
  - Number of Fourier terms **k**
  and immediate profile update.

## Requirements

- Julia ≥ 1.11
- Packages:
  ```jl
  using Pkg
  Pkg.activate(".")
  Pkg.instantiate()  # installs: PyCall, FFTW, Interpolations, PyPlot, Conda (for PyCall backend)
PythonCore installed and reachable by PyCall:

sh
Copier
Modifier
cd ../PythonCore
pip install -e .
A Qt5 backend for Matplotlib (via Conda):

jl
Copier
Modifier
using Pkg; Pkg.build("PyCall")
ENV["PYTHON"] = ""
using Conda; Conda.add("pyqt")
Run
sh
Copier
Modifier
julia --project=. gear_fourier_fit_polar.jl
julia --project=. gear_fourier_fit_polar_interactive.jl
Next
STFT_Gear_fit.jl: extend to Short-Time Fourier for dynamic signal fusion.

Makie GUI: pure-Julia interactive sliders without Python widgets.

Wear modeling: Archard’s law updates over cycles.

yaml
Copier
Modifier

---

### Final Steps

1. **Place** each README in the corresponding folder.
2. **Commit**:

   ```sh
   git add .
   git commit -m "Add README files for PythonCore, JuliaNotebooks, and top-level"