# GearREngineering

**Real-Time Analytical Digital Twin for Gear Degradation**

This repository implements a high-fidelity, real-time digital twin for gear geometry and wear modeling, combining:

1. **PythonCore**
   Pure-Python modules for parametric gear geometry (involute, trochoid fillets, rounds), export to DXF/JSON/CSV, plus a CLI/GUI front-end.
2. **JuliaNotebooks**
   Julia scripts to pull PythonCore profiles via PyCall, perform polar‑Fourier fitting, reconstruct profiles with FFT, and interactively explore gear parameters.
3. **VizExports**
   Helper scripts to export results and generate visualizations.

---

## 🚀 Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/wajdibox/GearEngineering.git
cd GearEngineering
```

### 2. PythonCore Setup

```bash
cd PythonCore
git switch main  # or ensure latest branch
git pull
python3 -m venv .venv
source .venv/bin/activate      # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                # makes `import PythonCore` work
```

### 3. JuliaNotebooks Setup

```bash
cd ../JuliaNotebooks
julia --project=.
julia> using Pkg; Pkg.instantiate()
```

### 4. Try it out

* **Static demo**:

  ```bash
  julia --project=. gear_fourier_fit_polar.jl
  ```
* **Interactive demo**:

  ```bash
  julia --project=. gear_fourier_fit_polar_interactive.jl
  ```

---

## 📂 Repository Layout

```
/
├── PythonCore/                 # Core geometry & export in Python
│   ├── gear_parameters.py
│   ├── geometry_generator.py
│   ├── exports.py
│   ├── gear_app.py
│   ├── utils_plotting.py
│   ├── README_PythonCore.md    # PythonCore usage & internals
│   └── requirements.txt
│
├── JuliaNotebooks/             # Julia scripts for FFT fitting & interactive demos
│   ├── gear_fourier_fit_polar.jl
│   ├── gear_fourier_fit_polar_interactive.jl
│   ├── gear_parametric_unified.jl
│   └── README_JuliaNotebooks.md # Julia notebook instructions
│
├── VizExports/                 # Visualization & export helpers
│   └── README_VizExports.md
│
├── AppWorks/                   # High‑level application examples & pipelines
│   └── USAGE_GUIDE.md
│
├── CommonIO/                   # Shared I/O utilities
│   └── README_CommonIO.md
│
├── OmniverseDev/               # Omniverse integration notes
│   └── README_OmniverseDev.md
│
├── README.md                   # ← This file
└── requirements.txt            # top‑level dependencies if any
```

---

## 🔧 Git & Contribution Guide

1. **Ignore large/temp files**: make sure `.gitignore` includes:

   * `JuliaNotebooks/Project.toml`
   * `JuliaNotebooks/Manifest.toml`
   * `**/__pycache__/`
   * `PythonCore/tests/`
   * `generate_structure.py`
   * `data/` and model checkpoints
2. **Branching & commits**:

   ```bash
   git checkout -b feature/YourFeature
   # work… then:
   git add <files>
   git commit -m "feat: add …"
   ```
3. **Sync with upstream**:

   ```bash
   git fetch origin
   git rebase origin/main
   ```
4. **Push & PR**:

   ```bash
   git push -u origin feature/YourFeature
   ```
5. **Tagging releases**:

   ```bash
   git tag v0.1.0 -m "v0.1.0: initial release"
   git push origin --tags
   ```

**Note**: avoid committing large data or generated caches—add them to `.gitignore` before staging.

---

## 🔜 Next Steps

* **STFT\_Gear-fit.jl** & **STFT\_Gear-fit\_interactive.jl**: add short‑time FFT analysis for time‑frequency gear features.
* **Wear evolution**: integrate Archard’s wear law for incremental geometry update.
* **Sensor fusion**: merge vibration/acoustic signals to drive real‑time twin updates.

---

## 📄 LICENSE & CREDITS

NOT YET IMPLEMENTED.

*Made with ❤️ by the WAJDI ABBASSI.*
