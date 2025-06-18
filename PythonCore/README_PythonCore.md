# 🧠 Scientific Gear Generator – Notion Documentation

This is the structured documentation for the PythonCore-based gear generator app. Designed for Notion-style readability and clarity.

---

## 🎯 Objective

To procedurally generate **2D spur gear profiles** with **scientific precision**, using an analytical, parametric approach. The system supports visualization, simulation, and export for digital twin systems.

---

## 🧱 System Architecture

### Modules Overview

| Module | Purpose |
|--------|---------|
| `main.py` | Entry point; launches GUI |
| `gear_parameters.py` | Stores and validates all gear inputs using a dataclass |
| `geometry_generator.py` | Core logic for generating involute flanks, root fillets, and full profiles |
| `utils_plotting.py` | Uses `matplotlib` to plot gear profiles and reference circles |
| `exports.py` | Exports the gear to JSON, CSV, DXF, and PNG |
| `gear_app.py` | GUI built with Tkinter to interface with the user |

---

## ⚙️ How It Works

### 1. **Parameter Handling**

Defined in `gear_parameters.py` using `@dataclass`.

- Input fields: module, teeth count, pressure angle, profile shift, etc.
- Validation: Ensures all fields are numeric and physically valid
- Conversion: Provides `.to_dict()` and `.from_dict()` for export/import

### 2. **Gear Geometry Generation**

Handled by `ScientificGearGeometry` in `geometry_generator.py`.

- Generates:
  - `involute flank` (from base circle to addendum)
  - `trochoidal root fillet` or rounded root
  - `reference circles` (base, pitch, addendum, root)
  - `full gear profile` by replicating single tooth

- Methods:
  - `generate_single_tooth()`
  - `generate_full_gear()`
  - `generate_reference_circles()`

### 3. **Plotting**

In `utils_plotting.py`:

- Two main plots:
  - `plot_tooth_profile()` – plots a single tooth with annotations
  - `plot_full_gear()` – rotates and assembles all teeth
- Uses consistent colors for each reference circle:
  - Base: Cyan
  - Pitch: Magenta
  - Outer: Brown
  - Root: Black
  - Offset/Profile shift: Grey

### 4. **GUI Interface**

Via `gear_app.py`:

- Built in Tkinter
- Field entries auto-initialize from defaults
- Buttons:
  - Generate
  - Export (JSON, CSV, PNG, DXF)
  - Save/Load settings

### 5. **Exports**

In `exports.py`:

- `export_json()`: Parameters, geometry arrays, and metadata
- `export_analysis_csv()`: Clean tabular summary for Julia or MATLAB
- `export_coordinates()`: Raw XY points of single tooth
- `export_full_gear_coordinates()`: Raw XY points of full gear
- `export_dxf()`: Simple 2D DXF with circles + tooth profile polyline
- `export_png()`: Snapshot of matplotlib plot

---

## 🧪 Example Flow

1. User launches app:
   ```
   python -m PythonCore.main
   ```

2. GUI loads with default gear parameters

3. User modifies parameters and clicks **Generate**

4. Visualization appears (via `matplotlib` window)

5. User clicks any export option

---

## 🧰 Use Cases

- Reverse engineering gears (input from over-pin measurements)
- Digital twin simulations (Omniverse/OpenUSD ready)
- Gear profile comparison (e.g., STFT or Fourier analysis)
- Design optimization (profile shift, edge smoothing)
- Wear detection & modeling (future phases)

---

## 🔬 Research Notes

- Based on **FGPG2** and Korean blog [https://tro.kr/47](https://tro.kr/47)
- Prioritizes:
  - G1 continuity between involute and fillet
  - Parametric control over all flank elements
  - Clean modular architecture for Julia/Python integration

---

## 📍 Next Steps

- Add Bézier or Fourier reconstruction for hybrid modeling
- Integrate simulation in Omniverse using OpenUSD
- Support planetary gear systems
- Gear meshing visualization

---



## 📄 `PythonCore/README_PythonCore.md`

```markdown
# PythonCore

Parametric gear geometry generation library in pure Python.

## Features

- **Analytical profiles**: involute flank, trochoid root fillet, rounded tip.
- **Export**: PNG, DXF, JSON, CSV of full-gear profile.
- **CLI**: `gear_app.py` for quick GUI.
- **Core modules**:
  - `gear_parameters.py` → `GearParameters` dataclass
  - `geometry_generator.py` → `ScientificGearGeometry`
  - `exports.py` → PNG/DXF/JSON/CSV writers
  - `utils_plotting.py` → Matplotlib helpers

## Installation

```sh
cd PythonCore
python3 -m venv .venv
source .venv/bin/activate       # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
Usage
From Python
python
Copier
Modifier
from scientific_gear_generator import gear_parameters, geometry_generator

params = gear_parameters.GearParameters(
    module=2.0, teeth=20, pressure_angle=20.0,
    profile_shift=0.0, addendum_factor=1.0, dedendum_factor=1.25,
    backlash_factor=0.0, edge_round_factor=0.1, root_round_factor=0.2
)
gear = geometry_generator.ScientificGearGeometry(params)
gear.generate()
x, y = gear.full_gear_x, gear.full_gear_y
CLI / GUI
sh
Copier
Modifier
python gear_app.py
Follow the prompts or use the sliders to export your gear.

Tests
sh
Copier
Modifier
pytest tests/
Next
Add wear evolution module (Archard’s law)

Integrate sensor-fusion routines

Publish to PyPI

yaml
Copier
Modifier
