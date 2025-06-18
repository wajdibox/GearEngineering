# C:\Users\DELL\Desktop\GearEngineering\PythonCore\exports.py
import json
import csv
import os
import math
from datetime import datetime
from typing import Tuple
from .gear_parameters import GearParameters
from .geometry_generator import ScientificGearGeometry
import matplotlib.pyplot as plt


def export_json(parameters: GearParameters,
                geometry: ScientificGearGeometry,
                filename: str) -> Tuple[bool, str]:
    """
    Export gear data to JSON format including parameters, calculations, and geometry
    """
    try:
        data = {
            "metadata": {
                "software": "Scientific Gear Generator",
                "method": "Fine Gear Profile Generator 2 Algorithm",
                "export_date": datetime.now().isoformat()
            },
            "parameters": parameters.to_dict(),
            "calculations": geometry.calculations,
            "geometry": {
                "tooth_profile_x": geometry.tooth_profile_x.tolist(),
                "tooth_profile_y": geometry.tooth_profile_y.tolist(),
                "full_gear_x": geometry.full_gear_x.tolist(),
                "full_gear_y": geometry.full_gear_y.tolist()
            }
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        return True, f"JSON exported to {os.path.basename(filename)}"
    except Exception as e:
        return False, f"JSON export failed: {e}"


def export_analysis_csv(parameters: GearParameters,
                        geometry: ScientificGearGeometry,
                        filename: str) -> Tuple[bool, str]:
    """
    Export key gear parameters for analysis in CSV format
    """
    try:
        calc = geometry.calculations
        rows = [
            ["Parameter", "Value", "Unit", "Description"],
            ["module", parameters.module, "mm", "Module"],
            ["teeth", parameters.teeth, "count", "Number of teeth"],
            ["pressure_angle", parameters.pressure_angle, "°", "Pressure angle"],
            ["profile_shift", parameters.profile_shift, "", "Profile shift"],
            ["addendum_factor", parameters.addendum_factor, "", "Addendum factor"],
            ["dedendum_factor", parameters.dedendum_factor, "", "Dedendum factor"],
            ["backlash_factor", parameters.backlash_factor, "", "Backlash factor"],
            ["edge_round_factor", parameters.edge_round_factor, "", "Edge rounding factor"],
            ["root_round_factor", parameters.root_round_factor, "", "Root rounding factor"],
            ["base_diameter", calc.get("base_dia", 0), "mm", "Base diameter"],
            ["pitch_diameter", calc.get("pitch_dia", 0), "mm", "Pitch diameter"],
            ["outer_diameter", calc.get("outer_dia", 0), "mm", "Outer diameter"],
            ["root_diameter", calc.get("root_dia", 0), "mm", "Root diameter"]
        ]
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        return True, f"Analysis CSV exported to {os.path.basename(filename)}"
    except Exception as e:
        return False, f"Analysis CSV export failed: {e}"


def export_coordinates(geometry: ScientificGearGeometry,
                       filename: str) -> Tuple[bool, str]:
    """
    Export single-tooth profile coordinates to CSV
    """
    try:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "radius", "angle_deg", "profile_type"])
            for x, y in zip(geometry.tooth_profile_x, geometry.tooth_profile_y):
                r = math.hypot(x, y)
                angle = math.degrees(math.atan2(y, x))
                writer.writerow([x, y, r, angle, "tooth_profile"])
        return True, f"Coordinates exported to {os.path.basename(filename)}"
    except Exception as e:
        return False, f"Coordinate export failed: {e}"


def export_full_gear_coordinates(geometry: ScientificGearGeometry,
                                 filename: str) -> Tuple[bool, str]:
    """
    Export full gear coordinates to CSV
    """
    try:
        pts = len(geometry.tooth_profile_x)
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "tooth_index"])
            for i in range(geometry.params.teeth):
                start = i * pts
                end = start + pts
                for x, y in zip(geometry.full_gear_x[start:end], geometry.full_gear_y[start:end]):
                    writer.writerow([x, y, i])
        return True, f"Full gear coordinates exported to {os.path.basename(filename)}"
    except Exception as e:
        return False, f"Full gear export failed: {e}"


def export_png(fig: plt.Figure,
               filename: str,
               dpi: int = 300) -> Tuple[bool, str]:
    """
    Export gear visualization to PNG image
    """
    try:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
        return True, f"PNG exported to {os.path.basename(filename)}"
    except Exception as e:
        return False, f"PNG export failed: {e}"


def export_dxf(geometry: ScientificGearGeometry,
               filename: str) -> Tuple[bool, str]:
    """
    Export the full gear outline as individual LINE segments in a DXF R12 file.
    Cinema 4D will import these as separate splines/lines visible in the viewport.
    """
    try:
        xs = geometry.full_gear_x
        ys = geometry.full_gear_y
        n = len(xs)
        with open(filename, "w") as out:
            # HEADER
            out.write("0\nSECTION\n2\nHEADER\n")
            out.write("9\n$ACADVER\n1\nAC1009\n")
            out.write("0\nENDSEC\n")
            # ENTITIES
            out.write("0\nSECTION\n2\nENTITIES\n")
            # LINE segments
            for i in range(n):
                x1, y1 = xs[i], ys[i]
                x2, y2 = xs[(i+1) % n], ys[(i+1) % n]
                out.write("0\nLINE\n8\n0\n")
                out.write(f"10\n{x1:.6f}\n20\n{y1:.6f}\n30\n0.000000\n")
                out.write(f"11\n{x2:.6f}\n21\n{y2:.6f}\n31\n0.000000\n")
            out.write("0\nENDSEC\n0\nEOF\n")
        return True, f"DXF exported to {os.path.basename(filename)}"
    except Exception as e:
        return False, f"DXF export failed: {e}"


def export_settings(parameters: GearParameters,
                    filename: str) -> Tuple[bool, str]:
    """
    Export gear parameters to JSON settings file
    """
    try:
        with open(filename, "w") as f:
            json.dump(parameters.to_dict(), f, indent=2)
        return True, "Settings saved successfully"
    except Exception as e:
        return False, f"Settings export failed: {e}"


def import_settings(filename: str) -> Tuple[GearParameters, str]:
    """
    Import gear parameters from JSON settings file
    """
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        params = GearParameters.from_dict(data)
        return params, "Settings loaded successfully"
    except Exception as e:
        return None, f"Settings import failed: {e}"
