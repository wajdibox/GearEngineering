### utils_plotting.py
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, Tuple
from .geometry_generator import ScientificGearGeometry


def create_gear_plot(
    fig: plt.Figure,
    ax: plt.Axes,
    geometry: ScientificGearGeometry,
    view_mode: str = "full",
    show_circles: Optional[Dict[str, bool]] = None
) -> None:
    """
    Draw a gear (full or single tooth) and its reference circles onto the given Matplotlib Axes.

    Parameters:
    - fig: the Matplotlib Figure containing ax
    - ax: the Axes to draw into
    - geometry: a generated ScientificGearGeometry instance (must have generate_full_gear(), etc. already called)
    - view_mode: "full" or "single"
    - show_circles: dict with keys 'base', 'pitch', 'offset', 'outer', 'root' controlling circle visibility
    """
    # Default circle flags
    if show_circles is None:
        show_circles = {key: True for key in ('base', 'pitch', 'offset', 'outer', 'root')}

    # Clear and configure axes
    ax.clear()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Draw gear geometry
    if view_mode == "single":
        x, y = geometry.tooth_profile_x, geometry.tooth_profile_y
        if x.size:
            ax.plot(x, y, '-', linewidth=2, label='Tooth Profile')
            ax.fill(x, y, alpha=0.3)
    else:
        Z = geometry.params.teeth
        P_ANGLE = 2 * np.pi / Z
        for i in range(Z):
            tx, ty = geometry.rotation(
                geometry.tooth_profile_x,
                geometry.tooth_profile_y,
                P_ANGLE,
                i
            )
            ax.plot(tx, ty, '-', linewidth=1.5, alpha=0.7)
            ax.fill(tx, ty, alpha=0.2)

    # Draw reference circles
    circle_map: Tuple[Tuple[str,str], ...] = (
        ('base_dia','base'),
        ('pitch_dia','pitch'),
        ('offset_dia','offset'),
        ('outer_dia','outer'),
        ('root_dia','root')
    )
    for calc_key, flag_key in circle_map:
        if show_circles.get(flag_key, False):
            dia = geometry.calculations.get(calc_key)
            if dia is not None and dia > 0:
                cx, cy = geometry.circle(dia)
                ax.plot(cx, cy, ':', linewidth=1.0, alpha=0.7, label=f"{flag_key.capitalize()} Circle")

    # Add parameter annotation
    outer = geometry.calculations.get('outer_dia', 1)
    text_y = (outer / 2) * 0.9
    ax.text(
        0, text_y,
        f"m={geometry.params.module} mm, z={geometry.params.teeth}",
        ha='center', va='center',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.6)
    )

    # Title and legend
    ax.set_title(f"Scientific Gear: {geometry.params.teeth} Teeth @ m={geometry.params.module}mm")
    ax.legend(loc='upper right', fontsize=8)

def plot_full_gear(gear_obj):
    """
    Convenience function to plot the full gear using default matplotlib settings.
    Returns the matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    create_gear_plot(fig, ax, gear_obj, view_mode="full")
    plt.show()
    return fig

def plot_single_tooth(gear_obj):
    fig, ax = plt.subplots(figsize=(5, 5))
    create_gear_plot(fig, ax, gear_obj, view_mode="single")
    plt.show()
    return fig
