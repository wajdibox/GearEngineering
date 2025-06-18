# __init__.py in PythonCore
from .gear_parameters import GearParameters
from .geometry_generator import ScientificGearGeometry
from .gear_app import ScientificGearApp
from .exports import export_dxf  # ✅ Add this

__all__ = [
    'GearParameters',
    'ScientificGearGeometry',
    'ScientificGearApp',
    'export_dxf'  # ✅ Add this
]

__version__ = "1.0.0"
