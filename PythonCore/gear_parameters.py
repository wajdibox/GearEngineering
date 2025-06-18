# C:\Users\DELL\Desktop\GearEngineering\PythonCore\gear_parameters.py
from dataclasses import dataclass

@dataclass
class GearParameters:
    """
    Scientific gear parameters following ISO standards
    Represents all geometric specifications for generating involute gears
    
    Attributes:
        module: Gear module (mm) - determines tooth size
        teeth: Number of teeth on the gear
        pressure_angle: Pressure angle (degrees) - affects tooth shape
        profile_shift: Profile shift coefficient - adjusts tooth profile
        addendum_factor: Addendum factor - controls tooth height above pitch circle
        dedendum_factor: Dedendum factor - controls tooth depth below pitch circle
        backlash_factor: Backlash factor - clearance between mating teeth
        edge_round_factor: Edge rounding factor - fillet at tooth tip
        root_round_factor: Root rounding factor - fillet at tooth base
    """
    module: float = 2.0           # Module (mm)
    teeth: int = 18               # Number of teeth
    pressure_angle: float = 20.0  # Pressure angle (degrees)
    profile_shift: float = 0.0    # Profile shift coefficient
    addendum_factor: float = 1.0  # Addendum factor
    dedendum_factor: float = 1.25 # Dedendum factor
    backlash_factor: float = 0.0  # Backlash factor
    edge_round_factor: float = 0.1 # Edge rounding factor
    root_round_factor: float = 0.2 # Root rounding factor

    def validate(self):
        """Validate parameter ranges and return error messages if any"""
        errors = []
        
        if not (0.5 <= self.module <= 10.0):
            errors.append("Module must be between 0.5 and 10.0 mm")
            
        if not (5 <= self.teeth <= 200):
            errors.append("Number of teeth must be between 5 and 200")
            
        if not (10.0 <= self.pressure_angle <= 35.0):
            errors.append("Pressure angle must be between 10° and 35°")
            
        if not (-0.8 <= self.profile_shift <= 0.8):
            errors.append("Profile shift must be between -0.8 and 0.8")
            
        if not (0.5 <= self.addendum_factor <= 2.0):
            errors.append("Addendum factor must be between 0.5 and 2.0")
            
        if not (0.8 <= self.dedendum_factor <= 2.0):
            errors.append("Dedendum factor must be between 0.8 and 2.0")
            
        if not (0.0 <= self.backlash_factor <= 0.5):
            errors.append("Backlash factor must be between 0.0 and 0.5")
            
        if not (0.0 <= self.edge_round_factor <= 0.3):
            errors.append("Edge round factor must be between 0.0 and 0.3")
            
        if not (0.0 <= self.root_round_factor <= 0.5):
            errors.append("Root round factor must be between 0.0 and 0.5")
            
        return errors

    def to_dict(self):
        """Convert parameters to dictionary"""
        return {
            "module": self.module,
            "teeth": self.teeth,
            "pressure_angle": self.pressure_angle,
            "profile_shift": self.profile_shift,
            "addendum_factor": self.addendum_factor,
            "dedendum_factor": self.dedendum_factor,
            "backlash_factor": self.backlash_factor,
            "edge_round_factor": self.edge_round_factor,
            "root_round_factor": self.root_round_factor
        }

    @classmethod
    def from_dict(cls, data):
        """Create parameters from dictionary"""
        return cls(
            module=data.get("module", 2.0),
            teeth=data.get("teeth", 18),
            pressure_angle=data.get("pressure_angle", 20.0),
            profile_shift=data.get("profile_shift", 0.0),
            addendum_factor=data.get("addendum_factor", 1.0),
            dedendum_factor=data.get("dedendum_factor", 1.25),
            backlash_factor=data.get("backlash_factor", 0.0),
            edge_round_factor=data.get("edge_round_factor", 0.1),
            root_round_factor=data.get("root_round_factor", 0.2)
        )