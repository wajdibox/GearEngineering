# C:\Users\DELL\Desktop\GearEngineering\PythonCore\geometry_generator.py
import numpy as np
import math
from .gear_parameters import GearParameters

class ScientificGearGeometry:
    """Scientific gear geometry generator using precise involute mathematics"""
    
    def __init__(self, params: GearParameters):
        self.params = params
        self.calculations = {}
        
        # Segment counts for precision
        self.seg_involute = 50
        self.seg_edge_round = 15
        self.seg_root_round = 15
        self.seg_outer = 10
        self.seg_root = 10
        self.seg_circle = 100
        
        # Generated geometry
        self.tooth_profile_x = np.array([])
        self.tooth_profile_y = np.array([])
        self.full_gear_x = np.array([])
        self.full_gear_y = np.array([])
        
        # Reference circles
        self.base_circle = np.array([])
        self.pitch_circle = np.array([])
        self.offset_circle = np.array([])
        self.outer_circle = np.array([])
        self.root_circle = np.array([])
        self.generate_full_gear()

    
    def calculate_parameters(self):
        """Calculate gear parameters using scientific formulas"""
        # Unpack parameters for readability
        M = self.params.module
        Z = self.params.teeth
        ALPHA = self.params.pressure_angle
        X = self.params.profile_shift
        A = self.params.addendum_factor
        D = self.params.dedendum_factor
        B = self.params.backlash_factor
        C = self.params.root_round_factor
        E = self.params.edge_round_factor
        
        # Convert to radians
        ALPHA_0 = np.radians(ALPHA)
        
        # Calculate key parameters
        ALPHA_M = np.pi / Z
        ALPHA_IS = ALPHA_0 + np.pi/(2*Z) + B/(Z*np.cos(ALPHA_0)) - (1+2*X/Z)*np.sin(ALPHA_0)/np.cos(ALPHA_0)
        THETA_IS = np.sin(ALPHA_0)/np.cos(ALPHA_0) + 2*(C*(1-np.sin(ALPHA_0))+X-D)/(Z*np.cos(ALPHA_0)*np.sin(ALPHA_0))
        THETA_IE = 2*E/(Z*np.cos(ALPHA_0)) + np.sqrt(((Z+2*(X+A-E))/(Z*np.cos(ALPHA_0)))**2-1)
        ALPHA_E = ALPHA_IS + THETA_IE - np.arctan(np.sqrt(((Z+2*(X+A-E))/(Z*np.cos(ALPHA_0)))**2-1))
        
        # Position calculations
        X_E = M*((Z/2)+X+A)*np.cos(ALPHA_E)
        Y_E = M*((Z/2)+X+A)*np.sin(ALPHA_E)
        X_E0 = M*(Z/2+X+A-E)*np.cos(ALPHA_E)
        Y_E0 = M*(Z/2+X+A-E)*np.sin(ALPHA_E)
        
        # Root calculations
        ALPHA_TS = (2*(C*(1-np.sin(ALPHA_0))-D)*np.sin(ALPHA_0)+B)/(Z*np.cos(ALPHA_0)) - 2*C*np.cos(ALPHA_0)/Z + np.pi/(2*Z)
        THETA_TE = 2*C*np.cos(ALPHA_0)/Z - 2*(D-X-C*(1-np.sin(ALPHA_0)))*np.cos(ALPHA_0)/(Z*np.sin(ALPHA_0))
        
        # Diameter calculations
        base_dia = M * Z * np.cos(ALPHA_0)
        pitch_dia = M * Z
        offset_dia = 2 * M * (Z/2 + X)
        outer_dia = 2 * M * (Z/2 + X + A)
        root_dia = 2 * M * (Z/2 + X - D)
        
        # Store calculations
        self.calculations = {
            'alpha_rad': ALPHA_0, 'alpha_m': ALPHA_M, 'alpha_is': ALPHA_IS,
            'theta_is': THETA_IS, 'theta_ie': THETA_IE, 'alpha_e': ALPHA_E,
            'x_e': X_E, 'y_e': Y_E, 'x_e0': X_E0, 'y_e0': Y_E0,
            'alpha_ts': ALPHA_TS, 'theta_te': THETA_TE,
            'base_dia': base_dia, 'pitch_dia': pitch_dia, 'offset_dia': offset_dia,
            'outer_dia': outer_dia, 'root_dia': root_dia
        }
        
        return self.calculations
    
    def symmetry_y(self, XX, YY):
        """Create mirror symmetry across Y axis"""
        XX2 = XX[::-1]
        YY2 = -YY[::-1]
        return XX2, YY2
    
    def involute_curve(self):
        """Generate involute curve using scientific parametric equations"""
        calc = self.calculations
        M = self.params.module
        Z = self.params.teeth
        
        # Generate parameter space
        THETA1 = np.linspace(calc['theta_is'], calc['theta_ie'], self.seg_involute)
        
        # Scientific involute curve equations
        X11 = (1/2) * M * Z * np.cos(calc['alpha_rad']) * np.sqrt(1 + THETA1**2) * np.cos(calc['alpha_is'] + THETA1 - np.arctan(THETA1))
        Y11 = (1/2) * M * Z * np.cos(calc['alpha_rad']) * np.sqrt(1 + THETA1**2) * np.sin(calc['alpha_is'] + THETA1 - np.arctan(THETA1))
        
        return X11, Y11
    
    def edge_round_curve(self, X11, Y11):
        """Generate edge rounding curve"""
        calc = self.calculations
        M = self.params.module
        E = self.params.edge_round_factor
        
        if len(X11) == 0 or len(Y11) == 0:
            return np.array([]), np.array([])
        
        try:
            # Calculate angle range for the fillet
            THETA3_MIN = np.arctan2(Y11[-1] - calc['y_e0'], X11[-1] - calc['x_e0'])
            THETA3_MAX = np.arctan2(calc['y_e'] - calc['y_e0'], calc['x_e'] - calc['x_e0'])
            
            # Ensure proper angle progression
            if THETA3_MAX < THETA3_MIN:
                THETA3_MAX += 2 * np.pi
            
            THETA3 = np.linspace(THETA3_MIN, THETA3_MAX, self.seg_edge_round)
            
            # Generate fillet points
            X21 = M * E * np.cos(THETA3) + calc['x_e0']
            Y21 = M * E * np.sin(THETA3) + calc['y_e0']
            
            return X21, Y21
        except Exception as e:
            print(f"Edge rounding error: {e}")
            return np.array([]), np.array([])
    
    def root_round_curve(self):
        """Generate root rounding curve"""
        calc = self.calculations
        M = self.params.module
        Z = self.params.teeth
        X = self.params.profile_shift
        D = self.params.dedendum_factor
        C = self.params.root_round_factor
        B = self.params.backlash_factor
        
        # Generate parameter space
        THETA_T = np.linspace(0, calc['theta_te'], self.seg_root_round)
        
        # Calculate root fillet angle
        if C != 0 and (D - X - C) == 0:
            THETA_S = (np.pi/2) * np.ones(len(THETA_T))
        elif (D - X - C) != 0:
            THETA_S = np.arctan((M * Z * THETA_T / 2) / (M * D - M * X - M * C))
        else:
            THETA_S = np.zeros(len(THETA_T))
        
        # Generate root fillet points
        X31 = M * ((Z/2 + X - D + C) * np.cos(THETA_T + calc['alpha_ts']) + 
                   (Z/2) * THETA_T * np.sin(THETA_T + calc['alpha_ts']) - 
                   C * np.cos(THETA_S + THETA_T + calc['alpha_ts']))
        Y31 = M * ((Z/2 + X - D + C) * np.sin(THETA_T + calc['alpha_ts']) - 
                   (Z/2) * THETA_T * np.cos(THETA_T + calc['alpha_ts']) - 
                   C * np.sin(THETA_S + THETA_T + calc['alpha_ts']))
        
        return X31, Y31
    
    def outer_arc(self):
        """Generate outer arc"""
        calc = self.calculations
        M = self.params.module
        Z = self.params.teeth
        X = self.params.profile_shift
        A = self.params.addendum_factor
        
        # Generate points along the tooth tip
        THETA6 = np.linspace(calc['alpha_e'], calc['alpha_m'], self.seg_outer)
        X41 = M * (Z/2 + A + X) * np.cos(THETA6)
        Y41 = M * (Z/2 + A + X) * np.sin(THETA6)
        
        return X41, Y41
    
    def root_arc(self):
        """Generate root arc"""
        calc = self.calculations
        M = self.params.module
        Z = self.params.teeth
        X = self.params.profile_shift
        D = self.params.dedendum_factor
        
        # Generate points along the tooth root
        THETA7 = np.linspace(0, calc['alpha_ts'], self.seg_root)
        X51 = M * (Z/2 - D + X) * np.cos(THETA7)
        Y51 = M * (Z/2 - D + X) * np.sin(THETA7)
        
        return X51, Y51
    
    def combine_tooth(self, X11, Y11, X21, Y21, X31, Y31, X41, Y41, X51, Y51):
        """Combine all curves to form a single tooth profile"""
        try:
            # Create symmetric segments
            X12, Y12 = self.symmetry_y(X11, Y11)
            X22, Y22 = self.symmetry_y(X21, Y21)
            X32, Y32 = self.symmetry_y(X31, Y31)
            X42, Y42 = self.symmetry_y(X41, Y41)
            X52, Y52 = self.symmetry_y(X51, Y51)
            
            # Remove duplicate points at connections
            segments = [
                (X42, Y42),  # Symmetric outer arc
                (X22, Y22),  # Symmetric edge round
                (X12, Y12),  # Symmetric involute
                (X32, Y32),  # Symmetric root round
                (X52, Y52),  # Symmetric root arc
                (X51, Y51),  # Root arc (original)
                (X31, Y31),  # Root round (original)
                (X11, Y11),  # Involute (original)
                (X21, Y21),  # Edge round (original)
                (X41, Y41)   # Outer arc (original)
            ]
            
            # Filter out empty segments
            valid_segments = [(x, y) for x, y in segments if len(x) > 0]
            
            # Combine all segments
            X1 = np.concatenate([x for x, y in valid_segments])
            Y1 = np.concatenate([y for x, y in valid_segments])
            
            return X1, Y1
            
        except Exception as e:
            print(f"Combining tooth segments failed: {e}")
            # Fallback to circular profile
            angles = np.linspace(-np.pi/self.params.teeth, np.pi/self.params.teeth, 50)
            radius = self.calculations.get('pitch_dia', 20) / 2
            return radius * np.cos(angles), radius * np.sin(angles)
    
    def rotation(self, Xtemp, Ytemp, ANGLE, i):
        """Rotate points by given angle"""
        XX = np.cos(ANGLE * i) * Xtemp - np.sin(ANGLE * i) * Ytemp
        YY = np.sin(ANGLE * i) * Xtemp + np.cos(ANGLE * i) * Ytemp
        return XX, YY
    
    def circle(self, diameter):
        """Generate circle points"""
        if diameter <= 0:
            return np.array([]), np.array([])
            
        THETA0 = np.linspace(0.0, 2 * np.pi, self.seg_circle)
        XX = diameter/2 * np.cos(THETA0)
        YY = diameter/2 * np.sin(THETA0)
        return XX, YY
    
    def generate_single_tooth(self):
        """Generate a single tooth profile using scientific method"""
        # Calculate parameters first
        self.calculate_parameters()
        
        try:
            # Generate involute curves
            X11, Y11 = self.involute_curve()
            
            # Generate edge rounding
            X21, Y21 = self.edge_round_curve(X11, Y11)
            
            # Generate root rounding
            X31, Y31 = self.root_round_curve()
            
            # Generate outer arc
            X41, Y41 = self.outer_arc()
            
            # Generate root arc
            X51, Y51 = self.root_arc()
            
            # Combine tooth profile
            X1, Y1 = self.combine_tooth(X11, Y11, X21, Y21, X31, Y31, X41, Y41, X51, Y51)
            
            # Align tooth to proper position
            ALIGN_ANGLE = np.pi/2 - np.pi/self.params.teeth
            self.tooth_profile_x, self.tooth_profile_y = self.rotation(X1, Y1, ALIGN_ANGLE, 1)
            
            return self.tooth_profile_x, self.tooth_profile_y
            
        except Exception as e:
            print(f"Error generating tooth: {e}")
            # Simple fallback
            angles = np.linspace(-np.pi/self.params.teeth, np.pi/self.params.teeth, 50)
            radius = self.calculations.get('pitch_dia', 20) / 2
            self.tooth_profile_x = radius * np.cos(angles)
            self.tooth_profile_y = radius * np.sin(angles)
            return self.tooth_profile_x, self.tooth_profile_y
    
    def generate_full_gear(self):
        """Generate complete gear with all teeth"""
        if len(self.tooth_profile_x) == 0:
            self.generate_single_tooth()
        
        Z = self.params.teeth
        P_ANGLE = 2 * np.pi / Z
        
        full_x = []
        full_y = []
        
        # Generate all teeth
        for i in range(Z):
            Xtemp, Ytemp = self.rotation(self.tooth_profile_x, self.tooth_profile_y, P_ANGLE, i)
            full_x.extend(Xtemp)
            full_y.extend(Ytemp)
        
        self.full_gear_x = np.array(full_x)
        self.full_gear_y = np.array(full_y)
        
        return self.full_gear_x, self.full_gear_y
    
    def generate_reference_circles(self):
        """Generate reference circles"""
        calc = self.calculations
        
        self.base_circle = self.circle(calc['base_dia'])
        self.pitch_circle = self.circle(calc['pitch_dia'])
        self.offset_circle = self.circle(calc['offset_dia'])
        self.outer_circle = self.circle(calc['outer_dia'])
        self.root_circle = self.circle(calc['root_dia'])
        
        return {
            'base': self.base_circle,
            'pitch': self.pitch_circle,
            'offset': self.offset_circle,
            'outer': self.outer_circle,
            'root': self.root_circle
        }
    
    def generate(self):
        self.calculate_parameters()
        self.generate_single_tooth()
        self.generate_full_gear()
        self.generate_reference_circles()
    
    # Add this method inside ScientificGearGeometry class
    def generate(self):
        """Convenience method to generate everything in one go."""
        self.calculate_parameters()
        self.generate_single_tooth()
        self.generate_full_gear()
        self.generate_reference_circles()

