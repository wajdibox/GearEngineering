# C:\Users\DELL\Desktop\GearEngineering\PythonCore\gear_app.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
from .gear_parameters import GearParameters
from .geometry_generator import ScientificGearGeometry
from .exports import (
    export_json, export_analysis_csv, export_coordinates,
    export_full_gear_coordinates, export_png, export_dxf,
    export_settings, import_settings
)
from .utils_plotting import create_gear_plot


class ScientificGearApp:
    """Main application for scientific gear generation with GUI controls."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Scientific Gear Generator")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        # Model
        self.parameters = GearParameters()
        self.param_vars = {
            'module': tk.DoubleVar(value=self.parameters.module),
            'teeth': tk.IntVar(value=self.parameters.teeth),
            'pressure_angle': tk.DoubleVar(value=self.parameters.pressure_angle),
            'profile_shift': tk.DoubleVar(value=self.parameters.profile_shift),
            'addendum_factor': tk.DoubleVar(value=self.parameters.addendum_factor),
            'dedendum_factor': tk.DoubleVar(value=self.parameters.dedendum_factor),
            'backlash_factor': tk.DoubleVar(value=self.parameters.backlash_factor),
            'edge_round_factor': tk.DoubleVar(value=self.parameters.edge_round_factor),
            'root_round_factor': tk.DoubleVar(value=self.parameters.root_round_factor)
        }
        self.geometry = None
        self.gear_generated = False

        # View options
        self.view_mode = tk.StringVar(value="full")
        self.show_circles = {k: tk.BooleanVar(value=True) for k in ['base','pitch','offset','outer','root']}

        # Matplotlib figure and canvas
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        # Build UI and bind events
        self._build_ui()
        self._bind_events()

        # Initial generation
        self.generate_gear()

    def _build_ui(self):
        # Main container
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=1)

        # Control panel
        self._create_control_panel(main)
        # Visualization panel
        self._create_visualization_panel(main)
        # Calculations panel
        self._create_calculations_panel(main)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN).pack(side=tk.BOTTOM, fill=tk.X)

    def _create_control_panel(self, parent):
        frame = ttk.Frame(parent, width=350)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0,5))
        frame.grid_propagate(False)

        notebook = ttk.Notebook(frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        self._create_params_tab(notebook)
        self._create_view_tab(notebook)
        self._create_actions_tab(notebook)

    def _create_params_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Parameters")
        canvas = tk.Canvas(tab); scr = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scr.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scr.pack(side=tk.RIGHT, fill=tk.Y)

        # Sections
        self._add_param_section(inner, "Basic Parameters", [
            ("Module (mm)", "module", 0.5, 10.0, 0.1),
            ("Teeth", "teeth", 5, 200, 1),
            ("Pressure Angle", "pressure_angle", 10.0, 35.0, 0.5),
            ("Profile Shift", "profile_shift", -0.8, 0.8, 0.01)
        ])
        self._add_param_section(inner, "Tooth Form", [
            ("Addendum Factor", "addendum_factor", 0.5, 2.0, 0.01),
            ("Dedendum Factor", "dedendum_factor", 0.8, 2.0, 0.01),
            ("Backlash Factor", "backlash_factor", 0.0, 0.5, 0.01),
            ("Edge Round Factor", "edge_round_factor", 0.0, 0.3, 0.01),
            ("Root Round Factor", "root_round_factor", 0.0, 0.5, 0.01)
        ])

    def _add_param_section(self, parent, title, items):
        lf = ttk.LabelFrame(parent, text=title, padding=10)
        lf.pack(fill=tk.X, padx=5, pady=5)
        for lbl, name, mn, mx, step in items:
            self._add_slider(lf, lbl, name, mn, mx, step)

    def _add_slider(self, parent, label, name, minv, maxv, step):
        # Frame
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.X, pady=3)
        ttk.Label(frm, text=label, width=16).pack(side=tk.LEFT)

        var = self.param_vars[name]
        # Entry
        entry_var = tk.StringVar(value=str(var.get()))
        entry = ttk.Entry(frm, textvariable=entry_var, width=6)
        entry.pack(side=tk.RIGHT, padx=(5,0))
        entry.bind("<Return>", lambda e, n=name, ev=entry_var: self._on_entry(n, ev))

        # Slider
        if name == "teeth":
            slider = tk.Scale(
                frm, from_=minv, to=maxv, orient="horizontal",
                resolution=1, variable=var, showvalue=False,
                command=lambda v, n=name: self._on_slider(n, v)
            )
        else:
            slider = ttk.Scale(
                frm, from_=minv, to=maxv, orient="horizontal",
                variable=var, command=lambda v, n=name: self._on_slider(n, v)
            )
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

    def _on_slider(self, name, value_str):
        try:
            v = int(value_str) if name=="teeth" else float(value_str)
        except ValueError:
            return
        setattr(self.parameters, name, v)
        # schedule
        if hasattr(self, '_timer'): self.root.after_cancel(self._timer)
        self._timer = self.root.after(300, self.generate_gear)

    def _on_entry(self, name, entry_var):
        text = entry_var.get().strip()
        try:
            v = int(text) if name=="teeth" else float(text)
        except ValueError:
            # revert
            prev = getattr(self.parameters, name)
            entry_var.set(str(prev))
            return
        setattr(self.parameters, name, v)
        self.param_vars[name].set(v)
        entry_var.set(str(v))
        self.generate_gear()

    def _create_view_tab(self, notebook):
        tab = ttk.Frame(notebook); notebook.add(tab, text="View")
        vf = ttk.LabelFrame(tab, text="View Mode", padding=10); vf.pack(fill=tk.X, padx=5, pady=5)
        ttk.Radiobutton(vf, text="Full Gear", variable=self.view_mode, value="full", command=self.update_view).pack(anchor=tk.W)
        ttk.Radiobutton(vf, text="Single Tooth", variable=self.view_mode, value="single", command=self.update_view).pack(anchor=tk.W)
        cf = ttk.LabelFrame(tab, text="Circles", padding=10); cf.pack(fill=tk.X, padx=5, pady=5)
        for txt,key in [("Base","base"),("Pitch","pitch"),("Offset","offset"),("Outer","outer"),("Root","root")]:
            ttk.Checkbutton(cf, text=txt+" Circle", variable=self.show_circles[key], command=self.update_view).pack(anchor=tk.W)

    def _create_actions_tab(self, notebook):
        tab = ttk.Frame(notebook); notebook.add(tab, text="Actions")
        ttk.Button(tab, text="Generate Gear", command=self.generate_gear).pack(fill=tk.X, pady=5)
        ef = ttk.LabelFrame(tab, text="Export", padding=10); ef.pack(fill=tk.X, padx=5, pady=5)
        for txt, cmd in [("JSON",self.export_json),("CSV",self.export_analysis_csv),
                         ("Coords",self.export_coordinates),("Full Coords",self.export_full_gear_coordinates),
                         ("PNG",self.export_png),("DXF",self.export_dxf)]:
            ttk.Button(ef, text=txt, command=cmd).pack(fill=tk.X, pady=2)
        sf = ttk.LabelFrame(tab, text="Settings", padding=10); sf.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(sf, text="Save Settings", command=self.save_settings).pack(fill=tk.X, pady=2)
        ttk.Button(sf, text="Load Settings", command=self.load_settings).pack(fill=tk.X, pady=2)
        ttk.Button(sf, text="Reset Defaults", command=self.reset_defaults).pack(fill=tk.X, pady=2)

    def _create_visualization_panel(self, parent):
        vf = ttk.Frame(parent); vf.grid(row=0, column=1, sticky="nsew", padx=5)
        vf.grid_rowconfigure(0, weight=1); vf.grid_columnconfigure(0, weight=1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=vf)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _create_calculations_panel(self, parent):
        cf = ttk.Frame(parent, width=300); cf.grid(row=0, column=2, sticky="nsew", padx=(5,0)); cf.grid_propagate(False)
        ttk.Label(cf, text="Calculations", font=(None,12,'bold')).pack(pady=10)
        tf = ttk.Frame(cf); tf.pack(fill=tk.BOTH, expand=True, padx=5)
        self.calc_text = tk.Text(tf, wrap="word", font=("Courier",9)); scr = ttk.Scrollbar(tf, command=self.calc_text.yview)
        self.calc_text.configure(yscrollcommand=scr.set); self.calc_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scr.pack(side=tk.RIGHT, fill=tk.Y)

    def _bind_events(self):
        self.root.bind('<Control-g>', lambda e: self.generate_gear())
        self.root.bind('<F5>', lambda e: self.generate_gear())

    def generate_gear(self):
        self.status_var.set("🔄 Generating...")
        self.root.update()
        try:
            self.geometry = ScientificGearGeometry(self.parameters)
            self.geometry.generate_single_tooth()
            self.geometry.generate_full_gear()
            self.geometry.generate_reference_circles()
            self.update_view()
            self.update_calculations()
            self.gear_generated = True
            self.status_var.set("✅ Ready")
        except Exception as e:
            self.status_var.set(f"❌ {e}")
            messagebox.showerror("Error", str(e))

    def update_view(self):
        if not self.gear_generated: return
        flags = {k:var.get() for k,var in self.show_circles.items()}
        create_gear_plot(self.fig, self.ax, self.geometry, self.view_mode.get(), flags)
        self.canvas.draw()

    def update_calculations(self):
        if not self.geometry: return
        calc = self.geometry.calculations; p = self.parameters
        txt = (
            f"GEAR PARAMETERS\n{'='*40}\n"
            f"Module: {p.module:.3f} mm\n"
            f"Teeth: {p.teeth}\n"
            f"Pressure: {p.pressure_angle:.1f}°\n\n"
            f"CALCULATED\n{'='*40}\n"
            f"Base DIA: {calc.get('base_dia',0):.3f} mm\n"
            f"Pitch DIA: {calc.get('pitch_dia',0):.3f} mm\n"
        )
        self.calc_text.delete(1.0, tk.END)
        self.calc_text.insert(tk.END, txt)

        # Export handlers
    def export_json(self):
        if not self.gear_generated:
            messagebox.showwarning("Warning", "Generate first")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Export JSON"
        )
        if filename:
            success, message = export_json(self.parameters, self.geometry, filename)
            if success:
                messagebox.showinfo("Export JSON", message)
                self.status_var.set(message)
            else:
                messagebox.showerror("Error", message)

    def export_analysis_csv(self):
        if not self.gear_generated:
            messagebox.showwarning("Warning", "Generate first")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Export Analysis CSV"
        )
        if filename:
            success, message = export_analysis_csv(self.parameters, self.geometry, filename)
            if success:
                self.status_var.set(message)
                messagebox.showinfo("Export CSV", message)
            else:
                messagebox.showerror("Error", message)

    def export_coordinates(self):
        if not self.gear_generated:
            messagebox.showwarning("Warning", "Generate first")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Export Coordinates"
        )
        if filename:
            success, message = export_coordinates(self.geometry, filename)
            if success:
                self.status_var.set(message)
                messagebox.showinfo("Export Coordinates", message)
            else:
                messagebox.showerror("Error", message)

    def export_full_gear_coordinates(self):
        if not self.gear_generated:
            messagebox.showwarning("Warning", "Generate first")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Export Full Gear Coordinates"
        )
        if filename:
            success, message = export_full_gear_coordinates(self.geometry, filename)
            if success:
                self.status_var.set(message)
                messagebox.showinfo("Export Full Gear Coords", message)
            else:
                messagebox.showerror("Error", message)

    def export_png(self):
        if not self.gear_generated:
            messagebox.showwarning("Warning", "Generate first")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
            title="Export PNG"
        )
        if filename:
            success, message = export_png(self.fig, filename)
            if success:
                messagebox.showinfo("Export PNG", message)
                self.status_var.set(message)
            else:
                messagebox.showerror("Error", message)

    def export_dxf(self):
        if not self.gear_generated:
            messagebox.showwarning("Warning", "Generate first")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".dxf",
            filetypes=[("DXF files", "*.dxf")],
            title="Export DXF"
        )
        if filename:
            success, message = export_dxf(self.geometry, filename)
            if success:
                self.status_var.set(message)
                messagebox.showinfo("Export DXF", message)
            else:
                messagebox.showerror("Error", message)

    def save_settings(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save Settings"
        )
        if filename:
            success, message = export_settings(self.parameters, filename)
            if success:
                self.status_var.set(message)
                messagebox.showinfo("Save Settings", message)
            else:
                messagebox.showerror("Error", message)

    def load_settings(self):
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Load Settings"
        )
        if filename:
            params, message = import_settings(filename)
            if params:
                self.parameters = params
                self.param_vars['module'].set(params.module)
                self.param_vars['teeth'].set(params.teeth)
                self.param_vars['pressure_angle'].set(params.pressure_angle)
                self.param_vars['profile_shift'].set(params.profile_shift)
                self.param_vars['addendum_factor'].set(params.addendum_factor)
                self.param_vars['dedendum_factor'].set(params.dedendum_factor)
                self.param_vars['backlash_factor'].set(params.backlash_factor)
                self.param_vars['edge_round_factor'].set(params.edge_round_factor)
                self.param_vars['root_round_factor'].set(params.root_round_factor)
                self.generate_gear()
                self.status_var.set(message)
            else:
                messagebox.showerror("Error", message)

    def reset_defaults(self):
        self.parameters = GearParameters()
        for k,var in self.param_vars.items():
            var.set(getattr(self.parameters, k))
        self.generate_gear()
        self.status_var.set("Reset to default parameters")


def main():
    root = tk.Tk()
    app = ScientificGearApp(root)
    root.mainloop()
