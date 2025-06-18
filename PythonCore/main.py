# C:\Users\DELL\Desktop\GearEngineering\PythonCore\main.py
import tkinter as tk
from .gear_app import ScientificGearApp

def main():
    print("🔬 Starting Scientific Gear Generator...")
    root = tk.Tk()
    app = ScientificGearApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()