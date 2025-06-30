@echo off
echo Installing AppWorks dependencies for Windows...
echo.

REM Install core Python packages
echo Installing matplotlib and other dependencies...
pip install matplotlib>=3.5.0
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install pandas>=1.3.0
pip install scikit-learn>=1.0.0

REM Install optional high-performance packages
echo Installing optional packages...
pip install pyfftw>=0.12.0
pip install gplearn>=0.4.0

echo.
echo Installation complete!
echo.
echo Now you can run:
echo   python -m AppWorks.main
echo   python simple_example.py
echo.
pause