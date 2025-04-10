@echo off
python -m venv venv
call venv\Scripts\activate
pip install dash==2.11.1
pip install dash-bootstrap-components==1.4.1
pip install pandas==1.5.3
pip install numpy==1.23.5
pip install plotly==5.13.1
pip install scikit-learn==1.2.2
echo.
echo All dependencies installed. Virtual environment ready!
pause
