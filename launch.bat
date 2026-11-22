@echo off

cd /d "%~dp0"

py -3 -m venv venv
call venv\Scripts\activate.bat

pip install -r requirements.txt

python gui.py

pause