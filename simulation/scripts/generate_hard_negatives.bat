@echo off
setlocal

REM Update this path to the Isaac Sim installation on this machine.
set "ISAAC_SIM_PATH=C:\Issac-Sim"

REM Resolve project paths from this batch file, regardless of the current directory.
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO_ROOT=%%~fI"
set "SCRIPT_PATH=%SCRIPT_DIR%palletjack_sdg\standalone_palletjack_sdg.py"
set "OUTPUT_DIR=%REPO_ROOT%\data\raw\experiment-03\distractors_only"

if not exist "%ISAAC_SIM_PATH%\python.bat" (
    echo ERROR: Cannot find Isaac Sim Python launcher:
    echo %ISAAC_SIM_PATH%\python.bat
    exit /b 1
)

if not exist "%SCRIPT_PATH%" (
    echo ERROR: Cannot find the SDG Python script:
    echo %SCRIPT_PATH%
    exit /b 1
)

echo Generating Experiment 3 target-free hard-negative data...
call "%ISAAC_SIM_PATH%\python.bat" "%SCRIPT_PATH%" --height 544 --width 960 --num_frames 1000 --distractors-only --data_dir "%OUTPUT_DIR%"
if errorlevel 1 exit /b 1

echo Hard-negative dataset generated at:
echo %OUTPUT_DIR%
