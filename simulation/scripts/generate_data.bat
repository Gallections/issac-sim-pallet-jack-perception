
@echo off
setlocal

REM Isaac Sim installation path
set "ISAAC_SIM_PATH=C:\Issac-Sim"

REM Resolve paths from this batch file, so it works regardless of the current directory.
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO_ROOT=%%~fI"

set "SCRIPT_PATH=%SCRIPT_DIR%palletjack_sdg\standalone_palletjack_sdg.py"

REM Keep Experiment 2 exports separate from the existing Experiment 1 data.
set "OUTPUT_ROOT=%REPO_ROOT%\data\raw\experiment-02"
set "OUTPUT_WAREHOUSE=%OUTPUT_ROOT%\distractors_warehouse"
set "OUTPUT_ADDITIONAL=%OUTPUT_ROOT%\distractors_additional"
set "OUTPUT_NO_DISTRACTORS=%OUTPUT_ROOT%\no_distractors"

REM Verify required files exist
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

echo Starting synthetic data generation...

REM Dataset 1: Warehouse distractors
call "%ISAAC_SIM_PATH%\python.bat" "%SCRIPT_PATH%" --height 544 --width 960 --num_frames 2000 --distractors warehouse --data_dir "%OUTPUT_WAREHOUSE%"
if errorlevel 1 exit /b 1

REM Dataset 2: Additional distractors
call "%ISAAC_SIM_PATH%\python.bat" "%SCRIPT_PATH%" --height 544 --width 960 --num_frames 2000 --distractors additional --data_dir "%OUTPUT_ADDITIONAL%"
if errorlevel 1 exit /b 1

REM Dataset 3: No distractors
call "%ISAAC_SIM_PATH%\python.bat" "%SCRIPT_PATH%" --height 544 --width 960 --num_frames 1000 --distractors None --data_dir "%OUTPUT_NO_DISTRACTORS%"
if errorlevel 1 exit /b 1

echo All three datasets generated successfully.
