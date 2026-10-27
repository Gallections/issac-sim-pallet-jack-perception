
@echo off
setlocal

REM Isaac Sim installation path
set "ISAAC_SIM_PATH=C:\Issac-Sim"

REM Resolve paths from this batch file, so it works regardless of the current directory.
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO_ROOT=%%~fI"

set "SCRIPT_PATH=%SCRIPT_DIR%palletjack_sdg\standalone_palletjack_sdg.py"

REM Keep this new collection separate from Experiments 1 and 2.
set "EXPERIMENT_ID=experiment-03"
set "OUTPUT_ROOT=%REPO_ROOT%\data\raw\%EXPERIMENT_ID%"
set "OUTPUT_WAREHOUSE=%OUTPUT_ROOT%\distractors_warehouse"
set "OUTPUT_ADDITIONAL=%OUTPUT_ROOT%\distractors_additional"
set "OUTPUT_NO_DISTRACTORS=%OUTPUT_ROOT%\no_distractors"
set "OUTPUT_HARD_NEGATIVES=%OUTPUT_ROOT%\hard_negatives"

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

REM Create the experiment folder before Isaac Sim starts writing the collections.
if not exist "%OUTPUT_ROOT%" mkdir "%OUTPUT_ROOT%"

echo Starting complete synthetic data generation for %EXPERIMENT_ID%...

REM Dataset 1: Warehouse distractors
call "%ISAAC_SIM_PATH%\python.bat" "%SCRIPT_PATH%" --height 544 --width 960 --num_frames 2000 --distractors warehouse --data_dir "%OUTPUT_WAREHOUSE%"
if errorlevel 1 exit /b 1

REM Dataset 2: Additional distractors
call "%ISAAC_SIM_PATH%\python.bat" "%SCRIPT_PATH%" --height 544 --width 960 --num_frames 2000 --distractors additional --data_dir "%OUTPUT_ADDITIONAL%"
if errorlevel 1 exit /b 1

REM Dataset 3: No distractors
call "%ISAAC_SIM_PATH%\python.bat" "%SCRIPT_PATH%" --height 544 --width 960 --num_frames 1000 --distractors None --data_dir "%OUTPUT_NO_DISTRACTORS%"
if errorlevel 1 exit /b 1

REM Dataset 4: Target-free hard negatives
call "%ISAAC_SIM_PATH%\python.bat" "%SCRIPT_PATH%" --height 544 --width 960 --num_frames 1000 --distractors-only --data_dir "%OUTPUT_HARD_NEGATIVES%"
if errorlevel 1 exit /b 1

echo All four datasets for %EXPERIMENT_ID% generated successfully.
