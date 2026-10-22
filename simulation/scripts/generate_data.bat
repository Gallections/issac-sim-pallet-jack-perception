
@echo off
setlocal

REM Isaac Sim installation path
set "ISAAC_SIM_PATH=C:\Issac-Sim"

REM Repository paths
set "REPO_PATH=%~dp0.."
set "SCRIPT_PATH=%REPO_PATH%\palletjack_sdg\standalone_palletjack_sdg.py"

set "OUTPUT_WAREHOUSE=%REPO_PATH%\palletjack_sdg\palletjack_data\distractors_warehouse"
set "OUTPUT_ADDITIONAL=%REPO_PATH%\palletjack_sdg\palletjack_data\distractors_additional"
set "OUTPUT_NO_DISTRACTORS=%REPO_PATH%\palletjack_sdg\palletjack_data\no_distractors"

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
