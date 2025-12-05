@echo off
echo =============================================
echo Running Experiment
echo =============================================

REM Example: "run_experiment.bat experiment2_2025-12-01"
REM (Optional) Activate conda/venv:
call D:\python_envs\DL_cuda\Scripts\activate.bat

@echo off
set EXP=%1
if "%EXP%"=="" (
    echo Usage: run_experiment.bat experiment_name
    exit /b 1
)

python generate_two_body_datasets.py --experiment %EXP%
python train_transformer.py          --experiment %EXP%
python use_models.py                 --experiment %EXP%

echo =============================================
echo Experiment finished successfully!
echo =============================================
pause