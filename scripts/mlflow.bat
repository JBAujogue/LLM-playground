@ECHO OFF
call C:/Users/jb/miniconda3/Scripts/activate.bat
cd %~dp0../experiments/logs-mlflow/
call conda activate llm-playground
mlflow server --host 127.0.0.1 --port 8080
call conda deactivate
pause