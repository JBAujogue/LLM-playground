@ECHO OFF
call C:/Users/jb/miniconda3/Scripts/activate.bat
cd %~dp0../mlruns
call conda activate llm-playground
mlflow server --host 127.0.0.1 --port 8080
call conda deactivate
pause