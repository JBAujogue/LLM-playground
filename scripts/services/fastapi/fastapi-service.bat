@ECHO OFF
call C:/Users/jb/miniconda3/Scripts/activate.bat
cd %~dp0
call conda activate llm-playground
uvicorn fastapi-service:app --root-path . --host 0.0.0.0 --port 8000
call conda deactivate
pause
