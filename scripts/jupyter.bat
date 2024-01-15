@ECHO OFF
call C:/Users/jb/miniconda3/Scripts/activate.bat
cd %~dp0..
call conda activate llm-playground
jupyter notebook
call conda deactivate
pause