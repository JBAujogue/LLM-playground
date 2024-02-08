@ECHO OFF
call C:/Users/jb/miniconda3/Scripts/activate.bat
cd %~dp0..
call conda activate llm-playground
tensorboard --logdir=mlruns
call conda deactivate
pause
