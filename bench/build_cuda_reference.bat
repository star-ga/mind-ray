@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist bench\bin mkdir bench\bin
nvcc -O3 -use_fast_math -o bench\bin\cuda_reference.exe archive\cuda_reference\main.cu
if %ERRORLEVEL% EQU 0 (
    echo Build successful: bench\bin\cuda_reference.exe
) else (
    echo Build failed
    exit /b 1
)
