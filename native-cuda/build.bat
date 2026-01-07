@echo off
setlocal

REM Build CUDA DLL for Mind Ray
REM Run from native-cuda directory

echo === Mind Ray CUDA Build ===

call "C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Auxiliary\Build\vcvarsall.bat" x64 > nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to set up Visual Studio environment
    exit /b 1
)

echo Compiling mindray_cuda.cu...
nvcc -O3 -use_fast_math -shared -o mindray_cuda.dll mindray_cuda.cu -Xcompiler "/MD /W3" --expt-relaxed-constexpr -Wno-deprecated-gpu-targets
if errorlevel 1 (
    echo ERROR: Compilation failed
    exit /b 1
)

echo.
echo Build successful!
if exist mindray_cuda.dll (
    for %%A in (mindray_cuda.dll) do echo Output: %%~fA [%%~zA bytes]
)

endlocal
