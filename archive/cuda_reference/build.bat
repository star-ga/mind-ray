@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist bin mkdir bin
nvcc -O3 -o bin\raytracer.exe main.cu
if %ERRORLEVEL% EQU 0 (
    echo Build successful
    bin\raytracer.exe --w 256 --h 256 --spp 16 --out out.ppm
    echo Output: out.ppm
) else (
    echo Build failed
)
