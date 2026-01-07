@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Auxiliary\Build\vcvars64.bat"
cd /d "C:\Users\Admin\projects\mind-ray\native-cuda"
nvcc -O3 -use_fast_math -arch=sm_89 -shared -Xcompiler "/MD" -o mindray_cuda_opt.dll mindray_cuda_opt.cu
if exist mindray_cuda_opt.dll (
    echo Build successful!
    copy /Y mindray_cuda_opt.dll mindray_cuda.dll
    copy /Y mindray_cuda.dll ..\bench\mindray_cuda.dll
) else (
    echo Build failed!
)
