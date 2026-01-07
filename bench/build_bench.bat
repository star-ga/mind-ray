call "C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Auxiliary\Build\vcvars64.bat"
cl /O2 cuda_benchmark.c /link /OUT:cuda_benchmark.exe
