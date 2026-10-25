#!/bin/bash
# run the dqgpu (torch+CUDA) python with the conda env's DLL folders on PATH
E=/c/Users/kaisa/miniforge3/envs/dqgpu
export PATH="$E:$E/Library/bin:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Scripts:$PATH"
export KMP_DUPLICATE_LIB_OK=TRUE
exec "$E/python.exe" -u "$@"
