# MIND-Ray Toolchain Guide

## MIND renderer

| Tool | Version | Why |
|------|---------|-----|
| `mindc` | v0.10.2 (pinned in `scripts/get_mindc.*`) | Type-checks and compiles `src/main.mind` |
| `mlir-opt`, `mlir-translate` | LLVM/MLIR 18 or 20 tested; CI uses 20 | `mindc build` lowers MIND through MLIR |
| `clang` | same LLVM install | Links the native executable |
| `python3` | any 3.x | Optional: `scripts/ppm_to_png.py` |

`mindc` finds the MLIR tools on `PATH`, or through the `MLIR_OPT`,
`MLIR_TRANSLATE` and `CLANG` environment variables. `mindc check` and
`mindc --emit-ir` need only the `mindc` binary.

### Install

```bash
# Ubuntu 24.04
sudo apt install mlir-20-tools clang-20
export PATH=/usr/lib/llvm-20/bin:$PATH

# macOS (Homebrew)
brew install llvm
export PATH="$(brew --prefix llvm)/bin:$PATH"

# mindc: downloads the release asset for this OS/arch from
# github.com/star-ga/mind/releases, checks it against the published .sha256
# and installs ./toolchain/mindc
./scripts/get_mindc.sh
./scripts/check_toolchain.sh
```

On Windows, `scripts\get_mindc.ps1` installs `toolchain\mindc.exe` and
`scripts\check_toolchain.ps1` checks for the MLIR tools. If your LLVM
distribution does not ship `mlir-opt`, build under WSL with the `.sh` scripts.

`scripts/build.sh` uses, in order: `$MINDC`, `./toolchain/mindc`, then `mindc`
on `PATH`.

The public `mindc` release emits an executable directly only when the MIND
CPU runtime library is installed, which the release archive does not include
(`error[build]: MIND runtime not found for backend 'cpu'`). `scripts/build.sh`
therefore builds the renderer with `--emit cdylib` (needs only libc) and links
`target/release/mind-ray` around the module's exported `main` with `clang`;
the executable loads `libmind-ray.so` from its own directory. The output is
bit-identical, so the recorded hashes are unchanged.

### Build, run, test

```bash
./scripts/build.sh                          # mindc check, mindc build --emit cdylib, link with clang
./scripts/run.sh spheres 320 180 16 4 42    # out/spheres.ppm
./scripts/test.sh                           # recorded image hashes for all scenes
```

### Troubleshooting

| Message | Fix |
|---------|-----|
| `MIND runtime not found for backend 'cpu'` | You ran `mindc build` without `--emit cdylib`; use `./scripts/build.sh` |
| `MLIR build failed: ... mlir-opt` not found | Install the MLIR tools and put them on `PATH` (see above) |
| `fmt::drift` from `mindc check` | Run `mindc fmt src/main.mind` |
| `FAIL <scene> fnv1a32=...` from `scripts/test.sh` | The renderer output changed; if intended, update the recorded hashes in `scripts/test.sh` |

## CUDA kernel

| Tool | Version |
|------|---------|
| CUDA Toolkit (`nvcc`) | 12.8 or newer (needed for `sm_120`); CI compiles with 13.4 |
| Host compiler | GCC/Clang on Linux; MSVC (Visual Studio Build Tools) on Windows |

```bash
./native-cuda/build.sh              # Linux: native-cuda/build/lib*.so
NVCC=/path/to/nvcc ./native-cuda/build.sh
```

```powershell
.\native-cuda\build_opt.ps1         # Windows: mindray_cuda.dll, copied to bench\
.\scripts\build_cuda.ps1            # Windows: baseline kernel
```

Compiling needs no GPU. Running the kernels needs an NVIDIA GPU of one of the
compiled architectures (`sm_75`, `sm_86`, `sm_89`, `sm_120`) or a newer GPU
that can JIT the embedded PTX.

## Continuous integration

`.github/workflows/ci.yml` runs two jobs on every push and pull request:

- **mind**: installs LLVM 20 MLIR tools and `mindc` v0.10.2, then runs
  `./scripts/test.sh` (type-check, build, image hashes).
- **cuda**: in `nvidia/cuda:13.4.2-devel-ubuntu24.04`, runs
  `./native-cuda/build.sh`.
