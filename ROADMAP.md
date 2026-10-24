# Roadmap

MIND-Ray is a path tracer written in MIND. Today the MIND renderer runs on the CPU
and a separate hand-written CUDA kernel exists for benchmarking; the two do not
share code. The roadmap closes that gap in the only direction that fits the
project: the GPU render comes from the MIND source.

1. **GPU rendering from the MIND renderer.** Compile `src/main.mind` to the GPU
   through the MIND toolchain and render the three scenes. Gate: the GPU build
   prints the same FNV-1a image hashes as the recorded CPU hashes
   (`b190b3bd`, `6a1239f6`, `842975cb`) for the same arguments; `scripts/test.sh`
   gains a `gpu` leg. When it passes, `native-cuda/` moves to
   `archive/cuda_reference/` beside the earlier reference and the README table is
   rewritten from the CI run. Status: blocked on the compiler's GPU target for
   this program's feature set (f64 scalar math, raw memory intrinsics, stdin/stdout).
2. **Native CPU build.** Build the CPU renderer through the compiler's native
   backend instead of the MLIR route, with unchanged hashes. Status: blocked on f64
   and the memory intrinsics entering the native construct allowlist.
3. **Benchmark honesty.** Until item 1 passes, every published MIND-Ray GPU number
   is labelled as the CUDA reference kernel's number, and the MIND renderer's
   numbers are CPU numbers.
