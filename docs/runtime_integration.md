# Optional: integrate as a Mind Runtime CUDA op

This demo is intentionally standalone.

If you want it as a first-class GPU op inside mind-runtime:

1) Add a new CUDA op kind (e.g. `RaytraceStep`).
2) Add an FFI launch function signature.
3) Add an op wrapper that validates tensor shapes and calls the launch function.
4) Compile the kernel along with existing CUDA kernels.

Suggested op interface:

- Inputs:
  - accumulation_f32: [H, W, 4] (rgba accum, f32)
  - frame_u32: scalar (frame index)
  - seed_u32: scalar
- Output:
  - rgba_u8: [H, W, 4]

Kernel does one progressive step and writes a display buffer.

You can reuse the exact device code from `cuda/main.cu`.
