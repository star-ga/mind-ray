# Mind Ray Toolchain Guide

This guide explains how to set up the Mind compiler toolchain for building Mind Ray.

---

## Quick Start

### Windows

```powershell
# Check if toolchain is already available
.\scripts\check_toolchain.ps1

# If not available, get Mind compiler
.\scripts\get_mindc.ps1

# Build and run in one command
.\scripts\build_and_run.ps1
```

### Linux/macOS

```bash
# Check if toolchain is already available
./scripts/check_toolchain.sh

# If not available, get Mind compiler
./scripts/get_mindc.sh

# Build and run in one command
./scripts/build_and_run.sh
```

---

## Toolchain Components

Mind Ray requires the following tools:

### Required

1. **Mind Compiler (mindc)**
   - Compiles `.mind` source files to executable binaries
   - Version: Latest from [github.com/cputer/mind](https://github.com/cputer/mind)
   - Install via: `scripts/get_mindc.*` or build from source

2. **Rust/Cargo** (for building Mind from source)
   - Required if building Mind compiler yourself
   - Install from: [rustup.rs](https://rustup.rs)

### Optional

3. **ImageMagick**
   - Converts PPM output to PNG/JPG
   - Windows: [imagemagick.org](https://imagemagick.org)
   - Linux: `sudo apt install imagemagick`
   - macOS: `brew install imagemagick`

---

## Installation Methods

### Method 1: Acquisition Script (Recommended)

The acquisition scripts will automatically download and install the Mind compiler.

**Note**: Pre-built binaries are not yet available. The script currently guides you through building from source.

**Windows:**
```powershell
.\scripts\get_mindc.ps1
```

**Linux/macOS:**
```bash
./scripts/get_mindc.sh
```

This installs `mindc` to `./toolchain/mindc` (or `mindc.exe` on Windows).

---

### Method 2: Build from Source

**Step 1: Clone Mind Repository**
```bash
git clone https://github.com/cputer/mind.git
cd mind
```

**Step 2: Build Compiler**
```bash
cargo build --release
```

**Step 3: Install to Mind Ray**

**Windows:**
```powershell
Copy-Item target\release\mindc.exe ..\mind-ray\toolchain\
```

**Linux/macOS:**
```bash
cp target/release/mindc ../mind-ray/toolchain/
chmod +x ../mind-ray/toolchain/mindc
```

**Step 4: Verify**
```bash
cd ../mind-ray
./scripts/check_toolchain.sh
```

---

### Method 3: Environment Variable

If you have Mind compiler installed elsewhere, set the `MINDC_PATH` environment variable:

**Windows (PowerShell):**
```powershell
$env:MINDC_PATH = "C:\path\to\mindc.exe"
```

**Linux/macOS:**
```bash
export MINDC_PATH=/path/to/mindc
```

**Permanent (Linux/macOS):**
```bash
echo 'export MINDC_PATH=/path/to/mindc' >> ~/.bashrc
source ~/.bashrc
```

---

## Toolchain Detection Order

Mind Ray scripts check for the compiler in this priority order:

1. **Local toolchain**: `./toolchain/mindc(.exe)`
2. **Environment variable**: `$MINDC_PATH` or `$env:MINDC_PATH`
3. **System PATH**: `mindc` command

This allows for:
- Project-local compiler versions (recommended)
- Per-user custom installations
- System-wide installations

---

## Verification

Run the toolchain check script to verify your setup:

**Windows:**
```powershell
.\scripts\check_toolchain.ps1
```

**Linux/macOS:**
```bash
./scripts/check_toolchain.sh
```

**Expected output:**
```
=== Mind Ray Toolchain Check ===

Checking for Mind compiler (mindc)... ✓ FOUND
  Location: ./toolchain/mindc (local toolchain)
  Version: mindc 0.1.0

Checking for Rust/Cargo... ✓ FOUND
  Version: cargo 1.70.0

Checking for ImageMagick (optional)... ✓ FOUND
  Can convert PPM to PNG/JPG

=== Toolchain Check Complete ===

✓ Ready to build Mind Ray!

Next steps:
  ./scripts/build.sh   # Build the renderer
  ./scripts/run.sh     # Run a test render
```

---

## Troubleshooting

### Windows

**Issue: "mindc is not recognized as a command"**
- Solution: Use `.\toolchain\mindc.exe` or run `.\scripts\check_toolchain.ps1`

**Issue: "Execution of scripts is disabled"**
- Solution: Run PowerShell as Administrator and execute:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

**Issue: Build fails with "Cargo not found"**
- Solution: Install Rust from [rustup.rs](https://rustup.rs)
- After install, restart PowerShell

**Issue: "Access denied" when copying to toolchain/**
- Solution: Close any programs using `mindc.exe` and retry

---

### Linux/macOS

**Issue: "Permission denied" when running scripts**
- Solution: Make scripts executable:
  ```bash
  chmod +x scripts/*.sh
  ```

**Issue: "mindc: command not found"**
- Solution: Use `./toolchain/mindc` or set `MINDC_PATH`

**Issue: Build fails with missing Rust**
- Solution: Install Rust via:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source $HOME/.cargo/env
  ```

**Issue: "./toolchain/mindc: No such file or directory"**
- Solution: Run `./scripts/get_mindc.sh` first

---

### Cross-Platform

**Issue: Mind compiler version mismatch**
- Solution: Rebuild from latest Mind repository:
  ```bash
  cd mind
  git pull
  cargo build --release
  # Copy to mind-ray/toolchain/
  ```

**Issue: Build succeeds but binary doesn't run**
- Check Mind compiler version: `mindc --version`
- Verify Mind standard library is compatible with Mind Ray source
- Check for missing dynamic libraries (Linux/macOS): `ldd bin/mind-ray`

---

## Updating the Compiler

To update to a newer version of the Mind compiler:

**Windows:**
```powershell
# Remove old compiler
Remove-Item .\toolchain\mindc.exe -ErrorAction SilentlyContinue

# Get new version
.\scripts\get_mindc.ps1

# Rebuild Mind Ray
.\scripts\build.ps1
```

**Linux/macOS:**
```bash
# Remove old compiler
rm -f ./toolchain/mindc

# Get new version
./scripts/get_mindc.sh

# Rebuild Mind Ray
./scripts/build.sh
```

---

## CI/CD Integration

For continuous integration, use the toolchain check script as a dependency:

```yaml
# Example GitHub Actions workflow
steps:
  - name: Check toolchain
    run: ./scripts/check_toolchain.sh

  - name: Build
    run: ./scripts/build.sh

  - name: Run tests
    run: ./bin/mind-ray verify --seed 42
```

---

## Additional Resources

- **Mind Language**: [github.com/cputer/mind](https://github.com/cputer/mind)
- **Mind Specification**: [github.com/cputer/mind-spec](https://github.com/cputer/mind-spec)
- **Mind Runtime**: [github.com/cputer/mind-runtime](https://github.com/cputer/mind-runtime)
- **Mind Ray Issues**: [github.com/cputer/mind-ray/issues](https://github.com/cputer/mind-ray/issues)
