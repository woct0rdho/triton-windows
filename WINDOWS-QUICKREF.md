# Triton Windows Quick Reference

## 🚀 Quick Start Commands

```powershell
# Clone and build (standard - Proton disabled)
git clone https://github.com/blap/triton-windows.git
cd triton-windows
powershell -ExecutionPolicy Bypass -File build.ps1

# Test installation
python -c "import triton; print(f'Triton {triton.__version__} ready!')"
```

## 🔧 Build Options

| Command | Purpose |
|---------|---------|
| `build.ps1` | Standard build |
| `build.ps1 -Clean` | Clean build (remove all artifacts) |
| `build.ps1 -Verbose` | Detailed build output |
| `build.ps1 -TimeoutMinutes 45` | Custom timeout |
| `build.ps1 -PythonPath "C:\Python312\python.exe"` | Custom Python |

## 📊 Proton Profiler Control

### Check Status
```python
import triton.profiler as profiler
print(f"Available: {profiler.is_available()}")
```

### Enable Proton
```powershell
$env:TRITON_BUILD_PROTON = "ON"
pip install -e . --no-cache-dir
```

### Disable Proton (Default)
```powershell
$env:TRITON_BUILD_PROTON = "OFF"  # or remove variable
pip install -e . --no-cache-dir
```

## 🧪 Testing Commands

| Test | Command | Coverage |
|------|---------|----------|
| Comprehensive | `python test_comprehensive_proton.py` | 100% (17 tests) |
| Default State | `python test_proton_disabled.py` | Core functionality |
| Enablement | `python test_proton_enabled.py` | Proton activation |
| Build System | `python test_proton_enablement_verification.py` | CMake validation |

## 🔍 Troubleshooting

### Common Issues & Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Build Timeout** | Build hangs/stops | `build.ps1 -TimeoutMinutes 60` |
| **Linker Error** | `LNK1181` cannot open file | `build.ps1 -Clean` |
| **Permission Error** | PowerShell policy | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| **Import Error** | Module not found | `pip install -e . --no-cache-dir` |
| **CUDA Not Found** | GPU not detected | Verify CUDA installation & PATH |

### Quick Diagnostics

```powershell
# Check environment
python -c "
import sys, torch
print(f'Python: {sys.executable}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
"

# Check build artifacts
dir python\triton\_C\libtriton.pyd
dir python\triton_windows.egg-info
```

## 📁 Key Files Reference

| File | Purpose |
|------|---------|
| `build.ps1` | Enhanced Windows build script |
| `BUILD-WINDOWS.md` | Comprehensive build guide |
| `CHANGELOG-WINDOWS.md` | Detailed change history |
| `python/triton/profiler.py` | Conditional profiler wrapper |
| `test_*.py` | Comprehensive test suite |

## 🎯 Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TRITON_BUILD_PROTON` | `OFF` | Enable/disable Proton profiling |
| `TRITON_CODEGEN_BACKENDS` | `nvidia` | GPU backend selection |
| `CMAKE_ARGS` | Auto | Additional CMake arguments |
| `MAX_JOBS` | Auto | Parallel compilation jobs |

## 📋 Prerequisites Checklist

- [ ] Windows 10/11 (64-bit)
- [ ] Python 3.9-3.13 (3.12+ recommended)
- [ ] Visual Studio 2022 (any edition)
- [ ] CUDA Toolkit 12.5+ (12.9 recommended)
- [ ] Git for Windows
- [ ] 16GB+ RAM (recommended)
- [ ] SSD storage (recommended)

## 🔄 Development Workflow

### 1. Initial Setup
```powershell
git clone https://github.com/blap/triton-windows.git
cd triton-windows
powershell -ExecutionPolicy Bypass -File build.ps1 -Clean
```

### 2. Code Changes
```powershell
# Quick rebuild for minor changes
pip install -e . --no-cache-dir

# Full rebuild for major changes
powershell -ExecutionPolicy Bypass -File build.ps1 -Clean
```

### 3. Testing
```powershell
# Quick verification
python test_proton_disabled.py

# Full test suite
python test_comprehensive_proton.py
```

### 4. Performance Testing
```python
import torch, triton, time

# Basic GPU kernel test
if torch.cuda.is_available():
    x = torch.rand(1024, device='cuda')
    y = torch.rand(1024, device='cuda')
    
    start = time.time()
    result = torch.add(x, y)
    torch.cuda.synchronize()
    print(f"GPU add time: {(time.time() - start)*1000:.2f} ms")
```

## 🎨 IDE Setup

### Visual Studio Code
1. Install C/C++ extension
2. Open folder: `triton-windows`
3. Configure Python interpreter
4. Use `compile_commands.json` from build directory

### Visual Studio 2022
1. Open Folder: `triton-windows`
2. CMake integration auto-configures
3. Set Python interpreter in Tools → Options

## 🔗 Useful Links

- **Main README**: [README.md](README.md)
- **Build Guide**: [BUILD-WINDOWS.md](BUILD-WINDOWS.md)
- **Changelog**: [CHANGELOG-WINDOWS.md](CHANGELOG-WINDOWS.md)
- **Upstream Triton**: [https://github.com/triton-lang/triton](https://github.com/triton-lang/triton)
- **CUDA Downloads**: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

## 💡 Pro Tips

1. **Use SSD**: Significantly faster builds
2. **Close Apps**: Free up RAM during compilation
3. **Windows Terminal**: Better output display
4. **Regular Cleanup**: `build.ps1 -Clean` for major changes
5. **Test Early**: Run tests after significant modifications

## 📞 Getting Help

1. **Check logs**: Look for `build_output.txt` and `errors_clean.txt`
2. **Run diagnostics**: Use commands in troubleshooting section
3. **Clean build**: Try `build.ps1 -Clean` for persistent issues
4. **Documentation**: Refer to `BUILD-WINDOWS.md` for detailed guidance

---

*Last updated: 2025-08-22 | Triton Windows 3.4.0*