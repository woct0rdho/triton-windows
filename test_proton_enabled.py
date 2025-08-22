#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify that Proton can be enabled when explicitly requested.

This script tests building Triton with TRITON_BUILD_PROTON=ON and verifies
that Proton functionality is available.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

def test_cmake_with_proton_enabled():
    """Test CMake configuration with Proton enabled."""
    print("Testing CMake configuration with Proton enabled...")
    
    # Create a temporary build directory
    with tempfile.TemporaryDirectory() as temp_dir:
        build_dir = Path(temp_dir) / "build_test"
        build_dir.mkdir()
        
        try:
            # Test CMake configure with Proton enabled
            cmake_cmd = [
                "cmake",
                "-S", ".",
                "-B", str(build_dir),
                "-DTRITON_BUILD_PYTHON_MODULE=ON",
                "-DTRITON_BUILD_PROTON=ON",  # Enable Proton
                "-DTRITON_CODEGEN_BACKENDS=nvidia"
            ]
            
            print(f"Running: {' '.join(cmake_cmd)}")
            result = subprocess.run(
                cmake_cmd,
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                print("[PASS] CMake configuration with Proton enabled succeeded")
                
                # Check if Proton-related files are included in build
                cmake_cache = build_dir / "CMakeCache.txt"
                if cmake_cache.exists():
                    cache_content = cmake_cache.read_text()
                    if "TRITON_BUILD_PROTON:BOOL=ON" in cache_content:
                        print("[PASS] TRITON_BUILD_PROTON=ON found in CMake cache")
                        return True
                    else:
                        print("[WARN] TRITON_BUILD_PROTON=ON not found in CMake cache")
                        return False
                else:
                    print("[WARN] CMakeCache.txt not found")
                    return False
            else:
                print(f"[FAIL] CMake configuration failed with return code {result.returncode}")
                print(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("[FAIL] CMake configuration timed out")
            return False
        except Exception as e:
            print(f"[FAIL] Error running CMake: {e}")
            return False

def test_proton_files_existence():
    """Test that Proton files exist in the source tree."""
    print("\nTesting Proton files existence...")
    
    proton_files = [
        "third_party/proton/CMakeLists.txt",
        "third_party/proton/proton/__init__.py",
        "third_party/proton/proton/proton.py",
        "third_party/proton/proton/viewer.py",
        "third_party/proton/dialect/CMakeLists.txt"
    ]
    
    missing_files = []
    for file_path in proton_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"[FAIL] Missing Proton files: {missing_files}")
        return False
    else:
        print("[PASS] All expected Proton files found")
        return True

def test_setup_py_proton_detection():
    """Test that setup.py correctly detects Proton flag."""
    print("\nTesting setup.py Proton detection...")
    
    try:
        # Test setup.py with TRITON_BUILD_PROTON=1
        env = os.environ.copy()
        env["TRITON_BUILD_PROTON"] = "1"
        
        # Just test the setup.py script parsing, not full build
        setup_test_cmd = [
            sys.executable, "-c",
            """
import os
import sys
sys.path.insert(0, '.')
from setup import check_env_flag

# Test the environment flag checking
result = check_env_flag('TRITON_BUILD_PROTON', '0')
print(f'TRITON_BUILD_PROTON check result: {result}')
if result:
    print('[PASS] setup.py correctly detects TRITON_BUILD_PROTON=1')
else:
    print('[FAIL] setup.py failed to detect TRITON_BUILD_PROTON=1')
"""
        ]
        
        result = subprocess.run(
            setup_test_cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and "[PASS]" in result.stdout:
            print("[PASS] setup.py Proton detection works correctly")
            return True
        else:
            print(f"[FAIL] setup.py test failed: {result.stdout}{result.stderr}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error testing setup.py: {e}")
        return False

def test_conditional_compilation_guards():
    """Test that C++ conditional compilation guards are in place."""
    print("\nTesting C++ conditional compilation guards...")
    
    files_to_check = [
        ("bin/RegisterTritonDialects.h", "#ifdef TRITON_BUILD_PROTON"),
        ("lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp", "#ifdef TRITON_BUILD_PROTON"),
        ("third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp", "#ifdef TRITON_BUILD_PROTON")
    ]
    
    all_good = True
    for file_path, expected_guard in files_to_check:
        if Path(file_path).exists():
            content = Path(file_path).read_text()
            if expected_guard in content:
                print(f"[PASS] {file_path} has conditional compilation guard")
            else:
                print(f"[FAIL] {file_path} missing conditional compilation guard")
                all_good = False
        else:
            print(f"[WARN] {file_path} not found")
            all_good = False
    
    return all_good

def test_cmake_defines_propagation():
    """Test that CMakeLists.txt properly propagates TRITON_BUILD_PROTON defines."""
    print("\nTesting CMakeLists.txt defines propagation...")
    
    cmake_file = Path("CMakeLists.txt")
    if not cmake_file.exists():
        print("[FAIL] CMakeLists.txt not found")
        return False
    
    content = cmake_file.read_text()
    
    # Check for add_compile_definitions(TRITON_BUILD_PROTON)
    if "add_compile_definitions(TRITON_BUILD_PROTON)" in content:
        print("[PASS] CMakeLists.txt properly adds TRITON_BUILD_PROTON compile definition")
        return True
    else:
        print("[FAIL] CMakeLists.txt missing TRITON_BUILD_PROTON compile definition")
        return False

def main():
    """Run all Proton enablement tests."""
    print("=" * 60)
    print("TRITON PROTON ENABLEMENT VERIFICATION")
    print("=" * 60)
    
    tests = [
        test_proton_files_existence,
        test_conditional_compilation_guards,
        test_cmake_defines_propagation,
        test_setup_py_proton_detection,
        test_cmake_with_proton_enabled
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"[FAIL] Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("PROTON ENABLEMENT TEST RESULTS")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "[PASS]" if result else "[FAIL]"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] ALL TESTS PASSED - Proton can be enabled when requested!")
        return 0
    else:
        print("[WARN] Some tests failed - check the output above")
        return 1

if __name__ == "__main__":
    sys.exit(main())