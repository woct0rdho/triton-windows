#!/usr/bin/env python3
"""
Simple verification script to test that Proton can be enabled when explicitly requested.

This script tests key enablement components without requiring a full build.
"""

import os
import sys
from pathlib import Path

def test_proton_files_exist():
    """Test that all required Proton source files exist."""
    print("1. Testing Proton source files existence...")
    
    required_files = [
        "third_party/proton/CMakeLists.txt",
        "third_party/proton/proton/__init__.py", 
        "third_party/proton/proton/proton.py",
        "third_party/proton/proton/viewer.py",
        "third_party/proton/dialect/CMakeLists.txt",
        "third_party/proton/dialect/lib/TritonProtonToLLVM/CMakeLists.txt"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def test_cmake_option_definition():
    """Test that CMakeLists.txt defines TRITON_BUILD_PROTON option."""
    print("\n2. Testing CMake option definition...")
    
    cmake_file = Path("CMakeLists.txt")
    if not cmake_file.exists():
        print("   ❌ CMakeLists.txt not found")
        return False
    
    content = cmake_file.read_text()
    
    # Check for option definition
    if 'option(TRITON_BUILD_PROTON "Build the Triton Proton profiler" OFF)' in content:
        print("   ✅ TRITON_BUILD_PROTON option properly defined with default OFF")
    else:
        print("   ❌ TRITON_BUILD_PROTON option not properly defined")
        return False
    
    # Check for conditional subdirectory inclusion
    if "if(TRITON_BUILD_PROTON)" in content and "add_subdirectory(third_party/proton)" in content:
        print("   ✅ Conditional Proton subdirectory inclusion found")
    else:
        print("   ❌ Conditional Proton subdirectory inclusion missing")
        return False
    
    # Check for compile definitions
    if "add_compile_definitions(TRITON_BUILD_PROTON)" in content:
        print("   ✅ TRITON_BUILD_PROTON compile definition found")
    else:
        print("   ❌ TRITON_BUILD_PROTON compile definition missing")
        return False
    
    return True

def test_setup_py_integration():
    """Test that setup.py correctly handles TRITON_BUILD_PROTON flag."""
    print("\n3. Testing setup.py integration...")
    
    setup_file = Path("setup.py")
    if not setup_file.exists():
        print("   ❌ setup.py not found")
        return False
    
    content = setup_file.read_text()
    
    # Check for check_env_flag usage
    proton_checks = content.count('check_env_flag("TRITON_BUILD_PROTON"')
    if proton_checks >= 5:  # Should appear multiple times
        print(f"   ✅ TRITON_BUILD_PROTON checks found {proton_checks} times in setup.py")
    else:
        print(f"   ❌ Insufficient TRITON_BUILD_PROTON checks in setup.py (found {proton_checks})")
        return False
    
    # Check for passthrough args
    if '"TRITON_BUILD_PROTON"' in content and "passthrough_args" in content:
        print("   ✅ TRITON_BUILD_PROTON in passthrough_args")
    else:
        print("   ❌ TRITON_BUILD_PROTON not in passthrough_args")
        return False
    
    return True

def test_conditional_compilation_guards():
    """Test that C++ files have proper conditional compilation guards."""
    print("\n4. Testing C++ conditional compilation guards...")
    
    cpp_files_to_check = [
        ("lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp", "#ifdef TRITON_BUILD_PROTON"),
        ("third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/CMakeLists.txt", "if (TRITON_BUILD_PROTON)")
    ]
    
    all_good = True
    for file_path, expected_guard in cpp_files_to_check:
        if Path(file_path).exists():
            content = Path(file_path).read_text()
            if expected_guard in content:
                print(f"   ✅ {file_path} has guard: {expected_guard}")
            else:
                print(f"   ❌ {file_path} missing guard: {expected_guard}")
                all_good = False
        else:
            print(f"   ⚠️  {file_path} not found")
            all_good = False
    
    return all_good

def test_proton_enablement_mechanism():
    """Test the mechanism that enables Proton when flag is set."""
    print("\n5. Testing Proton enablement mechanism...")
    
    # Test environment variable detection
    test_script = '''
import os
import sys
sys.path.insert(0, '.')
from setup import check_env_flag

# Test with TRITON_BUILD_PROTON=ON
os.environ["TRITON_BUILD_PROTON"] = "ON"
result = check_env_flag("TRITON_BUILD_PROTON", "0")
print(f"TRITON_BUILD_PROTON=ON result: {result}")

# Test with TRITON_BUILD_PROTON=1  
os.environ["TRITON_BUILD_PROTON"] = "1"
result = check_env_flag("TRITON_BUILD_PROTON", "0")
print(f"TRITON_BUILD_PROTON=1 result: {result}")

# Test with TRITON_BUILD_PROTON=OFF
os.environ["TRITON_BUILD_PROTON"] = "OFF"
result = check_env_flag("TRITON_BUILD_PROTON", "0")
print(f"TRITON_BUILD_PROTON=OFF result: {result}")

# Test with default (no env var)
if "TRITON_BUILD_PROTON" in os.environ:
    del os.environ["TRITON_BUILD_PROTON"]
result = check_env_flag("TRITON_BUILD_PROTON", "0")
print(f"TRITON_BUILD_PROTON default result: {result}")
'''
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-c", test_script], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            output = result.stdout
            if ("TRITON_BUILD_PROTON=ON result: True" in output and 
                "TRITON_BUILD_PROTON=1 result: True" in output and
                "TRITON_BUILD_PROTON=OFF result: False" in output and
                "TRITON_BUILD_PROTON default result: False" in output):
                print("   ✅ Environment variable detection works correctly")
                return True
            else:
                print("   ❌ Environment variable detection failed")
                print(f"   Output: {output}")
                return False
        else:
            print(f"   ❌ Script execution failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing enablement mechanism: {e}")
        return False

def test_documentation_references():
    """Test that documentation mentions how to enable Proton."""
    print("\n6. Testing documentation references...")
    
    docs_to_check = [
        ("third_party/proton/README.md", "TRITON_BUILD_PROTON"),
        ("python/tutorials/12-optional-profiler-usage.py", "TRITON_BUILD_PROTON"),
        ("BUILD.md", "TRITON_BUILD_PROTON")
    ]
    
    found_docs = 0
    for doc_path, expected_text in docs_to_check:
        if Path(doc_path).exists():
            content = Path(doc_path).read_text()
            if expected_text in content:
                print(f"   ✅ {doc_path} contains {expected_text}")
                found_docs += 1
            else:
                print(f"   ⚠️  {doc_path} missing {expected_text}")
        else:
            print(f"   ⚠️  {doc_path} not found")
    
    if found_docs >= 2:
        print(f"   ✅ Found documentation references in {found_docs} files")
        return True
    else:
        print(f"   ❌ Insufficient documentation references (found {found_docs})")
        return False

def main():
    """Run all verification tests."""
    print("=" * 70)
    print("TRITON PROTON ENABLEMENT VERIFICATION")
    print("=" * 70)
    print("This test verifies that Proton can be enabled when explicitly requested")
    print("without requiring a full build.\n")
    
    tests = [
        test_proton_files_exist,
        test_cmake_option_definition, 
        test_setup_py_integration,
        test_conditional_compilation_guards,
        test_proton_enablement_mechanism,
        test_documentation_references
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 70)
    print("PROTON ENABLEMENT VERIFICATION RESULTS")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL" 
        test_name = test.__name__.replace('test_', '').replace('_', ' ').title()
        print(f"{i+1}. {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Proton can be successfully enabled when TRITON_BUILD_PROTON=ON")
        print("✅ All conditional compilation mechanisms are in place")
        print("✅ Setup infrastructure supports both enabled and disabled states")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("Some components may need attention for full Proton enablement support")
        return 1

if __name__ == "__main__":
    sys.exit(main())