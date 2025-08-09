# Project: Triton para Windows (somente CUDA)

## General Instructions:

- Lembre-se: estamos adaptando este projeto triton para executar em Windows, somente CUDA e em hardware antigo.
- execute scripts usando poweshell.exe.
- All code should be compatible with Windows, CUDA and CUDA Pascal.

## Specific Component: 'include/triton/Conversion/TritonGPUToLLVM/TritonFPConversion.h'

- Adaptações para compilar triton em CUDA em hardware antigo.

## Specific Component: 'run_tests.ps1'

- executa os testes em 'python/test/unit/combined_test.py' para 'include/triton/Conversion/TritonGPUToLLVM/TritonFPConversion.h' com as especificações necessárias.

## Specific Component: 'python/test/unit/combined_test.py'

- testes para 'include/triton/Conversion/TritonGPUToLLVM/TritonFPConversion.h'.

## Specific Component: 'build.ps1'

- compila triton com as especificações necessárias.