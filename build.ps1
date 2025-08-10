git clean -dfX
Remove-Item -Recurse -Force build

# 🚧 Variáveis de ambiente para o build
$envPaths = @(
  "C:\Windows\System32",
  "C:\Program Files\Python312",
  "C:\Program Files\Python312\Scripts",
  "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin",
  "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64",
  "C:\Program Files (x86)\Windows Kits\10\bin\10.0.20348.0\x64",
  "C:\Program Files\Git\cmd"
) -join ";"
[System.Environment]::SetEnvironmentVariable('Path', $envPaths, 'Process')

[System.Environment]::SetEnvironmentVariable('INCLUDE',
  'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.20348.0\shared;C:\Program Files (x86)\Windows Kits\10\Include\10.0.20348.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.20348.0\um;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\extras\CUPTI\include',
  'Process')

[System.Environment]::SetEnvironmentVariable('LIB',
  'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.20348.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.20348.0\um\x64',
  'Process')

# 🔧 Flags Triton
[System.Environment]::SetEnvironmentVariable('TRITON_OFFLINE_BUILD', '0', 'Process')
[System.Environment]::SetEnvironmentVariable('TRITON_BUILD_UT', '1', 'Process')
[System.Environment]::SetEnvironmentVariable('TRITON_BUILD_BINARY', '1', 'Process')
[System.Environment]::SetEnvironmentVariable('TRITON_BUILD_PROTON', '0', 'Process')
[System.Environment]::SetEnvironmentVariable('TRITON_BUILD_WITH_CCACHE', '1', 'Process')

# ⚙️ Flags CMake e MSVC
[System.Environment]::SetEnvironmentVariable('CMAKE_ARGS', '-DCMAKE_CXX_STANDARD=17', 'Process')
[System.Environment]::SetEnvironmentVariable('CL', '/Zc:__cplusplus /std:c++17', 'Process')

# 🎯 Build only for NVIDIA
[System.Environment]::SetEnvironmentVariable('TRITON_CODEGEN_BACKENDS', 'nvidia', 'Process')

# 🧪 Instalação + log de erros filtrado
try {
  & "C:\Program Files\Python312\python.exe" -m pip install -e . 2> errors.txt
  Select-String -Path errors.txt -Pattern "error" | Out-File -FilePath errors_clean.txt -Encoding utf8
} catch {
  Write-Host "❌ Erro ao instalar dependências via pip"
}
