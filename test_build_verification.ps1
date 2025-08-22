# Test build verification logic
$expectedOutputs = @("python\triton\_C\libtriton.pyd")

# Check for egg-info directory
$eggInfoFound = $false
$possibleEggInfoDirs = @("python\triton.egg-info", "python\triton_windows.egg-info")

foreach ($eggInfoDir in $possibleEggInfoDirs) {
    if (Test-Path $eggInfoDir) {
        $expectedOutputs += $eggInfoDir
        $eggInfoFound = $true
        Write-Host "Found package info: $eggInfoDir" -ForegroundColor Green
        break
    }
}

if (-not $eggInfoFound) {
    Write-Host "Warning: No egg-info directory found. Expected one of: $($possibleEggInfoDirs -join ', ')" -ForegroundColor Yellow
}

Write-Host "Expected outputs: $($expectedOutputs -join ', ')" -ForegroundColor Cyan

$buildSuccess = $true
foreach ($file in $expectedOutputs) {
    if (!(Test-Path $file)) {
        Write-Host "Expected output not found: $file" -ForegroundColor Yellow
        $buildSuccess = $false
    } else {
        Write-Host "Verified: $file" -ForegroundColor Green
    }
}

if ($buildSuccess) {
    Write-Host "Build verification: PASSED" -ForegroundColor Green
} else {
    Write-Host "Build verification: FAILED" -ForegroundColor Red
}