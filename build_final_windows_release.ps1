$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Project ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Run SETUP_OMEGA_FISH.bat first to create the build environment."
}
$env:PYTHONPATH = $Project
& $Python -m unittest discover -s (Join-Path $Project "tests") -v
if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
Push-Location $Project
try {
    & $Python -m PyInstaller --noconfirm --clean (Join-Path $Project "omega_fish_model.spec")
    if ($LASTEXITCODE -ne 0) { throw "Executable build failed." }
    $Iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if (-not $Iscc) {
        $Known = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
        if (Test-Path -LiteralPath $Known) { $Iscc = Get-Item $Known }
    }
    if ($Iscc) {
        & $Iscc.Source (Join-Path $Project "installer\OmegaFISH.iss")
        if ($LASTEXITCODE -ne 0) { throw "Installer build failed." }
        Write-Host "Installer created under release\."
    } else {
        Write-Host "Inno Setup 6 was not found. Portable executable created under dist\."
    }
} finally {
    Pop-Location
}
