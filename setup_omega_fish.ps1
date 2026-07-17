$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Version = (Get-Content -LiteralPath (Join-Path $Project "VERSION.txt") -Raw).Trim()

function Find-Python {
    $Candidates = @(
        (Join-Path $Project ".venv\Scripts\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"),
        "C:\Program Files\Python312\python.exe",
        "C:\Program Files\Python311\python.exe"
    )
    foreach ($Candidate in $Candidates) {
        if (-not $Candidate -or -not (Test-Path -LiteralPath $Candidate)) { continue }
        try {
            & $Candidate -c "import sys; assert sys.version_info >= (3,11)" 2>$null
            if ($LASTEXITCODE -eq 0) { return @{ File=$Candidate; Args=@() } }
        } catch {}
    }

    $Py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($Py) {
        foreach ($Selector in @("-3.12", "-3.11", "-3")) {
            try {
                & $Py.Source $Selector -c "import sys; assert sys.version_info >= (3,11)" 2>$null
                if ($LASTEXITCODE -eq 0) { return @{ File=$Py.Source; Args=@($Selector) } }
            } catch {}
        }
    }

    $PythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($PythonCommand) {
        try {
            & $PythonCommand.Source -c "import sys; assert sys.version_info >= (3,11)" 2>$null
            if ($LASTEXITCODE -eq 0) { return @{ File=$PythonCommand.Source; Args=@() } }
        } catch {}
    }
    return $null
}

Write-Host ""
Write-Host "Omega FISH Model $Version — Complete Windows Setup"
Write-Host "--------------------------------------------------"
$Python = Find-Python
if (-not $Python) {
    $Winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $Winget) {
        throw "Python 3.11+ was not found and winget is unavailable. Install Python 3.12 from python.org, enable Add Python to PATH, then rerun setup."
    }
    Write-Host "Python 3.12 was not found. Installing it with winget..."
    & $Winget.Source install --id Python.Python.3.12 --exact --scope user --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "Python installation failed." }
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
    $Python = Find-Python
    if (-not $Python) { throw "Python was installed but could not be located. Restart Windows and rerun setup." }
}

$Venv = Join-Path $Project ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Creating isolated Python environment..."
    $BaseArgs = @($Python.Args)
    & $Python.File @BaseArgs -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}

Write-Host "Installing runtime and build dependencies..."
& $VenvPython -m pip install --upgrade pip wheel
& $VenvPython -m pip install -r (Join-Path $Project "requirements_build.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
$env:Path = (Join-Path $Venv "Scripts") + ";" + $env:Path

Write-Host "Building the compiled C++ numerical backend..."
$NativeBuilt = $false
Push-Location $Project
try {
    & $VenvPython (Join-Path $Project "build_native_backend.py") --clean
    if ($LASTEXITCODE -eq 0) {
        $NativeBuilt = $true
        Write-Host "Compiled Omega native engine built and tested."
    } else {
        Write-Warning "The native backend could not be built. Omega will retain the Python fallback. Install Visual Studio Build Tools with the Desktop development with C++ workload, then run BUILD_OMEGA_NATIVE_BACKEND.bat."
    }
} catch {
    Write-Warning "Native build unavailable: $($_.Exception.Message). Omega will retain the tested Python fallback."
} finally {
    Pop-Location
}

Write-Host "Running combined validation..."
$env:PYTHONPATH = $Project
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
& $VenvPython -m compileall (Join-Path $Project "omega_desktop.py") (Join-Path $Project "omega_self_check.py") (Join-Path $Project "expert_workflow_app.py") (Join-Path $Project "chart_studio_app.py") (Join-Path $Project "priority_diagnostics_app.py") (Join-Path $Project "build_native_backend.py") (Join-Path $Project "stock_model") (Join-Path $Project "tests")
if ($LASTEXITCODE -ne 0) { throw "Python compilation validation failed." }
& $VenvPython -m pytest -q (Join-Path $Project "tests")
if ($LASTEXITCODE -ne 0) { throw "Combined unit tests failed. The executable was not built." }
& $VenvPython (Join-Path $Project "omega_cli.py") native-status
if ($LASTEXITCODE -ne 0) { throw "Native backend status check failed." }
& $VenvPython (Join-Path $Project "omega_cli.py") profile (Join-Path $Project "Data_Sets\Data_set_Age_Structured_Demo\model_ready_timeseries.csv") k --points 5 --multistarts 1 --output (Join-Path $Project "reports\setup_profile_smoke.json")
if ($LASTEXITCODE -ne 0) { throw "Refitted likelihood-profile smoke test failed." }
& $VenvPython (Join-Path $Project "omega_self_check.py") --quick
if ($LASTEXITCODE -ne 0) { throw "Omega self-check failed. The executable was not built." }

Write-Host "Building the Windows desktop application..."
Push-Location $Project
try {
    & $VenvPython -m PyInstaller --noconfirm --clean (Join-Path $Project "omega_fish_model.spec")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }
} finally {
    Pop-Location
}

$Built = Join-Path $Project "dist\Omega FISH Model"
$BuiltExe = Join-Path $Built "Omega FISH Model.exe"
if (-not (Test-Path -LiteralPath $BuiltExe)) { throw "The expected executable was not created: $BuiltExe" }

$InstallRoot = Join-Path $env:LOCALAPPDATA "Programs\Omega FISH Model"
if (Test-Path -LiteralPath $InstallRoot) { Remove-Item -LiteralPath $InstallRoot -Recurse -Force }
New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null
Copy-Item -Path (Join-Path $Built "*") -Destination $InstallRoot -Recurse -Force

$Shell = New-Object -ComObject WScript.Shell
$DesktopShortcut = $Shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "Omega FISH Model.lnk"))
$DesktopShortcut.TargetPath = Join-Path $InstallRoot "Omega FISH Model.exe"
$DesktopShortcut.WorkingDirectory = $InstallRoot
$DesktopShortcut.IconLocation = (Join-Path $InstallRoot "Omega FISH Model.exe") + ",0"
$DesktopShortcut.Description = "Omega FISH stock assessment and quantitative modelling platform"
$DesktopShortcut.Save()

$StartMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Omega FISH Model"
New-Item -ItemType Directory -Path $StartMenu -Force | Out-Null
$MenuShortcut = $Shell.CreateShortcut((Join-Path $StartMenu "Omega FISH Model.lnk"))
$MenuShortcut.TargetPath = Join-Path $InstallRoot "Omega FISH Model.exe"
$MenuShortcut.WorkingDirectory = $InstallRoot
$MenuShortcut.IconLocation = (Join-Path $InstallRoot "Omega FISH Model.exe") + ",0"
$MenuShortcut.Save()

@{
    version = $Version
    installed = (Get-Date).ToString("o")
    source = $Project
    executable = (Join-Path $InstallRoot "Omega FISH Model.exe")
    native_backend_built = $NativeBuilt
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $InstallRoot "install.json") -Encoding UTF8

Write-Host ""
Write-Host "Omega FISH Model installed successfully."
Write-Host "Executable: $(Join-Path $InstallRoot 'Omega FISH Model.exe')"
Write-Host "Desktop and Start Menu shortcuts were created."
Start-Process -FilePath (Join-Path $InstallRoot "Omega FISH Model.exe") -WorkingDirectory $InstallRoot
