# Sets up Research Radar on Windows:
#   1. a virtual environment in this folder (.venv) with Research Radar installed,
#   2. your data folder with a starter interests.yaml (an existing one is never overwritten),
#   3. a desktop icon that opens the terminal interface,
#   4. a check of Ollama and the two models a scan uses.
# Run it once, and again after moving the folder:
#   powershell -ExecutionPolicy Bypass -File .\install.ps1
# The data folder is %RADAR_HOME% when that is set, otherwise %USERPROFILE%\.research-radar.

$ErrorActionPreference = "Stop"
$App = $PSScriptRoot

# Problems that do not stop the install (Ollama, models, the icon) are collected and printed at the end.
$Notes = New-Object System.Collections.Generic.List[string]

# Python: the copy from the Python install manager, then the py launcher, then python on PATH.
# The python.exe in WindowsApps only opens the Microsoft Store, so it is skipped.
$PyHelp = "Install Python 3.10 or newer from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then run this script again."
$Py = $null
$PyArgs = @()
$managed = Get-ChildItem (Join-Path $env:LOCALAPPDATA "Python") -Directory -Filter "pythoncore-3*" -ErrorAction SilentlyContinue |
    Sort-Object { try { [version]($_.Name -replace '^pythoncore-(\d+\.\d+).*$', '$1') } catch { [version]"0.0" } } -Descending |
    ForEach-Object { Join-Path $_.FullName "python.exe" } |
    Where-Object { Test-Path $_ } |
    Select-Object -First 1
if ($managed) {
    $Py = $managed
} else {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $Py = $launcher.Source
        $PyArgs = @("-3")
    } else {
        $python = Get-Command python -ErrorAction SilentlyContinue
        if ($python -and $python.Source -notlike "*\WindowsApps\*") { $Py = $python.Source }
    }
}
if (-not $Py) { throw "Python was not found. $PyHelp" }
$PyVersion = & $Py @PyArgs -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ($LASTEXITCODE -ne 0 -or -not $PyVersion) { throw "Python could not be started ($Py). $PyHelp" }
$PyVersion = "$PyVersion".Trim()
$PyParts = $PyVersion.Split(".")
if ([int]$PyParts[0] -lt 3 -or ([int]$PyParts[0] -eq 3 -and [int]$PyParts[1] -lt 10)) {
    throw "Python $PyVersion is too old. $PyHelp"
}
Write-Host "Python $PyVersion found."

# The program, installed in editable mode, so the update check can replace its files in place.
$VPy = Join-Path $App ".venv\Scripts\python.exe"
if (-not (Test-Path $VPy)) {
    Write-Host "Creating the virtual environment..."
    & $Py @PyArgs -m venv (Join-Path $App ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Python could not create the .venv folder in $App." }
}
Write-Host "Installing Research Radar and the packages it needs (this can take a minute)..."
& $VPy -m pip install --disable-pip-version-check -q -e $App
if ($LASTEXITCODE -ne 0) { throw "pip could not install Research Radar. Check the internet connection, then run this script again." }
$Radar = Join-Path $App ".venv\Scripts\radar.exe"
& $Radar --version

# The data folder and a starter interests.yaml.
$Init = @(& $Radar init)
if ($LASTEXITCODE -ne 0) { throw "radar init could not create the data folder." }
$Init | Select-Object -First 2 | ForEach-Object { Write-Host $_ }
$Interests = "interests.yaml in your data folder"
$Line = $Init | Where-Object { "$_" -like "Edit your interests here:*" } | Select-Object -First 1
if ($Line) { $Interests = ("$Line" -replace '^Edit your interests here:\s*', '') }

# The desktop icon. It opens Windows Terminal when it is installed, the plain console otherwise.
& $Radar shortcut
if ($LASTEXITCODE -ne 0) { $Notes.Add("The desktop icon could not be created. Try again later with: .venv\Scripts\radar.exe shortcut") }

# Ollama and the models. These checks only warn: the program itself is installed by now.
$OllamaExe = $null
$Found = Get-Command ollama -ErrorAction SilentlyContinue
if ($Found) { $OllamaExe = $Found.Source }
else {
    $Guess = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path $Guess) { $OllamaExe = $Guess }
}
if (-not $OllamaExe) {
    $Notes.Add("Ollama is not installed. Get it from https://ollama.com/download . When it asks you to sign in, choose 'No thanks, I'll use Ollama locally'. Then run this script again to check the models.")
} else {
    # radar doctor --local asks Ollama for its models and prints one line per problem.
    $Prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $Check = @(& $Radar doctor --local)
    $Passed = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $Prev
    if ($Passed) {
        $Check | ForEach-Object { Write-Host $_ }
    } else {
        foreach ($Item in $Check) {
            $Text = ("$Item".Trim() -replace '^FAIL\s+ollama at \S+:\s*', '')
            if ($Text) { $Notes.Add($Text) }
        }
    }
}

Write-Host ""
if ($Notes.Count -eq 0) {
    Write-Host "All checks passed." -ForegroundColor Green
} else {
    Write-Host "Before you start:" -ForegroundColor Yellow
    foreach ($Note in $Notes) { Write-Host "  - $Note" }
}
Write-Host ""
Write-Host "Next: write your own topics in $Interests (the two topics in it are examples), then open Research Radar from the desktop icon."
Write-Host "Commands in a terminal: .venv\Scripts\radar.exe followed by doctor, scan, send or update."
