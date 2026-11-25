# Load the MSVC developer environment for one target architecture into a GitHub
# Actions job: runs vcvarsall.bat of the newest Visual Studio found by vswhere and
# appends every variable it sets or changes to $env:GITHUB_ENV, so later steps
# see cl, link, the Windows SDK and the bundled CMake/Ninja.
#
#   powershell -ExecutionPolicy Bypass -File scripts/msvc_dev_env.ps1 -Arch amd64
#
# -Arch is a vcvarsall.bat argument: amd64 (x64 host and target) or arm64
# (ARM64 host and target). Without GITHUB_ENV (a local run) the variables are
# printed instead. Local builds normally use Launch-VsDevShell.ps1 instead
# (docs/setup.md).
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('amd64', 'arm64')]
    [string]$Arch
)

$ErrorActionPreference = 'Stop'

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found at $vswhere" }
$component = if ($Arch -eq 'arm64') { 'Microsoft.VisualStudio.Component.VC.Tools.ARM64' }
             else { 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64' }
$vs = & $vswhere -latest -products * -requires $component -property installationPath
if (-not $vs) { throw "no Visual Studio installation with $component" }
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvarsall.bat'
if (-not (Test-Path $vcvars)) { throw "vcvarsall.bat not found at $vcvars" }
Write-Host "Visual Studio: $vs"

$before = @{}
Get-ChildItem env: | ForEach-Object { $before[$_.Name] = $_.Value }

$lines = & cmd.exe /d /c "`"$vcvars`" $Arch >nul 2>&1 && set"
if ($LASTEXITCODE -ne 0) { throw "vcvarsall.bat $Arch failed with exit code $LASTEXITCODE" }

$changed = [ordered]@{}
foreach ($line in $lines) {
    $i = $line.IndexOf('=')
    if ($i -le 0) { continue }
    $name = $line.Substring(0, $i)
    $value = $line.Substring($i + 1)
    if (-not $before.ContainsKey($name) -or $before[$name] -ne $value) { $changed[$name] = $value }
}
if (-not $changed.Contains('VCToolsInstallDir')) { throw "vcvarsall.bat $Arch did not set up MSVC" }

$text = ($changed.Keys | ForEach-Object { "$_=$($changed[$_])" }) -join "`n"
if ($env:GITHUB_ENV) {
    # UTF-8 without BOM: Windows PowerShell's Out-File -Encoding utf8 would add one.
    [System.IO.File]::AppendAllText($env:GITHUB_ENV, "$text`n", (New-Object System.Text.UTF8Encoding $false))
} else {
    $text
}
Write-Host "MSVC $($changed['VCToolsVersion']) for $Arch ($($changed.Count) variables set)"
