<#
.SYNOPSIS
    Download and extract the native ONNX Runtime and QNN execution provider
    binaries used by the EdgeBench C++ tools.

.DESCRIPTION
    Fetches pinned, checksum-verified archives into third_party/downloads/ and
    extracts them into:

        -Arch arm64 (default)
        third_party/onnxruntime/        include/ and lib/ (win-arm64) from the
                                        Microsoft.ML.OnnxRuntime NuGet package
        third_party/onnxruntime-qnn/    QNN plugin EP + QNN backend DLLs from the
                                        onnxruntime-qnn GitHub release

        -Arch x64 (CPU-only builds, e.g. continuous integration)
        third_party/onnxruntime-x64/    include/ and lib/ (win-x64) from the same
                                        NuGet package; the QNN execution provider
                                        exists for ARM64 only and is not fetched

    Versions, URLs and SHA256 pins (of the archives and of the extracted DLLs)
    come from cmake/bundle-pins.json, which CMake also reads, so edgebench doctor
    checks the bundle against the same pins.

    Safe to re-run: a directory already extracted from the expected archive is
    left alone (the archive is not even downloaded), and archives already present
    with the expected SHA256 are not downloaded again. Every run checks the pinned
    files of the extracted directories. Use -Force to re-extract.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\fetch_deps.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\fetch_deps.ps1 -Arch x64
#>
[CmdletBinding()]
param(
    [ValidateSet('arm64', 'x64')]
    [string]$Arch = 'arm64',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is much faster without the progress bar
Add-Type -AssemblyName System.IO.Compression.FileSystem

$RepoRoot = Split-Path -Parent $PSScriptRoot
# The single source of the versions and SHA256 pins (also read by CMake). Keep the
# onnxruntime version in sync with requirements.txt.
$Pins = Get-Content -LiteralPath (Join-Path $RepoRoot 'cmake\bundle-pins.json') -Raw | ConvertFrom-Json

# One NuGet package holds the ONNX Runtime binaries for both architectures.
function New-OrtDependency([string]$Name, [string]$Runtime) {
    $ort = $Pins.onnxruntime
    return @{
        Name    = $Name
        Version = $ort.version
        Url     = $ort.archive.url
        File    = $ort.archive.file
        Sha256  = $ort.archive.sha256
        # Archive prefix -> destination subdirectory
        Extract = [ordered]@{
            'build/native/include/'         = 'include'
            "runtimes/$Runtime/native/"     = 'lib'
            'LICENSE'                       = '.'
            'ThirdPartyNotices.txt'         = '.'
        }
        # Pinned files, relative to the destination
        PinnedDir = 'lib'
        Pinned    = $ort.files.$Arch
    }
}

$Deps = if ($Arch -eq 'x64') { @(New-OrtDependency 'onnxruntime-x64' 'win-x64') } else { @(
    (New-OrtDependency 'onnxruntime' 'win-arm64'),
    @{
        Name      = 'onnxruntime-qnn'
        Version   = $Pins.onnxruntime_qnn.version
        Url       = $Pins.onnxruntime_qnn.archive.url
        File      = $Pins.onnxruntime_qnn.archive.file
        Sha256    = $Pins.onnxruntime_qnn.archive.sha256
        Extract   = [ordered]@{
            ''  = '.'   # everything, flat
        }
        PinnedDir = '.'
        Pinned    = $Pins.onnxruntime_qnn.files.$Arch
    }
) }

$ThirdParty  = Join-Path $RepoRoot 'third_party'
$DownloadDir = Join-Path $ThirdParty 'downloads'
New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null

function Get-Sha256([string]$Path) {
    (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Get-VerifiedArchive($Dep) {
    $path = Join-Path $DownloadDir $Dep.File
    if (Test-Path -LiteralPath $path) {
        if ((Get-Sha256 $path) -eq $Dep.Sha256) {
            Write-Host "  cached   $($Dep.File)"
            return $path
        }
        Write-Warning "Checksum mismatch for cached $($Dep.File); downloading again."
        Remove-Item -LiteralPath $path -Force
    }
    Write-Host "  download $($Dep.Url)"
    $partial = "$path.partial"
    try {
        Invoke-WebRequest -Uri $Dep.Url -OutFile $partial -UseBasicParsing
    } catch {
        if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
        throw "Failed to download $($Dep.Url): $($_.Exception.Message)"
    }
    $actual = Get-Sha256 $partial
    if ($actual -ne $Dep.Sha256) {
        Remove-Item -LiteralPath $partial -Force
        throw "SHA256 mismatch for $($Dep.File)`n  expected $($Dep.Sha256)`n  actual   $actual"
    }
    Move-Item -LiteralPath $partial -Destination $path -Force
    return $path
}

function Test-Extracted($Dep) {
    $marker = Join-Path (Join-Path $ThirdParty $Dep.Name) '.fetched'
    return (Test-Path -LiteralPath $marker) -and
        ((Get-Content -LiteralPath $marker -Raw).Trim() -eq "$($Dep.Version) $($Dep.Sha256)")
}

function Expand-Dependency($Dep, [string]$Archive) {
    $dest   = Join-Path $ThirdParty $Dep.Name
    $marker = Join-Path $dest '.fetched'
    $stamp  = "$($Dep.Version) $($Dep.Sha256)"
    if (Test-Path -LiteralPath $dest) { Remove-Item -LiteralPath $dest -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null

    $zip = [System.IO.Compression.ZipFile]::OpenRead($Archive)
    try {
        $count = 0
        foreach ($entry in $zip.Entries) {
            if ($entry.FullName.EndsWith('/')) { continue }   # directory entry
            foreach ($prefix in $Dep.Extract.Keys) {
                $isMatch = if ($prefix -eq '' -or $prefix.EndsWith('/')) {
                    $entry.FullName.StartsWith($prefix)
                } else {
                    $entry.FullName -eq $prefix
                }
                if (-not $isMatch) { continue }
                $relative = if ($prefix.EndsWith('/')) { $entry.FullName.Substring($prefix.Length) } else { $entry.FullName }
                $target = [System.IO.Path]::GetFullPath(
                    (Join-Path (Join-Path $dest $Dep.Extract[$prefix]) $relative))
                if (-not $target.StartsWith([System.IO.Path]::GetFullPath($dest))) {
                    throw "Refusing to extract outside destination: $($entry.FullName)"
                }
                New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
                $count++
                break
            }
        }
    } finally {
        $zip.Dispose()
    }
    if ($count -eq 0) { throw "No files extracted from $Archive" }
    Set-Content -LiteralPath $marker -Value $stamp -Encoding ascii
    Write-Host "  extracted $count files -> third_party\$($Dep.Name)"
}

# The pinned files of an extracted dependency must match cmake/bundle-pins.json.
function Test-PinnedFiles($Dep) {
    $dir = Join-Path (Join-Path $ThirdParty $Dep.Name) $Dep.PinnedDir
    foreach ($file in $Dep.Pinned) {
        $path = Join-Path $dir $file.name
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Pinned file missing: $path (re-extract with -Force)"
        }
        $actual = Get-Sha256 $path
        if ($actual -ne $file.sha256) {
            throw "SHA256 mismatch for $path`n  expected $($file.sha256)`n  actual   $actual"
        }
    }
    Write-Host "  $(@($Dep.Pinned).Count) pinned files match cmake\bundle-pins.json"
}

Write-Host "Architecture: $Arch"
foreach ($dep in $Deps) {
    Write-Host "$($dep.Name) $($dep.Version)"
    if (-not $Force -and (Test-Extracted $dep)) {
        Write-Host "  up to date: third_party\$($dep.Name) ($($dep.Version))"
    } else {
        $archive = Get-VerifiedArchive $dep
        Expand-Dependency $dep $archive
    }
    Test-PinnedFiles $dep
}
Write-Host "Done."
