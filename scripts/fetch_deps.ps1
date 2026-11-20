<#
.SYNOPSIS
    Download and extract the native ONNX Runtime and QNN execution provider
    binaries used by the EdgeBench C++ tools.

.DESCRIPTION
    Fetches pinned, checksum-verified archives into third_party/downloads/ and
    extracts them into:

        third_party/onnxruntime/        include/ and lib/ (win-arm64) from the
                                        Microsoft.ML.OnnxRuntime NuGet package
        third_party/onnxruntime-qnn/    QNN plugin EP + QNN backend DLLs from the
                                        onnxruntime-qnn GitHub release

    Safe to re-run: archives already present with the expected SHA256 are not
    downloaded again, and an extracted directory is reused if it was produced
    from the same archive. Use -Force to re-extract.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\fetch_deps.ps1
#>
[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is much faster without the progress bar
Add-Type -AssemblyName System.IO.Compression.FileSystem

# Keep the onnxruntime version in sync with requirements.txt.
$OrtVersion = '1.32.0'
$QnnVersion = '2.8.0'

$Deps = @(
    @{
        Name    = 'onnxruntime'
        Version = $OrtVersion
        Url     = "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$OrtVersion/microsoft.ml.onnxruntime.$OrtVersion.nupkg"
        File    = "microsoft.ml.onnxruntime.$OrtVersion.nupkg"
        Sha256  = 'a9150bc2992cbbb6722070da9910e1ffa989a8bfc67bdd5bb8e745ca755641db'
        # Archive prefix -> destination subdirectory
        Extract = [ordered]@{
            'build/native/include/'         = 'include'
            'runtimes/win-arm64/native/'    = 'lib'
            'LICENSE'                       = '.'
            'ThirdPartyNotices.txt'         = '.'
        }
    },
    @{
        Name    = 'onnxruntime-qnn'
        Version = $QnnVersion
        Url     = "https://github.com/onnxruntime/onnxruntime-qnn/releases/download/v$QnnVersion/onnxruntime-qnn-$QnnVersion-win-arm64.zip"
        File    = "onnxruntime-qnn-$QnnVersion-win-arm64.zip"
        Sha256  = 'c63ffed1567c241dab60ec2af2251f73bc68e38b86fa0a8ac833d1b14af7d2d4'
        Extract = [ordered]@{
            ''  = '.'   # everything, flat
        }
    }
)

$RepoRoot    = Split-Path -Parent $PSScriptRoot
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

function Expand-Dependency($Dep, [string]$Archive) {
    $dest   = Join-Path $ThirdParty $Dep.Name
    $marker = Join-Path $dest '.fetched'
    $stamp  = "$($Dep.Version) $($Dep.Sha256)"
    if (-not $Force -and (Test-Path -LiteralPath $marker) -and
        ((Get-Content -LiteralPath $marker -Raw).Trim() -eq $stamp)) {
        Write-Host "  up to date: third_party\$($Dep.Name) ($($Dep.Version))"
        return
    }
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

foreach ($dep in $Deps) {
    Write-Host "$($dep.Name) $($dep.Version)"
    $archive = Get-VerifiedArchive $dep
    Expand-Dependency $dep $archive
}
Write-Host "Done."
