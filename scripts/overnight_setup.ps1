# Overnight setup: wait for the FasterLivePortrait package download, extract it,
# and convert all ONNX models to TensorRT engines. Logs to overnight_setup.log.
$ErrorActionPreference = "Continue"
$root = "E:\Users\1192027\eva talks"
$zip = "$root\third_party\FasterLivePortrait-windows.zip"
$dest = "$root\third_party\FLP-win"
$log = "$root\overnight_setup.log"
$expectedBytes = 8.6GB

function Log($msg) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Out-File $log -Append -Encoding utf8
}

Log "=== overnight setup started ==="

# 1. wait for the download to finish (size complete and stable), up to 6 h
$deadline = (Get-Date).AddHours(6)
$lastSize = -1
while ($true) {
    if (-not (Test-Path $zip)) { Log "zip not found yet"; Start-Sleep 60; continue }
    $size = (Get-Item $zip).Length
    if ($size -ge $expectedBytes -and $size -eq $lastSize) { Log "download complete: $([math]::Round($size/1GB,2)) GB"; break }
    $lastSize = $size
    if ((Get-Date) -gt $deadline) { Log "TIMEOUT waiting for download ($([math]::Round($size/1GB,2)) GB)"; exit 1 }
    Start-Sleep 60
}

# 2. verify the archive is readable
tar.exe -tf $zip 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Log "ERROR: zip failed integrity listing"; exit 1 }
Log "zip integrity OK"

# 3. extract
if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Force $dest | Out-Null }
Log "extracting..."
tar.exe -xf $zip -C $dest 2>&1 | ForEach-Object { Log "tar: $_" }
if ($LASTEXITCODE -ne 0) { Log "ERROR: extraction failed"; exit 1 }
Log "extraction done"

# 4. find and run the onnx -> tensorrt conversion script (builds engines for THIS gpu)
$bat = Get-ChildItem $dest -Recurse -Filter "all_onnx2trt.bat" | Select-Object -First 1
if (-not $bat) { Log "ERROR: all_onnx2trt.bat not found in package"; exit 1 }
Log "running conversion: $($bat.FullName) (this takes a while)"
Set-Location $bat.DirectoryName
# some package bats live in scripts\ and expect the package root as cwd
if ($bat.DirectoryName -match "scripts$") { Set-Location (Split-Path $bat.DirectoryName) }
cmd /c "call `"$($bat.FullName)`" < NUL" 2>&1 | ForEach-Object { Log "trt: $_" }
Log "conversion script finished with exit code $LASTEXITCODE"

# 5. inventory the built engines
Get-ChildItem (Get-Location).Path -Recurse -Filter "*.trt" -ErrorAction SilentlyContinue |
    ForEach-Object { Log "engine: $($_.FullName.Substring($dest.Length)) ($([math]::Round($_.Length/1MB,0)) MB)" }

Log "=== overnight setup finished ==="
"DONE" | Out-File "$root\overnight_setup.DONE" -Encoding utf8
