# Stop the IVF tool services started by start_services.ps1 (Windows PowerShell 5.1 or 7).
#
#   powershell -ExecutionPolicy Bypass -File task\scripts\stop_services.ps1
$RunDir = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) "outputs/run"
$pidFiles = Get-ChildItem -Path $RunDir -Filter "service-*.pid" -ErrorAction SilentlyContinue
if (-not $pidFiles) { Write-Host "no services recorded in $RunDir"; exit 0 }
foreach ($file in $pidFiles) {
    $name = $file.BaseName -replace '^service-', ''
    $process = Get-Process -Id ([int](Get-Content $file.FullName)) -ErrorAction SilentlyContinue
    if ($process) {
        Stop-Process -Id $process.Id -Force
        Write-Host "stopped $name (process $($process.Id))"
    } else {
        Write-Host "$name was not running"
    }
    Remove-Item $file.FullName
}
