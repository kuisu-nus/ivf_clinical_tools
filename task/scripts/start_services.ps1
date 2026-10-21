# Start both IVF tool services in the background on Windows (PowerShell 5.1 or 7) and wait
# until both answer /health. The Windows counterpart of start_services.sh.
#
#   powershell -ExecutionPolicy Bypass -File task\scripts\start_services.ps1
#   powershell -ExecutionPolicy Bypass -File task\scripts\start_services.ps1 -Lan     # reachable from the LAN
#   powershell -ExecutionPolicy Bypass -File task\scripts\stop_services.ps1
#
# -TrainCsv  cleaned training CSV (default: $env:IVF_TRAIN_CSV, else outputs\clean_v1\train_clean.csv)
# -Model     model artifact (default: task\services\models\lr_35_indicators.json)
# -Lan       listen on all interfaces (0.0.0.0) instead of 127.0.0.1 and print the LAN addresses
# Logs: outputs\logs (service logs, rotated) and outputs\run (stdout/stderr, process ids).
param(
    [string]$TrainCsv = $env:IVF_TRAIN_CSV,
    [string]$Model = $env:IVF_MODEL,
    [switch]$Lan,
    [int]$SimilarPort = 9201,
    [int]$PredictPort = 9202
)
# Errors are checked explicitly: with "Stop", Windows PowerShell 5.1 aborts on any stderr line of python.
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Services = Join-Path $Root "task/services"
$RunDir = Join-Path $Root "outputs/run"
$LogDir = Join-Path $Root "outputs/logs"
$OnWindows = ($PSVersionTable.PSEdition -eq "Desktop") -or $IsWindows

function Fail([string]$Message) { Write-Host "ERROR: $Message" -ForegroundColor Red; exit 1 }

# Python: the "py" launcher if present (python.exe may be the Microsoft Store stub), else python / python3.
$Python = $null
foreach ($candidate in @("py", "python", "python3")) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        $found = & $candidate -c "import sys; print(sys.executable if sys.version_info >= (3, 10) else '')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $found) { $Python = "$found".Trim(); break }
    }
}
if (-not $Python) { Fail "Python 3.10 or newer not found (tried py, python, python3)" }
& $Python -c "import numpy, fastapi, pydantic, uvicorn" 2>$null
if ($LASTEXITCODE -ne 0) { Fail "missing packages: `"$Python`" -m pip install -r requirements.txt" }

if (-not $TrainCsv) { $TrainCsv = Join-Path $Root "outputs/clean_v1/train_clean.csv" }
if (-not $Model) { $Model = Join-Path $Services "models/lr_35_indicators.json" }
if (-not (Test-Path $TrainCsv)) { Fail "training CSV not found: $TrainCsv (use -TrainCsv)" }
if (-not (Test-Path $Model)) { Fail "model artifact not found: $Model (use -Model)" }
$TrainCsv = (Resolve-Path $TrainCsv -ErrorAction Stop).Path
$Model = (Resolve-Path $Model -ErrorAction Stop).Path

New-Item -ItemType Directory -Force -Path $RunDir, $LogDir -ErrorAction Stop | Out-Null
foreach ($name in @("ivf_similar", "ivf_predict")) {
    $pidFile = Join-Path $RunDir "service-$name.pid"
    if (Test-Path $pidFile) {
        $old = Get-Process -Id ([int](Get-Content $pidFile)) -ErrorAction SilentlyContinue
        if ($old) { Fail "$name is already running (process $($old.Id)); stop it first with stop_services.ps1" }
        Remove-Item $pidFile
    }
}

# Settings read by the services (inherited by the processes started below).
$env:PYTHONPATH = $Services
$env:IVF_TRAIN_CSV = $TrainCsv
$env:IVF_MODEL = $Model
$env:IVF_LOG_DIR = $LogDir
$env:IVF_SIMILAR_PORT = "$SimilarPort"
$env:IVF_PREDICT_PORT = "$PredictPort"
$env:IVF_TOOLS_HOST = $(if ($Lan) { "0.0.0.0" } else { "127.0.0.1" })
$env:PYTHONIOENCODING = "utf-8"

function Start-IvfService([string]$Name, [string]$Folder) {
    $options = @{
        FilePath               = $Python
        ArgumentList           = @("`"$(Join-Path $Services "ivf_services/$Folder/server.py")`"")
        RedirectStandardOutput = (Join-Path $RunDir "service-$Name.out.log")
        RedirectStandardError  = (Join-Path $RunDir "service-$Name.err.log")
        PassThru               = $true
    }
    if ($OnWindows) { $options.WindowStyle = "Hidden" }
    $process = Start-Process @options -ErrorAction Stop
    Set-Content -Path (Join-Path $RunDir "service-$Name.pid") -Value $process.Id
    Write-Host "started $Name (process $($process.Id))"
}
Start-IvfService "ivf_similar" "similar"
Start-IvfService "ivf_predict" "predict"

# Wait for both /health endpoints (no proxy, so a system proxy or TUN cannot get in the way).
$check = @"
import sys, time, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for port in sys.argv[1:]:
    url = 'http://127.0.0.1:%s/health' % port
    deadline = time.time() + 60
    while True:
        try:
            with opener.open(urllib.request.Request(url, headers={'X-Health-Check': '1'}), timeout=2) as r:
                print('ready: %s' % url)
                break
        except OSError:
            if time.time() > deadline:
                sys.stderr.write('service did not become ready at %s\n' % url)
                sys.exit(1)
            time.sleep(0.5)
"@
& $Python -c $check $SimilarPort $PredictPort
if ($LASTEXITCODE -ne 0) {
    foreach ($name in @("ivf_similar", "ivf_predict")) {
        Write-Host "--- last lines of $RunDir/service-$name.err.log ---" -ForegroundColor Yellow
        Get-Content (Join-Path $RunDir "service-$name.err.log") -Tail 8 -ErrorAction SilentlyContinue
    }
    Write-Host "Startup failed (details above and in $LogDir); clean up with stop_services.ps1." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Both services are running (logs: $LogDir; stop: stop_services.ps1)." -ForegroundColor Green
$hosts = @("127.0.0.1")
if ($Lan -and $OnWindows) {
    $hosts += Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.|198\.1[89]\.)' } |
        ForEach-Object { $_.IPAddress }
}
foreach ($h in $hosts) {
    Write-Host "  similar patients: http://${h}:$SimilarPort/docs"
    Write-Host "  pregnancy predict: http://${h}:$PredictPort/docs"
}
