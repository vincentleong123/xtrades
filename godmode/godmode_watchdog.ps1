# godmode_watchdog.ps1
# Keeps the XAU/USD demo automation alive: mt5_bot_v2.py (trader) + bridge.py (dashboard).
# - global mutex: single instance
# - starts them if missing (only when MT5 terminal is running)
# - restarts if a process died or the bot's log went silent (hang) while terminal is up
# - bridge health = real HTTP probe on 127.0.0.1:8765
# - restart-storm backoff: >=5 starts in 10 min -> wait 10 min
# - kill switch: create file watchdog-disable.flag  -> stops bot+bridge, watchdog idles
#   (delete the flag to resume; watchdog resumes automatically)
# Launch: start-godmode.vbs in shell:startup (runs hidden at logon)

$ErrorActionPreference = 'SilentlyContinue'

$mutex = New-Object System.Threading.Mutex($false, 'Global\GodmodeWatchdog')
try { if (-not $mutex.WaitOne(0)) { exit } } catch { exit }

$g = 'C:\Users\User\Desktop\xtrades\godmode'
$disable = Join-Path $g 'watchdog-disable.flag'
$log = Join-Path $g 'watchdog.log'

function Log([string]$m) {
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m
    Add-Content -LiteralPath $log -Value $line
    if ((Get-Item -LiteralPath $log).Length -gt 1MB) {
        Get-Content -LiteralPath $log -Tail 400 | Set-Content -LiteralPath $log
    }
}

function Find-Proc([string]$needle) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like $needle }
}

function Test-Bridge {
    try {
        $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8765/api/status' -UseBasicParsing -TimeoutSec 4
        return ($r.StatusCode -eq 200)
    } catch { return $false }
}

$history = @()
$termWasDown = $false

Log "watchdog started (pid $PID)"
Start-Sleep -Seconds 45   # boot grace: logon scripts / MT5 terminal first

while ($true) {
    Start-Sleep -Seconds 30

    # ---- kill switch ----
    if (Test-Path -LiteralPath $disable) {
        $b = Find-Proc '*mt5_bot_v2.py*'
        if ($b) { $b | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; Log 'FLAG: trader stopped' }
        $r = Find-Proc '*bridge.py*'
        if ($r) { $r | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; Log 'FLAG: bridge stopped' }
        continue
    }

    # ---- restart-storm backoff ----
    $now = Get-Date
    $history = @($history | Where-Object { $_ -gt $now.AddMinutes(-10) })
    if ($history.Count -ge 5) {
        Log "restart storm ($($history.Count) starts in 10 min) - backing off 10 min"
        Start-Sleep -Seconds 600
        continue
    }

    $termRunning = [bool](Get-Process -Name terminal64 -ErrorAction SilentlyContinue)

    # ---- bridge: HTTP probe, kill if hung ----
    if (-not (Test-Bridge)) {
        $bp = Find-Proc '*bridge.py*'
        if ($bp) { $bp | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; Log 'bridge HTTP dead - killed' }
        Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', 'python -u bridge.py >> bridge.log 2>> bridge_err.log' -WorkingDirectory $g -WindowStyle Hidden
        $history += (Get-Date)
        Log 'bridge started'
        Start-Sleep -Seconds 5
        continue
    }

    # ---- trader ----
    $botProc = Find-Proc '*mt5_bot_v2.py*'
    if ($botProc) {
        # hang check: bot logs a cycle line every 5s; silent >6 min while terminal
        # is up means stuck (e.g. MT5 IPC deadlock)
        if ($termRunning -and (Test-Path (Join-Path $g 'bot_live.log'))) {
            $age = (Get-Date) - (Get-Item (Join-Path $g 'bot_live.log')).LastWriteTime
            if ($age.TotalMinutes -gt 6) {
                $botProc | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
                Log ("bot hung (log stale {0:N0}s) - killed, will restart" -f $age.TotalSeconds)
                $history += (Get-Date)
                Start-Sleep -Seconds 3
                continue
            }
        }
        if ($termWasDown) { Log 'terminal64 up again'; $termWasDown = $false }
    } else {
        if ($termRunning) {
            Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', 'python -u mt5_bot_v2.py >> bot_live.log 2>> bot_live_err.log' -WorkingDirectory $g -WindowStyle Hidden
            $history += (Get-Date)
            Log 'trader started (terminal up)'
            Start-Sleep -Seconds 6
        } else {
            if (-not $termWasDown) { Log 'terminal64 not running - waiting'; $termWasDown = $true }
        }
    }
}
