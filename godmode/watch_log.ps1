$log = 'C:\Users\User\Desktop\xtrades\godmode\bot_live.log'

Write-Host ''
Write-Host '  XAUUSD BOT - LIVE LOG' -ForegroundColor White
Write-Host '  cyan = what the bot is thinking   green = trades/fills   yellow = waiting/standing down   red = errors' -ForegroundColor DarkGray
Write-Host "  file  = $log" -ForegroundColor DarkGray
Write-Host '  press Ctrl+C then Y, or just close this window, to stop watching (bot keeps running)' -ForegroundColor DarkGray
Write-Host ''

try {
    $s = Invoke-RestMethod 'http://127.0.0.1:8765/api/status' -TimeoutSec 4
    Write-Host ('  now: equity ${0} | positions {1} | radar {2} (score {3}) | trades {4} | guard target ${5} | learner {6} trades' -f `
        $s.account.equity, $s.positions.Count, $s.radar.mode, $s.radar.score, $s.config.timeframe, $s.cooldown.profit_guard.target, $s.learning.trades_in_db) -ForegroundColor White
    if ($s.cooldown.autotrading) {
        Write-Host '  Algo Trading: ON - bot can open trades' -ForegroundColor Green
    } else {
        Write-Host '  Algo Trading: OFF - the bot CANNOT trade. Press Ctrl+E inside MT5 now.' -ForegroundColor Red
    }
} catch {
    Write-Host '  bridge unreachable - is bridge.py running?' -ForegroundColor Red
}

Write-Host ''

Get-Content -LiteralPath $log -Tail 12 -Wait | ForEach-Object {
    $l = $_
    if ($l -match '\[think\]') {
        Write-Host $l -ForegroundColor Cyan
    } elseif ($l -match 'ERROR|Exception|reject|failed|retcode=[1-9]') {
        Write-Host $l -ForegroundColor Red
    } elseif ($l -match 'OPEN|CLOSE|-> ok|WICK-STOP|TARGET HIT|fill|autotrading') {
        Write-Host $l -ForegroundColor Green
    } elseif ($l -match 'flatten|stand|cooldown|learn|waiting|loaded|config|profit-guard') {
        Write-Host $l -ForegroundColor Yellow
    } else {
        Write-Host $l -ForegroundColor Gray
    }
}