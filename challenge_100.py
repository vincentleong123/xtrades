"""$100-in-100-CANDLES CHALLENGE (quick round).
Real XAUUSD M1 candles; gold-medalist stack (sine LIMIT, triple-rejection
burst, vwap band fade LIMIT) + the exact exit-engine peels. Score /100.
"""
import sys
sys.path.insert(0, r'godmode')
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from mt5_bot_v2 import (sine_chop_signal, triple_rejection, session_vwap_zone,
                        heiken_ashi, size_lots)

GOAL, BAL0, RISK = 100.0, 444.88, 0.035
SDF, DIS = 2.0, 1.5
CH = 100

mt5.initialize()
bars = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M1, 0, CH + 130)
df = pd.DataFrame({"time": [int(r["time"]) for r in bars],
                   "open": [float(r["open"]) for r in bars],
                   "high": [float(r["high"]) for r in bars],
                   "low": [float(r["low"]) for r in bars],
                   "close": [float(r["close"]) for r in bars]})
mt5.shutdown()
print(f"bars: {len(df)} | challenge = last {CH} | last close {df.close.iloc[-1]:.2f}")

def atr_series(d, per=14):
    tr = np.zeros(len(d))
    tr[1:] = np.maximum(d["high"].values[1:] - d["low"].values[1:],
                        np.maximum(abs(d["high"].values[1:] - d["close"].values[:-1]),
                                   abs(d["low"].values[1:] - d["close"].values[:-1])))
    out = np.full(len(d), np.nan)
    for i in range(per, len(d)):
        out[i] = tr[i - per + 1:i + 1].mean()
    return out

df = df.reset_index(drop=True)
atr_a = atr_series(df)
start = len(df) - CH
bal = BAL0
open_pos, ledger, zone_ts, vwf = [], [], {}, {}
last_loss_i = None

def lots_for(dist):
    if dist <= 0.25:
        return 0.0
    l = size_lots(bal, bal * RISK, dist, {"risk_pct": RISK * 100, "atr_stop_mult": SDF, "base_lots": 0.03})
    return 0.0 if l is None else l

class Pos:
    def __init__(self, d, l_, px, i):
        self.dir, self.lots, self.entry, self.i0 = d, l_, px, i
        self.peak, self.tp, self.tp_moved = 0.0, None, False
        self.sd = SDF * atr_a[i]
        self.soft = px - self.sd if d == "LONG" else px + self.sd
        self.dead = False
    def fl(self, px):
        d = 1 if self.dir == "LONG" else -1
        return d * (px - self.entry) * 100.0 * self.lots
    def hit(self, px, why, i):
        global bal
        bal += self.fl(px)
        ledger.append((i, self.dir, why, self.fl(px)))
        self.dead = True

for i in range(start, len(df)):
    row = df.iloc[i]
    av = atr_a[i]
    if not av or np.isnan(av) or bal < BAL0 * 0.4:
        continue
    hist = df.iloc[:i + 1]
    _ha = heiken_ashi(hist)
    gprev = bool(_ha.close.iloc[-2] > _ha.open.iloc[-2])
    lc_ok = last_loss_i is None or (i - last_loss_i) >= 1  # 60s brake

    if lc_ok and len(open_pos) < 10:
        v = session_vwap_zone(hist)
        if v:
            up, dn = v["vwap"] + 2.01 * v["sd"], v["vwap"] - 2.01 * v["sd"]
            if row.high >= up - 0.12 * av and row.close < up and i - vwf.get("S", -99) > 3:
                l_ = lots_for(0.92 * av)
                if l_ >= 0.01:
                    open_pos.append(Pos("SHORT", l_, row.close, i))
                    vwf["S"] = i
                    ledger.append((i, "SHORT", f"vwapfade in @{row.close:.2f}", None))
            elif row.low <= dn + 0.12 * av and row.close > dn and i - vwf.get("B", -99) > 3:
                l_ = lots_for(0.92 * av)
                if l_ >= 0.01:
                    open_pos.append(Pos("LONG", l_, row.close, i))
                    vwf["B"] = i
                    ledger.append((i, "LONG", f"vwapfade in @{row.close:.2f}", None))

        sw = sine_chop_signal(hist, av)
        if sw and len(open_pos) < 10:
            d_, wh, lvl = sw
            key = f"SW{d_}{lvl:.1f}"
            if i - zone_ts.get(key, -99) > 6:
                l_ = lots_for(1.1 * av)
                if l_ >= 0.01:
                    open_pos.append(Pos(d_, l_, row.close, i))
                    zone_ts[key] = i
                    ledger.append((i, d_, f"sinewave in @{row.close:.2f}", None))

        tr_ = triple_rejection(hist, av)
        if tr_ and len(open_pos) < 6:
            td_, twy, tz = tr_
            key = f"TR{td_}{tz:.1f}"
            if i - zone_ts.get(key, -99) > 9:
                sl_tr = (tz - 0.9 * av) if td_ == "LONG" else (tz + 0.9 * av)
                l_ = lots_for(abs(row.close - sl_tr))
                if l_ >= 0.01:
                    open_pos.append(Pos(td_, l_, row.close, i))
                    zone_ts[key] = i
                    ledger.append((i, td_, f"bottomhunt in @{row.close:.2f}", None))

    # exits
    for pos in open_pos:
        if pos.dead:
            continue
        g = 1 if pos.dir == "LONG" else -1
        fl_ex = pos.fl(row.low if pos.dir == "LONG" else row.high)   # best float this bar
        fl_cl = pos.fl(row.close)
        pos.peak = max(pos.peak, fl_ex, fl_cl)
        dis = pos.soft - 0.5 * pos.sd if pos.dir == "LONG" else pos.soft + 0.5 * pos.sd
        if (pos.dir == "LONG" and row.low <= dis) or (pos.dir == "SHORT" and row.high >= dis):
            pos.hit(dis, "disaster-SL", i)
            if pos.i0 == i:
                last_loss_i = i
            continue
        if (pos.dir == "LONG" and row.close < pos.soft) or (pos.dir == "SHORT" and row.close > pos.soft):
            pos.hit(row.close, "wick-stop", i)
            last_loss_i = i
            continue
        if pos.peak < 2.0:
            continue
        if fl_cl <= 0:
            pos.hit(row.close, "peak-prot", i)
            continue
        if not pos.tp_moved:
            e2 = max(df.high.iloc[max(0, i - 1):i + 1]) if pos.dir == "LONG" else min(df.low.iloc[max(0, i - 1):i + 1])
            tp = e2 + 0.12 * av if pos.dir == "LONG" else e2 - 0.12 * av
            if (pos.dir == "LONG" and tp > row.close) or (pos.dir == "SHORT" and tp < row.close):
                pos.tp, pos.tp_moved = tp, True
        if pos.tp_moved:
            if (pos.dir == "LONG" and row.high >= pos.tp) or (pos.dir == "SHORT" and row.low <= pos.tp):
                pos.hit(pos.tp, "tp-chop-fill", i)
                continue
        if pos.peak > 2.0 and (pos.peak - fl_cl) >= 0.30 * pos.peak:
            pos.hit(row.close, "chop-out", i)
            continue
        if (pos.dir == "LONG" and not gprev) or (pos.dir == "SHORT" and gprev):
            pos.hit(row.close, "ha-flip", i)
            continue
        if (i - pos.i0) > 3 and fl_cl < 0.25 * pos.peak:
            pos.hit(row.close, "stale", i)
    open_pos = [p for p in open_pos if not p.dead]

# windup: force-flat at the end
for pos in open_pos:
    pos.hit(df.close.iloc[-1], "challenge-end-flat", len(df) - 1)

pnl = bal - BAL0
wins = [t for t in ledger if t[3] is not None]  # hit entries excluded
just_close = [t for t in ledger if t[3] is not None]
w_ = [t for t in ledger if t[3] is not None]  # ledger entries -> close ones

close_led = [t for t in ledger if t[3] is not None]
wins = [t for t in close_led if t[3] > 0]
losses = [t for t in close_led if t[3] <= 0]
eq = [BAL0]
for t in close_led:
    eq.append(eq[-1] + t[3])
peak_eq = max(eq)
dd_ix = eq.index(peak_eq)
dd = peak_eq - min(eq[dd_ix:])
tot_win, tot_loss = sum(t[3] for t in wins), sum(abs(t[3]) for t in losses)
wr = len(wins) / len(close_led) if close_led else 0
score = (60 * max(0, pnl / GOAL)) + (15 * wr if close_led else 0) + \
    15 * max(0, 1.0 - max(0, dd - 25) / 37.5) + (10 * (tot_win / tot_loss / 3) if tot_loss else 0)
score = max(0, min(100, score))

print(f"\n=== CHALLENGE: quick round over {CH} candles ===")
print(f"closed: {len(close_led)} | win rate {wr:.0%} | PnL ${pnl:+.2f} (goal $100)")
print(f"gross wins ${tot_win:.2f} / losses ${tot_loss:.2f} | max dd ${dd:.2f}")
print(f"SCORE: {score:.1f} / 100")
print("\nledger:")
for t in ledger:
    print(f"  bar {t[0]:>4} {t[1]:5} {t[2]:30}" + (f" -> ${t[3]:+.2f}" if t[3] is not None else ""))
