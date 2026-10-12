"""SHEEP TAX study (read-only): when we chase a 1-min move with a market
order, how often does the chop slap us back? And what would have happened
if we instead parked a limit order on the OPPOSITE side?

Leg = an M1 candle with body >= 1.2 x ATR14 (a real impulse push).
For each leg:
  CHASE   = market entry at the leg close (the sheep).
  LIMIT   = limit order at the leg extreme so the chop fills us on the
            poke-back (price reversal towards/against the leg).
Stats computed on real XAUUSD M1 bars.
"""
import MetaTrader5 as mt5
import numpy as np
from datetime import datetime, timedelta, timezone

mt5.initialize()
to = datetime.now(timezone.utc)
frm = to - timedelta(days=2)
bars = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M1, frm, to)
o = np.array([r["open"] for r in bars])
h = np.array([r["high"] for r in bars])
l = np.array([r["low"] for r in bars])
c = np.array([r["close"] for r in bars])
n = len(c)
print(f"bars: {n}  ({datetime.fromtimestamp(bars[0]['time'], tz=timezone.utc)} -> "
      f"{datetime.fromtimestamp(bars[-1]['time'], tz=timezone.utc)} UTC)")

# ATR14
tr = np.zeros(n)
tr[1:] = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
atr = None
atr = np.full(n, np.nan)
for i in range(14, n):
    atr[i] = tr[i - 13:i + 1].mean()

BODY_ATR = 1.2
HORIZON = 2          # how many M1 bars to evaluate the aftermath
legs = [i for i in range(15, n - 3) if not np.isnan(atr[i])
        and abs(c[i] - o[i]) >= BODY_ATR * atr[i]
        and abs(c[i + 1] - c[i]) < abs(c[i] - o[i])]   # reduce overlap: leg close ~ stable next bar
print(f"impulse legs detected (body >= {BODY_ATR}x ATR): {len(legs)}")

chase_wins = chase_slap = 0
chase_advs, chase_ends = [], []
lim_fills = 0
lim_results, lim_misses = [], []

for i in legs:
    up = c[i] > o[i]
    a = atr[i]
    nxt_lo = min(l[i + 1:i + 1 + HORIZON])
    nxt_hi = max(h[i + 1:i + 1 + HORIZON])
    end = c[min(i + HORIZON, n - 1)]
    # CHASE: buy the up-leg / sell the down-leg at its close
    if up:
        adv = (c[i] - nxt_lo)      # adverse in $
        res = end - c[i]
    else:
        adv = (nxt_hi - c[i])
        res = c[i] - end
    chase_advs.append(adv / a)
    chase_ends.append(res / a)
    if adv >= 0.5 * a: chase_slap += 1
    if res > 0: chase_wins += 1
    # LIMIT opposite: for an up-leg, park a SHORT limit at leg high+0.1A
    if up:
        lim_px = h[i] + 0.10 * a
        filled = nxt_hi >= lim_px
        res4 = lim_px - end if filled else None
    else:
        lim_px = l[i] - 0.10 * a
        filled = nxt_lo <= lim_px
        res4 = end - lim_px if filled else None
    if filled:
        lim_fills += 1
        lim_results.append(res4 / a)
    else:
        lim_misses.append(abs(end - lim_px) / a)

ca = np.array(chase_advs); ce = np.array(chase_ends)
lr = np.array(lim_results) if lim_results else np.array([0.0])
print("\n=== CHASE (market, right after the move) ===")
print(f"  legs {len(legs)}: slapped (adverse >=0.5 ATR within {HORIZON} min): "
      f"{chase_slap} ({chase_slap/len(legs):.0%})")
print(f"  net-positive after {HORIZON} min: {chase_wins} ({chase_wins/len(legs):.0%})")
print(f"  median adverse: {np.median(ca):.2f} ATR | p75: {np.percentile(ca,75):.2f} ATR")
print(f"  median end: {np.median(ce):+2f} ATR")
print("\n=== LIMIT on the OPPOSITE side (the medalist) ===")
print(f"  filled: {lim_fills}/{len(legs)} ({lim_fills/len(legs):.0%})")
if lim_results:
    print(f"  fill-> result vs leg extreme: median {np.median(lr):+.2f} ATR, "
          f"positive rate: {(lr > 0).mean():.0%}")
mt5.shutdown()
