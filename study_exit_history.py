"""STUDY: did closed trades give back big open profit to SL / small exits?

Read-only. For every closed position (last N days), reconstruct:
  - entry/exit time, type, volume, entry price, realized profit
  - original SL & TP as set at entry (from the entry deal)
  - MFE/MAE in $ via XAUUSD M1 bars over the position's lifetime
  - realized R (vs entry SL risk), MFE R, hold minutes, exit style

Print a per-trade table + give-back aggregates. Run while MT5 is open.
"""
import sys
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np

MAGIC_BOT = 987001
DAYS = 7


def main():
    if not mt5.initialize():
        print("MT5 init FAILED:", mt5.last_error())
        return 1
    acc = mt5.account_info()
    print(f"acct {acc.login} bal {acc.balance:.2f} eq {acc.equity:.2f}")

    to = datetime.now(timezone.utc) + timedelta(hours=3)  # server time approx UTC+3
    frm = to - timedelta(days=DAYS)
    deals = mt5.history_deals_get(frm, to)
    if deals is None or len(deals) == 0:
        print("no deals")
        mt5.shutdown()
        return 0
    deals = [d for d in deals if d.symbol == "XAUUSD" and d.entry in (mt5.DEAL_ENTRY_IN, mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT, mt5.DEAL_ENTRY_OUT_BY)]

    by_pos = {}
    for d in deals:
        by_pos.setdefault(d.position_id, []).append(d)

    # M1 bars for the window
    bars = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M1, frm, to)
    times = np.array([b["time"] for b in bars], dtype=np.int64)
    highs = np.array([b["high"] for b in bars])
    lows = np.array([b["low"] for b in bars])

    # original SL/TP come from ORDERS (deals have no sl/tp)
    ords = mt5.history_orders_get(frm, to) or []
    pos_sltp = {}
    for o in ords:
        if o.symbol != "XAUUSD" or o.state not in (mt5.ORDER_STATE_FILLED, mt5.ORDER_STATE_PARTIAL):
            continue
        # keep the FIRST (entry) order per position
        if o.position_id not in pos_sltp or o.time_setup < pos_sltp[o.position_id][2]:
            pos_sltp[o.position_id] = (o.sl, o.tp, o.time_setup)

    rows = []
    for pid, ds in by_pos.items():
        ds = sorted(ds, key=lambda x: x.time)
        entry = next((d for d in ds if d.entry == mt5.DEAL_ENTRY_IN), None)
        outs = [d for d in ds if d.entry == mt5.DEAL_ENTRY_OUT]
        if entry is None or not outs:
            continue
        vol_in = entry.volume
        closed_vol = sum(o.volume for o in outs)
        profit = sum(o.profit for o in outs) + sum(o.commission for o in ds) + sum(o.swap for o in ds)
        price_in = entry.price
        is_long = entry.type == mt5.DEAL_TYPE_BUY
        t_in, t_out = entry.time, max(o.time for o in outs)
        hold_min = (t_out - t_in) / 60.0
        out_sl, out_tp, _sto = pos_sltp.get(pid, (0.0, 0.0, 0))
        risk_per_unit = abs(price_in - out_sl) if out_sl else 0.0
        risk_usd = risk_per_unit * vol_in * 100 if risk_per_unit else 0.0  # XAUUSD contract=100oz
        contract = vol_in * 100
        # favorable extreme during lifetime (entry candle time -> exit)
        mask = (times >= t_in) & (times <= t_out)
        if mask.any():
            ext = np.max(highs[mask]) if is_long else np.min(lows[mask])
        else:
            ext = price_in
        mfe_usd = (ext - price_in) * contract if is_long else (price_in - ext) * contract
        mae_usd = 0.0  # fill below
        realized_r = profit / risk_usd if risk_usd else 0.0
        mfe_r = mfe_usd / risk_usd if risk_usd else 0.0
        exit_deal = max(outs, key=lambda o: o.volume)
        reason = exit_deal.comment if exit_deal.comment else ('out')
        rows.append(dict(pid=pid, long=is_long, vol=vol_in, vout=closed_vol,
                         tin=datetime.fromtimestamp(t_in, tz=timezone.utc).strftime("%m-%d %H:%M"),
                         tmin=hold_min, pin=price_in, pout=exit_deal.price,
                         magic=entry.magic,
                         sl=out_sl, tp=out_tp or 0, profit=profit,
                         risk=risk_usd, mfe=mfe_usd, rr=realized_r, mr=mfe_r, reason=reason[:28]))

    rows.sort(key=lambda r: r["tin"])
    magic_b = any(abs(r["profit"]) > 0 for r in rows)
    print("\n{:<14} {:>4} {:>5} {:>9} {:>7} {:>8} {:>7} {:>7} {:>7} {:>6} {:>6} {:>6} {}".format(
        "time", "dir", "vol", "in->out", "hold_m", "profit", "risk", "MFE", "realR", "mfeR", "tp", "magic", "exit"))
    for r in rows:
        out = f"{r['pin']:.2f}->{r['pout']:.2f}"
        print("{:<14} {:>4} {:>5.2f} {:>9} {:>7.1f} {:>8.2f} {:>7.0f} {:>7.2f} {:>7.2f} {:>6.2f} {:>6.0f} {:>6} {}".format(
            r["tin"], "LONG" if r["long"] else "SHRT", r["vol"], out, r["tmin"], r["profit"],
            r["risk"], r["mfe"], r["rr"], r["mr"], r["tp"], r["magic"], r["reason"]))

    good = [r for r in rows if r["risk"]]
    gave = [r for r in good if r["mr"] >= 0.5 and r["rr"] < 0.2]
    winners = [r for r in good if r["rr"] >= 0.5]
    print(f"\n=== aggregates ===")
    print(f"closed positions: {len(rows)} (risk-known: {len(good)})")
    if good:
        print(f"winners realized>=0.5R: {len(winners)}  |  GAVE-BACKS (mfe>=0.5R realized<0.2R): {len(gave)}")
        print(f"total give-back $ (mfe-|realized| on gave-backs): {sum(r['mfe'] - abs(r['profit']) for r in gave):.2f}")
        print(f"sum profit all: {sum(r['profit'] for r in rows):.2f}  sum realized on good: {sum(r['profit'] for r in good):.2f}")
        held = [r['tmin'] for r in good]
        print(f"hold minutes: median {sorted(held)[len(held)//2]:.1f}  max {max(held):.1f}")
    mt5.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
