"""
XAU/USD GOD-MODE BOT (v2)
=========================
EMA crossover engine + Liquidation Radar circuit breaker.

Modes:
  Live demo:  python mt5_bot_v2.py            (needs MT5 terminal + demo login)
  Paper:      python mt5_bot_v2.py --paper   (synthetic feed, full pipeline,
                                              radar included, zero risk)

Flow:
  1. Auto-tune in trading_journey_v2.html, export/copy best params
     into strategy_config.json (or edit from the dashboard itself)
  2. Run this bot    -> writes god_data.json + trades.csv every cycle
  3. Run bridge.py   -> dashboard at http://localhost:8765
  4. Watch the radar. GREEN = bot trades. STAND_DOWN = no new trades.
     FLATTEN = close everything. Config hot-reloads every cycle.

This is a DEMO-account research tool. Not financial advice.
"""

import json
import math
import os
import sys
import time
import csv
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from liquidation_radar import LiquidationRadar, RadarConfig, rolling_atr
from sl_learner import SLLearner

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE, "strategy_config.json")
DATA_PATH = os.path.join(BASE, "god_data.json")
TRADES_CSV = os.path.join(BASE, "trades.csv")
CMD_PATH = os.path.join(BASE, "cmd.json")
CMD_RESULT_PATH = os.path.join(BASE, "cmd_result.json")
PROFIT_GUARD_PATH = os.path.join(BASE, "profit_guard.json")

DEFAULT_CONFIG = {
    "symbol": "XAUUSD",
    "timeframe": "1hour",
    "ema_fast": 9,
    "ema_slow": 21,
    "atr_period": 14,
    "atr_stop_mult": 2.0,
    "rr_ratio": 2.0,
    "risk_pct": 2.0,
    "base_lots": 0.03,
    "max_concurrent": 2,
    "wick_proof_stop": False,
    "disaster_stop_mult": 1.5,
    "profit_target_usd": 0,
    "equity_floor_usd": 0,    # 0 = off; downhill twin of the profit guard
    "entry_mode": "cross",
    "mission_hours": 0,
    "time_stop_s": 216,
    "auto_close": True,
    "cycle_seconds": 60,
    "ultra_mode": False,
    # gold-medalist brakes
    "loss_cooldown_s": 60,      # no new signal latch right after a loss
    # PERMISSION GATE (user 2026-10-09: "an indicator that says should I
    # allow this strategy to trade at ALL right now") - direction-blind.
    # Determined on 5000 real bars at live params (EMA 10/35, risk 3%,
    # GIVE, 216s cap): ADX>=13 + whip 2/45 = profit flat ($934 vs $962),
    # worst loss streak 4 -> 2, PF 3.15 -> 3.49. mode: adx/er/chop/off.
    "regime_gate_mode": "adx",
    "regime_gate_thr": 13,
    "whip_break_n": 2,          # N consecutive losing closes -> stand down
    "whip_break_bars": 45,      # ...for this many 1min bars (0 = off)
    # entry quality gates ("fewer, better trades") - see evaluate_gates()
    "gates": {
        "breakout_n": 0,          # 0 = off; else require close beyond N-bar extreme
        "atr_expansion": False,   # require ATR > its own moving average
        "atr_expansion_period": 50,
        "breakeven_r": 0.0,       # 0 = off; else park stop at entry after +N R
    },
    "radar": {
        "stand_down_score": 70,
        "flatten_score": 90,
    },
    "exit_engine": {
        "on": True,
        "min_peak": 2.0,
        "lock_atr_off": 0.12,
        "retrace_pct": 30,
        "ha_flip": True,
        "hold_max_s": 135,
        "stale_frac": 0.25,
    },
}


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            merged = {**DEFAULT_CONFIG, **cfg}
            merged["radar"] = {**DEFAULT_CONFIG["radar"], **cfg.get("radar", {})}
            return merged
        except Exception as e:
            print(f"[config] broken json, using defaults ({e})")
            return dict(DEFAULT_CONFIG)
    with open(CONFIG_PATH, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    return dict(DEFAULT_CONFIG)


@dataclass
class BtTrade:
    num: int
    direction: str
    lots: float
    entry_price: float
    exit_price: Optional[float]
    entry_time: str
    exit_time: Optional[str]
    pnl: float
    reason: str
    balance_after: Optional[float] = None
    is_win: Optional[bool] = None


class PaperFeed:
    """Synthetic XAU feed: pre-seeds history, evolves one H1 bar per tick."""

    def __init__(self, n_seed: int = 1100, seed: int = 42):
        rng = np.random.default_rng(seed)
        price = 2350.0
        rows = []
        trend, vol = 0.0, 1.2
        for i in range(n_seed):
            if rng.random() < 0.02:
                trend = rng.normal(0, 0.3)
            if rng.random() < 0.01:
                vol = rng.uniform(0.6, 2.2)
            ret = (trend + rng.normal(0, vol)) / 100
            o = price
            price = float(np.clip(price * (1 + ret), 1800, 3200))
            c = price
            w = abs(rng.normal(0, vol * 0.4)) / 100 * price
            rows.append({"open": o, "high": max(o, c) + w,
                         "low": min(o, c) - w, "close": c})
        self.df = pd.DataFrame(rows)
        self.rng = rng
        self.now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        self.spread = 0.30
        self._vol = vol
        self._trend = trend

    def h1(self) -> pd.DataFrame:
        return self.df

    def h4(self) -> pd.DataFrame:
        return self.df.iloc[::4].reset_index(drop=True)

    def tick(self):
        self._vol = max(0.5, self._vol + self.rng.normal(0, 0.05))
        if self.rng.random() < 0.01:
            self._vol = self.rng.uniform(0.8, 3.5)   # regime change
        if self.rng.random() < 0.005:
            # occasional violent liquidation sweep candle
            self._vol *= self.rng.uniform(3.0, 6.0)
        if self.rng.random() < 0.02:
            self._trend = self.rng.normal(0, 0.3)
        ret = (self._trend + self.rng.normal(0, self._vol)) / 100
        o = float(self.df["close"].iloc[-1])
        c = float(np.clip(o * (1 + ret), 1800, 3200))
        w = abs(self.rng.normal(0, self._vol * 0.4)) / 100 * c
        self.df = pd.concat([self.df, pd.DataFrame([{
            "open": o, "high": max(o, c) + w, "low": min(o, c) - w, "close": c
        }])], ignore_index=True)
        self.now = self.now + timedelta(hours=1)
        # spread widens when vol regime is hot
        self.spread = 0.30 * (1.0 + max(0.0, (self._vol - 1.2)))

    def bid_ask(self):
        c = float(self.df["close"].iloc[-1])
        half = self.spread / 2.0
        return c - half, c + half


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def signal(df: pd.DataFrame, cfg: dict) -> Optional[str]:
    """Cross mode: classic fast/slow crossover on the last closed bar.
    Burst mode (aggressive): enter on the breakout bar aligned with the
    EMA trend - no fresh cross needed. Fires bar after bar while the
    trend persists (pyramids up to max_concurrent), which is what makes
    the fast +$100-150 mission windows hit in hours instead of days.
    """
    if len(df) < cfg["ema_slow"] + 2:
        return None
    fast = ema(df["close"], cfg["ema_fast"])
    slow = ema(df["close"], cfg["ema_slow"])
    if str(cfg.get("entry_mode", "cross")).lower() in ("burst", "auto"):
        trend_up = fast.iloc[-1] > slow.iloc[-1]
        n = int((cfg.get("gates") or {}).get("breakout_n", 20) or 20)
        if n > len(df) - 1:
            return None
        hi = float(df["high"].iloc[-(n + 1):-1].max())
        lo = float(df["low"].iloc[-(n + 1):-1].min())
        close = float(df["close"].iloc[-1])
        if trend_up and close > hi:
            return "LONG"
        if not trend_up and close < lo:
            return "SHORT"
        return None
    f_now, f_prev = fast.iloc[-1] > slow.iloc[-1], fast.iloc[-2] > slow.iloc[-2]
    s_now, s_prev = fast.iloc[-1] < slow.iloc[-1], fast.iloc[-2] < slow.iloc[-2]
    if f_now and not f_prev:
        return "LONG"
    if s_now and not s_prev:
        return "SHORT"
    return None


# ---------- PERMISSION GATE: "may the tape be traded at all?" --------
# Direction-blind layer under every entry path (user 2026-10-09): it
# never picks a side, it only says whether current conditions are clean
# enough to let the strategy fire. Mirrors the dashboard's adxArr/
# erArr/chopArr loop-for-loop so live == backtest.
_WHIP = {"run": 0, "pause_until": 0.0}
_TIDE = {"info": None, "trap": False, "ts": 0.0}
_RG = {"last": ""}
# fade family = range/channel reversion entries. These trade THE CHOP
# the trend family stands down from (user's manual sigma-band playbook:
# chop day = fade both extremes, trend day = go with it).
_FADE_TAGS = frozenset({
    "vwapfade", "trap-fade", "psych-fade", "sine-buy", "sine-sell",
    "whip-buy", "whip-sell", "arrow", "candle-lap-buy", "candle-lap-sell",
})


def adx_value(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder ADX - same seeding as dashboard adxArr (sum of first
    period values, then Wilder, count-seeded DX smoothing)."""
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(h)
    if n <= period:
        return 100.0
    tr_s = pdm_s = ndm_s = dx_s = 0.0
    cnt = 0
    ph, pl, pc = h[0], l[0], c[0]
    out = 100.0
    for i in range(1, n):
        up = h[i] - ph
        dn = pl - l[i]
        pdm = up if (up > dn and up > 0) else 0.0
        ndm = dn if (dn > up and dn > 0) else 0.0
        tr = max(h[i] - l[i], abs(h[i] - pc), abs(l[i] - pc))
        ph, pl, pc = h[i], l[i], c[i]
        if i <= period:
            tr_s += tr
            pdm_s += pdm
            ndm_s += ndm
        else:
            tr_s = tr_s - tr_s / period + tr
            pdm_s = pdm_s - pdm_s / period + pdm
            ndm_s = ndm_s - ndm_s / period + ndm
        if i >= period:
            pdi = 100.0 * pdm_s / tr_s if tr_s > 0 else 0.0
            ndi = 100.0 * ndm_s / tr_s if tr_s > 0 else 0.0
            dx = (100.0 * abs(pdi - ndi) / (pdi + ndi)
                  if (pdi + ndi) > 0 else 0.0)
            cnt += 1
            if cnt <= period:
                dx_s += dx
                if cnt == period:
                    out = dx_s / period
            else:
                dx_s = dx_s - dx_s / period + dx
                out = dx_s / period
    return float(out)


def er_value(df: pd.DataFrame, n: int = 10) -> float:
    """Kaufman Efficiency Ratio - mirror of dashboard erArr (0..1)."""
    c = df["close"].to_numpy(dtype=float)
    if len(c) <= n:
        return 1.0
    path = 0.0
    for j in range(len(c) - n, len(c)):
        path += abs(c[j] - c[j - 1])
    net = abs(c[-1] - c[-1 - n])
    return (net / path) if path > 0 else 0.0


def chop_value(df: pd.DataFrame, n: int = 14) -> float:
    """Ehlers Choppiness Index - mirror of dashboard chopArr (0..100)."""
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    if len(c) <= n:
        return 100.0
    tr_sum = 0.0
    for j in range(len(c) - n, len(c)):
        p = c[j - 1]
        tr_sum += max(h[j] - l[j], abs(h[j] - p), abs(l[j] - p))
    rng = float(max(h[-n:]) - min(l[-n:]))
    if rng <= 0:
        return 100.0
    return 100.0 * float(np.log10(tr_sum / rng)) / float(np.log10(n))


def regime_ok(df, cfg) -> tuple:
    """(ok, why) for the configured detector. thr semantics match the
    dashboard: adx/er pass when value >= thr, chop passes when <= thr."""
    mode = str(cfg.get("regime_gate_mode", "off") or "off").lower()
    if mode in ("off", "none", "") or df is None or len(df) < 40:
        return True, "off"
    try:
        thr = float(cfg.get("regime_gate_thr", 0) or 0)
    except (TypeError, ValueError):
        return True, "?"
    if mode == "adx":
        v = adx_value(df, 14)
        return v >= thr, (f"ADX {v:.1f} {'>=' if v >= thr else '<'} "
                          f"{thr:g}")
    if mode == "er":
        v = er_value(df, 10)
        return v >= thr, (f"ER {v:.2f} {'>=' if v >= thr else '<'} "
                          f"{thr:g}")
    if mode == "chop":
        v = chop_value(df, 14)
        return v <= thr, (f"CHOP {v:.1f} {'<=' if v <= thr else '>'} "
                          f"{thr:g}")
    return True, mode


def regime_gate_allows(df, cfg, tag: str = "") -> bool:
    """SPLIT PERMISSION (user 2026-10-08, option "a"): one detector, two
    mirrored families. TREND entries (cross/ultra/army/macd/reload) fire
    only when permission is GRANTED (ADX>=thr) - the measured win
    (streak 4->2, PF 5.24). FADE entries fire only when it is DENIED:
    range reversion wants the sideways tape (user's sigma-band playbook).
    Whipsaw breaker binds BOTH families; gate=off trades everything;
    fomorian (manual demo) stays exempt. Logs only on state flips."""
    if tag == "fomorian":
        return True
    wb_b = int(cfg.get("whip_break_bars", 0) or 0)
    now = time.time()
    mode = str(cfg.get("regime_gate_mode", "off") or "off").lower()
    gate_on = mode not in ("off", "none", "")
    if now < _WHIP.get("pause_until", 0.0):
        st = "whip"
        why = f"{int(_WHIP['pause_until'] - now)}s left of {wb_b}min pause"
    elif not gate_on:
        st = "off"
        why = "gate off"
    else:
        ok, why = regime_ok(df, cfg)
        st = "ok" if ok else "deny"
    if st != _RG["last"]:
        if st == "ok":
            if _RG["last"]:
                print(f"[regime] PERMISSION GRANTED ({why}) - trend "
                      "family ACTIVE (cross/ultra), fade family off")
        elif st == "deny":
            print(f"[regime] PERMISSION DENIED ({why}) - trend family "
                  "off, FADE family ACTIVE (fade the chop)")
        elif st == "whip":
            print(f"[regime] WHIPSAW BREAKER: {why} - standing down "
                  "(all families)")
        elif st == "off" and _RG["last"]:
            print("[regime] gate off - all families active")
        _RG["last"] = st
    if st == "whip":
        return False
    if st == "off":
        return True
    if tag in _FADE_TAGS:
        return st == "deny"      # fades: only IN the chop
    return st == "ok"            # trend: only WITH permission


def build_entry_ctx(direction: str, reading, df_h1: pd.DataFrame, cfg: dict,
                    spread: float, hour: int, price: float) -> dict:
    """Snapshot entry conditions - what the SL learner autopsies later."""
    fast = ema(df_h1["close"], cfg["ema_fast"])
    slow = ema(df_h1["close"], cfg["ema_slow"])
    gap_pct = abs(float(fast.iloc[-1]) - float(slow.iloc[-1])) / price * 100.0
    ma_dist = reading.ma_distance_atr
    if ma_dist != ma_dist:          # NaN guard
        ma_dist = 99.0
    return SLLearner.market_ctx(
        direction=direction,
        atr_ratio=reading.atr_ratio,
        radar_score=reading.score,
        ma_dist_atr=ma_dist,
        sweep_ratio=reading.sweep_ratio,
        spread=spread,
        hour=hour,
        ema_gap_pct=gap_pct,
    )


def size_lots(balance: float, risk_amount: float, stop_dist: float, cfg: dict) -> float:
    # XAUUSD: 1 lot = 100 oz -> $1 price move = $100 per lot
    if stop_dist <= 0:
        return 0.0
    raw = risk_amount / (stop_dist * 100.0)
    lots = min(raw, cfg["base_lots"] * 10, 1.0)
    return round(max(0.01, lots), 2)


class Engine:
    """Strategy + risk + radar core shared by live and paper modes."""

    def __init__(self, start_balance: float):
        self.balance = start_balance
        self.peak = start_balance
        self.max_dd_pct = 0.0
        self.open_positions: List[dict] = []
        self.closed: List[BtTrade] = []
        self.trade_counter = 0
        self.radar = LiquidationRadar()
        self.green_streak = 0
        self.learner: Optional[SLLearner] = None
        self.relearn_every = 10
        self._relearn_bucket = 0
        self.last_filter = ""

    def risk_amount(self, cfg) -> float:
        return self.balance * (cfg["risk_pct"] / 100.0)

    def open_trade(self, direction: str, price: float, cfg, ts: str,
                   ctx: Optional[dict] = None) -> bool:
        # SL-learner gate: block entries into learned no-trade zones
        if ctx is not None and self.learner is not None:
            ok, why = self.learner.gate(ctx)
            if not ok:
                self.last_filter = why
                return False
        atr = float(rolling_atr(self._last_h1).iloc[-1])
        stop_dist = cfg["atr_stop_mult"] * atr
        if stop_dist <= 0 or price <= 0:
            return False
        lots = size_lots(self.balance, self.risk_amount(cfg), stop_dist, cfg)
        if lots <= 0:
            return False
        self.trade_counter += 1
        self.open_positions.append({
            "num": self.trade_counter,
            "direction": direction,
            "lots": lots,
            "entry": price,
            "stop": price - stop_dist if direction == "LONG" else price + stop_dist,
            "target": price + stop_dist * cfg["rr_ratio"] if direction == "LONG"
                      else price - stop_dist * cfg["rr_ratio"],
            "entry_time": ts,
            "ctx": ctx or {},
        })
        return True

    def _record_close(self, p: dict, exit_price: float, ts: str,
                      reason: str, pnl: float) -> None:
        self.balance += pnl
        self.peak = max(self.peak, self.balance)
        if self.peak > 0:
            self.max_dd_pct = max(self.max_dd_pct,
                                  (self.peak - self.balance) / self.peak * 100)
        self.closed.append(BtTrade(
            num=p["num"], direction=p["direction"], lots=p["lots"],
            entry_price=round(p["entry"], 2), exit_price=round(exit_price, 2),
            entry_time=p["entry_time"], exit_time=ts,
            pnl=round(pnl, 2), reason=reason,
            balance_after=round(self.balance, 2), is_win=pnl > 0,
        ))
        # autopsy for the learner: SL trades teach, wins calibrate
        if self.learner is not None:
            self.learner.record_completed(
                p.get("ctx", {}), pnl, reason, p["lots"], p["entry"], exit_price)
            if len(self.closed) % self.relearn_every == 0:
                report = self.learner.relearn()
                print(f"  [relearn] trades={report['trades_in_db']} "
                      f"SL rate={report['overall_sl_rate']:.0%} "
                      f"rules={report['rules_active']}")

    def check_exits(self, high: float, low: float, ts: str) -> None:
        still_open = []
        for p in self.open_positions:
            exit_price, reason = None, ""
            if p["direction"] == "LONG":
                if low <= p["stop"]:
                    exit_price, reason = p["stop"], "SL"
                elif high >= p["target"]:
                    exit_price, reason = p["target"], "TP"
            else:
                if high >= p["stop"]:
                    exit_price, reason = p["stop"], "SL"
                elif low <= p["target"]:
                    exit_price, reason = p["target"], "TP"
            if exit_price is None:
                still_open.append(p)
                continue
            move = (exit_price - p["entry"]) if p["direction"] == "LONG" \
                else (p["entry"] - exit_price)
            pnl = move * p["lots"] * 100.0
            self._record_close(p, exit_price, ts, reason, pnl)
        self.open_positions = still_open

    def flatten_all(self, price: float, ts: str) -> None:
        for p in list(self.open_positions):
            move = (price - p["entry"]) if p["direction"] == "LONG" \
                else (p["entry"] - price)
            pnl = move * p["lots"] * 100.0
            self._record_close(p, price, ts, "FLATTEN", pnl)
        self.open_positions = []

    _last_h1: Optional[pd.DataFrame] = None

    def stats(self) -> dict:
        n = len(self.closed)
        wins = [t for t in self.closed if t.pnl > 0]
        losses = [t for t in self.closed if t.pnl <= 0]
        gross_w = sum(t.pnl for t in wins)
        gross_l = abs(sum(t.pnl for t in losses))
        return {
            "trades": n,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / n * 100, 1) if n else 0.0,
            "profit_factor": round(gross_w / gross_l, 2) if gross_l else None,
            "stop_losses": sum(1 for t in self.closed if t.reason == "SL"),
            "flatten_exits": sum(1 for t in self.closed if t.reason == "FLATTEN"),
            "max_dd_pct": round(self.max_dd_pct, 1),
        }


SEEN_DEALS_PATH = os.path.join(BASE, "learner_seen.json")


def load_seen_deals() -> set:
    """Deal tickets already fed to the learner.

    Must survive restarts: without this every restart re-autopsied the
    whole 30-day deal history, duplicating rows in the learner DB
    (41 -> 51 -> 61 -> 71 trades in one morning) and skewing its rules.
    """

    try:
        with open(SEEN_DEALS_PATH) as f:
            return {int(x) for x in json.load(f)}
    except Exception:
        return set()


def save_seen_deals(seen: set) -> None:
    try:
        keep = sorted(seen)[-3000:]   # bounded: last 3000 tickets is plenty
        tmp = SEEN_DEALS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(keep, f)
        os.replace(tmp, SEEN_DEALS_PATH)
    except (OSError, TypeError, ValueError):
        pass


def save_trades_csv(trades: List[BtTrade]):
    if not trades:
        return
    with open(TRADES_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num", "direction", "lots", "entry", "exit", "entry_time",
                     "exit_time", "pnl", "reason", "balance_after"])
        for t in trades:
            w.writerow([t.num, t.direction, t.lots, t.entry_price, t.exit_price,
                         t.entry_time, t.exit_time, t.pnl, t.reason, t.balance_after])


def write_god_data(cfg, engine, radar_reading, feed, mode: str, extra: dict = None,
                   live_positions=None, live_account=None,
                   live_stats=None, cost_meter=None, ladder_meter=None,
                   tactics=None):
    last_10 = engine.closed[-10:]
    # LIVE mode must never publish the paper engine's fake money: the
    # dashboard was showing engine.balance + engine.open_positions, i.e.
    # "$162.95 / 0 positions" while a real trade was open and losing.
    if live_account is not None:
        account_block = {
            "balance": round(live_account.balance, 2),
            "equity": round(live_account.equity, 2),
            "peak": round(live_account.balance, 2),
        }
    else:
        account_block = {
            "balance": round(engine.balance, 2),
            "equity": round(engine.balance, 2),
            "peak": round(engine.peak, 2),
        }
    if live_positions is not None:
        # MetaTrader5 constants are hardcoded here: the mt5 module is
        # imported INSIDE run_live, so it is not visible in this helper.
        # POSITION_TYPE_BUY = 0, POSITION_TYPE_SELL = 1
        positions_block = [
            {"num": p.ticket,
             "direction": "BUY" if p.type == 0 else "SELL",
             "lots": p.volume,
             "entry": round(p.price_open, 2),
             "stop": round(p.sl, 2) if p.sl else None,
             "target": round(p.tp, 2) if p.tp else None,
             "floating": round(p.profit, 2),
             "entry_time": datetime.fromtimestamp(
                 p.time, timezone.utc).isoformat()}
            for p in live_positions
        ]
    else:
        positions_block = [
            {"num": p["num"], "direction": p["direction"], "lots": p["lots"],
             "entry": round(p["entry"], 2), "stop": round(p["stop"], 2),
             "target": round(p["target"], 2), "entry_time": p["entry_time"]}
            for p in engine.open_positions
        ]
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "config": cfg,
        "account": account_block,
        "radar": {
            "score": radar_reading.score,
            "mode": radar_reading.mode,
            "triggers": radar_reading.triggers,
            "ma_distance_atr": None if pd.isna(radar_reading.ma_distance_atr)
                              else round(radar_reading.ma_distance_atr, 2),
            "atr_ratio": round(radar_reading.atr_ratio, 2),
            "sweep_ratio": round(radar_reading.sweep_ratio, 2),
            "spread": round(feed.spread, 2),
        },
        "positions": positions_block,
        "recent_trades": [asdict(t) for t in reversed(last_10)],
        "events": engine.radar.events[-20:],
        "stats": live_stats if live_stats is not None else engine.stats(),
        "cost": cost_meter,
        "ladder": ladder_meter or {},
        "learning": engine.learner.summary() if engine.learner else None,
        "tactics": tactics or [],
        "cooldown": extra or {},
    }
    tmp = DATA_PATH + ".tmp"
    last_err = None
    for _ in range(5):
        try:
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, DATA_PATH)
            return
        except (PermissionError, OSError) as e:
            # transient Windows file lock (AV / dashboard read / racing process):
            # never crash the trading loop over a status-file write, but retry
            # briefly because the bridge opens god_data.json on every poll
            last_err = e
            time.sleep(0.05)
    print(f"[warn] god_data.json write skipped (file locked: {last_err})")


def build_tactics(cfg, df_h1, reading, brain_orders, engine):
    """Per-cycle tactical board: one row per flag the engine actually
    weighs before acting. Published to the dashboard so it is visible
    that every move considers the whole tape, not just the EMA cross.
    on=True = flag is live/considered this cycle; block=True = it is
    actively suppressing entries right now."""
    flags = []
    _now = time.time()
    _w = float(_WHIP.get("pause_until", 0.0) or 0.0)
    try:
        flags.append({
            "id": "radar", "group": "core", "label": "Radar",
            "on": reading.mode in ("GREEN", "AMBER"),
            "detail": f"{reading.mode} score {reading.score}"
                      + (f" \u00b7 {reading.triggers[0]}"
                         if reading.triggers else ""),
        })
        try:
            _ok, _why = regime_ok(df_h1, cfg)
        except Exception:
            _ok, _why = True, "?"
        flags.append({
            "id": "regime", "group": "core",
            "label": "Regime gate (trend family)", "on": bool(_ok),
            "detail": f"{cfg.get('regime_gate_mode')} \u00b7 {_why}",
        })
        _sd = _now < _w
        flags.append({
            "id": "whipsaw", "group": "core", "label": "Whipsaw breaker",
            "on": _sd, "block": _sd,
            "detail": (f"STAND DOWN {int(_w - _now)}s left" if _sd
                       else ("hold (trigger: "
                             f"{cfg.get('whip_break_n')} losing closes in "
                             f"{cfg.get('whip_break_bars')} bars)")),
        })
        try:
            _ha = ha_regime(df_h1)
        except Exception:
            _ha = None
        flags.append({
            "id": "ha", "group": "tape", "label": "Heiken-Ashi",
            "on": _ha is not None,
            "detail": ("?" if not _ha else
                       f"{_ha.get('mode')} "
                       f"{float(_ha.get('ratio', 0) or 0) * 100:.0f}% green"
                       f" \u00b7 run {_ha.get('run')} "
                       f" \u00b7 slope {float(_ha.get('slope', 0) or 0):+.1f}"
                       + ("" if _ha.get("fade_ok")
                          else " \u00b7 fade brains silenced")),
        })
        try:
            _sv = session_vwap_zone(df_h1)
        except Exception:
            _sv = None
        flags.append({
            "id": "vwap", "group": "tape", "label": "VWAP sigma",
            "on": _sv is not None,
            "detail": ("?" if not _sv else
                       f"sigma {float(_sv.get('sig', 0) or 0):+.2f}"
                       + (" \u00b7 fade zone"
                          if abs(float(_sv.get('sig', 0) or 0)) >= 1.28
                          else "")),
        })
        try:
            _fp = fomo_phase(df_h1)
        except Exception:
            _fp = None
        flags.append({
            "id": "fomo", "group": "tape", "label": "FOMO phase",
            "on": _fp is not None,
            "detail": ("?" if not _fp else
                       f"{_fp.get('phase')}"
                       + (f" {_fp.get('dir')}" if _fp.get("dir") else "")
                       + f" \u00b7 net {float(_fp.get('net_sig', 0) or 0):+.1f}"
                       + "\u03c3"),
        })
        _ti = _TIDE.get("info") or {}
        _tr = bool(_TIDE.get("trap"))
        flags.append({
            "id": "tide", "group": "armed", "label": "Whipsaw recon",
            "on": _tr,
            "detail": (("WHIPSAW CHOP" if _tr
                        else ("SQUEEZE" if _ti.get("squeeze")
                              else "TRENDING"))
                       + f" \u00b7 {_ti.get('flips', 0)} flips ~12h"
                       + (f" \u00b7 fail {float(_ti.get('fail_rate', 0) or 0) * 100:.0f}%"
                          if _ti else ""))
                       if _ti else "no recon yet",
        })
        flags.append({
            "id": "brains", "group": "armed", "label": "Brain orders",
            "on": bool(brain_orders),
            "detail": f"{len(brain_orders)} parked (bow-and-arrow peg orders)",
        })
        _rules = 0
        if engine is not None and engine.learner is not None:
            try:
                _rules = len((engine.learner.summary() or {}).get("rules")
                             or [])
            except Exception:
                _rules = 0
        flags.append({
            "id": "learner", "group": "armed", "label": "Learning filters",
            "on": _rules > 0,
            "detail": (f"{_rules} active no-trade rule(s)"
                       if _rules else "none active"),
        })
    except Exception as e:
        print(f"[tactics] board build error: {e}")
    return flags


def run_paper(cfg):
    print("=" * 66)
    print("  GOD-MODE BOT - PAPER MODE (synthetic feed, zero risk)")
    print("  dashboard: run bridge.py then open http://localhost:8765")
    print("=" * 66)
    feed = PaperFeed(n_seed=1100, seed=42)
    engine = Engine(cfg.get("start_balance_paper", 500.0))
    engine.learner = SLLearner()
    rc = RadarConfig(**{**asdict(RadarConfig()),
                        **{"stand_down_score": cfg["radar"]["stand_down_score"],
                           "flatten_score": cfg["radar"]["flatten_score"]}})
    engine.radar = LiquidationRadar(rc)
    print(f"learner db: {len(engine.learner.trades)} past trades, "
          f"{len(engine.learner.rules)} active no-trade filters")
    cycles = 0
    fast = cfg["cycle_seconds"]
    try:
        while True:
            cfg = load_config()          # hot-reload from dashboard
            feed.tick()
            df_h1, df_h4 = feed.h1(), feed.h4()
            engine._last_h1 = df_h1
            bid, ask = feed.bid_ask()
            price = float(df_h1["close"].iloc[-1])

            recent_stops = sum(
                1 for t in engine.closed[-10:] if t.reason == "SL")
            reading = engine.radar.assess(
                df_h1, df_h4, feed.spread, 0.30,
                recent_stop_outs=recent_stops, now=feed.now)

            engine.check_exits(float(df_h1["high"].iloc[-1]),
                               float(df_h1["low"].iloc[-1]),
                               feed.now.isoformat())

            if reading.should_flatten:
                engine.flatten_all(price, feed.now.isoformat())
                engine.green_streak = 0
            elif reading.can_open_new and len(engine.open_positions) < cfg["max_concurrent"]:
                sig = signal(df_h1, cfg)
                if sig in ("LONG", "SHORT"):
                    px = ask if sig == "LONG" else bid
                    ctx = build_entry_ctx(sig, reading, df_h1, cfg,
                                          feed.spread, feed.now.hour, price)
                    opened = engine.open_trade(sig, px, cfg,
                                                feed.now.isoformat(), ctx)
                    if opened is False:
                        print(f"  [learner] FILTERED {sig} entry -> {engine.last_filter}")
                engine.green_streak += 1
            elif reading.mode == "STAND_DOWN":
                engine.green_streak = 0

            cycles += 1
            write_god_data(cfg, engine, reading, feed, "paper",
                           {"cycle": cycles, "synthetic_time": feed.now.isoformat(),
                            "green_streak": engine.green_streak})
            print(f"[{feed.now:%Y-%m-%d %H:%M}] radar={reading.mode:<10} "
                  f"score={reading.score:<3} bal=${engine.balance:>9.2f} "
                  f"pos={len(engine.open_positions)} trades={len(engine.closed)}")
            time.sleep(max(1, fast))
    except KeyboardInterrupt:
        print("\nshutting down, saving trades.csv")
        save_trades_csv(engine.closed)
        print(f"final balance ${engine.balance:.2f} | stats: {engine.stats()}")


def evaluate_gates(cfg: dict, df: "pd.DataFrame") -> tuple:
    """ENTRY QUALITY GATES - the 'fewer, better trades' layer.

    Discovered by sweep on real MT5 bars (relearn2.js): requiring the
    signal candle to be an actual N-bar breakout, plus live volatility,
    cut trades 131 -> 21 while lifting win rate 24% -> 57% and PF
    1.61 -> 4.89. Verified as a plateau (45/45 neighbouring
    parameter sets profitable) and robust to a 4x worse spread.

    Returns (ok: bool, reasons: list[str]).
    """
    g = cfg.get("gates") or {}
    reasons = []

    n = int(g.get("breakout_n", 0) or 0)
    if n > 0:
        if len(df) < n + 2:
            reasons.append("warming up")
        else:
            # closed bars only; the reference window EXCLUDES the signal bar
            prev_high = float(df["high"].iloc[-(n + 1):-1].max())
            prev_low = float(df["low"].iloc[-(n + 1):-1].min())
            close = float(df["close"].iloc[-1])
            fast = ema(df["close"], cfg["ema_fast"])
            slow = ema(df["close"], cfg["ema_slow"])
            if len(fast) < 2 or pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]):
                reasons.append("warming up")
            else:
                above = fast.iloc[-1] > slow.iloc[-1]
                below = fast.iloc[-1] < slow.iloc[-1]
                if above and close <= prev_high:
                    reasons.append(f"no breakout (close {close:.2f} <= {n}-bar high {prev_high:.2f})")
                if below and close >= prev_low:
                    reasons.append(f"no breakdown (close {close:.2f} >= {n}-bar low {prev_low:.2f})")

    if g.get("atr_expansion", False):
        p = int(g.get("atr_expansion_period", 50) or 50)
        atr = rolling_atr(df, cfg["atr_period"])
        if len(atr) < p + 2 or pd.isna(atr.iloc[-1]):
            reasons.append("warming up")
        else:
            a_now = float(atr.iloc[-1])
            a_avg = float(atr.iloc[-p:].mean())
            if a_now <= a_avg:
                reasons.append(f"volatility flat (ATR {a_now:.2f} <= avg {a_avg:.2f})")

    return (len(reasons) == 0), reasons


TRAP_CFG = {
    # validated on real 1-min bars (combo dual-brain hit +$100 in 17/19
    # 24h missions, median 4.4h): fade CONFIRMED whipsaw sweeps
    "look": 180,          # 3h range window
    "sweep": 0.1,         # must knife the edge by this x ATR
    "slbuf": 0.4,         # SL sits this x ATR beyond the sweep extreme
    "tpfrac": 0.9,        # TP ~90% of the way to the opposite edge
    "min_rg": 3,          # range must be >= 3x ATR to be fade-able
    "time_stop_min": 45,  # dead fade positions exit after this many minutes
}


def trap_signal(df: "pd.DataFrame") -> Optional[tuple]:
    """BRAIN 2: whipsaw trap-fade. The trap is CONFIRMED on the closed
    bar: a knife through the 3h floor that CLOSES back inside = bull
    trap -> fade it long (bear trap mirrored). Enters at the snapback
    close (limit-grade price, no chasing), SL just beyond the sweep
    extreme, TP most of the way to the opposite edge.
    Returns (dir, sl, tp, why) or None."""
    look = TRAP_CFG["look"]
    if len(df) < look + 2:
        return None
    atr_s = rolling_atr(df)
    if atr_s is None or pd.isna(atr_s.iloc[-1]):
        return None
    atr = float(atr_s.iloc[-1])
    if atr <= 0:
        return None
    RL = float(df["low"].iloc[-(look + 1):-1].min())
    RH = float(df["high"].iloc[-(look + 1):-1].max())
    rng = RH - RL
    if rng < TRAP_CFG["min_rg"] * atr:
        return None
    sweep = TRAP_CFG["sweep"] * atr
    bar = df.iloc[-1]
    close = float(bar["close"])
    if float(bar["low"]) < RL - sweep and close > RL:
        sl = min(float(bar["low"]), RL) - TRAP_CFG["slbuf"] * atr
        tp = RL + TRAP_CFG["tpfrac"] * rng
        if sl < close:
            return ("LONG", sl, tp,
                    "bull-trap: dipped below 3h floor, closed back inside")
    if float(bar["high"]) > RH + sweep and close < RH:
        sl = max(float(bar["high"]), RH) + TRAP_CFG["slbuf"] * atr
        tp = RH - TRAP_CFG["tpfrac"] * rng
        if sl > close:
            return ("SHORT", sl, tp,
                    "bear-trap: spiked above 3h ceiling, closed back inside")
    return None


def triple_rejection(df, av3, lookback=30):
    """~triple bottom that REJECTS on the 1-min chart: three wobbly
    (non-linear) low pivots in one band, then price lifts off - the
    user's gold-medalist chop read: this tends to pump violently.
    Returns ("LONG"|"SHORT" , why, zone_level) or None."""
    if df is None or len(df) < 24 or not av3 or av3 <= 0:
        return None
    win = df.iloc[-lookback:]
    lows = win["low"].tolist()
    highs = win["high"].tolist()
    c = float(df["close"].iloc[-1])

    def bands(vals, hi_side):
        zone = max(vals) if hi_side else min(vals)
        band = 0.55 * av3
        piv = []
        n = len(vals)
        for i in range(2, n - 2):
            v = vals[i]
            if hi_side:
                near = v >= zone - band
                pivot = (v >= vals[i - 1] and v >= vals[i + 1]
                         and v >= vals[i - 2] and v >= vals[i + 2])
            else:
                near = v <= zone + band
                pivot = (v <= vals[i - 1] and v <= vals[i + 1]
                         and v <= vals[i - 2] and v <= vals[i + 2])
            if near and pivot:
                piv.append(i)
        merged = []
        for i in piv:
            if merged and i - merged[-1] <= 2:
                better = vals[i] > vals[merged[-1]] if hi_side \
                    else vals[i] < vals[merged[-1]]
                if better:
                    merged[-1] = i
                continue
            merged.append(i)
        return zone, merged

    zone_b, m_b = bands(lows, hi_side=False)
    if (len(m_b) >= 3 and zone_b + 0.30 * av3 < c < zone_b + 2.5 * av3
            and min(lows[-2:]) > zone_b + 0.55 * av3):
        return ("LONG", f"triple-bottom {zone_b:.2f}", zone_b)
    zone_t, m_t = bands(highs, hi_side=True)
    if (len(m_t) >= 3 and zone_t - 2.5 * av3 < c < zone_t - 0.30 * av3
            and max(highs[-2:]) < zone_t - 0.55 * av3):
        return ("SHORT", f"triple-top {zone_t:.2f}", zone_t)
    return None


def session_vwap_zone(df: "pd.DataFrame") -> Optional[dict]:
    """VWAP BAND V2 gauge — the user trades this BY HAND: session-
    anchored VWAP (resets at the UTC day roll) + band ladder at
    1 / 1.28 / 2.01 sigma. Narration only for now; a fade-brain can be
    grafted on later without rebuilding the math."""
    if df is None or len(df) < 8:
        return None
    day0 = int(df["time"].iloc[-1] // 86400 * 86400)
    sub = df["close"][df["time"] >= day0]
    if len(sub) < 5:
        return None
    vwap = float(sub.mean())
    sd = float(sub.std() or 1.0)
    if sd <= 0.05:
        return None
    price = float(df["close"].iloc[-1])
    sig = (price - vwap) / sd
    return {"sig": sig, "vwap": vwap, "sd": sd}


ROUND_PSYCH = {"grid": 25.0, "sweep": 0.1, "slbuf": 0.4, "pad": 0.3}

SINE = {
    # validated on real M1 bars (combo burst+sine: 94% of 24h missions,
    # dd $162-195, median 1.7-2.2h; standalone sniper: 81-87% WR)
    "W": 25,          # detection window
    "amp_cv": 0.45,   # amplitude coefficient-of-variation max
    "sp_cv": 0.55,    # spacing CV max (50-candle study 10-08: amp_cv med
                      # 0.27 vs gate 0.45, sp_cv med 0.52 vs old gate 0.40;
                      # ALL 21 windows failed spacing only -> 0.55 arms 16/21,
                      # rhythm stays gated, amplitude untouched)
    "proj_off": 0.05, # limit sits this x ATR INSIDE the pivot
    "tp_frac": 0.75,  # TP this far into the opposite pivot
    "sl_pad": 0.4,    # disaster stop this x ATR beyond the pivot
}


HA_REGIME = {
    # the "roasted" lesson (16:58: six stacked buys stopped in one burst,
    # 17:02 two sells, 17:35 seven small buys): the brains traded AGAINST
    # a dominant Heiken-Ashi color regime. Learn the last 200 candles:
    # color ratio + current run decide CHOP vs TREND; fade-family brains
    # fire ONLY inside CHOP; bursts only WITH the trend.
    "lookback": 200,
    "dom": 0.65,
}


def ha_regime(df: "pd.DataFrame") -> Optional[dict]:
    """Heiken-Ashi regime over the last `lookback` 1-min candles."""
    if df is None or len(df) < HA_REGIME["lookback"] + 12:
        return None
    atr_s = rolling_atr(df)
    if atr_s is None or pd.isna(atr_s.iloc[-1]) \
            or float(atr_s.iloc[-1]) <= 0:
        return None
    av = float(atr_s.iloc[-1])
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty(len(c))
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(c)):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    j = len(c) - 1
    W = HA_REGIME["lookback"]
    greens = int(((ha_c[j - W + 1:j + 1] - ha_o[j - W + 1:j + 1]) > 0).sum())
    ratio = greens / W
    run = 0
    for k in range(j, j - 10, -1):
        up = ha_c[k] > ha_o[k]
        if run == 0:
            run = 1 if up else -1
        elif (run > 0) == up:
            run += 1 if up else -1
        else:
            break
    slope = (ha_c[j] - ha_c[j - 10]) / av
    if ratio >= HA_REGIME["dom"]:
        mode = "GREEN-TREND"
    elif ratio <= (1.0 - HA_REGIME["dom"]):
        mode = "RED-TREND"
    else:
        mode = "CHOP"
    return {"ratio": round(ratio, 3), "run": int(run),
            "slope": round(slope, 2), "mode": mode,
            "fade_ok": mode == "CHOP",
            "with_dir": ("LONG" if mode == "GREEN-TREND"
                         else "SHORT" if mode == "RED-TREND" else None)}


ARMED_TTL = {
    # "wait for nicer entry, then fire MARKET" - the patience window per
    # brain (seconds the armed order stays live waiting for the tape)
    "trap-fade": 300, "psych-fade": 300, "army-ride": 300, "arrow": 300,
    "sine-buy": 600, "sine-sell": 600, "whip-buy": 600, "whip-sell": 600,
    "candle-lap-buy": 45, "candle-lap-sell": 45,
}


ARROW = {
    # double-arrow learner brain — backtest verdict (arrow2.js, real bars):
    # every-bar both-side = -$700..-$2,800/day (churn-death); the ONLY
    # +EV slice: SEDUCE phase -> CONTRA side (65-70% TP rate, ~+$1 avg).
    # 0.04 lots, TP +$4 = $1.00 gold move; 1.0 ATR stop; ha-flip kill;
    # relearn every 2 candles (user). exit engine marks it market-out.
    "phase": "SEDUCE",
    "side": "contra",
    "lots": 0.04,
    "tp_dist": 1.0,
    "sl_atr": 1.0,
    "relearn_every": 1,   # 1-candle mode: arrow considers every candle
}
ARROW_CSV_PATH = os.path.join(BASE, "arrow_log.csv")


def write_arrow_row(row: dict) -> None:
    """one row per arrow FILL: the intended-vs-contrarian learning book."""
    try:
        new = not os.path.exists(ARROW_CSV_PATH)
        with open(ARROW_CSV_PATH, "a", newline="") as fcsv:
            w = csv.writer(fcsv)
            if new:
                w.writerow(["time", "ticket", "phase_dir", "side", "lots",
                            "entry", "sl", "tp", "run_dir"])
            w.writerow([datetime.now().isoformat(timespec="seconds"),
                        row.get("ticket"), row.get("phase_dir"),
                        row.get("side"), row.get("lots"), row.get("entry"),
                        row.get("sl"), row.get("tp"), row.get("run_dir")])
    except OSError:
        pass


WHIPSAW = {
    # 10+10-candle whipsaw brain (whip2.js on real bars: 100% mission
    # hit, median 0.1h, 94% WR, dd $69): the last 10 candles flip
    # repeatedly AND the prior 10 did too, amplitude honest -> fade
    # both wall of the 20-bar channel at the extremes (puffer pegs).
    "flips_a": 3,
    "flips_b": 2,
    "amp_hold": 0.35,
    "min_rng_atr": 1.5,
    "tie_in_atr": 0.5,   # 1-candle mode: walls sit 0.5 ATR inside the
                         # channel so the tape reaches them THIS candle
    "tp_atr": 1.2,
    "sl_atr": 1.0,
    "relearn_every": 1,
    "side_cap": 2,
    "lots_cap": 0.20,   # panic mode (user: 40-birthday sprint): walls 0.15->0.20
}


def whip_channel(df: "pd.DataFrame") -> Optional[dict]:
    """10+10 whipsaw confirm + channel (trough/crest) levels."""
    W = 10
    if df is None or len(df) < W * 2 + 8:
        return None
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values

    def raz(j):
        f = 0
        hi = -1e9
        lo = 1e9
        for k in range(j - W + 1, j):
            hi = max(hi, float(h[k]))
            lo = min(lo, float(l[k]))
            if (c[k] > c[k - 1]) != (c[k + 1] > c[k]):
                f += 1
        return f, hi, lo

    fa, ha, la = raz(len(df) - 1)
    fb, hb, lb = raz(len(df) - 1 - W)
    if fa < WHIPSAW["flips_a"] or fb < WHIPSAW["flips_b"]:
        return None
    hia = max(ha, hb); lia = min(la, lb)
    rng_a = (ha - la); rng_b = (hb - lb)
    if max(rng_a, rng_b) < WHIPSAW["min_rng_atr"] * 0.3:
        return None
    hold = min(rng_a, rng_b) / max(0.3, max(rng_a, rng_b))
    if hold < WHIPSAW["amp_hold"]:
        return None
    return {"trough": lia, "crest": hia}


def macd_trix(df: "pd.DataFrame", fast=5, slow=10, sig=3,
              trx=9, trx_sig=3) -> Optional[dict]:
    """AGGRESSIVE MACD + TRIX (user spec 2026-10-08): quick-setting MACD
    (5,10,3) gives momentum sign/acceleration, TRIX(9) with signal(3) gives
    trend-confirm. Returns bias (LONG/SHORT/None), fresh (a cross this
    bar), hist/accel/trix values for instrumentation."""
    if df is None or len(df) < 60:
        return None
    c = df["close"].astype(float)
    macd = c.ewm(span=fast, adjust=False).mean() \
        - c.ewm(span=slow, adjust=False).mean()
    m_sig = macd.ewm(span=sig, adjust=False).mean()
    hist = macd - m_sig
    e1 = c.ewm(span=trx, adjust=False).mean()
    e2 = e1.ewm(span=trx, adjust=False).mean()
    e3 = e2.ewm(span=trx, adjust=False).mean()
    tline = e3.pct_change() * 100.0
    tsig = tline.ewm(span=trx_sig, adjust=False).mean()
    h0 = float(hist.iloc[-1])
    h1 = float(hist.iloc[-2])
    accel = h0 - h1
    t0 = float(tline.iloc[-1])
    t1 = float(tline.iloc[-2])
    ts0 = float(tsig.iloc[-1])
    ts1 = float(tsig.iloc[-2])
    crossed = (h1 <= 0 < h0) or (h1 >= 0 > h0)
    tcross = (t1 <= ts1 < t0) or (t1 >= ts1 > t0)
    if h0 > 0 and accel >= 0 and t0 >= ts0:
        nb = "LONG"
    elif h0 < 0 and accel <= 0 and t0 <= ts0:
        nb = "SHORT"
    else:
        nb = None
    return dict(bias=nb, fresh=(nb is not None and (crossed or tcross)),
                hist=h0, accel=accel, trix=t0, tsig=ts0, h1=h1)


def regime_classify(df: "pd.DataFrame") -> Optional[dict]:
    """PRACTICE TUNING READOUT (user spec 2026-10-08): classify the LAST
    100 candles as SIDEWAYS / TRENDING UP-DOWN / BREAKOUT / CHOPPY UP-DOWN,
    PLUS a Heiken-Ashi reading of the same window. Uses the efficiency
    ratio (net move vs total range), the regression-ish slope normalized by
    ATR, and ATR expansion (100 vs 500) to spot breakouts. Returns a dict
    of regime + ha + metrics, or None if the frame is too small."""
    if df is None or len(df) < 510:
        return None
    sub = df.tail(100)
    o = sub["open"].astype(float).values
    h = sub["high"].astype(float).values
    l = sub["low"].astype(float).values
    c = sub["close"].astype(float).values
    tr = h - l
    atr_100 = float(tr.mean()) if len(tr) else 0.0
    tr500 = df["high"].tail(500).astype(float).values \
        - df["low"].tail(500).astype(float).values
    atr_500 = float(tr500.mean()) if len(tr500) else atr_100
    net = float(c[-1] - c[0])
    sumrng = float(tr.sum())
    erl = abs(net) / sumrng if sumrng > 0 else 0.0
    slope_atr = net / atr_100 if atr_100 > 0 else 0.0
    vol_frac = atr_100 / atr_500 if atr_500 > 0 else 1.0
    # breakout: a move of 2+ ATR beyond the prior 50-bar extreme with vol
    # expansion
    prior_hi = float(h[-51:-1].max()) if len(h) > 51 else float(h[:-1].max())
    prior_lo = float(l[-51:-1].min()) if len(l) > 51 else float(l[:-1].min())
    up_brk = (c[-1] > prior_hi and slope_atr >= 2.0 and vol_frac >= 1.35)
    dn_brk = (c[-1] < prior_lo and slope_atr <= -2.0 and vol_frac >= 1.35)
    if up_brk:
        regime = "BREAKOUT UP"
    elif dn_brk:
        regime = "BREAKOUT DOWN"
    elif erl >= 0.30:
        regime = "TRENDING UP" if net > 0 else "TRENDING DOWN"
    elif erl <= 0.11:
        regime = "SIDEWAYS"
    elif abs(slope_atr) >= 0.2:
        regime = "CHOPPY UP" if net > 0 else "CHOPPY DOWN"
    else:
        regime = "CHOP"
    # heiken-ashi view of the same 100 candles
    ha_o = [0.0] * len(o)
    ha_c = [0.0] * len(c)
    ha_o[0] = (o[0] + c[0]) / 2.0
    ha_c[0] = (o[0] + h[0] + l[0] + c[0]) / 4.0
    flips = 0
    streak = 1
    last_g = ha_c[0] >= ha_o[0]
    for i in range(1, len(o)):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
        ha_c[i] = (o[i] + h[i] + l[i] + c[i]) / 4.0
        g = ha_c[i] >= ha_o[i]
        if g != last_g:
            flips += 1
            streak = 1
        else:
            streak += 1
        last_g = g
    ha_net = float(ha_c[-1] - ha_c[0])
    ha_erl = abs(ha_net) / (np.abs(np.diff(ha_c)).sum() or 1.0)
    if streak >= 6:
        ha = "HA " + ("trend up" if last_g else "trend down")
    elif flips >= 30:
        ha = "HA chop"
    else:
        ha = "HA mixed"
    return dict(regime=regime, erl=round(erl, 2),
                slope_atr=round(slope_atr, 2), vol_frac=round(vol_frac, 2),
                ha=ha, ha_flips=flips, ha_streak=streak,
                ha_erl=round(ha_erl, 2),
                bias=("up" if net > 0 else "down"))


def sine_signal(df: "pd.DataFrame") -> Optional[dict]:
    """SINE WAVE brain: regular alternating pivots over the last W
    candles -> trade BOTH sides with limit orders at the projections:
    BUY LIMIT at trough+off (TP toward the crest), SELL LIMIT at
    crest-off (TP toward the trough). Wave-alive gate: the last close
    must still be inside the channel. Returns {} / entries / None."""
    W = SINE["W"]
    if df is None or len(df) < W + 4:
        return None
    atr_s = rolling_atr(df)
    if atr_s is None or pd.isna(atr_s.iloc[-1]):
        return None
    av = float(atr_s.iloc[-1])
    if av <= 0:
        return None
    h = df["high"].values
    l = df["low"].values
    n = len(df)
    piv = []
    for i in range(n - W, n - 1):
        if h[i] > h[i - 1] and h[i] > h[i + 1]:
            piv.append(("H", float(h[i]), i))
        elif l[i] < l[i - 1] and l[i] < l[i + 1]:
            piv.append(("L", float(l[i]), i))
    cut = []
    for t, p, i in piv:
        if cut and cut[-1][0] == t:
            if (t == "H" and p > cut[-1][1]) or (t == "L" and p < cut[-1][1]):
                cut[-1] = (t, p, i)
        else:
            cut.append((t, p, i))
    if len(cut) < 3:
        return None
    amps = [abs(cut[k][1] - cut[k - 1][1]) for k in range(1, len(cut))]
    spac = [cut[k][2] - cut[k - 1][2] for k in range(1, len(cut))]

    def cvs(a: list) -> float:
        m = sum(a) / len(a)
        sd = (sum((x - m) ** 2 for x in a) / len(a)) ** 0.5
        return (sd / m) if m > 0.05 else 9.0

    if cvs(amps) > SINE["amp_cv"] or cvs(spac) > SINE["sp_cv"]:
        return None
    trough = [p for t, p, i in cut if t == "L"]
    crest = [p for t, p, i in cut if t == "H"]
    if not trough or not crest:
        return None
    trough, crest = trough[-1], crest[-1]
    amp = crest - trough
    if amp < 1.5 * av:
        return None
    px = float(df["close"].iloc[-1])
    if px < trough - 0.3 * amp or px > crest + 0.3 * amp:
        return None
    out = {}
    buy_lvl = trough + SINE["proj_off"] * av
    if px > buy_lvl:
        out["buy"] = (buy_lvl, trough - SINE["sl_pad"] * av,
                      trough + SINE["tp_frac"] * amp)
    sell_lvl = crest - SINE["proj_off"] * av
    if px < sell_lvl:
        out["sell"] = (sell_lvl, crest + SINE["sl_pad"] * av,
                       crest - SINE["tp_frac"] * amp)
    return out or None


FOMO_CSV_PATH = os.path.join(BASE, "fomo_phases.csv")
FOMO_CFG = {
    # the user's gold-medalist thesis: the tape is a MOVEMENT ANIMATION -
    # accelerate -> pause -> seduce (counter-poke that traps FOMO orders;
    # the learner proved BUY|ma-far = 92% SL-rate) -> dance -> accelerate.
    # This watcher classifies the phase every bar (narration + CSV only;
    # a trap-brain can be grafted onto the phase labels later).
    "run_sig": 2.0,
    "seduce_min": 0.35,
    "pause_sig": 1.0,
    "alts_min": 3,
}


def fomo_phase(df: "pd.DataFrame") -> Optional[dict]:
    """Classify the current FOMO-cycle phase of the 1-min tape."""
    if df is None or len(df) < 36:
        return None
    atr_s = rolling_atr(df)
    if atr_s is None or pd.isna(atr_s.iloc[-1]) or float(atr_s.iloc[-1]) <= 0:
        return None
    av = float(atr_s.iloc[-1])
    clos = df["close"].values
    j = len(df) - 1
    net8 = float(clos[j] - clos[j - 8])
    net2 = float(clos[j] - clos[j - 2])
    up_run = net8 >= FOMO_CFG["run_sig"] * av
    dn_run = net8 <= -FOMO_CFG["run_sig"] * av
    direction = ""
    if up_run or dn_run:
        direction = "UP" if up_run else "DOWN"
        run_from = float(clos[j - 8])
        if up_run:
            far = float(df["high"].iloc[j - 8:j + 1].max())
            retr = (far - float(clos[j])) / max(0.1, far - run_from)
        else:
            far = float(df["low"].iloc[j - 8:j + 1].min())
            retr = (float(clos[j]) - far) / max(0.1, run_from - far)
        if retr >= FOMO_CFG["seduce_min"]:
            return {"phase": "SEDUCE", "dir": direction,
                    "depth_sig": round(retr, 2), "net_sig": round(net8 / av, 2)}
        return {"phase": "RUN", "dir": direction,
                "depth_sig": round(retr, 2), "net_sig": round(net8 / av, 2)}
    if abs(net8) < FOMO_CFG["pause_sig"] * av:
        return {"phase": "PAUSE", "dir": "", "depth_sig": 0.0,
                "net_sig": round(net8 / av, 2)}
    # alternation count in the last 8 candles -> dance
    alts = sum(1 for i in range(j - 7, j)
               if (clos[i] > clos[i - 1]) != (clos[i + 1] > clos[i]))
    if alts >= FOMO_CFG["alts_min"]:
        return {"phase": "DANCE", "dir": "", "depth_sig": round(alts / 8, 2),
                "net_sig": round(net8 / av, 2)}
    return {"phase": "DRIFT", "dir": "", "depth_sig": 0.0,
            "net_sig": round(net8 / av, 2)}


def append_fomo(event: dict) -> None:
    """Phase log - the read-only memory the future trap-brain trains on."""
    try:
        new = not os.path.exists(FOMO_CSV_PATH)
        with open(FOMO_CSV_PATH, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "phase", "dir", "depth_sig", "net_sig"])
            w.writerow([datetime.now().isoformat(timespec="seconds"),
                        event.get("phase"), event.get("dir"),
                        event.get("depth_sig"), event.get("net_sig")])
    except OSError:
        pass


def recon_tide(df: "pd.DataFrame", cfg: dict) -> dict:
    """The practice run before the fight: read the sea state on the
    most recent ~12h of bars - how many EMA whipsaws, how many 20-bar
    breakouts failed within 10 bars (traps), how compressed the range
    is. Prints one [recon] line and decides whether the trap-fade
    brain (BRAIN 2) is armed."""
    fast = ema(df["close"], cfg["ema_fast"])
    slow = ema(df["close"], cfg["ema_slow"])
    span = min(len(df) - 2, 720)          # last ~12h on the 1-min TF
    side = (fast.iloc[-span:] > slow.iloc[-span:]).astype(int)
    flips = int(side.diff().abs().sum())
    n_bo = 0
    n_fail = 0
    closes = df["close"]
    highs = df["high"]
    lows = df["low"]
    j_end = len(df) - 12
    for j in range(max(21, len(df) - span - 1), j_end):
        ph = float(highs.iloc[j - 20:j].max())
        pl = float(lows.iloc[j - 20:j].min())
        c = float(closes.iloc[j])
        if c > ph:
            n_bo += 1
            if float(closes.iloc[j + 1:j + 11].min()) < ph:
                n_fail += 1
        elif c < pl:
            n_bo += 1
            if float(closes.iloc[j + 1:j + 11].max()) > pl:
                n_fail += 1
    rate = (n_fail / n_bo) if n_bo else 0.0
    r2 = float(highs.iloc[-120:].max() - lows.iloc[-120:].min())
    atr_s = rolling_atr(df)
    atr = float(atr_s.iloc[-1]) if atr_s is not None and not pd.isna(
        atr_s.iloc[-1]) else 1.0
    trend_run = 0
    if len(df) > 30:
        for k in range(len(df) - 1, 5, -1):
            if (float(highs.iloc[k]) < float(highs.iloc[k - 2])
                    and float(lows.iloc[k]) < float(lows.iloc[k - 2])):
                trend_run += 1
            else:
                break
    trap = flips >= 4 or rate >= 0.35
    squeeze = r2 < 12 * atr
    tide = "WHIPSAW CHOP" if trap else ("SQUEEZE" if squeeze else "TRENDING")
    hours = span / 60.0
    print(f"[recon] tide: {tide} | EMA whipsaws {flips} in last {hours:.0f}h "
          f"| failed breakouts {n_fail}/{n_bo} ({rate * 100:.0f}%) "
          f"| 2h range {r2:.2f} = {r2 / atr:.1f}x ATR "
          f"| LL-run {trend_run} bars")
    return {"trap": trap, "squeeze": squeeze, "flips": flips,
            "fail_rate": rate}


def heiken_ashi(df: "pd.DataFrame") -> "pd.DataFrame":
    """Standard HA transform: close = o+h+l+c /4, open = prev open/close
    midpoint. The army gauge reads the growth of these candles."""
    o = df["open"]
    c = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_o = [(float(o.iloc[0]) + float(c.iloc[0])) / 2.0]
    for j in range(1, len(df)):
        ha_o.append((ha_o[-1] + float(c.iloc[j - 1])) / 2.0)
    ha_o = pd.Series(ha_o, index=df.index)
    ha_h = pd.concat([df["high"], ha_o, c], axis=1).max(axis=1)
    ha_l = pd.concat([df["low"], ha_o, c], axis=1).min(axis=1)
    return pd.DataFrame({"open": ha_o, "high": ha_h, "low": ha_l,
                         "close": c})


def htf_bias(df30: "pd.DataFrame", df_h4: "pd.DataFrame") -> dict:
    """HTF SIDE-LOCK (user spec 2026-10-08 13:2x): 30m Heiken-Ashi red +
    4h HA red (or sloping down = "turning red soon") -> SHORT lock; both
    green -> LONG lock; anything else -> None (both sides open)."""
    if df30 is None or df_h4 is None:
        return {"bias": None, "red30": None, "red4h": None, "slope4h": 0.0}
    h30c = df30["close"].astype(float)
    h30o = df30["open"].astype(float)
    h4c = df_h4["close"].astype(float)
    h4o = df_h4["open"].astype(float)
    red30 = bool(h30c.iloc[-1] < h30o.iloc[-1])
    red4h = bool(h4c.iloc[-1] < h4o.iloc[-1])
    slope4h = float(h4c.iloc[-1] - h4c.iloc[-min(len(h4c), 9)])
    if red30 and (red4h or slope4h < 0):
        nb = "SHORT"
    elif (not red30) and ((not red4h) or slope4h >= 0):
        nb = "LONG"
    else:
        nb = None
    return {"bias": nb, "red30": red30, "red4h": red4h,
            "slope4h": round(slope4h, 2)}


def army_gauge(df_ha: "pd.DataFrame") -> dict:
    """Green vs red army: consecutive HA candle count (signed) + body
    ACCELERATION (are the candles still growing?). consec>0 = green
    army, <0 = red army; accel is normalized (about 0 = steady)."""
    up = (df_ha["close"] > df_ha["open"]).astype(int).tolist()
    consec = 0
    for x in reversed(up):
        if consec == 0:
            consec = 1 if x == 1 else -1
        else:
            more = (consec > 0 and x == 1) or (consec < 0 and x == 0)
            if more:
                consec += 1 if x == 1 else -1
            else:
                break
    body = (df_ha["close"] - df_ha["open"]).abs()
    last3 = float(body.iloc[-3:].mean())
    prev3 = float(body.iloc[-6:-3].mean()) if len(body) >= 6 else 0.0
    accel = 0.0
    if prev3 > 0:
        accel = (2 * last3 - prev3) / prev3
    return {"consec": int(consec), "accel": round(accel, 3),
            "green": bool(up[-1])}


ROUND_PSYCH = {"grid": 25.0, "sweep": 0.1, "slbuf": 0.4, "pad": 0.3}

BOUNCE_GUARD = {
    # learned from live: shorts peaked at ~$50 floating, one big reversal
    # candle gave back ~$57. Take the money on: (a) a single-candle
    # give-back of >=60% of the peak (hard), or (b) >=35% of the peak
    # with 2+ consecutive declines (the bounce is accelerating).
    "frac_soft": 0.28,
    "frac_hard": 0.55,
    "min_peak": 8.0,       # only positions that had a real floating gain
    "dip_min": 2,
}

EXIT_ENGINE = {
    # CHOP SCALP EXITS (2026-10-07): 1-2 minute life on XAUUSD chop.
    # History lesson (34 closed trades): 4 trades reached +0.6R+ and
    # donated it back waiting for far TPs/serves (worst: +$11.40 open,
    # stopped at -$21.54). The money is made in the first 1-2 candles.
    "on": True,
    "min_peak": 2.0,        # engine wakes once floating has a real peak ($)
    "lock_atr_off": 0.12,   # TP parked beyond the recent extreme by this x ATR
    "retrace_pct": 30,      # lost 30% of the peak -> take the money NOW
    "ha_flip": True,        # last CLOSED heiken-ashi candle against us -> out
    "hold_max_s": 135,      # stale: >2.2 min in and weak -> flat (1-2 candle life)
    "stale_frac": 0.25,     # stale only if profit < this fraction of peak
}

LIMIT_CFG = {
    # BOW-AND-ARROW: fade-family entries park LIMIT orders slightly away
    # from the market (peg x ATR) for better fills. The peg dies after
    # ttl_s (in/out within one candle) or when the next candle opens.
    "peg_atr": 0.15,
    "ttl_s": 45,
    "repeg_atr": 0.6,
    "use_limits": True,    # MIXED ROUTING (best-of-evidence): limits for fade walls,
    "market_tags": {"army-ride", "re-ride", "reload"},   # momentum = market chase
    # PUFFER-FISH anchor (user): park fade limits at the MID of the
    # forming candle - the violent huff-and-puff puff passes through
    # the middle on its way to the extremes, then snaps back; the mid
    # fill rides the reversal from inside the fight.
    "mid_tags": {"trap-fade", "psych-fade", "reload", "re-ride",
                 "army-ride", "sine-buy", "sine-sell"},
    "tag_peg": {"trap-fade": 0.15, "psych-fade": 0.15,
                "army-ride": 0.25, "reload": 0.15, "re-ride": 0.15,
                "vwapfade": 0.10, "sine-buy": 0.05, "sine-sell": 0.05,
                "arrow": 0.05, "whip-buy": 0.0, "whip-sell": 0.0},
}
BOUNCE_LOG = os.path.join(BASE, "bounce_log.csv")

# LIVE METER (user 2026-10-08): measured spread + realized slippage per
# market fill, and the routing decision made by the cost-aware gate.
# "slip" holds EWMA US$ per lot by side (ask/bid), "last" the most recent
# measurement, market/rises count market fills vs spread->limit falls.
LIVE_METER = {"slip": {}, "last": {}, "market": 0, "rises": 0}

# SPREAD-CAPTURE LADDER (user 2026-10-08): the main cross signal parks
# 3 staggered LIMIT orders on the spread-favorable side instead of ONE
# market order. LONG layers sit on/under the BID, SHORT layers on/over
# the ASK - every fill GRABS the spread instead of paying it, and deeper
# layers only fill as price comes to us (strictly better fills, never
# worse). Unfilled layers expire and are cancelled by the ladder sweep.
LADDER_CFG_DEFAULTS = {
    "on": True,          # main cross entry => layered limit, no market
    "layers": 3,
    "weights": [0.5, 0.3, 0.2],   # near layer takes the biggest slice
    "step_atr": 0.20,    # each deeper layer sits this x ATR further out
    "ttl_s": 300,        # unfilled layers live 5 min, then cancel cleanly
    # user 2026-10-12: hunt-spaced layers - 2nd near the 1st stop, 3rd AT
    # it (x stop_dist). When set with >= layers entries, replaces step_atr.
    "offsets": None,
}
LADDER_CFG = dict(LADDER_CFG_DEFAULTS)
LADDER_STATE = {"active": {}, "fills": 0, "cancels": 0}
VANTAGE_CFG_DEFAULTS = {
    "on": True,          # delayed 4th bullet near the stop shelf
    "delay_s": 120,      # let the pullback develop before parking it
    "price_atr": 0.8,    # reserve sits this x ATR adverse of signal ref
    "size_frac": 0.3,    # of the signal's total lots (one light bullet)
    "ttl_s": 300,        # unfilled reserve cancels like a layer
    "scratch_trig_atr": 0.6,   # pump toward stops that scratches layers
    "scratch_within_s": 180,   # ...only this soon after the signal
}
VANTAGE_CFG = dict(VANTAGE_CFG_DEFAULTS)
VANTAGE_STATE = {"scheduled": None, "order": None, "scratched_sig": None}


def _streak_best(seq, sign):
    """longest run of same-sign pnl rows in [(time, pnl), ...]; 0 counts as loss."""
    best = cur = 0
    for _, pf in seq:
        if (pf > 0) == (sign > 0) and pf != 0:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best

RELOAD_CFG = {
    # DOUBLE TRADE after a bounce-guard exit: ride the bounce (reverse
    # of the closed trade), exit it when the ACCELERATION dies (the
    # small pause), then re-ride the original side to its destination TP.
    "sl_atr": 0.8,        # bounce-side stop: 0.8 x ATR back from fills
    "accel_stall": 0.05,  # per-5s profit growth under this = stall
    "stall_steps": 2,     # 2 consecutive stalls -> bounce is dying
    "min_profit": 0.4,    # only ride a bounce that has real profit
    "dead_min": 1200,     # bounce ride dead after 20 min (no re-ride)
    "reride_sl_atr": 0.35,
}


def bounce_close_reload_plan(closed_ctx, p_type, closed_px, price_open):
    """After a bounce-guard exit (not from a reload chain), plan the
    bounce-side ride immediately. mt5 constants are local to run_live:
    POSITION_TYPE_SELL == 1, BUY == 0 (hardcoded, per house rule)."""
    if closed_ctx and closed_ctx.get("reload"):
        return None   # chains of depth 1 only: no ping-pong churn
    return {"dir": "LONG" if p_type == 1 else "SHORT",
            "px_close": closed_px, "src_entry": price_open, "t_due": time.time()}


def log_bounce(event: dict) -> None:
    """Every bounce-guard exit is the surface the learning reads; the
    CSV stays small and later tunes BOUNCE_GUARD fractions."""
    try:
        new = not os.path.exists(BOUNCE_LOG)
        with open(BOUNCE_LOG, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "ticket", "peak", "floating",
                            "give", "pct", "dips", "closed", "retcode"])
            w.writerow([datetime.now().isoformat(timespec="seconds"),
                        event.get("ticket"), round(event.get("peak", 0), 2),
                        round(event.get("floating", 0), 2),
                        round(event.get("give", 0), 2),
                        round(event.get("pct", 0), 1),
                        event.get("dips"), event.get("closed"),
                        event.get("retcode")])
    except OSError:
        pass


def load_profit_baseline(current_balance: float) -> float:
    """Profit-guard baseline. Persists across crash-restarts; a fresh
    cycle starts when profit_guard.json is deleted (RESUME-TRADING.bat)."""
    base = None
    try:
        with open(PROFIT_GUARD_PATH, "r") as f:
            base = float(json.load(f).get("baseline"))
    except Exception:
        base = None
    if base is None or current_balance < base:
        base = current_balance
        try:
            with open(PROFIT_GUARD_PATH, "w") as f:
                json.dump({"baseline": round(base, 2)}, f)
        except OSError:
            pass
    return base


MAIN_WALLET_PATH = os.path.join(BASE, "main_wallet.json")


def load_wallet() -> dict:
    """Persistent $100-block vault. Survives restarts; `today` resets by
    calendar date so the dashboard can count blocks banked in 1 day."""
    try:
        with open(MAIN_WALLET_PATH, "r") as f:
            w = json.load(f)
        if not isinstance(w, dict):
            raise ValueError("not a dict")
    except Exception:
        w = {"total": 0.0, "day": "", "today": 0, "blocks": []}
    today = datetime.now().strftime("%Y-%m-%d")
    if w.get("day") != today:
        w["day"] = today
        w["today"] = 0
    if not isinstance(w.get("blocks"), list):
        w["blocks"] = []
    return w


def bank_block(cfg: dict, equity: float, balance: float) -> dict:
    """Record one banked profit block (default $100) into the vault."""
    amount = float(cfg.get("profit_target_usd", 100) or 100)
    w = load_wallet()
    w["total"] = round(float(w.get("total", 0) or 0) + amount, 2)
    w["today"] = int(w.get("today", 0)) + 1
    w["blocks"].append({
        "t": datetime.now().isoformat(timespec="seconds"),
        "amount": amount,
        "equity_at_bank": round(equity, 2),
    })
    w["blocks"] = w["blocks"][-50:]
    w["balance"] = round(balance, 2)
    tmp = MAIN_WALLET_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(w, f, indent=2)
    os.replace(tmp, MAIN_WALLET_PATH)
    return w


def resolve_tf(cfg: dict, mt5) -> int:
    """Config timeframe string -> MT5 constant (page profile names)."""
    tf_map = {"1min": mt5.TIMEFRAME_M1, "2min": mt5.TIMEFRAME_M2,
              "5min": mt5.TIMEFRAME_M5, "15min": mt5.TIMEFRAME_M15,
              "30min": mt5.TIMEFRAME_M30, "1hour": mt5.TIMEFRAME_H1,
              "4hour": mt5.TIMEFRAME_H4, "daily": mt5.TIMEFRAME_D1}
    return tf_map.get(str(cfg.get("timeframe", "1hour")).lower(),
                      mt5.TIMEFRAME_H1)


TF_SECONDS = {"1min": 60, "2min": 120, "5min": 300, "15min": 900,
              "30min": 1800, "1hour": 3600, "4hour": 14400,
              "daily": 86400}


def narrate(cfg, reading, positions, live_ctx, df_h1, acc_now) -> str:
    """One human-readable line: what the bot is 'thinking' this cycle."""
    tf_name = str(cfg.get("timeframe", "1hour"))
    tf_sec = TF_SECONDS.get(tf_name, 3600)
    close_in = tf_sec - (int(time.time()) % tf_sec)
    if reading.should_flatten:
        return (f"radar RED (score {reading.score}) - flattening "
                f"everything, no questions asked")
    if positions:
        flot = (acc_now.equity - acc_now.balance) if acc_now else 0.0
        opened = ", ".join(
            f"{(live_ctx.get(p.ticket) or {}).get('direction', '?')} "
            f"{p.volume}@{p.price_open:.2f}" for p in positions)
        guard = ("defending wick-stops and TP"
                 if cfg.get("wick_proof_stop")
                 else "broker SL/TP working")
        return (f"holding {len(positions)} ({opened}) floating "
                f"${flot:+.2f} - {guard}")
    if reading.mode == "STAND_DOWN":
        tg = ", ".join(reading.triggers[:2]) or "score elevated"
        return (f"radar amber (score {reading.score}: {tg}) - standing "
                f"down, being patient")
    if not reading.can_open_new:
        return "cooldown active - resting after recent stops, patient"
    if len(positions) >= cfg["max_concurrent"]:
        return f"fully loaded ({len(positions)}/{cfg['max_concurrent']}) - managing exits first"
    ef_l = float(ema(df_h1["close"], cfg["ema_fast"]).iloc[-1])
    es_l = float(ema(df_h1["close"], cfg["ema_slow"]).iloc[-1])
    gap = ef_l - es_l
    side = "above" if gap >= 0 else "below"
    return (f"radar GREEN - watching {tf_name}: fast EMA {side} slow by "
            f"${abs(gap):.2f}; bar closes in {close_in}s - waiting for "
            f"the cross")


def run_command(cfg, mt5, fill_mode: int, magic: int, cmd: dict, live_ctx: dict) -> dict:
    """Execute one cockpit command. Returns result dict for cmd_result.json."""
    action = str(cmd.get("action", "")).lower()
    out = {"id": cmd.get("id"), "action": action,
           "ts": datetime.now(timezone.utc).isoformat()}
    sym_name = cfg["symbol"]

    if action in ("buy", "sell"):
        sym = mt5.symbol_info(sym_name)
        if sym is None:
            return {**out, "ok": False, "detail": f"symbol {sym_name} unavailable"}
        try:
            lots = float(cmd.get("lots") or 0.01)
        except (TypeError, ValueError):
            lots = 0.01
        step = sym.volume_step or 0.01
        lots = max(sym.volume_min, min(lots, sym.volume_max))
        lots = round(round(lots / step) * step, 2)
        # ATR-protective stops, same math as the strategy (on its timeframe)
        bars = mt5.copy_rates_from_pos(sym_name, resolve_tf(cfg, mt5), 0, 300)
        atr = None
        if bars is not None and len(bars) > cfg["atr_period"] + 2:
            atr = float(rolling_atr(pd.DataFrame(bars).iloc[:-1]).iloc[-1])
        stop_dist = (cfg["atr_stop_mult"] * atr) if atr else 6.0
        ok, detail = False, "no tick"
        for attempt in range(1, 4):
            tick = mt5.symbol_info_tick(sym_name)
            if tick is None:
                break
            if action == "buy":
                px, otype = tick.ask, mt5.ORDER_TYPE_BUY
                soft_sl = px - stop_dist
                tp = px + stop_dist * cfg["rr_ratio"]
            else:
                px, otype = tick.bid, mt5.ORDER_TYPE_SELL
                soft_sl = px + stop_dist
                tp = px - stop_dist * cfg["rr_ratio"]
            if cfg.get("wick_proof_stop"):
                dm = float(cfg.get("disaster_stop_mult", 1.5))
                sl = (px - stop_dist * dm) if action == "buy" \
                    else (px + stop_dist * dm)
            else:
                sl = soft_sl
            req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": sym_name,
                   "volume": lots, "type": otype, "price": px, "deviation": 30,
                   "sl": round(sl, 2), "tp": round(tp, 2), "magic": magic,
                   "comment": "cockpit", "type_time": mt5.ORDER_TIME_GTC,
                   "type_filling": fill_mode}
            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                ok = True
                detail = (f"{action.upper()} {lots} XAUUSD @ {px:.2f} "
                          f"SL {sl:.2f} TP {tp:.2f}")
                time.sleep(0.3)
                for lp in (mt5.positions_get(symbol=sym_name) or []):
                    if lp.magic == magic and lp.ticket not in live_ctx:
                        live_ctx[lp.ticket] = {"direction": action.upper(),
                                               "lots": lots, "atr": atr,
                                               "soft_sl": soft_sl}
                break
            detail = (f"retcode={res.retcode if res else 'None'} "
                      f"comment={res.comment if res else 'no result'}")
            time.sleep(0.5)
        print(f"[cmd] {action} {lots} -> {'ok' if ok else detail}")
        return {**out, "ok": ok, "detail": detail}

    if action == "close_all":
        closed, fails = 0, []
        for p in (mt5.positions_get(symbol=sym_name) or []):
            ctype = (mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY
                     else mt5.ORDER_TYPE_BUY)
            tick = mt5.symbol_info_tick(sym_name)
            if tick is None:
                fails.append(f"#{p.ticket} no tick")
                continue
            px = tick.bid if ctype == mt5.ORDER_TYPE_SELL else tick.ask
            req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": sym_name,
                   "volume": p.volume, "type": ctype, "position": p.ticket,
                   "price": px, "deviation": 30, "magic": p.magic,
                   "comment": "cockpit close", "type_time": mt5.ORDER_TIME_GTC,
                   "type_filling": fill_mode}
            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                closed += 1
            else:
                fails.append(f"#{p.ticket} retcode="
                             f"{res.retcode if res else 'None'}")
        detail = f"closed {closed} position(s)"
        if fails:
            detail += "; " + "; ".join(fails)
        print(f"[cmd] close_all -> {detail}")
        return {**out, "ok": not fails, "detail": detail}

    if action == "status":
        acc = mt5.account_info()
        if acc is None:
            return {**out, "ok": False, "detail": "no account"}
        ti = mt5.terminal_info()
        algo = "on" if (ti and ti.trade_allowed) else "OFF"
        return {**out, "ok": True,
                "detail": (f"balance ${acc.balance:.2f} equity "
                           f"${acc.equity:.2f}, autotrading {algo}")}
    return {**out, "ok": False, "detail": f"unknown action {action!r}"}


def handle_cmd_file(cfg, mt5, fill_mode: int, magic: int, live_ctx: dict) -> None:
    """Poll cmd.json (written by the bridge), execute, report cmd_result.json."""
    if not os.path.exists(CMD_PATH):
        return
    cmd = None
    try:
        with open(CMD_PATH, "r") as f:
            cmd = json.load(f)
    except Exception as e:
        print(f"[cmd] broken cmd.json ignored ({e})")
    try:
        os.remove(CMD_PATH)
    except OSError:
        pass
    if not isinstance(cmd, dict) or not cmd.get("action"):
        return
    try:
        result = run_command(cfg, mt5, fill_mode, magic, cmd, live_ctx)
    except Exception as e:
        result = {"id": cmd.get("id"), "action": str(cmd.get("action")),
                  "ts": datetime.now(timezone.utc).isoformat(),
                  "ok": False, "detail": f"exception: {e}"}
    try:
        tmp = CMD_RESULT_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f)
        os.replace(tmp, CMD_RESULT_PATH)
    except OSError as e:
        print(f"[cmd] could not write result ({e})")


def run_live(cfg):
    # backend switch: strategy_config.json "backend": "mt4" swaps the
    # MetaTrader5 module for the xtbridge MT4 file bridge (mt4link.py)
    try:
        from mt4link import install_backend
        if install_backend(cfg):
            print("[backend] MetaTrader5 API -> MT4 xtbridge file bridge")
    except Exception as _be:
        print(f"[backend] mt4link not engaged: {_be}")
    try:
        import faulthandler as _fh
        print("[watchdog] faulthandler available - per-cycle stall tripwire active")
    except Exception as _fh_e:
        print(f"[watchdog] faulthandler unavailable: {_fh_e}")
        _fh = None
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 package not installed.")
        print("  pip install MetaTrader5")
        print("or run paper mode:  python mt5_bot_v2.py --paper")
        return
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return
    acc = mt5.account_info()
    if acc is None:
        print("no account - open MT5 and log into your DEMO account first")
        return
    print(f"connected: {acc.login} | ${acc.balance:.2f} | {acc.server}")
    if not acc.trade_mode == 0:
        print("WARNING: this does not look like a demo account. aborting.")
        mt5.shutdown()
        return
    sym = mt5.symbol_info(cfg["symbol"])
    if sym is None or not mt5.symbol_select(cfg["symbol"], True):
        print(f"symbol {cfg['symbol']} unavailable on this broker")
        mt5.shutdown()
        return
    # pick the filling mode the symbol actually supports (bit1=FOK, bit2=IOC)
    if sym.filling_mode & 2:
        fill_mode = mt5.ORDER_FILLING_IOC
    elif sym.filling_mode & 1:
        fill_mode = mt5.ORDER_FILLING_FOK
    else:
        fill_mode = mt5.ORDER_FILLING_RETURN
    print(f"filling mode: {fill_mode} (symbol filling_mode bits {sym.filling_mode})")
    magic = 987001
    engine = Engine(acc.balance)
    engine.learner = SLLearner()
    print(f"learner db: {len(engine.learner.trades)} past trades, "
          f"{len(engine.learner.rules)} active no-trade filters")
    rc = RadarConfig(stand_down_score=cfg["radar"]["stand_down_score"],
                     flatten_score=cfg["radar"]["flatten_score"])
    engine.radar = LiquidationRadar(rc)
    normal_spread = float(sym.spread) or 0.30
    live_ctx: dict = {}          # position ticket -> entry ctx
    for p0 in (mt5.positions_get(symbol=cfg["symbol"]) or []):
        if p0.magic == magic:
            live_ctx[p0.ticket] = {
                "direction": "BUY" if p0.type == mt5.POSITION_TYPE_BUY
                else "SELL", "lots": p0.volume}
    # (full ctx reconstruction happens in the loop - see the [ctx] block)
    recorded_deals: set = load_seen_deals()  # deal tickets already autopsied
    print(f"learner memory: {len(recorded_deals)} deals already recorded "
          f"(persisted across restarts)")
    last_bar: Optional[int] = None   # last CLOSED bar time of the signal TF
    pending: Optional[dict] = None    # latched signal awaiting a fill
    last_gate_note: str = ""         # last "why we didn't qualify" (narrated)
    last_tf: Optional[int] = None    # printed-timeframe change tracker
    last_wick: Optional[bool] = None # printed wick-proof change tracker
    last_gates: tuple = None        # printed entry-gate change tracker
    last_strat: tuple = None        # printed full-strategy change tracker
    last_exsig: str = None          # printed exit-engine change tracker
    last_entry_mode: Optional[str] = None  # printed burst/cross tracker
    profit_target = float(cfg.get("profit_target_usd", 0) or 0)
    expected_login = getattr(acc, "login", None)
    guard_baseline = load_profit_baseline(acc.balance) if profit_target > 0 else None
    guard_target = (guard_baseline + profit_target) if profit_target > 0 else None
    if profit_target > 0:
        print(f"[profit-guard] ON: stop everything at equity "
              f"${guard_target:.2f} (baseline ${guard_baseline:.2f} "
              f"+ ${profit_target:.2f})")
    # aggressive mission: optional hard time deadline (config mission_hours,
    # >0). Start time persists across crash-restarts so a deadline is
    # absolute, not per-process. Cleared when the mission ends.
    mission_hours = float(cfg.get("mission_hours", 0) or 0)
    mission_file = os.path.join(BASE, "mission_start.json")
    mission_t0 = time.time()
    if mission_hours > 0:
        if os.path.exists(mission_file):
            try:
                with open(mission_file) as f:
                    mission_t0 = float(json.load(f).get("t", mission_t0))
            except (OSError, ValueError):
                pass
        try:
            with open(mission_file, "w") as f:
                json.dump({"t": mission_t0}, f)
        except OSError:
            pass
        print(f"[mission] AGGRESSIVE: burst entry target +${profit_target:.0f} "
              f"| deadline {mission_hours:.0f}h "
              f"(t0 {time.strftime('%m-%d %H:%M', time.localtime(mission_t0))}) "
              f"-> stop the bot at +target or at the deadline, whichever first")
    REASON_SL = getattr(mt5, "DEAL_REASON_SL", 4)
    REASON_TP = getattr(mt5, "DEAL_REASON_TP", 5)
    DEAL_OUT = getattr(mt5, "DEAL_ENTRY_OUT", 1)
    last_recon_ts = 0.0    # sea-state recon runs at boot + hourly
    last_army_note = ""    # HTF army gauge narration change-tracker
    last_vwap_bkt = 99.0   # [vwap] session-sigma narration bucket tracker
    last_fomo_phase = ""   # [fomo] phase-change narration tracker
    last_fom_flag = False  # [fomorian] mode-change narration tracker
    last_ultra_flag = False  # [ultra] mode-change narration tracker
    last_arrow_bar = None  # arrow re-learn per-bar gate
    arrow_n = 0            # arrow fire counter (relearn every N candles)
    last_whip_bar = None   # whipsaw per-bar fire gate
    armed_entries = {}     # patience layer: nice-price watch, fires MARKET
    whip_n = 0             # whipsaw fire counter
    last_ha_mode = ""      # [ha] 200-candle regime narration tracker
    trap_armed = False     # BRAIN 2 (whipsaw trap-fade) armed by recon
    brain_orders = {}      # parked bow-and-arrow limit orders, by ticket
    reload_plan = None     # pending bounce-ride after a bounce-guard exit
    last_loss_ts = None    # gold-medal brake: time of last losing close
    bh_zone_ts = {}        # bottomhunt zone -> last fire ts
    vwf_ts = {}            # vwapfade side -> last fire ts
    _last_sym = None       # last good quote - a bridge hiccup must not kill the loop
    live_spread = 0.30     # fallback spread until the first good quote
    try:
        while True:
            cfg = load_config()
            try:
                if _fh is not None:
                    # stall tripwire: re-armed every healthy cycle, so a dump
                    # in the err log can only mean a real >180s wedge - with
                    # the exact hung line attached.
                    _fh.cancel_dump_traceback_later()
                    _fh.dump_traceback_later(180)
            except Exception:
                pass
            # LADDER + VANTAGE config hot-reload with cfg every cycle
            LADDER_CFG.clear()
            LADDER_CFG.update(LADDER_CFG_DEFAULTS)
            LADDER_CFG.update(cfg.get("limit_ladder") or {})
            VANTAGE_CFG.clear()
            VANTAGE_CFG.update(VANTAGE_CFG_DEFAULTS)
            VANTAGE_CFG.update(cfg.get("vantage") or {})
            gate_note = ""      # set only when a signal exists; else nothing to say
            # account lock: never trade / never guard on a terminal that is
            # not the account this bot was started on (switched login counts)
            acc_live = mt5.account_info()
            if acc_live is None or acc_live.login != expected_login:
                print(f"[SAFE] terminal account is "
                      f"{acc_live.login if acc_live else 'None'} but this bot "
                      f"is locked to {expected_login} - standing down "
                      f"(no trades, no cmds, no guard)")
                time.sleep(cfg.get("cycle_seconds", 5))
                continue
            handle_cmd_file(cfg, mt5, fill_mode, magic, live_ctx)
            tf = resolve_tf(cfg, mt5)
            if tf != last_tf:
                print(f"[config] signal timeframe: "
                      f"{cfg.get('timeframe', '1hour')} (radar stays H1/H4)")
                last_tf = tf
            wick_on = bool(cfg.get("wick_proof_stop"))
            strat_sig = (cfg.get("timeframe"), cfg.get("ema_fast"),
                         cfg.get("ema_slow"), cfg.get("atr_stop_mult"),
                         cfg.get("rr_ratio"), cfg.get("risk_pct"),
                         cfg.get("base_lots"), cfg.get("max_concurrent"),
                         cfg.get("profit_target_usd"),
                         cfg.get("regime_gate_mode"), cfg.get("regime_gate_thr"),
                         cfg.get("whip_break_n"), cfg.get("whip_break_bars"))
            if strat_sig != last_strat:
                print(f"[config] strategy: TF {cfg.get('timeframe')} "
                      f"EMA {cfg.get('ema_fast')}/{cfg.get('ema_slow')} "
                      f"stop x{cfg.get('atr_stop_mult')} "
                      f"RR {cfg.get('rr_ratio')} "
                      f"risk {cfg.get('risk_pct')}% "
                      f"lots {cfg.get('base_lots')} "
                      f"max {cfg.get('max_concurrent')} "
                      f"target ${cfg.get('profit_target_usd')}")
                _rm = str(cfg.get("regime_gate_mode", "off") or "off").lower()
                if _rm in ("off", "none", ""):
                    print("[regime] PERMISSION GATE: OFF")
                else:
                    _wn = int(cfg.get("whip_break_n", 0) or 0)
                    _wb = int(cfg.get("whip_break_bars", 0) or 0)
                    print(f"[regime] PERMISSION GATE: {_rm.upper()} "
                          f"thr {cfg.get('regime_gate_thr')} "
                          + (f"+ whip {_wn} losses -> {_wb}min"
                             if _wn and _wb else "(whip off)"))
                last_strat = strat_sig
            if bool(cfg.get("fomorian_mode")) != last_fom_flag:
                last_fom_flag = bool(cfg.get("fomorian_mode"))
                print(f"[fomorian] MODE "
                      f"{'ON' if last_fom_flag else 'OFF'} - "
                      f"martingale demo: mult "
                      f"{cfg.get('fomorian_mult', 2.0)} start "
                      f"{cfg.get('fomorian_initial', 0.02)} cap "
                      f"{cfg.get('fomorian_cap', 0.5)} - "
                      + (("MANUAL martingale: steps have NO "
                          "SL/TP - YOU cut each trade, the bot "
                          "doubles off your cut"
                          if not bool(cfg.get("auto_close", True))
                          else "WATCH THE SPIRAL")
                         if last_fom_flag
                         else "normal brains back"))
            if bool(cfg.get("ultra_mode")) != last_ultra_flag:
                last_ultra_flag = bool(cfg.get("ultra_mode"))
                print(f"[ultra] MODE "
                      f"{'ON' if last_ultra_flag else 'OFF'} - "
                      + ("ultra-fast in/out: MACD(SMA30/144) timing "
                         "lows & highs on 1min, SL 0.5 ATR / TP "
                         "0.75 ATR, 1h+30m precaution must agree, "
                         "hist flip exits, all other brains muted"
                         if last_ultra_flag
                         else "classic brains back"))
            ex_sig = json.dumps(cfg.get("exit_engine") or {}, sort_keys=True)
            if ex_sig != last_exsig:
                ex0 = cfg.get("exit_engine") or {}
                print(f"[exit-engine] "
                      f"{'ON' if ex0.get('on') else 'OFF'} "
                      f"hold<={ex0.get('hold_max_s')}s "
                      f"peak-prot@${ex0.get('min_peak')} "
                      f"tp-tighten({ex0.get('lock_atr_off')}xATR) "
                      f"retrace {ex0.get('retrace_pct')}% "
                      f"ha-flip {'ON' if ex0.get('ha_flip') else 'OFF'}")
                last_exsig = ex_sig
            if wick_on != last_wick:
                print(f"[config] wick-proof stops: {'ON' if wick_on else 'OFF'} "
                      f"(disaster x{cfg.get('disaster_stop_mult', 1.5)})")
                last_wick = wick_on
            gates = cfg.get("gates") or {}
            gate_sig = (int(gates.get("breakout_n", 0) or 0), bool(gates.get("atr_expansion")),
                        float(gates.get("breakeven_r", 0) or 0))
            if gate_sig != last_gates:
                bn, ax, br = gate_sig
                print(f"[config] entry gates: breakout={bn or 'OFF'} "
                      f"atr_expansion={'ON' if ax else 'OFF'} breakeven={br or 'OFF'}R"
                      + ("" if (bn or ax) else "  (WARNING: ungated - every cross is an entry)"))
                last_gates = gate_sig
            mode = str(cfg.get("entry_mode", "cross")).lower()
            if mode != last_entry_mode:
                if mode == "auto":
                    print("[config] entry mode: AUTO (triple brain: burst "
                          "timing + trap-fade + HTF army ride + round-"
                          "number fade, recon-gated)")
                else:
                    print("[config] entry mode: " + ("BURST (trend-aligned "
                          "breakout entries, pyramids up to max_concurrent "
                          "- no fresh cross needed)" if mode == "burst"
                          else "cross (classic EMA crossover)"))
                last_entry_mode = mode
            bars = mt5.copy_rates_from_pos(cfg["symbol"], tf, 0, 900)
            h4 = mt5.copy_rates_from_pos(cfg["symbol"], mt5.TIMEFRAME_H4, 0, 260)
            m30 = mt5.copy_rates_from_pos(cfg["symbol"], mt5.TIMEFRAME_M30, 0, 350)
            if bars is None or len(bars) < 60 or h4 is None or len(h4) < 60:
                time.sleep(cfg["cycle_seconds"]); continue
            # use CLOSED bars only: last row from copy_rates_from_pos is the
            # forming candle - its close flickers with every tick
            df_h1 = pd.DataFrame(bars).iloc[:-1].reset_index(drop=True)
            df_h4 = pd.DataFrame(h4).iloc[:-1].reset_index(drop=True)
            df30 = (pd.DataFrame(m30).iloc[:-1].reset_index(drop=True)
                    if m30 is not None and len(m30) > 60 else None)
            engine._last_h1 = df_h1
            # THE PRACTICE RUN: sea-state recon at boot + hourly
            nowts = time.time()
            if nowts - last_recon_ts > 3600:
                tide_info = recon_tide(df_h1, cfg)
                trap_armed = bool(tide_info["trap"])
                _TIDE.update({"info": tide_info, "trap": trap_armed,
                              "ts": nowts})
                if trap_armed:
                    print("[recon] whipsaws confirmed -> BRAIN 2 trap-fade "
                          "ARMED (fade confirmed sweeps: fast in, fast out)")
                last_recon_ts = nowts
            # BOW-AND-ARROW lifecycle: stale peg orders die after ttl_s,
            # when the next candle opens, or when price drifts too far
            # off the peg (fresh sweep = fresh peg anyway); fills become
            # managed positions right here.
            if brain_orders:
                open_now = {o.ticket: True for o in
                            (mt5.orders_get(symbol=cfg["symbol"]) or [])}
                for ot, od in list(brain_orders.items()):
                    if ot not in open_now:
                        brain_orders.pop(ot, None)    # filled or vanished
                        continue
                    age = time.time() - float(od.get("born", time.time()))
                    drift_bad = (abs(od.get("peg_px", price) - price)
                                 > LIMIT_CFG["repeg_atr"] * 0.6)
                    if (age > LIMIT_CFG["ttl_s"] or drift_bad
                            or od.get("bar") != current_bar):
                        rmar = mt5.order_send(
                            {"action": mt5.TRADE_ACTION_REMOVE, "order": ot})
                        if rmar and rmar.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"[peg] cancelled stale {od['tag']} "
                                  f"limit (age {age:.0f}s)")
                            brain_orders.pop(ot, None)
            # SPREAD-CAPTURE LADDER lifecycle: a layer that leaves the
            # order book filled (the position scan below registers it); a
            # layer that survives past its TTL is cancelled so no stale
            # bet lingers into the next signal.
            if LADDER_STATE["active"]:
                open_lad = {o.ticket for o in
                            (mt5.orders_get(symbol=cfg["symbol"]) or [])}
                for ot2, ld2 in list(LADDER_STATE["active"].items()):
                    if ot2 not in open_lad:
                        LADDER_STATE["active"].pop(ot2, None)
                        LADDER_STATE["fills"] += 1
                        print(f"[ladder] layer #{ot2} FILLED "
                              f"({ld2['dir']} @ {ld2['level']:.2f}) - "
                              f"spread captured")
                        continue
                    age2 = time.time() - float(ld2.get("born", time.time()))
                    if age2 > LADDER_CFG.get("ttl_s", 300):
                        rl = mt5.order_send(
                            {"action": mt5.TRADE_ACTION_REMOVE, "order": ot2})
                        if rl and rl.retcode == mt5.TRADE_RETCODE_DONE:
                            LADDER_STATE["active"].pop(ot2, None)
                            LADDER_STATE["cancels"] += 1
                            print(f"[ladder] stale layer #{ot2} cancelled "
                                  f"(age {age2:.0f}s, no fill)")
                    elif age2 < 0.15:
                        print(f"[ladder] {ld2['dir']} layer #{ot2} parked "
                              f"@ {ld2['level']:.2f} SL {ld2['sl']:.2f} "
                              f"TP {ld2['tp']:.2f} (spread-grab fill)")
            # VANTAGE RELOAD (user 2026-10-12 "the 3 stooges"): the delayed
            # 4th bullet for the stop-hunt pullback. One reserve LIMIT is
            # parked NEAR the stop shelf after the pullback has had time to
            # develop - the hunt fills it at a vantage price while early
            # layers scratch out light. Bounded: one per signal, TTL expiry,
            # counts toward max_concurrent, stood down on opposite book.
            _vg = VANTAGE_CFG
            _vs = VANTAGE_STATE
            if _vg.get("on") and _vs.get("scheduled") is not None:
                _sc = _vs["scheduled"]
                _vpos = [pp for pp in
                         (mt5.positions_get(symbol=cfg["symbol"]) or [])
                         if pp.magic == magic]
                _vopp = [pp for pp in _vpos
                         if ("LONG" if pp.type == 0 else "SHORT")
                         != _sc["dir"]]
                if _vopp or any((vv.get("dir") != _sc["dir"])
                                for vv in LADDER_STATE["active"].values()):
                    if _vs.get("order") is not None:
                        _rmv = mt5.order_send(
                            {"action": mt5.TRADE_ACTION_REMOVE,
                             "order": _vs["order"]["ticket"]})
                        if _rmv and _rmv.retcode == \
                                mt5.TRADE_RETCODE_DONE:
                            print(f"[vantage] {_sc['dir']} stood down - "
                                  f"opposite book")
                        _vs["order"] = None
                    _vs["scheduled"] = None
                    _vs["scratched_sig"] = None
                elif time.time() - _sc["born"] > \
                        float(_vg.get("delay_s", 120) or 0) + \
                        float(_vg.get("ttl_s", 300) or 0):
                    _vs["scheduled"] = None
                    _vs["scratched_sig"] = None
                    print(f"[vantage] {_sc['dir']} schedule expired "
                          f"(no hunt to fade)")
                else:
                    # hunt-scratch: the pump toward our stops is the tell.
                    # Scratch early layers light (once per signal) so the
                    # stooges don't take the full whack together.
                    if _vs.get("scratched_sig") != _sc.get("sig_id"):
                        _tkv = mt5.symbol_info_tick(cfg["symbol"])
                        if _tkv is not None:
                            _pxv = (_tkv.ask if _sc["dir"] == "SHORT"
                                    else _tkv.bid)
                            _adv = ((_pxv - _sc["ref"])
                                    if _sc["dir"] == "SHORT"
                                    else (_sc["ref"] - _pxv))
                            _sdv = _sc["stop_dist"] or 1.0
                            if _adv >= float(_vg.get("scratch_trig_atr",
                                                      0.6) or 0) * _sdv \
                                    and time.time() - _sc["born"] <= \
                                    float(_vg.get("scratch_within_s",
                                                  180) or 0):
                                _nscr = 0
                                for _sp in _vpos:
                                    _sdir = ("LONG" if _sp.type == 0
                                             else "SHORT")
                                    if _sdir != _sc["dir"]:
                                        continue
                                    _ct = (mt5.ORDER_TYPE_SELL
                                           if _sdir == "LONG"
                                           else mt5.ORDER_TYPE_BUY)
                                    _t2 = mt5.symbol_info_tick(
                                        cfg["symbol"])
                                    if _t2 is None:
                                        break
                                    _px2 = (_t2.bid if _ct ==
                                            mt5.ORDER_TYPE_SELL
                                            else _t2.ask)
                                    _rr2 = mt5.order_send(
                                        {"action": mt5.TRADE_ACTION_DEAL,
                                         "symbol": cfg["symbol"],
                                         "volume": _sp.volume, "type": _ct,
                                         "position": _sp.ticket,
                                         "price": _px2, "deviation": 30,
                                         "magic": magic,
                                         "comment": "vantage scratch",
                                         "type_time": mt5.ORDER_TIME_GTC,
                                         "type_filling": fill_mode})
                                    if _rr2 and _rr2.retcode == \
                                            mt5.TRADE_RETCODE_DONE:
                                        _nscr += 1
                                _vs["scratched_sig"] = _sc.get("sig_id")
                                print(f"[vantage] hunt-scratch: closed "
                                      f"{_nscr} early {_sc['dir']} layer(s) "
                                      f"into the pump (adv {_adv:.2f})")
                    # park the reserve bullet once the delay has played out
                    if _vs.get("order") is None and time.time() >= \
                            float(_sc.get("arm_at", 0) or 0):
                        _tk3 = mt5.symbol_info_tick(cfg["symbol"])
                        _ac3 = mt5.account_info()
                        _sy3 = mt5.symbol_info(cfg["symbol"])
                        if _tk3 is not None and _ac3 is not None \
                                and _sy3 is not None:
                            _sd3 = _sc["stop_dist"] or 1.0
                            _pa = float(_vg.get("price_atr", 0.8) or 0)
                            _lvl = (_tk3.ask + _pa * _sd3) \
                                if _sc["dir"] == "SHORT" \
                                else (_tk3.bid - _pa * _sd3)
                            _ok_lvl = (_lvl > _tk3.bid + 0.05) \
                                if _sc["dir"] == "SHORT" \
                                else (_lvl < _tk3.ask - 0.05)
                            _room = len(_vpos) < min(
                                4, int(cfg.get("max_concurrent", 3) or 3))
                            _lots_v = max(
                                _sy3.volume_min or 0.01,
                                round(size_lots(
                                    _ac3.balance,
                                    _ac3.balance * cfg["risk_pct"] / 100.0,
                                    _sd3, cfg)
                                    * float(_vg.get("size_frac", 0.3)
                                            or 0), 2))
                            if _ok_lvl and _room:
                                _dm3 = float(cfg.get("disaster_stop_mult",
                                                     1.5))
                                _sl3 = (_lvl - _sd3 * _dm3) \
                                    if _sc["dir"] == "LONG" \
                                    else (_lvl + _sd3 * _dm3)
                                _tp3 = (_lvl + _sd3 * _sc["rr"]) \
                                    if _sc["dir"] == "LONG" \
                                    else (_lvl - _sd3 * _sc["rr"])
                                _ot3 = (mt5.ORDER_TYPE_BUY_LIMIT
                                        if _sc["dir"] == "LONG"
                                        else mt5.ORDER_TYPE_SELL_LIMIT)
                                _rv = mt5.order_send(
                                    {"action": mt5.TRADE_ACTION_PENDING,
                                     "symbol": cfg["symbol"],
                                     "volume": _lots_v, "type": _ot3,
                                     "price": round(_lvl, 2),
                                     "deviation": 30,
                                     "sl": round(_sl3, 2),
                                     "tp": round(_tp3, 2), "magic": magic,
                                     "comment": "VAN",
                                     "type_time": mt5.ORDER_TIME_GTC,
                                     "type_filling": fill_mode})
                                if _rv and _rv.retcode == \
                                        mt5.TRADE_RETCODE_DONE:
                                    _otv = getattr(_rv, "order", 0)
                                    if _otv:
                                        _vs["order"] = {
                                            "ticket": _otv,
                                            "born": time.time(),
                                            "level": _lvl}
                                        print(f"[vantage] {_sc['dir']} "
                                              f"reserve parked @ "
                                              f"{_lvl:.2f} ({_lots_v} lots, "
                                              f"TTL "
                                              f"{_vg.get('ttl_s', 300)}s) - "
                                              f"the hunt fills it, not us")
                            elif not _room:
                                print("[vantage] book full - reserve waits "
                                      "for a free slot")
                    # parked reserve lifecycle: fills register via the
                    # position scan; stale reserve cancels like a layer
                    if _vs.get("order") is not None:
                        _vo2 = _vs["order"]
                        _open_v = {oo.ticket for oo in
                                   (mt5.orders_get(
                                       symbol=cfg["symbol"]) or [])}
                        if _vo2["ticket"] not in _open_v:
                            _vs["order"] = None
                            print(f"[vantage] reserve #{_vo2['ticket']} "
                                  f"FILLED ({_sc['dir']} @ "
                                  f"{_vo2['level']:.2f}) - hunting from "
                                  f"the vantage")
                        elif time.time() - _vo2["born"] > \
                                float(_vg.get("ttl_s", 300) or 0):
                            _rm2 = mt5.order_send(
                                {"action": mt5.TRADE_ACTION_REMOVE,
                                 "order": _vo2["ticket"]})
                            if _rm2 and _rm2.retcode == \
                                    mt5.TRADE_RETCODE_DONE:
                                _vs["order"] = None
                                print(f"[vantage] stale reserve "
                                      f"#{_vo2['ticket']} cancelled "
                                      f"(no hunt came)")
            # register limit fills as managed ctx (tag from order comment)
            for lp in (mt5.positions_get(symbol=cfg["symbol"]) or []):
                if lp.magic == magic and lp.ticket not in live_ctx:
                    ltag = (lp.comment or "").strip()
                    if ltag in LIMIT_CFG["tag_peg"] or ltag.startswith("LAD"):
                        lextra = {}
                        for od in brain_orders.values():
                            if od["tag"] == ltag:
                                lextra = od.get("extra") or {}
                                break
                        lcd = {ltag: True, "born": time.time(),
                               "soft_sl": None}
                        lcd.update(lextra)
                        live_ctx[lp.ticket] = lcd
                        print(f"[peg] limit FILL -> #{lp.ticket} {ltag} "
                              f"{lp.volume} lots, ctx registered")
            sym = mt5.symbol_info(cfg["symbol"])
            if sym is None:
                sym = _last_sym   # transient EA/bridge hiccup: reuse last good quote
            if sym is None:
                print("[warn] symbol_info unavailable - skipping cycle")
                time.sleep(2)
                continue
            _last_sym = sym
            live_spread = float(sym.spread) or 0.30
            tick = mt5.symbol_info_tick(cfg["symbol"])
            # keep the cost meter alive even between orders: live spread &
            # far-side quotes every cycle (same math send_brain uses)
            _ctk = float(getattr(sym, "trade_contract_size", 100.0) or 100.0)
            _ptk = float(getattr(sym, "point", 0.01) or 0.01)
            LIVE_METER["last"] = {
                "spread_pts": float(sym.spread or 0.0),
                "spread_usd": round(float(sym.spread or 0.0) * _ptk * _ctk, 3),
                "bid": round(float(tick.bid), 2) if tick is not None else None,
                "ask": round(float(tick.ask), 2) if tick is not None else None,
                "ts": time.time(),
            }
            price = float(df_h1["close"].iloc[-1])
            current_bar = int(df_h1["time"].iloc[-1])
            new_bar = (current_bar != last_bar)
            if new_bar:
                last_bar = current_bar   # one signal evaluation per closed bar

            # sync closed trades from deal history (last 30 days)
            deals = mt5.history_deals_get(
                datetime.now(timezone.utc) - timedelta(days=30),
                datetime.now(timezone.utc) + timedelta(days=1))
            closed_by_magic = [d for d in (deals or [])
                               if d.magic == magic and d.entry == DEAL_OUT]
            # meter truth: EVERY close on the symbol (bot + manual fills),
            # so the dashboard win-rate tells the full account story
            all_closed = [d for d in (deals or [])
                          if d.entry == DEAL_OUT
                          and d.symbol == cfg.get("symbol", "XAUEUR")]
            recent_stops = sum(1 for d in closed_by_magic[-10:]
                               if d.reason == REASON_SL)

            # autopsy every broker-side closed trade into the learner
            new_seen = False
            for d in closed_by_magic:
                if d.ticket in recorded_deals:
                    continue
                recorded_deals.add(d.ticket)
                new_seen = True
                reason = "SL" if d.reason == REASON_SL else \
                         ("TP" if d.reason == REASON_TP else "EXPERT")
                if d.profit < 0:
                    last_loss_ts = time.time()
                    # whip breaker: count the consecutive-loss run
                    _WHIP["run"] = _WHIP.get("run", 0) + 1
                    _wb_n = int(cfg.get("whip_break_n", 0) or 0)
                    _wb_b = int(cfg.get("whip_break_bars", 0) or 0)
                    if _wb_n > 0 and _wb_b > 0 and _WHIP["run"] >= _wb_n:
                        _WHIP["pause_until"] = time.time() + _wb_b * 60
                        _WHIP["run"] = 0
                        print(f"[whip] {_wb_n}+ consecutive losses - "
                              f"whipsaw breaker: stand down {_wb_b}min")
                else:
                    _WHIP["run"] = 0
                ctx = live_ctx.pop(d.position_id, {})
                if engine.learner is not None:
                    engine.learner.record_completed(
                        ctx, d.profit, reason, d.volume, d.price, d.price)
                acc_now = mt5.account_info()
                engine.trade_counter += 1
                engine.closed.append(BtTrade(
                    num=engine.trade_counter,
                    direction=ctx.get("direction", "?"), lots=d.volume,
                    entry_price=round(d.price, 2), exit_price=round(d.price, 2),
                    entry_time=datetime.now(timezone.utc).isoformat(),
                    exit_time=datetime.now(timezone.utc).isoformat(),
                    pnl=round(d.profit, 2), reason=reason,
                    balance_after=round(acc_now.balance, 2) if acc_now else None,
                    is_win=d.profit > 0,
                ))
            if new_seen:
                save_seen_deals(recorded_deals)
            if (engine.learner is not None
                    and len(engine.closed) >= engine.relearn_every
                    and len(engine.closed) // engine.relearn_every
                        != engine._relearn_bucket):
                engine._relearn_bucket = len(engine.closed) // engine.relearn_every
                report = engine.learner.relearn()
                print(f"  [relearn] trades={report['trades_in_db']} "
                      f"SL rate={report['overall_sl_rate']:.0%} "
                      f"rules={report['rules_active']}")

            if tf == mt5.TIMEFRAME_H1:
                df_radar_h1 = df_h1
            else:
                r1 = mt5.copy_rates_from_pos(cfg["symbol"],
                                             mt5.TIMEFRAME_H1, 0, 900)
                df_radar_h1 = (pd.DataFrame(r1).iloc[:-1].reset_index(drop=True)
                               if r1 is not None and len(r1) > 60 else df_h1)
            reading = engine.radar.assess(
                df_radar_h1, df_h4, live_spread, normal_spread,
                recent_stop_outs=recent_stops)

            positions = [p for p in (mt5.positions_get(symbol=cfg["symbol"]) or [])
                         if p.magic == magic]

            def place_spread_ladder(direction, stop_dist, rr_u, av3):
                """SPREAD-CAPTURE LADDER (user 2026-10-08 "make the
                spread work in our favour"): the main cross signal no
                longer sends ONE market order that pays the cross.
                Instead it parks `layers` staggered LIMIT orders on the
                spread-favorable side - LONG buys on/under the BID,
                SHORT sells on/over the ASK. Every layer that fills has
                GRABBED the spread the market order would have paid;
                deeper layers fill only if price keeps coming to us, and
                each layer carries its own broker SL/TP (wick-proof
                disaster stop). Returns number of layers parked (0 =
                blocked/no-op - the signal is swallowed, not market-ed)."""
                lc = LADDER_CFG
                if not lc.get("on"):
                    return 0
                if (fom_active or ultra_active) \
                        and direction != "fomorian":
                    return 0
                if not auto_close:
                    return 0
                if not regime_gate_allows(df_h1, cfg, "ladder"):
                    return 0
                tick_l = mt5.symbol_info_tick(cfg["symbol"])
                acc_l = mt5.account_info()
                sym_l = mt5.symbol_info(cfg["symbol"])
                if tick_l is None or acc_l is None or sym_l is None:
                    return 0
                ct = float(getattr(sym_l, "trade_contract_size", 100.0)
                           or 100.0)
                spread_pts = float(sym_l.spread or 0.0)
                spread_usd = spread_pts * float(sym_l.point or 0.01) * ct
                # hedge guard: the ladder must not stack onto a live
                # opposite leg (2x spread, no edge).
                for _p in positions:
                    _po = "LONG" if _p.type == 0 else "SHORT"
                    if _po != direction:
                        print(f"[ladder] {direction} BLOCKED - "
                              f"#{_p.ticket} {_po} still open "
                              f"(netting = 2x spread)")
                        return 0
                if LADDER_STATE["active"]:
                    print(f"[ladder] {len(LADDER_STATE['active'])} "
                          f"layer(s) already parked - no restack")
                    return 0
                lots_total = size_lots(
                    acc_l.balance,
                    acc_l.balance * cfg["risk_pct"] / 100.0,
                    stop_dist, cfg)
                if lots_total < sym_l.volume_min:
                    return 0
                n_layers = max(1, int(lc.get("layers", 3)))
                wts = list(lc.get("weights") or [0.5, 0.3, 0.2])
                while len(wts) < n_layers:
                    wts.append(0.0)
                step = float(lc.get("step_atr", 0.2)) * (
                    av3 if (av3 is not None and av3 > 0) else stop_dist)
                offs = lc.get("offsets")
                use_offs = isinstance(offs, (list, tuple)) and \
                    len(offs) >= n_layers and all(
                        isinstance(x, (int, float)) and not
                        isinstance(x, bool) for x in offs[:n_layers])
                ref = (tick_l.bid if direction == "LONG"
                       else tick_l.ask)
                dm = float(cfg.get("disaster_stop_mult", 1.5))
                placed = 0
                for i in range(n_layers):
                    wi = wts[i] if i < len(wts) else 0.0
                    if wi <= 0:
                        continue
                    lots_i = max(sym_l.volume_min,
                                 round(lots_total * wi, 2))
                    if use_offs:
                        px_i = (ref - float(offs[i]) * stop_dist) \
                            if direction == "LONG" \
                            else (ref + float(offs[i]) * stop_dist)
                    else:
                        px_i = (ref - i * step) if direction == "LONG" \
                            else (ref + i * step)
                    if direction == "LONG":
                        px_i = min(px_i, tick_l.ask - 0.05)
                        if px_i >= tick_l.ask - 0.05:
                            continue
                    else:
                        px_i = max(px_i, tick_l.bid + 0.05)
                        if px_i <= tick_l.bid + 0.05:
                            continue
                    sl_b = (px_i - stop_dist * dm) \
                        if direction == "LONG" \
                        else (px_i + stop_dist * dm)
                    tp_i = (px_i + stop_dist * rr_u) \
                        if direction == "LONG" \
                        else (px_i - stop_dist * rr_u)
                    otype_i = (mt5.ORDER_TYPE_BUY_LIMIT
                               if direction == "LONG"
                               else mt5.ORDER_TYPE_SELL_LIMIT)
                    req_l = {"action": mt5.TRADE_ACTION_PENDING,
                             "symbol": cfg["symbol"], "volume": lots_i,
                             "type": otype_i,
                             "price": round(px_i, 2),
                             "deviation": 30,
                             "sl": round(sl_b, 2),
                             "tp": round(tp_i, 2),
                             "magic": magic,
                             "comment": f"LAD{i + 1}",
                             "type_time": mt5.ORDER_TIME_GTC,
                             "type_filling": fill_mode}
                    res_l = mt5.order_send(req_l)
                    if res_l and res_l.retcode == \
                            mt5.TRADE_RETCODE_DONE:
                        ot_l = getattr(res_l, "order", 0)
                        if ot_l:
                            LADDER_STATE["active"][ot_l] = {
                                "dir": direction, "born": time.time(),
                                "level": px_i, "sl": sl_b, "tp": tp_i,
                                "layer": i + 1}
                            placed += 1
                    else:
                        rc_l = res_l.retcode if res_l else "None"
                        cm_l = (res_l.comment if res_l
                                else "no result")
                        print(f"[ladder] LAD{i + 1} rejected "
                              f"retcode={rc_l} comment={cm_l!r}")
                if placed:
                    lvls = " ".join(
                        f"{v['level']:.2f}" for v in
                        sorted(LADDER_STATE["active"].values(),
                               key=lambda x: x["layer"]))
                    print(f"[ladder] {direction} {placed}/{n_layers} "
                          f"layers parked [{lvls}] "
                          f"(spread ${spread_usd:.2f}/lot GRABBED per "
                          f"fill instead of paid) TTL "
                          f"{lc.get('ttl_s', 300)}s")
                return placed

            # crash/restart safety: a position whose manage-context was lost
            # (bot restarted while it was open, or ctx predates wick-proof)
            # gets a reconstructed one from broker truth, so soft-stop and
            # breakeven management are never silently inert.
            seed_atr = float(rolling_atr(df_h1).iloc[-1]) if len(df_h1) else 0.0
            for p in positions:
                c = live_ctx.get(p.ticket)
                if c is not None and c.get("soft_sl") is not None \
                        and c.get("sd") is not None:
                    continue
                buy = p.type == mt5.POSITION_TYPE_BUY
                # best sd: the distance the broker SL is actually sitting at
                # (= disaster dist when wick-proof ON, = soft level when OFF)
                sd = None
                if p.sl and p.price_open:
                    d = (p.price_open - p.sl) if buy else (p.sl - p.price_open)
                    if d > 0:
                        dm = (float(cfg.get("disaster_stop_mult", 1.5))
                              if cfg.get("wick_proof_stop") else 1.0)
                        sd = d / dm
                if not sd or not (0 < sd < 500.0):
                    sd = ((cfg["atr_stop_mult"] * seed_atr) if seed_atr
                          else 6.0)
                c = {"direction": "BUY" if buy else "SELL", "lots": p.volume,
                     "soft_sl": (p.price_open - sd) if buy
                     else (p.price_open + sd),
                     "sd": sd, "recovered": True}
                live_ctx[p.ticket] = c
                print(f"[ctx] #{p.ticket} context reconstructed: "
                      f"{c['direction']} {p.volume} sd={sd:.2f} "
                      f"soft_sl={c['soft_sl']:.2f}")

            # wick-proof soft stops: close when a CLOSED signal-TF bar
            # settles beyond the soft SL (momentary wicks are held)
            if cfg.get("wick_proof_stop"):
                last_close = float(df_h1["close"].iloc[-1])
                be_r = float((cfg.get("gates") or {}).get("breakeven_r", 0) or 0)
                for p in positions:
                    c = live_ctx.get(p.ticket) or {}
                    soft = c.get("soft_sl")
                    sd = c.get("sd")
                    # --- breakeven promotion: after +be_r x risk, the soft
                    # stop moves to entry. Cuts the "was up $1.5, closed at
                    # -$4" loss events that dominated the loss record.
                    if (be_r > 0 and sd and soft is not None
                            and not c.get("be")):
                        risk_usd = sd * p.volume * 100.0
                        if p.profit >= be_r * risk_usd:
                            c["soft_sl"] = p.price_open
                            c["be"] = True
                            live_ctx[p.ticket] = c
                            print(f"[be] #{p.ticket} +{p.profit:.2f} "
                                  f"(+{be_r}R) -> stop moved to breakeven "
                                  f"{p.price_open:.2f}")
                    if soft is None:
                        continue
                    soft = c.get("soft_sl")
                    if ((p.type == mt5.POSITION_TYPE_BUY and last_close < soft)
                            or (p.type == mt5.POSITION_TYPE_SELL
                                and last_close > soft)):
                        ctype = (mt5.ORDER_TYPE_SELL
                                 if p.type == mt5.POSITION_TYPE_BUY
                                 else mt5.ORDER_TYPE_BUY)
                        tick_now = mt5.symbol_info_tick(cfg["symbol"])
                        if tick_now is None:
                            continue
                        px = (tick_now.bid if ctype == mt5.ORDER_TYPE_SELL
                              else tick_now.ask)
                        req = {"action": mt5.TRADE_ACTION_DEAL,
                               "symbol": cfg["symbol"], "volume": p.volume,
                               "type": ctype, "position": p.ticket,
                               "price": px, "deviation": 30, "magic": p.magic,
                               "comment": "wick-stop",
                               "type_time": mt5.ORDER_TIME_GTC,
                               "type_filling": fill_mode}
                        res = mt5.order_send(req)
                        rc = res.retcode if res else "None"
                        cm = res.comment if res else "no result"
                        print(f"WICK-STOP #{p.ticket} bar-close "
                              f"{last_close:.2f} beyond soft SL {soft:.2f} "
                              f"-> retcode={rc} comment={cm!r}")

            # Heiken-ashi state for the exit engine (last CLOSED signal-TF
            # HA candle direction; the forming bar is df_h1[-1]).
            ha_gprev = None
            try:
                if (cfg.get("exit_engine") or {}).get("on") and len(df_h1) >= 6:
                    _ha = heiken_ashi(df_h1)
                    ha_gprev = bool(_ha["close"].iloc[-2] > _ha["open"].iloc[-2])
            except Exception:
                ha_gprev = None

            # PROFIT-BOUNCE GUARD (learned behavior): a strong reversal
            # bounce gives back most of an open trade's peak profit.
            # The give-back AMOUNT plus its ACCELERATION (consecutive
            # declining cycles) decide when to take the money and run.
            # ---- MANUAL MODE (user 2026-10-08): auto_close=false kills
            # EVERY automatic exit (216s cap, 1-candle cut, exit-engine,
            # bounce/reload guards, radar flatten). Only the USER closes
            # trades (CLOSE on the dashboard / close_all cmd). fomorian
            # steps run the manual martingale: no SL/TP attached, cut them
            # yourself, the bot doubles the next one off your cut.
            auto_close = bool(cfg.get("auto_close", True))
            fom_active = bool(cfg.get("fomorian_mode", False))
            ultra_active = bool(cfg.get("ultra_mode", False))
            for p in (positions if auto_close else []):
                bgc = live_ctx.get(p.ticket) or {}
                bg_peak = float(bgc.get("peak", p.profit))
                bg_give = bg_peak - p.profit
                bg_dips = int(bgc.get("dips", 0))
                if p.profit < float(bgc.get("last_p", p.profit)):
                    bg_dips = bg_dips + 1 if bg_give > 0 else 0
                else:
                    bg_dips = 0

                def bg_close(comment):
                    """market-close one position (bounce/reload family)"""
                    ctype = (mt5.ORDER_TYPE_SELL
                             if p.type == mt5.POSITION_TYPE_BUY
                             else mt5.ORDER_TYPE_BUY)
                    tick_now = mt5.symbol_info_tick(cfg["symbol"])
                    if tick_now is None:
                        return None
                    cpx = (tick_now.bid if ctype == mt5.ORDER_TYPE_SELL
                           else tick_now.ask)
                    req = {"action": mt5.TRADE_ACTION_DEAL,
                           "symbol": cfg["symbol"], "volume": p.volume,
                           "type": ctype, "position": p.ticket,
                           "price": cpx, "deviation": 30, "magic": p.magic,
                           "comment": comment,
                           "type_time": mt5.ORDER_TIME_GTC,
                           "type_filling": fill_mode}
                    rr2 = mt5.order_send(req)
                    rc2 = rr2.retcode if rr2 else "None"
                    return rc2, cpx

                # ---- HARD TIME-CAP (user 2026-10-08: close EVERY trade
                # after time_stop_s seconds REGARDLESS of profit or loss).
                # Supersedes the 1-candle cut and the exit-engine
                # hold/stale time-outs so 216s is the single max hold;
                # profit-taking (TP/peak-prot/tp-tighten/ha-flip) still
                # closes winners earlier.
                ts_cap = float(cfg.get("time_stop_s", 0) or 0)
                born0 = float(bgc.get("born") or 0)
                if born0:
                    life0 = time.time() - born0
                    if ts_cap > 0 and life0 >= ts_cap:
                        rc4, cpx4 = bg_close("tstop")
                        print(f"[exit] #{p.ticket} TIME-CAP "
                              f"{ts_cap:.0f}s: held {life0:.0f}s "
                              f"=> flat @ {cpx4} retcode={rc4} "
                              f"(pnl {p.profit:+.2f} regardless)")
                        live_ctx.pop(p.ticket, None)
                        continue
                    end0 = (born0 // 60 + 1) * 60
                    if ts_cap <= 0 and life0 >= 8 and time.time() >= end0 - 2:
                        rc4, cpx4 = bg_close("1candle")
                        print(f"[exit] #{p.ticket} 1-CANDLE CUT: held "
                              f"{life0:.0f}s to the candle edge -> flat "
                              f"@ {cpx4} retcode={rc4}")
                        live_ctx.pop(p.ticket, None)
                        continue

                # ---- EXIT ENGINE (chop-scalp 1-2 candle life) ----
                # Wakes only once the trade has a REAL peak. Manual closes
                # via the cockpit keep priority - engine never blocks them.
                ex = cfg.get("exit_engine") or {}
                if (not bgc.get("reload")) and ex.get("on") \
                        and bg_peak >= float(ex.get("min_peak", 2.0)):
                    try:
                        is_buy = p.type == mt5.POSITION_TYPE_BUY

                        # (1) PEAK PROTECTION: never positive -> negative.
                        if p.profit <= 0.0:
                            rc3, cpx = bg_close("peak-prot")
                            print(f"[exit] #{p.ticket} PEAK-PROT: peak "
                                  f"${bg_peak:+.2f} -> ${p.profit:+.2f} "
                                  f"(never give it back) -> retcode={rc3}")
                            live_ctx.pop(p.ticket, None)
                            continue

                        # (2) TP-TIGHTEN: park the take profit just beyond
                        # the recent candle extreme so the exit fills near
                        # the top/bottom of THIS chop leg.
                        if (not bgc.get("tp_moved")) and len(df_h1) >= 3 \
                                and seed_atr > 0:
                            tick_now = mt5.symbol_info_tick(cfg["symbol"])
                            if tick_now is not None:
                                ext = (max(float(df_h1["high"].iloc[-2]),
                                           float(df_h1["high"].iloc[-3]))
                                       if is_buy else
                                       min(float(df_h1["low"].iloc[-2]),
                                           float(df_h1["low"].iloc[-3])))
                                off = float(ex.get("lock_atr_off", 0.12)) \
                                    * seed_atr
                                gap = max(0.15, 0.05 * seed_atr)
                                if is_buy:
                                    cand = ext + off
                                    cand = max(cand, tick_now.bid + gap)
                                    cand = round(min(cand, p.tp), 2) \
                                        if (p.tp and p.tp > 0) else round(cand, 2)
                                    ok_tp = p.tp == 0 or cand < p.tp - gap
                                else:
                                    cand = ext - off
                                    cand = min(cand, tick_now.ask - gap)
                                    cand = round(max(cand, p.tp), 2) \
                                        if (p.tp and p.tp > 0) else round(cand, 2)
                                    ok_tp = p.tp == 0 or cand > p.tp + gap
                                if ok_tp:
                                    req = {"action": mt5.TRADE_ACTION_SLTP,
                                           "position": p.ticket,
                                           "symbol": cfg["symbol"],
                                           "sl": p.sl or 0.0,
                                           "tp": cand}
                                    r4 = mt5.order_send(req)
                                    rc4 = r4.retcode if r4 else "None"
                                    print(f"[exit] #{p.ticket} TP-TIGHTEN: "
                                          f"peak ${bg_peak:+.2f} -> TP "
                                          f"{(p.tp or 0):.2f}>{cand:.2f} "
                                          f"chop-extreme -> retcode={rc4}")
                                    bgc["tp_moved"] = True
                                    live_ctx[p.ticket] = bgc

                        # (3) RETRACE OUT: gave back retrace_pct of peak.
                        if p.profit > 0 and bg_give > 0 and \
                                bg_give >= float(ex.get("retrace_pct", 30)) / 100.0 * bg_peak:
                            rc3, cpx = bg_close("chop-out")
                            pct3 = bg_give / bg_peak * 100.0
                            print(f"[exit] #{p.ticket} CHOP-OUT: peak "
                                  f"${bg_peak:+.2f} -> ${p.profit:+.2f} "
                                  f"(back ${bg_give:.2f} = {pct3:.0f}%) "
                                  f"banked -> retcode={rc3}")
                            log_bounce({"ticket": p.ticket, "peak": bg_peak,
                                        "floating": p.profit, "give": bg_give,
                                        "pct": pct3, "dips": bg_dips,
                                        "closed": bool(rc3 == mt5.TRADE_RETCODE_DONE),
                                        "retcode": rc3})
                            if rc3 == mt5.TRADE_RETCODE_DONE \
                                    and reload_plan is None:
                                reload_plan = bounce_close_reload_plan(
                                    bgc, p.type, cpx, p.price_open)
                            live_ctx.pop(p.ticket, None)
                            continue

                        # (4) HA-FLIP: last closed HA candle turned against
                        # our side -> this 1-min chop leg is over, get out.
                        if ex.get("ha_flip") and ha_gprev is not None:
                            if (is_buy and not ha_gprev) or \
                                    ((not is_buy) and ha_gprev):
                                rc3, cpx = bg_close("ha-flip")
                                print(f"[exit] #{p.ticket} HA-FLIP: peak "
                                      f"${bg_peak:+.2f} -> ${p.profit:+.2f} "
                                      f"-> HA candle against "
                                      f"{'LONG' if is_buy else 'SHORT'} -> "
                                      f"retcode={rc3}")
                                if rc3 == mt5.TRADE_RETCODE_DONE \
                                        and reload_plan is None:
                                    reload_plan = bounce_close_reload_plan(
                                        bgc, p.type, cpx, p.price_open)
                                live_ctx.pop(p.ticket, None)
                                continue

                        # (5) STALE: 1-2 candle life; weak after hold_max_s. Only
                        # when the flat 216s time-cap is OFF (the cap is
                        # the single time exit).
                        born = float(bgc.get("born", 0) or 0)
                        if ts_cap <= 0 and born and time.time() - born > \
                                float(ex.get("hold_max_s", 135)):
                            if p.profit < float(ex.get("stale_frac", 0.25)) * bg_peak:
                                rc3, cpx = bg_close("stale-1-2c")
                                print(f"[exit] #{p.ticket} STALE: "
                                      f"{time.time()-born:.0f}s in, "
                                      f"${p.profit:+.2f} vs peak "
                                      f"${bg_peak:+.2f} -> out -> retcode={rc3}")
                                live_ctx.pop(p.ticket, None)
                                continue
                    except Exception as _ex_err:
                        print(f"[exit-engine] error "
                              f"#{getattr(p, 'ticket', '?')}: {_ex_err}")

                # DOUBLE-TRADE step 2 (only for the bounce ride itself):
                # exit when the up/bounce ACCELERATION dies (the small
                # pause), then immediately plan the re-ride to the
                # original destination TP; a dead bounce closes silently
                if bgc.get("reload") and (cfg.get("exit_engine") or {}).get("on", True) \
                        and bg_peak >= 2.0 and p.profit <= 0.0:
                    rc3, cpx = bg_close("peak-prot")
                    print(f"[exit] #{p.ticket} PEAK-PROT (reload ride): peak "
                          f"${bg_peak:+.2f} -> ${p.profit:+.2f} -> retcode={rc3}")
                    live_ctx.pop(p.ticket, None)
                    continue
                if bgc.get("reload") and time.time() - float(bgc.get("born", 0)) > 10:
                    hist = bgc.setdefault("hist", [])
                    hist.append(p.profit)
                    if len(hist) > 12:
                        bgc["hist"] = hist[-12:]
                    if (time.time() - float(bgc.get("born", 0))
                            > RELOAD_CFG["dead_min"]):
                        rc3, cpx = bg_close("reload-dead")
                        print(f"[reload] dead bounce #{p.ticket} "
                              f"{p.profit:+.2f} -> closed (retcode={rc3})")
                        live_ctx.pop(p.ticket, None)
                        continue
                    if (p.profit >= RELOAD_CFG["min_profit"] and len(hist) >= 3
                            and (hist[-1] - hist[-2]) <= RELOAD_CFG["accel_stall"]
                            and (hist[-2] - hist[-3]) <= RELOAD_CFG["accel_stall"]):
                        rc3, cpx = bg_close("reload-accel-dead")
                        print(f"[reload] bounce slowing #{p.ticket} "
                              f"{p.profit:+.2f} (2 stalls) -> retcode={rc3} "
                              f"-> re-riding the ORIGINAL side")
                        if rc3 != "None":
                            reload_plan = {"dir": ("SHORT"
                                                   if p.type == 0 else "LONG"),
                                           "px_close": cpx,
                                           "dst_tp": float(bgc.get("src_entry", 0)) or None,
                                           "re_ride": True,
                                           "t_due": time.time()}
                        live_ctx.pop(p.ticket, None)
                        continue
                if bg_peak > BOUNCE_GUARD["min_peak"] and p.profit > 0:
                    pct = bg_give / bg_peak * 100.0
                    hard = bg_give >= BOUNCE_GUARD["frac_hard"] * bg_peak
                    soft = (bg_give >= BOUNCE_GUARD["frac_soft"] * bg_peak
                            and bg_dips >= BOUNCE_GUARD["dip_min"])
                    if hard or soft:
                        ctype = (mt5.ORDER_TYPE_SELL
                                 if p.type == mt5.POSITION_TYPE_BUY
                                 else mt5.ORDER_TYPE_BUY)
                        tick_now = mt5.symbol_info_tick(cfg["symbol"])
                        if tick_now is None:
                            continue
                        px = (tick_now.bid
                              if ctype == mt5.ORDER_TYPE_SELL else tick_now.ask)
                        req = {"action": mt5.TRADE_ACTION_DEAL,
                               "symbol": cfg["symbol"], "volume": p.volume,
                               "type": ctype, "position": p.ticket,
                               "price": px, "deviation": 30, "magic": p.magic,
                               "comment": "bounce-guard",
                               "type_time": mt5.ORDER_TIME_GTC,
                               "type_filling": fill_mode}
                        res = mt5.order_send(req)
                        rc = res.retcode if res else "None"
                        print(f"[bounce] #{p.ticket} PEAK ${bg_peak:+.2f} -> "
                              f"${p.profit:+.2f} (give-back ${bg_give:.2f} = "
                              f"{pct:.0f}%, {bg_dips} down-steps "
                              f"{'HARD' if hard else 'accel'}) -> taking the "
                              f"money -> retcode={rc}")
                        log_bounce({"ticket": p.ticket, "peak": bg_peak,
                                    "floating": p.profit, "give": bg_give,
                                    "pct": pct, "dips": bg_dips,
                                    "closed": bool(res and
                                                   res.retcode ==
                                                   mt5.TRADE_RETCODE_DONE),
                                    "retcode": rc})
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE \
                                and reload_plan is None:
                            reload_plan = bounce_close_reload_plan(
                                bgc, p.type, px, p.price_open)
                            if reload_plan:
                                print(f"[reload] bounce-side ride planned: "
                                      f"{reload_plan['dir']} from the "
                                      f"give-back, then re-ride to "
                                      f"destination")
                        live_ctx.pop(p.ticket, None)
                        continue
                bgc["peak"] = max(bg_peak, p.profit)
                bgc["last_p"] = p.profit
                bgc["dips"] = bg_dips
                live_ctx[p.ticket] = bgc

            if auto_close and reading.should_flatten:
                for p in positions:
                    close_type = (mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY
                                  else mt5.ORDER_TYPE_BUY)
                    tick_now = mt5.symbol_info_tick(cfg["symbol"])
                    px = tick_now.bid if close_type == mt5.ORDER_TYPE_SELL else tick_now.ask
                    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": cfg["symbol"],
                           "volume": p.volume, "type": close_type, "position": p.ticket,
                           "price": px, "deviation": 30, "magic": magic,
                           "comment": "radar FLATTEN",
                           "type_time": mt5.ORDER_TIME_GTC,
                           "type_filling": fill_mode}
                    res = mt5.order_send(req)
                    rc = res.retcode if res else "None"
                    cm = res.comment if res else "no result"
                    print(f"FLATTEN close #{p.ticket}: retcode={rc} comment={cm!r}")

            elif reading.can_open_new and len(positions) < min(4, cfg["max_concurrent"]):
                # signal latch: fire on the bar close, then KEEP retrying
                # while that same closed bar is still the signal bar. A
                # transient broker condition (Algo Trading off, network
                # blip, requote) used to throw the whole setup away - the
                # bot then idled until the next cross, minutes or an hour
                # later, with the setup silently lost.
                lc_brake = float(cfg.get("loss_cooldown_s", 60) or 0)
                if lc_brake and last_loss_ts \
                        and time.time() - last_loss_ts < lc_brake:
                    # 09:01 lesson: a violent flush swallows every revenge
                    # entry stacked right after a loss - silence the latch.
                    sig_now = None
                else:
                    sig_now = signal(df_h1, cfg)
                # hreg is computed later in this cycle (regime gate); bind
                # a default so an early burst signal can never hit an
                # UnboundLocalError before the first ha_regime() run.
                if "hreg" in locals():
                    hreg = hreg
                else:
                    hreg = None
                if sig_now and hreg is not None:
                    # HA regime veto: never burst AGAINST a dominant army
                    if hreg["mode"] == "RED-TREND" and sig_now == "LONG":
                        sig_now = None
                        if last_ha_mode != "VETO-LONG":
                            print("[ha] VETO LONG burst - 200c HA is "
                                  f"{hreg['ratio'] * 100:.0f}% red "
                                  "({0} run {1})".format(hreg["mode"],
                                                         hreg["run"]))
                            last_ha_mode = "VETO-LONG"
                    elif hreg["mode"] == "GREEN-TREND" and sig_now == "SHORT":
                        sig_now = None
                        if last_ha_mode != "VETO-SHORT":
                            print("[ha] VETO SHORT burst - 200c HA is "
                                  f"{hreg['ratio'] * 100:.0f}% green "
                                  "(run {0})".format(hreg["run"]))
                            last_ha_mode = "VETO-SHORT"
                # PERMISSION GATE: chop says "not now" regardless of side
                if sig_now and not regime_gate_allows(df_h1, cfg):
                    sig_now = None
                if sig_now:
                    if (pending is None or pending["bar"] != current_bar
                            or pending["sig"] != sig_now):
                        pending = {"sig": sig_now, "bar": current_bar,
                                   "tries": 0}
                else:
                    pending = None
                sig = None
                if (pending is not None and pending["bar"] == current_bar
                        and pending["sig"] == sig_now
                        and (new_bar or pending["tries"] > 0)):
                    sig = pending["sig"]
                if sig:
                    # entry quality gates: only take qualified setups
                    ok_gate, whys = evaluate_gates(cfg, df_h1)
                    if not ok_gate:
                        gate_note = f"{sig}: " + "; ".join(whys)
                        if gate_note != last_gate_note:
                            print(f"  [gate] SKIP {gate_note}")
                            last_gate_note = gate_note
                        sig = None
                        pending = None   # not a retry: the setup isn't qualified
                    elif last_gate_note:
                        print("  [gate] conditions now satisfied")
                        last_gate_note = None
                if sig:
                    atr = float(rolling_atr(df_h1).iloc[-1])
                    stop_dist = cfg["atr_stop_mult"] * atr
                    acc_now = mt5.account_info()
                    if acc_now is None or sym is None:
                        print("[warn] account/symbol unavailable - signal skipped, retry next bar")
                        sig = None
                        pending = None
                        risk_amt = 0.0
                        lots = 0.0
                    else:
                        risk_amt = acc_now.balance * cfg["risk_pct"] / 100.0
                        lots = size_lots(acc_now.balance, risk_amt, stop_dist, cfg)
                    ctx = build_entry_ctx(sig, reading, df_h1, cfg,
                                          live_spread,
                                          datetime.now().hour, price)
                    if lots < (sym.volume_min if sym is not None else 0.01):
                        pending = None   # sizing impossible - do not latch
                    if sig and lots >= sym.volume_min:
                        if engine.learner is not None:
                            ok_gate, why = engine.learner.gate(ctx)
                            if not ok_gate:
                                print(f"  [learner] FILTERED {sig} entry -> {why}")
                                sig = None
                                pending = None   # a veto is final, not a retry
                    if sig:
                        # SPREAD-CAPTURE LADDER: the main cross entry parks
                        # layered LIMITs on the spread-favorable side rather
                        # than paying the cross with one market order. If
                        # the ladder is on and places, the market send below
                        # is skipped for this bar; if it is blocked/no-op,
                        # the signal is swallowed (never a surprise market
                        # fill after the ladder decided the spread was the
                        # enemy).
                        if LADDER_CFG.get("on") and not LADDER_STATE["active"]:
                            _rr_use = float(cfg.get("rr",
                                                    cfg["rr_ratio"]) or 2)
                            _np = place_spread_ladder(sig, stop_dist,
                                                      _rr_use, atr)
                            if _np:
                                print(f"[ladder] bar signal {sig} -> "
                                      f"{_np} layered limits parked "
                                      f"(no market cross)")
                                _vg0 = VANTAGE_CFG
                                if _vg0.get("on"):
                                    _old_v = VANTAGE_STATE.get("order")
                                    if _old_v is not None:
                                        _rmo = mt5.order_send(
                                            {"action":
                                             mt5.TRADE_ACTION_REMOVE,
                                             "order": _old_v["ticket"]})
                                        if _rmo and _rmo.retcode == \
                                                mt5.TRADE_RETCODE_DONE:
                                            print("[vantage] prior reserve "
                                                  "superseded by new signal")
                                        VANTAGE_STATE["order"] = None
                                    VANTAGE_STATE["scheduled"] = {
                                        "dir": sig, "stop_dist": stop_dist,
                                        "rr": _rr_use, "ref": price,
                                        "born": time.time(),
                                        "arm_at": time.time() + float(
                                            _vg0.get("delay_s", 120) or 0),
                                        "sig_id": (str(current_bar), sig)}
                                    VANTAGE_STATE["scratched_sig"] = None
                                    print(f"[vantage] {sig} reserve "
                                          f"scheduled (parks in "
                                          f"{float(_vg0.get('delay_s', 120) or 0):.0f}s "
                                          f"near the stop shelf)")
                            sig = None
                            pending = None
                    if sig:
                        known = {p.ticket for p in positions}
                        ok = False
                        # retry: a stale price / requote used to kill the trade
                        # for a whole hour (signals fire once per new H1 bar)
                        for attempt in range(1, 4):
                            tick2 = mt5.symbol_info_tick(cfg["symbol"])
                            if tick2 is None:
                                print("no tick, cannot send")
                                break
                            if sig == "LONG":
                                soft_sl = tick2.ask - stop_dist
                                tp = tick2.ask + stop_dist * cfg["rr_ratio"]
                                otype, px = mt5.ORDER_TYPE_BUY, tick2.ask
                            else:
                                soft_sl = tick2.bid + stop_dist
                                tp = tick2.bid - stop_dist * cfg["rr_ratio"]
                                otype, px = mt5.ORDER_TYPE_SELL, tick2.bid
                            # wick-proof: broker SL sits at the disaster
                            # distance; the soft SL is managed by
                            # close-confirm in the loop (wicks are held)
                            if cfg.get("wick_proof_stop"):
                                dm = float(cfg.get("disaster_stop_mult", 1.5))
                                sl = (px - stop_dist * dm) if sig == "LONG" \
                                    else (px + stop_dist * dm)
                            else:
                                sl = soft_sl
                            ctx["soft_sl"] = soft_sl
                            ctx["sd"] = stop_dist      # for +R breakeven math
                            ctx["born"] = time.time()   # 1-2 candle life clock
                            if not auto_close:
                                # manual mode: entry goes in NAKED - only
                                # the user closes it
                                sl = 0.0
                                tp = 0.0
                            req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": cfg["symbol"],
                                   "volume": lots, "type": otype, "price": px,
                                   "deviation": 30,
                                   "sl": round(sl, 2), "tp": round(tp, 2),
                                   "magic": magic, "comment": "godmode v2",
                                   "type_time": mt5.ORDER_TIME_GTC,
                                   "type_filling": fill_mode}
                            res = mt5.order_send(req)
                            ok = res and res.retcode == mt5.TRADE_RETCODE_DONE
                            if ok:
                                print(f"OPEN {sig} {lots} lots @ {px:.2f} "
                                      f"SL {sl:.2f} TP {tp:.2f} -> ok (attempt {attempt})")
                                pending = None
                                break
                            rc = res.retcode if res else "None"
                            cm = res.comment if res else "no result"
                            print(f"OPEN {sig} attempt {attempt} -> rejected "
                                  f"retcode={rc} comment={cm!r}")
                            # TRANSIENT broker conditions must not eat the
                            # setup: keep it latched and retry next cycle.
                            # (10027 Algo Trading off, 10031 no connection,
                            #  10030/10025/10032/10039 price/quote churn)
                            transient = {10024, 10025, 10027, 10030, 10031,
                                         10032, 10033, 10039, None}
                            if rc in transient:
                                pending["tries"] = pending.get("tries", 0) + 1
                                print(f"  [pending] {sig} signal HELD - broker is "
                                      f"unavailable ({cm!r}); will re-try in "
                                      f"{cfg['cycle_seconds']}s on the same bar")
                                break
                            else:
                                print(f"  [pending] {sig} signal DROPPED - "
                                      f"permanent rejection, not retrying")
                                pending = None
                            time.sleep(0.5)
                        if ok:
                            time.sleep(0.3)
                            for lp in (mt5.positions_get(symbol=cfg["symbol"]) or []):
                                if lp.magic == magic and lp.ticket not in known:
                                    live_ctx[lp.ticket] = ctx

            else:
                pending = None   # radar says stand down / book is full:
                                 # a held signal is no longer actionable

            hreg = ha_regime(df_h1)
            if hreg is not None and hreg["mode"] != last_ha_mode:
                print(f"[ha] 200c HA {hreg['ratio'] * 100:.0f}% green "
                      f"| run {hreg['run']} | slope {hreg['slope']:+.1f} "
                      f"-> {hreg['mode']}"
                      + ("" if hreg["fade_ok"] else
                         " - fade brains SILENCED (trend regime)"))
                last_ha_mode = hreg["mode"]
            fade_ok = (hreg is None) or bool(hreg["fade_ok"])
            if hreg is None:
                last_ha_mode = ""   # short data; don't lock the print

            # AUX BRAINS (auto mode): 1) trap-fade (recon-gated) 2) army
            # ride on HTF Heiken Ashi momentum 3) round-number level fade.
            # Same-bar stacking is INTENTIONAL - the ensemble who beat
            # every single brain (18/18 24h missions, median 3.7h).
            entry_mode_now = str(cfg.get("entry_mode", "cross")).lower()
            # HTF army gauge runs EVERY cycle (dashboard tide narration)
            ga30 = ga4h = None
            if entry_mode_now == "auto" and df30 is not None:
                ga30 = army_gauge(heiken_ashi(df30))
                ga4h = army_gauge(heiken_ashi(df_h4))
                note = (f"30M {'GREEN' if ga30['green'] else 'RED'} "
                        f"x{abs(ga30['consec'])} {ga30['accel']:+.2f} | "
                        f"4H {'GREEN' if ga4h['green'] else 'RED'} "
                        f"x{abs(ga4h['consec'])} {ga4h['accel']:+.2f}")
                if note != last_army_note:
                    print(f"[army] {note} - "
                          f"{'green army pressing' if ga30['green'] and ga4h['green'] else 'red army pressing' if (not ga30['green']) and (not ga4h['green']) else 'armies clashing (chop risk)'}")
                    last_army_note = note
            # HTF LOCK -> ADVISORY (heartbeat doctrine 2026-10-08: the
            # chart moves in an S-shape heartbeat - we trade ALL
            # directions; htf_state is only used for narration + the
            # ultra brain's agree/mixed read, never as a side-lock).
            _tnb = htf_bias(df30, df_h4) if entry_mode_now == "auto" \
                else None
            _ghs = _tnb if _tnb is not None \
                else {"bias": None, "red30": None, "red4h": None,
                      "slope4h": 0.0}
            globals()["htf_state"] = _ghs
            if _ghs["bias"] != globals().get("htf_old"):
                globals()["htf_old"] = _ghs["bias"]
                _r30 = ("RED" if _ghs["red30"] else "GREEN") \
                    if _ghs["red30"] is not None else "??"
                _r4h = ("RED" if _ghs["red4h"] else "GREEN") \
                    if _ghs["red4h"] is not None else "??"
                if _ghs["bias"]:
                    print(f"[htf] tide: {_ghs['bias']} "
                          f"(30m {_r30}, 4h {_r4h} "
                          f"slope {_ghs['slope4h']:+.1f}) - "
                          f"advisory (all directions still trade)")
                else:
                    print(f"[htf] HTF lock released "
                          f"(30m {_r30}, 4h {_r4h} "
                          f"slope {_ghs['slope4h']:+.1f}) - "
                          f"both sides open")
            svz = session_vwap_zone(df_h1)
            if svz is not None:
                bkt = round(svz["sig"], 1)
                if bkt != last_vwap_bkt:
                    upper = svz["vwap"] + 2.01 * svz["sd"]
                    lower = svz["vwap"] - 2.01 * svz["sd"]
                    print(f"[vwap] session sigma {svz['sig']:+.2f} | "
                          f"vwap {svz['vwap']:.2f} | upper 2.01sig "
                          f"{upper:.2f} | lower 2.01sig {lower:.2f}")
                    last_vwap_bkt = bkt
            # FOMO phase watcher (user's movement-animation thesis)
            fph = fomo_phase(df_h1)
            if fph is not None and fph["phase"] != last_fomo_phase:
                print(f"[fomo] {fph['phase']}"
                      + (f" {fph['dir']}" if fph["dir"] else "")
                      + (f" (depth {fph['depth_sig']:.2f}x)"
                         if fph["dir"] else "")
                      + f" | net {fph['net_sig']:+.1f} sigma")
                last_fomo_phase = fph["phase"]
                append_fomo(fph)
            if (entry_mode_now == "auto"
                    and reading.can_open_new
                    and len(positions) < cfg["max_concurrent"]):
                bar_now = df_h1.iloc[-1]
                atr_s3 = rolling_atr(df_h1)
                av3 = (float(atr_s3.iloc[-1]) if atr_s3 is not None
                       and not pd.isna(atr_s3.iloc[-1]) else None)
                cur_close = float(bar_now["close"])
                cur_high = float(bar_now["high"])
                cur_low = float(bar_now["low"])

                def place_brain_limit(direction, sl, tp, tag, peg_mult,
                                      extra=None, level=None, lots=None):
                    """BOW-AND-ARROW: park a limit order slightly away
                    from the market instead of paying the cross.
                    BUY LIMIT sits peg*ATR under the ask; SELL LIMIT sits
                    peg*ATR above the bid - OR at an EXACT level (sine
                    pivots pass level=). The peg dies after ttl_s or
                    when the next candle opens (sweep); registered ctx
                    on fill so every manager still works."""
                    if av3 is None or av3 <= 0:
                        return None
                    # HEARTBEAT: HTF lock advisory only - all directions
                    # allowed through the limit path.
                    if (fom_active or ultra_active) \
                            and tag not in ("fomorian", "ultra", "vwapfade"):
                        return None
                    if not regime_gate_allows(df_h1, cfg, tag):
                        return None
                    if not auto_close:
                        sl = 0.0
                        tp = 0.0
                    for attempt in range(1, 3):
                        tick2 = mt5.symbol_info_tick(cfg["symbol"])
                        acc2 = mt5.account_info()
                        if tick2 is None:
                            return None
                        peg = peg_mult * av3
                        use_mid = tag in LIMIT_CFG.get("mid_tags", ())
                        mid_c = None
                        try:
                            fh = float(bars["high"][-1])
                            flw = float(bars["low"][-1])
                            mid_c = (fh + flw) / 2.0
                        except Exception:
                            mid_c = None
                        if direction == "LONG":
                            if level is not None:
                                px_lim = level
                            elif use_mid and mid_c is not None:
                                px_lim = mid_c
                            else:
                                px_lim = tick2.ask - peg
                            if use_mid and mid_c is not None:
                                # puffer-fish: never out of the fight -
                                # mid-candle or deeper, whichever is closer
                                # to the market (fill sooner)
                                px_lim = (max(px_lim, mid_c)
                                          if level is not None else px_lim)
                            otype2 = mt5.ORDER_TYPE_BUY_LIMIT
                            px_lim = min(px_lim, tick2.ask - 0.05)
                            if px_lim >= tick2.ask - 0.05:
                                return None
                            dist = (px_lim if isinstance(sl, (int, float))
                                    else tick2.ask) - sl
                        else:
                            if level is not None:
                                px_lim = level
                            elif use_mid and mid_c is not None:
                                px_lim = mid_c
                            else:
                                px_lim = tick2.bid + peg
                            if use_mid and mid_c is not None:
                                px_lim = (min(px_lim, mid_c)
                                          if level is not None else px_lim)
                            otype2 = mt5.ORDER_TYPE_SELL_LIMIT
                            px_lim = max(px_lim, tick2.bid + 0.05)
                            if px_lim <= tick2.bid + 0.05:
                                return None
                            dist = sl - px_lim
                        if dist <= 0.25:
                            return None
                        tp2 = tp
                        if tp2 is None:
                            tp2 = (px_lim + dist * 3.0 if direction == "LONG"
                                   else px_lim - dist * 3.0)
                        accbal = acc2.balance if acc2 is not None else 100.0
                        lots_t = (lots if lots is not None else
                                  size_lots(accbal,
                                            accbal * cfg["risk_pct"] / 100.0,
                                            dist, cfg))
                        sym2 = mt5.symbol_info(cfg["symbol"])
                        if lots_t < sym2.volume_min:
                            print(f"  [{tag}] limit sizing impossible")
                            return None
                        req = {"action": mt5.TRADE_ACTION_PENDING,
                               "symbol": cfg["symbol"],
                               "volume": lots_t, "type": otype2,
                               "price": round(px_lim, 2),
                               "deviation": 30,
                               "sl": round(sl, 2), "tp": round(tp2, 2),
                               "magic": magic, "comment": tag,
                               "type_time": mt5.ORDER_TIME_GTC,
                               "type_filling": (mt5.ORDER_FILLING_RETURN
                                                if attempt == 1
                                                else fill_mode)}
                        res2 = mt5.order_send(req)
                        if res2 and res2.retcode == mt5.TRADE_RETCODE_DONE:
                            tp2 = round(tp2, 2)
                            if getattr(res2, "deal", 0):
                                known2 = {p.ticket for p in positions}
                                time.sleep(0.3)
                                for lp in (mt5.positions_get(symbol=cfg["symbol"]) or []):
                                    if lp.magic == magic and lp.ticket not in known2:
                                        cd2 = {tag: True, "born": time.time(),
                                               "soft_sl": None}
                                        cd2.update(extra or {})
                                        live_ctx[lp.ticket] = cd2
                                return (lots_t, px_lim, tp2)
                            ot = getattr(res2, "order", 0)
                            if ot:
                                brain_orders[ot] = {"tag": tag,
                                                    "dir": direction,
                                                    "bar": current_bar,
                                                    "born": time.time(),
                                                    "peg_px": px_lim,
                                                    "extra": extra or {}}
                            print(f"[peg] {tag} {direction} LIMIT parked "
                                  f"@ {px_lim:.2f} SL {sl:.2f} TP {tp2:.2f} "
                                  f"(ttl {LIMIT_CFG['ttl_s']}s)")
                            return (lots_t, px_lim, tp2)
                        rc2 = res2.retcode if res2 else "None"
                        cm2 = res2.comment if res2 else "no result"
                        print(f"[peg] {tag} {direction} attempt {attempt} "
                              f"-> rejected retcode={rc2} comment={cm2!r}")
                    return None

                def send_brain(direction, sl, tp, tag, rr=None, extra=None,
                               peg=None, level=None, lots=None):
                    """entry sender: MARKET ALWAYS (user 2026-10-07: change
                    the limit buys to market buys - the pegs missed fills).
                    Every brain fills at the ask/bid with 3-retry; peg/level
                    params remain accepted but are IGNORED (revert later by
                    flipping LIMIT_CFG["use_limits"] back to True)."""
                    if fom_active and tag != "fomorian":
                        # manual martingale keeps the board pure: only the
                        # doubling game may enter while fomorian_mode is ON
                        return None
                    if not auto_close:
                        # manual mode: entries carry NO stops - the user
                        # decides every exit
                        sl = 0.0
                        tp = 0.0
                    lpegs = peg if peg is not None \
                        else LIMIT_CFG["tag_peg"].get(tag)
                    # ULTRA MODE: while ultra_mode is ON only the ultra
                    # brain may enter through this chokepoint.
                    if (fom_active or ultra_active) \
                            and tag not in ("fomorian", "ultra", "vwapfade"):
                        return None
                    if not regime_gate_allows(df_h1, cfg, tag):
                        return None
                    # HEARTBEAT DOCTRINE (user 2026-10-08: "chart moves in a
                    # constant heartbeat / S-shape - we trade in ALL
                    # directions"): the 30m+4h lock is ADVISORY now, never
                    # a hard side-lock.
                    _hs = globals().get("htf_state") or {}
                    # optimizer freeze (user spec 2026-10-08): after a
                    # failing 10-min window all entries stand down for the
                    # stated period so the bot re-learns instead of revenge-trades.
                    if "opt_state" in globals() and tag != "fomorian":
                        _opg = globals()["opt_state"]
                        if time.time() < _opg.get("frozen_until", 0):
                            if _opg.get("frozen_logged") != _opg.get("frozen_until"):
                                _opg["frozen_logged"] = _opg["frozen_until"]
                                print("[optimize] entry FREEZE active "
                                      "(failure window) - all entries "
                                      "stand down while it re-learns")
                            return None
                    momentum = tag in LIMIT_CFG.get("market_tags", ())
                    if (LIMIT_CFG.get("use_limits", True)
                            and not momentum
                            and (lpegs is not None or level is not None)):
                        return place_brain_limit(direction, sl, tp, tag,
                                                 lpegs, extra, level, lots)
                    # HEDGE GUARD (user 2026-10-08: "the 2 current trades
                    # opened the same time long and short"): a live opposite
                    # leg is pure dead cost (2x spread for zero net edge).
                    # Reversal-family tags flip a trade that is already
                    # CLOSED (no opposite position live) so they are exempt.
                    if tag not in ("fomo", "fomorian", "ultra",
                                   "reload", "re-ride"):
                        for _p in positions:
                            _po = "LONG" if _p.type == 0 else "SHORT"
                            if _po != direction:
                                print(f"[hedge] {tag} {direction} BLOCKED - "
                                      f"#{_p.ticket} {_po} still open "
                                      f"(netting = 2x spread, no edge)")
                                return None
                    # COST-AWARE ROUTING (meter + user 2026-10-08): measure
                    # the LIVE spread + realized slippage at order time; when
                    # the round-trip cost eats a meaningful slice of the
                    # per-trade risk, DON'T pay the cross - park a LIMIT
                    # instead so the fill waits for a tighter spread.
                    cost_cfg = cfg.get("cost", {}) or {}
                    sym_ck = mt5.symbol_info(cfg["symbol"])
                    tick_ck = mt5.symbol_info_tick(cfg["symbol"])
                    acc_ck = mt5.account_info()
                    if sym_ck is None or tick_ck is None or acc_ck is None:
                        return None
                    ct = float(getattr(sym_ck, "trade_contract_size", 100.0)
                               or 100.0)
                    spread_pts = float(sym_ck.spread or 0.0)
                    spread_usd = spread_pts * float(sym_ck.point or 0.01) * ct
                    slip_dir = "ask" if direction == "LONG" else "bid"
                    slip_usd = float(LIVE_METER["slip"].get(slip_dir, 0.0))
                    px_far = tick_ck.ask if direction == "LONG" else tick_ck.bid
                    dist_far = (px_far - sl) if direction == "LONG" \
                        else (sl - px_far)
                    if dist_far <= 0.25:
                        return None
                    rr_u2 = (rr if (rr is not None and rr > 0)
                             else float(cfg.get("rr", 2) or 2))
                    tp_far = (px_far + (dist_far * rr_u2 if direction == "LONG"
                                        else -dist_far * rr_u2)
                              ) if tp is None else tp
                    lots_ck = (lots if lots is not None else size_lots(
                        acc_ck.balance,
                        acc_ck.balance * cfg["risk_pct"] / 100.0,
                        dist_far, cfg))
                    if lots_ck < sym_ck.volume_min:
                        lots_ck = sym_ck.volume_min
                    rt_usd = (spread_usd + 2.0 * slip_usd) * lots_ck
                    risk_usd = max(acc_ck.balance * cfg["risk_pct"] / 100.0,
                                   0.01)
                    max_rt = float(cost_cfg.get("max_rt_pct", 0.20)) * risk_usd
                    min_rt = float(cost_cfg.get("min_rt_usd", 0.25))
                    max_rt = max(max_rt, min_rt)
                    spread_bad = rt_usd > max_rt
                    LIVE_METER["last"] = {
                        "spread_pts": spread_pts,
                        "spread_usd": round(spread_usd, 3),
                        "slip_usd": round(slip_usd, 3),
                        "rt_usd": round(rt_usd, 3),
                        "risk_usd": round(risk_usd, 3),
                        "budget_pct": round(max_rt / risk_usd * 100 if risk_usd else 0, 1),
                        "route": "limit" if spread_bad else "market",
                        "tag": tag, "dir": direction,
                        "lots": lots_ck, "ts": time.time(),
                    }
                    if spread_bad and not momentum:
                        LIVE_METER["rises"] += 1
                        print(f"[cost] {tag} {direction} spread {spread_pts:.0f}pt "
                              f"=${spread_usd:.2f}/lot + slip ~${slip_usd:.2f}/lot "
                              f"=> rt ${rt_usd:.2f} > ${max_rt:.2f} budget "
                              f"- parking LIMIT (wait for tight spread)")
                        return place_brain_limit(
                            direction, sl, tp, tag,
                            float(cost_cfg.get("limit_peg_atr", 0.04)),
                            extra, level, lots)
                    if not spread_bad:
                        LIVE_METER["market"] += 1
                    for t_att in range(1, 4):
                        tick2 = mt5.symbol_info_tick(cfg["symbol"])
                        acc2 = mt5.account_info()
                        if tick2 is None or acc2 is None:
                            return None
                        px2 = (tick2.ask if direction == "LONG"
                               else tick2.bid)
                        dist = (px2 - sl) if direction == "LONG" else (sl - px2)
                        if dist <= 0.25:
                            return None
                        # rr can arrive None (arms that let the exit-engine
                        # manage the out) - default to the config RR so a
                        # tp=None entry never crashes with dist * None.
                        rr_u = (rr if (rr is not None and rr > 0)
                                else float(cfg.get("rr", 2) or 2))
                        tp2 = (px2 + (dist * rr_u if direction == "LONG"
                                      else -dist * rr_u)) if tp is None else tp
                        lots_t = (lots if lots is not None else
                                  size_lots(acc2.balance,
                                            acc2.balance * cfg["risk_pct"] / 100.0,
                                            dist, cfg))
                        one = mt5.symbol_info(cfg["symbol"])
                        if lots_t < one.volume_min:
                            print(f"  [{tag}] sizing impossible "
                                  f"({direction})")
                            return None
                        otype2 = (mt5.ORDER_TYPE_BUY if direction == "LONG"
                                  else mt5.ORDER_TYPE_SELL)
                        req2 = {"action": mt5.TRADE_ACTION_DEAL,
                                "symbol": cfg["symbol"], "volume": lots_t,
                                "type": otype2, "price": px2, "deviation": 30,
                                "sl": round(sl, 2), "tp": round(tp2, 2),
                                "magic": magic, "comment": tag,
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": fill_mode}
                        res2 = mt5.order_send(req2)
                        if res2 and res2.retcode == mt5.TRADE_RETCODE_DONE:
                            # realized slippage vs the price we asked for:
                            # EWMA per side, shown on the dashboard meter
                            slip_fill = abs(res2.price - px2) * ct
                            old_slip = float(LIVE_METER["slip"].get(slip_dir, 0.0))
                            LIVE_METER["slip"][slip_dir] = \
                                0.8 * old_slip + 0.2 * slip_fill
                            LIVE_METER["last"]["fill_slip_usd"] = \
                                round(slip_fill, 3)
                            LIVE_METER["last"]["fill_px"] = round(res2.price, 2)
                            known2 = {p.ticket for p in positions}
                            time.sleep(0.3)
                            for lp in (mt5.positions_get(symbol=cfg["symbol"]) or []):
                                if lp.magic == magic and lp.ticket not in known2:
                                    cd2 = {tag: True, "born": time.time(),
                                           "soft_sl": None}
                                    cd2.update(extra or {})
                                    live_ctx[lp.ticket] = cd2
                            return (lots_t, px2, tp2)
                        rc2 = res2.retcode if res2 else "None"
                        cm2 = res2.comment if res2 else "no result"
                        print(f"[{tag}] {direction} attempt {t_att} -> "
                              f"rejected retcode={rc2} comment={cm2!r}")
                        if rc2 in {10024, 10025, 10027, 10030, 10031, 10032,
                                   10033, 10039, None}:
                            return None
                        time.sleep(0.5)
                    return None

                # DOUBLE-TRADE step 1: the bounce-ride (reverse of the
                # trade the bounce-guard just closed) - entered from the
                # give-back; step 2 (re-ride to destination) is planned
                # by the reload-exit logic below on manager stall.
                if reload_plan is not None and av3 is not None and (time.time()
                        >= reload_plan.get("t_due", 0)):
                    rp = reload_plan
                    src = rp.get("src_entry", 0.0)
                    px_c = rp.get("px_close", cur_close)
                    if rp.get("re_ride"):
                        rel_sl = (px_c + RELOAD_CFG["reride_sl_atr"] * av3
                                  if rp["dir"] == "SHORT"
                                  else px_c - RELOAD_CFG["reride_sl_atr"] * av3)
                        rel_tp = rp.get("dst_tp") or None
                        got = send_brain(rp["dir"], rel_sl, rel_tp,
                                         "re-ride",
                                         extra={"reride": True,
                                                "born": time.time(),
                                                "soft_sl": None})
                    else:
                        rel_sl = (px_c - RELOAD_CFG["sl_atr"] * av3
                                  if rp["dir"] == "LONG"
                                  else px_c + RELOAD_CFG["sl_atr"] * av3)
                        cap = 4.0 * av3
                        if rp["dir"] == "LONG":
                            rel_tp = (min(src, px_c + cap) if src else None)
                            # learned from live: when the source entry is
                            # too close to the close price the TP would be
                            # razor-thin (RR < 1) - fall back to a real
                            # 2.5-ATR target instead
                            if rel_tp is not None and (rel_tp - px_c) < 1.2 * av3:
                                rel_tp = px_c + 2.5 * av3
                        else:
                            rel_tp = (max(src, px_c - cap) if src else None)
                            if rel_tp is not None and (px_c - rel_tp) < 1.2 * av3:
                                rel_tp = px_c - 2.5 * av3
                        got = send_brain(rp["dir"], rel_sl, rel_tp, "reload",
                                         extra={"reload": True,
                                                "src_entry": src,
                                                "born": time.time(),
                                                "soft_sl": None})
                    if got:
                        print(f"[reload] {rp['dir']} bounce-side order "
                              f"{got[0]} lots @ {got[1]:.2f} SL {rel_sl:.2f} "
                              f"TP {got[2]:.2f} "
                              f"({'re-ride to destination' if rp.get('re_ride') else 'riding the bounce'})")
                    else:
                        print("[reload] bounce-side rejected - dropping plan")
                    reload_plan = None

                def arm_entry(tag, direction, nice_px, sl=None, tp=None,
                              extra=None, lots=None):
                    """PATIENCE LAYER: no order leaves instantly. The
                    armed watch waits for the tape to come to the nice
                    price, then fires the MARKET order (3-retry)."""
                    if (fom_active or ultra_active) \
                            and tag not in ("fomorian", "ultra", "vwapfade"):
                        return None
                    if not regime_gate_allows(df_h1, cfg, tag):
                        return None
                    armed_entries[f"{tag}|{direction}"] = {
                        "tag": tag, "dir": direction, "nice": nice_px,
                        "born": time.time(), "ttl": ARMED_TTL.get(tag, 300),
                        "sl": sl, "tp": tp, "extra": extra or {},
                        "lots": lots}
                    return True

                # fire armed watches whose nice price the tape reached
                loop_ar = list(armed_entries.items())
                for arkey, arml in loop_ar:
                    if time.time() - arml["born"] > arml["ttl"]:
                        print(f"[wait] {arml['tag']} {arml['dir']} window "
                              "expired unfilled - standing down")
                        armed_entries.pop(arkey, None)
                        continue
                    # permission may have flipped while armed - recheck
                    if not regime_gate_allows(df_h1, cfg, arml["tag"]):
                        continue
                    tick_ar = mt5.symbol_info_tick(cfg["symbol"])
                    if tick_ar is None:
                        continue
                    live_px_ar = (tick_ar.bid if arml["dir"] == "LONG"
                                  else tick_ar.ask)
                    # 1-CANDLE aggression: fire when the tape is within
                    # 0.15 ATR of the wall (near-market fill, keeps the
                    # wall's SL/TP) - not only on an exact touch.
                    band_ar = (0.15 * av3) if (av3 and av3 > 0) else 0.4
                    if arml["dir"] == "LONG":
                        hit_px = live_px_ar <= arml["nice"] + band_ar
                    else:
                        hit_px = live_px_ar >= arml["nice"] - band_ar
                    if hit_px:
                        # PYRAMID SPACING GUARD (2026-10-09): block a
                        # fill within pyramid_min_atr x ATR of any open
                        # position's entry - kills cluster pyramids
                        # (3 fills inside a $0.21 window = one bar stop,
                        # -$41.56). Stays armed: fires only after price
                        # has moved the min distance off the cluster.
                        _pma = float(cfg.get("pyramid_min_atr", 0) or 0)
                        _spaced_ok = True
                        if _pma > 0 and positions:
                            _thr_ar = _pma * (av3 if (av3 and av3 > 0)
                                              else 1.0)
                            for _pp in positions:
                                if abs(live_px_ar
                                       - _pp.price_open) < _thr_ar:
                                    _spaced_ok = False
                                    _plr = globals().setdefault(
                                        "_pyro_log", {})
                                    if _plr.get(arkey, 0) \
                                            < time.time() - 60:
                                        _plr[arkey] = time.time()
                                        print(f"[pyro] spacing block "
                                              f"{arml['tag']} "
                                              f"{arml['dir']} - within "
                                              f"{_thr_ar:.2f} of "
                                              f"#{_pp.ticket} "
                                              f"@{_pp.price_open:.2f}")
                                    break
                        if not _spaced_ok:
                            continue
                        armed_entries.pop(arkey, None)
                        got_ar = send_brain(arml["dir"], arml["sl"],
                                            arml["tp"], arml["tag"],
                                            lots=arml.get("lots"))
                        if got_ar:
                            print(f"[wait] {arml['tag']} {arml['dir']} "
                                  f"patience paid - MARKET fill "
                                  f"{got_ar[0]} lots @ {got_ar[1]:.2f} "
                                  f"TP {got_ar[2]:.2f}")

                # BRAIN 1 (ref gate): whipsaw trap, recon-gated
                if trap_armed and fade_ok and av3 is not None:
                    t_entry = trap_signal(df_h1)
                    if t_entry is not None:
                        t_dir, t_sl, t_tp, t_why = t_entry
                        # nice = retest near THIS price (the old t_tp-based
                        # value armed SHORTs $40 below entry - never fillable)
                        t_nice = (cur_close - 0.25 * av3 if t_dir == "LONG"
                                  else cur_close + 0.25 * av3)
                        arm_entry("trap-fade", t_dir, t_nice, sl=t_sl,
                                  tp=t_tp)
                        print(f"[trap] FADED {t_why} -> {t_dir} ARMED "
                              f"nice {t_nice:.2f} SL {t_sl:.2f} "
                              f"TP {t_tp:.2f}")
                # BRAIN 2 (army ride + 1-min timing entry)
                if (ga30 is not None and ga4h is not None and av3 is not None
                        and min(ga30["consec"], ga4h["consec"]) >= 1):
                    aligned = (ga30["consec"] >= 1 and ga4h["consec"] >= 1
                               and ga30["accel"] >= 0)
                    red = (ga30["consec"] <= -1 and ga4h["consec"] <= -1
                           and ga30["accel"] <= 0)
                    if aligned or red:
                        pull_dir = "LONG" if aligned else "SHORT"
                        p_sl = (cur_low - 0.6 * av3 if aligned
                                else cur_high + 0.6 * av3)
                        a_nice = (cur_close - 0.45 * av3 if aligned
                                  else cur_close + 0.45 * av3)
                        arm_entry("army-ride", pull_dir, a_nice, sl=p_sl)
                        print(f"[pull] {pull_dir} with the army ARMED "
                              f"(nice entry {a_nice:.2f}, RR3 on fire) "
                              f"30M x{ga30['consec']} 4H x{ga4h['consec']}")
                # BRAIN 3: round-number psychology fade ($25 grid)
                if fade_ok and av3 is not None:
                    lvl_dn = (math.floor(cur_close / ROUND_PSYCH["grid"])
                              * ROUND_PSYCH["grid"])
                    lvl_up = lvl_dn + ROUND_PSYCH["grid"]
                    if (cur_low < lvl_dn - ROUND_PSYCH["sweep"] * av3
                            and cur_close > lvl_dn):
                        sl = (min(cur_low, lvl_dn)
                              - ROUND_PSYCH["slbuf"] * av3)
                        arm_entry("psych-fade", "LONG",
                                  cur_close - 0.25 * av3, sl=sl,
                                  tp=lvl_up - ROUND_PSYCH["pad"])
                        print(f"[psych] FADED round-level sweep "
                              f"{lvl_dn:.2f} -> LONG ARMED "
                              f"nice {cur_close - 0.25 * av3:.2f} "
                              f"SL {sl:.2f} "
                              f"TP {lvl_up - ROUND_PSYCH['pad']:.2f}")
                    elif (cur_high > lvl_up + ROUND_PSYCH["sweep"] * av3
                            and cur_close < lvl_up):
                        sl = (max(cur_high, lvl_up)
                              + ROUND_PSYCH["slbuf"] * av3)
                        arm_entry("psych-fade", "SHORT",
                                  cur_close + 0.25 * av3, sl=sl,
                                  tp=lvl_dn + ROUND_PSYCH["pad"])
                        print(f"[psych] FADED round-level sweep "
                              f"{lvl_up:.2f} -> SHORT ARMED "
                              f"nice {cur_close + 0.25 * av3:.2f} "
                              f"SL {sl:.2f} "
                              f"TP {lvl_dn + ROUND_PSYCH['pad']:.2f}")

                # BRAIN 4: gold-medalist M-chop - the user's triple-
                # bottom/top rejection read: wobbly triple bottom in a
                # band + price lifting off = pump coming; ride it
                # (market burst; the exit engine scalps the top).
                if av3 is not None and len(df_h1) >= 40:
                    tr_now = triple_rejection(df_h1, av3)
                    if tr_now is not None:
                        tr_dir, tr_why, tr_zone = tr_now
                        tr_now_ts = time.time()
                        lc4 = float(cfg.get('loss_cooldown_s', 60) or 0)
                        cooled = (not last_loss_ts
                                  or tr_now_ts - last_loss_ts >= lc4)
                        zc_key = f"{tr_dir}{tr_zone:.1f}"
                        army_dn4 = (ga30 is not None and ga30["consec"] <= -1)
                        army_up4 = (ga30 is not None and ga30["consec"] >= 1)
                        if (cooled
                                and tr_now_ts - bh_zone_ts.get(zc_key, 0) > 900
                                and not (army_dn4 and tr_dir == "LONG")
                                and not (army_up4 and tr_dir == "SHORT")):
                            b_sl = (tr_zone - 0.9 * av3 if tr_dir == "LONG"
                                    else tr_zone + 0.9 * av3)
                            got = send_brain(tr_dir, b_sl, None,
                                             "bottomhunt", rr=2.0)
                            if got:
                                bh_zone_ts[zc_key] = tr_now_ts
                                print(f"[chop] {tr_why.upper()} REJECTION "
                                      f"-> {tr_dir} burst {got[0]} lots "
                                      f"@ {got[1]:.2f} SL {b_sl:.2f} RR2 "
                                      f"(exits handled by the engine)")

                # BRAIN 5: vwap 2-sigma band fade - the user's hand read:
                # poke the upper/lower band and close back inside = the
                # band acts like a rejection line; fade with a LIMIT peg
                # into the band (never chase - limit = the medalist order).
                if av3 is not None:
                    v5 = session_vwap_zone(df_h1)
                    if v5 is not None:
                        v_up = v5["vwap"] + 2.01 * v5["sd"]
                        v_dn = v5["vwap"] - 2.01 * v5["sd"]
                        hv = float(df_h1["high"].iloc[-1])
                        lv = float(df_h1["low"].iloc[-1])
                        cv = cur_close
                        vf_ts = time.time()
                        lc5 = float(cfg.get("loss_cooldown_s", 60) or 0)
                        cooled5 = (not last_loss_ts
                                   or vf_ts - last_loss_ts >= lc5)
                        army_dn5 = (ga30 is not None and ga30["consec"] <= -1)
                        army_up5 = (ga30 is not None and ga30["consec"] >= 1)
                        if (cooled5 and hv >= v_up - 0.12 * av3
                                and cv < v_up and not army_dn5
                                and vf_ts - vwf_ts.get("S", 0) > 300):
                            s5 = v_up + 0.8 * av3
                            got = send_brain("SHORT", s5, v5["vwap"],
                                             "vwapfade")
                            if got:
                                vwf_ts["S"] = vf_ts
                                print(f"[vwap] BAND FADE upper {v_up:.2f} "
                                      f"-> SHORT {got[0]} lots (limit peg "
                                      f"~{got[1]:.2f}) SL {s5:.2f} "
                                      f"TP {v5['vwap']:.2f}")
                        elif (cooled5 and lv <= v_dn + 0.12 * av3
                                and cv > v_dn and not army_up5
                                and vf_ts - vwf_ts.get("B", 0) > 300):
                            s5 = v_dn - 0.8 * av3
                            got = send_brain("LONG", s5, v5["vwap"],
                                             "vwapfade")
                            if got:
                                vwf_ts["B"] = vf_ts
                                print(f"[vwap] BAND FADE lower {v_dn:.2f} "
                                      f"-> LONG {got[0]} lots (limit peg "
                                      f"~{got[1]:.2f}) SL {s5:.2f} "
                                      f"TP {v5['vwap']:.2f}")

                # BRAIN 6 (REBUILT per user): the red-drawn sine wave —
                # BOTH sides traded at once with EXACT-LEVEL limit
                # orders: BUY LIMIT at the projected trough, SELL LIMIT
                # at the projected crest. Pivot regularity (amplitude
                # + spacing CV) is the sine gate; the exit engine does
                # the market-out near swing; army gate respects the tide.
                # sine_brain master switch (2026-10-09): 0/29 lifetime
                # sets + R2 0.03 fit = pure drain, killed by config.
                sine_on = bool(cfg.get("sine_brain", True))
                if sine_on and av3 is not None:
                    sw = sine_signal(df_h1)
                    if sw:
                        vf_ts = time.time()
                        lc6 = float(cfg.get("loss_cooldown_s", 60) or 0)
                        cooled6 = (not last_loss_ts
                                   or vf_ts - last_loss_ts >= lc6)
                        army_dn6 = (ga30 is not None and ga30["consec"] <= -1)
                        army_up6 = (ga30 is not None and ga30["consec"] >= 1)
                        if fade_ok and cooled6 and "buy" in sw and not army_dn6:
                            b_lvl, b_sl, b_tp = sw["buy"]
                            zkey = f"SB{b_lvl:.0f}"
                            if vf_ts - bh_zone_ts.get(zkey, 0) > 600:
                                arm_entry("sine-buy", "LONG", b_lvl,
                                          sl=b_sl, tp=b_tp)
                                bh_zone_ts[zkey] = vf_ts
                                print(f"[sine] TROUGH {b_lvl:.2f} -> "
                                      f"BUY wall ARMED SL {b_sl:.2f} "
                                      f"TP {b_tp:.2f}")
                        if fade_ok and cooled6 and "sell" in sw and not army_up6:
                            s_lvl, s_sl, s_tp = sw["sell"]
                            zkey = f"SS{s_lvl:.0f}"
                            if vf_ts - bh_zone_ts.get(zkey, 0) > 600:
                                arm_entry("sine-sell", "SHORT", s_lvl,
                                          sl=s_sl, tp=s_tp)
                                bh_zone_ts[zkey] = vf_ts
                                print(f"[sine] CREST {s_lvl:.2f} -> "
                                      f"SELL wall ARMED SL {s_sl:.2f} "
                                      f"TP {s_tp:.2f}")

                # SINE-RIDE (user spec, ADAPTED 2026-10-08): forced HEARTBEAT
                # trade whose PERIOD ADAPTS to the live best-fit sine (scan
                # 4-45 candles, re-fit every 60s) - buys the low zone, sells
                # the high zone, unlimited adds per wave, per-set profit
                # counter at every phase flip.
                if "sr_state" not in globals():
                    globals()["sr_state"] = {"last_force": 0.0, "last_fit": 0.0,
                                             "set": 1, "won": 0,
                                             "zone": None, "set_bal": None,
                                             "set_trades": 0, "forced": 0,
                                             "last_set_pnl": 0.0, "brake_ts": 0.0,
                                             "w10": 2.0 * math.pi / 600.0,
                                             "ph_off": 0.0, "pt0": 0.0,
                                             "period_min": 10.0,
                                             "r2": 0.0, "gated": None}
                sr = globals()["sr_state"]
                try:
                    _sr_now = time.time()
                    _sr_rates = mt5.copy_rates_from_pos(
                        cfg["symbol"], mt5.TIMEFRAME_M1, 0, 300)
                    if _sr_rates is not None and len(_sr_rates) >= 60:
                        if _sr_now - sr["last_fit"] >= 60.0:
                            _t10 = _sr_rates["time"].astype("float64")
                            _c10 = _sr_rates["close"].astype("float64")
                            _tb = _t10 - _t10[0]
                            _best_r2, _best = -1.0, None
                            for _pc in np.arange(4.0, 45.5, 0.5):
                                _w = 2.0 * math.pi / (_pc * 60.0)
                                _X = np.column_stack([np.cos(_w * _tb),
                                                      np.sin(_w * _tb),
                                                      np.ones(len(_t10))])
                                _cf, *_ = np.linalg.lstsq(_X, _c10, rcond=None)
                                _sres = float(np.sum((_c10 - _X @ _cf) ** 2))
                                _r2 = 1.0 - _sres / float(np.sum(
                                    (_c10 - _c10.mean()) ** 2))
                                if _r2 > _best_r2:
                                    _best_r2, _best = _r2, (_w, _cf, _pc)
                            if _best is not None:
                                sr["w10"], _cf10, _pc10 = _best
                                sr["ph_off"] = math.atan2(-_cf10[1], _cf10[0])
                                sr["pt0"] = _t10[0]
                                sr["period_min"] = float(_pc10)
                                sr["r2"] = float(_best_r2)
                                if _best_r2 < 0.05:
                                    if sr["gated"] is not True:
                                        sr["gated"] = True
                                        print(f"[sine-ride] heartbeat LOW "
                                              f"QUALITY (R2 "
                                              f"{_best_r2:.2f}) - fallback "
                                              f"chop force in 900s")
                                else:
                                    if sr["gated"] is not False:
                                        sr["gated"] = False
                                        print(f"[sine-ride] heartbeat "
                                              f"UNFILTERED - real wave "
                                              f"(R2 {_best_r2:.2f})")
                                sr["last_fit"] = _sr_now
                                print(f"[sine-ride] adjust period "
                                      f"{_pc10:.0f} min (R2 {_best_r2:.2f})")
                        _ph10 = (sr["w10"] * (_sr_rates["time"].astype(
                                    "float64")[-1] - sr["pt0"])
                                 + sr["ph_off"]) % (2 * math.pi)
                        _z10 = ("low"
                                if math.pi / 2 < _ph10 < 3 * math.pi / 2
                                else "high")
                        if sr["zone"] is not None and _z10 != sr["zone"]:
                            _bal10 = mt5.account_info().balance
                            _dp10 = ((_bal10 - sr["set_bal"])
                                     if sr["set_bal"] else 0.0)
                            if _dp10 > 0:
                                sr["won"] += 1
                            sr["last_set_pnl"] = _dp10
                            if _dp10 < -25.0:
                                sr["brake_ts"] = (time.time()
                                                 + max(240.0,
                                                       sr["period_min"] * 60.0))
                                print(f"[sine-ride] BRAKE armed: wave lost "
                                      f"{_dp10:+.2f} - skipping one heartbeat "
                                      f"({max(4.0, sr['period_min']):.0f} min)")
                            print(f"[sine-ride] SET #{sr['set']} CLOSED "
                                  f"({sr['zone']}->{_z10}) trades "
                                  f"{sr['set_trades']} pnl {_dp10:+.2f} -> "
                                  f"{'PROFIT' if _dp10 > 0 else 'LOSS'} | "
                                  f"sets won {sr['won']}/{sr['set']}")
                            sr["set"] += 1
                            sr["set_trades"] = 0
                            sr["set_bal"] = _bal10
                        if sr["set_bal"] is None:
                            sr["set_bal"] = mt5.account_info().balance
                        sr["zone"] = _z10
                        _per_s = max(240.0, min(2700.0,
                                                sr["period_min"] * 60.0))
                        # wave-loss brake: a wave that just lost > $18 skips
                        # the NEXT forced heartbeat (arms still trade) so a
                        # fresh force is never stacked into a bleeding wave.
                        _braking = (_sr_now < sr["brake_ts"])
                        # wave-quality gate (user chose 2026-10-08 01:3x):
                        # force a heartbeat when the sine fit is rideable
                        # (R2 >= 0.05, loosened for the clawback sprint); in
                        # limp chop force a mean-reversion scalp every 900s
                        # so the account keeps in-motion instead of waiting.
                        # sine_min_r2 (2026-10-09): user's 0.15 floor -
                        # never force a heartbeat on a garbage fit.
                        _r2min = float(cfg.get("sine_min_r2", 0.05) or 0.05)
                        _wave_ok = sr["r2"] >= _r2min
                        _since_f = _sr_now - sr["last_force"]
                        _fallback_f = (not _wave_ok) and _since_f >= 900.0
                        if _braking:
                            sr["last_force"] = _sr_now
                        elif (cfg.get("sine_brain", True)
                              and (_fallback_f
                                   or (_wave_ok
                                       and _since_f >= _per_s))):
                            _tk10 = mt5.symbol_info_tick(cfg["symbol"])
                            _mid10 = ((_tk10.bid + _tk10.ask) / 2.0
                                      if _tk10 else cur_close)
                            _av10 = av3 if (av3 and av3 > 0) else 1.0
                            _bias10 = "LONG" if _z10 == "low" else "SHORT"
                            # chop-tuned (2026-10-08 01:1x): SL 0.6 ATR /
                            # TP 0.5 ATR so the forced wave rides complete
                            # inside the band instead of paying the full
                            # 1-ATR stop every whipsaw.
                            _sl10 = (_mid10 - 0.6 * _av10
                                     if _bias10 == "LONG"
                                     else _mid10 + 0.6 * _av10)
                            _tp10 = (_mid10 + 0.5 * _av10
                                     if _bias10 == "LONG"
                                     else _mid10 - 0.5 * _av10)
                            _got10 = send_brain(
                                _bias10, _sl10, _tp10, "sine-ride",
                                extra={"sineride": sr["set"],
                                       "heartbeat": True,
                                       "born": _sr_now,
                                       "soft_sl": None})
                            if _got10:
                                sr["last_force"] = _sr_now
                                sr["forced"] += 1
                                sr["set_trades"] += 1
                                print(f"[sine-ride] HEARTBEAT FORCE "
                                      f"#{sr['forced']} {_bias10} @ "
                                      f"{_mid10:.2f} (phase {_z10} zone, "
                                      f"period {sr['period_min']:.0f} min, "
                                      f"set #{sr['set']})")
                            else:
                                # entry blocked (cap/cooldown) - retry 2min
                                sr["last_force"] = _sr_now - (_per_s - 120.0)
                except Exception as _sr_e:
                    print(f"[sine-ride] heartbeat error: {_sr_e}")

                # SELF-OPTIMIZER (user spec 2026-10-08 10:1x): every 10 min
                # re-learn the last window from the learner DB and re-deploy
                # live by mutating the in-memory cfg (no restart): failure
                # window -> shrink risk/stop + 5-min entry freeze; hot
                # window -> restore toward base risk. Persists to the config
                # file so the next boot inherits the re-deploy too.
                if "opt_state" not in globals():
                    globals()["opt_state"] = {"last_opt": 0.0, "frozen_until": 0.0,
                                              "frozen_logged": 0.0, "last_chg": 0.0}
                opg = globals()["opt_state"]
                try:
                    # auto_tune master switch (false = report-only, never mutate)
                    if (_sr_now - opg["last_opt"] >= 600.0) and cfg.get("auto_tune", True):
                        opg["last_opt"] = _sr_now
                        try:
                            with open(os.path.join(BASE, "learning_db.json"),
                                      "r", encoding="utf-8") as _lf:
                                _ldb = json.load(_lf)
                        except Exception:
                            _ldb = {}
                        from datetime import datetime as _dt
                        _cut = time.time() - 600.0
                        _trs = [t for t in _ldb.get("trades", [])
                                if isinstance(t.get("ts"), str)
                                and _dt.fromisoformat(t["ts"]).timestamp() >= _cut]
                        _n = len(_trs)
                        _net = sum(float(t.get("pnl") or 0) for t in _trs)
                        _wr = (100.0 * sum(1 for t in _trs
                                            if float(t.get("pnl") or 0) > 0) / _n
                                if _n else 0.0)
                        # proportional thresholds: a 10-min window of 1-2 normal
                        # stops must NOT trigger - only real bleeds and real runs
                        _bal = 500.0
                        try:
                            _ai = mt5.account_info()
                            if _ai is not None and _ai.balance:
                                _bal = float(_ai.balance)
                        except Exception:
                            pass
                        _rcur = float(cfg.get("risk_pct", 3) or 3)
                        _scur = float(cfg.get("atr_stop_mult", 3) or 3)
                        _ch = []
                        _deployed = False
                        _cool = (_sr_now - opg.get("last_chg", 0.0)) >= 3600.0
                        if _n >= 5:
                            if _net <= -0.06 * _bal:
                                _nr = max(1.0, _rcur - 1.0)
                                _ns = max(1.5, _scur - 0.5)
                                if _cool:
                                    cfg["risk_pct"] = _nr
                                    cfg["atr_stop_mult"] = _ns
                                    opg["frozen_until"] = _sr_now + 180.0
                                    opg["last_chg"] = _sr_now
                                    _deployed = True
                                    _ch.append(f"risk {_rcur:g}->{_nr:g} stop "
                                                f"{_scur:g}->{_ns:g} freeze 5m")
                                else:
                                    _ch.append(f"HELD by cooldown: cut to risk {_nr:g} deferred")
                            elif _net >= 0.05 * _bal:
                                _nr = min(5.0, _rcur + 1.0)
                                if _cool:
                                    cfg["risk_pct"] = _nr
                                    opg["last_chg"] = _sr_now
                                    _deployed = True
                                    _ch.append(f"hot: risk {_rcur:g}->{_nr:g}")
                                else:
                                    _ch.append(f"HELD by cooldown: raise to risk {_nr:g} deferred")
                            if _deployed:
                                try:
                                    with open(CONFIG_PATH, "w",
                                              encoding="utf-8") as _cf:
                                        json.dump(cfg, _cf, indent=2,
                                                  default=str)
                                except Exception:
                                    pass
                        print(f"[optimize] 10min window n={_n} "
                              f"net={_net:+.1f} wr={_wr:.0f}% -> "
                              + ("deployed: " + "; ".join(_ch) if _ch
                                 else "no change"))
                except Exception as _oe:
                    print(f"[optimize] error: {_oe}")

                # REGIME TUNER (user spec 2026-10-08): every 5 min read the
                # last-100-candle regime + Heiken-Ashi and PRACTICE-TUNE the
                # live config: chop/sideways -> shorter RR + tighter stop,
                # trending/breakout -> full RR + wider stop.
                if "regime_state" not in globals():
                    globals()["regime_state"] = {"last_rc": 0.0, "regime": None}
                rgs = globals()["regime_state"]
                try:
                    if _sr_now - rgs["last_rc"] >= 300.0:
                        rgs["last_rc"] = _sr_now
                        _rc = regime_classify(df_h1)
                        if _rc:
                            rgs["regime"] = _rc
                            _chg = []
                            # clawback (user 2026-10-08): settings are
                            # LOCKED at EMA 10/35 ATR3 RR5 - the tuner only
                            # REPORTS the regime unless config gains
                            # "regime_tune": true
                            if cfg.get("regime_tune"):
                                _rr_cur = float(
                                    cfg.get("rr_ratio", 5) or 5)
                                _st_cur = float(
                                    cfg.get("atr_stop_mult", 2) or 2)
                                if ("SIDEWAYS" in _rc["regime"]
                                        or _rc["regime"].startswith(
                                            "CHOP")):
                                    if _rr_cur != 3.0:
                                        cfg["rr_ratio"] = 3.0
                                        _chg.append(
                                            f"rr {_rr_cur:g}->3")
                                    if _st_cur != 2.0:
                                        cfg["atr_stop_mult"] = 2.0
                                        _chg.append(
                                            f"stop {_st_cur:g}->2")
                                else:
                                    if _rr_cur != 5.0:
                                        cfg["rr_ratio"] = 5.0
                                        _chg.append(
                                            f"rr {_rr_cur:g}->5")
                                    if _st_cur != 2.5:
                                        cfg["atr_stop_mult"] = 2.5
                                        _chg.append(
                                            f"stop {_st_cur:g}->2.5")
                                if _chg:
                                    try:
                                        with open(CONFIG_PATH, "w",
                                                  encoding="utf-8") as _cf:
                                            json.dump(cfg, _cf, indent=2,
                                                      default=str)
                                    except Exception:
                                        pass
                            print(f"[recon] regime: {_rc['regime']} "
                                  f"(erl {_rc['erl']} slope "
                                  f"{_rc['slope_atr']:+.2f}A atr "
                                  f"x{_rc['vol_frac']}) | "
                                  f"{_rc['ha']} (flips {_rc['ha_flips']}, "
                                  f"streak {_rc['ha_streak']}) -> "
                                  + ("tuned: " + "; ".join(_chg) if _chg
                                     else "hold"))
                except Exception as _re:
                    print(f"[recon] regime error: {_re}")

                # MACD-TRIX momentum brain (user spec 2026-10-08): aggressive
                # MACD(5,10,3)+TRIX(9,3) confirm momentum for entries at 0.03
                # lots; PROGRESSIVE scale-ins (max 3 adds, spaced >=90s,
                # extension-only >=0.6ATR - never averaging down); histogram
                # flip against the ride = exit.
                if "mts" not in globals():
                    globals()["mts"] = {"ride": None, "adds": 0,
                                        "last_add": 0.0, "last_px": None}
                _mts = globals()["mts"]
                try:
                    _mtm = macd_trix(df_h1)
                    _nbm = _mtm["bias"] if _mtm else None
                    if _mtm is not None:
                        _lvm = [p for p in (mt5.positions_get(symbol=cfg["symbol"]) or [])
                                if "macd-trix" in live_ctx.get(p.ticket, {})]
                        if not _lvm:
                            _mts["ride"] = None
                            _mts["adds"] = 0
                            _mts["last_px"] = None
                        else:
                            _mts["last_px"] = _mts["last_px"] or float(_lvm[0].price_open)
                            if auto_close and (_nbm is None or _nbm != _mts["ride"]):
                                for _lp in list(_lvm):
                                    _ctp = (mt5.ORDER_TYPE_SELL
                                            if _lp.type == mt5.POSITION_TYPE_BUY
                                            else mt5.ORDER_TYPE_BUY)
                                    mt5.order_send({"action": mt5.TRADE_ACTION_DEAL,
                                                    "symbol": cfg["symbol"],
                                                    "volume": float(_lp.volume),
                                                    "type": _ctp,
                                                    "position": _lp.ticket,
                                                    "deviation": 30,
                                                    "magic": magic,
                                                    "comment": "macd-flip",
                                                    "type_time": mt5.ORDER_TIME_GTC,
                                                    "type_filling": fill_mode})
                                    print(f"[macd] flip exit {_lp.ticket} "
                                          f"({_nbm or 'momentum gone'} vs "
                                          f"ride {_mts['ride']})")
                                _mts["ride"] = None
                                _mts["adds"] = 0
                                _mts["last_px"] = None
                            else:
                                _tkmt = mt5.symbol_info_tick(cfg["symbol"])
                                _pxmt = ((_tkmt.bid + _tkmt.ask) / 2.0
                                         if _tkmt else cur_close)
                                _avmt = av3 if (av3 and av3 > 0) else 1.0
                                _ext = ((_pxmt - _mts["last_px"]) >= 0.6 * _avmt
                                        if _mts["ride"] == "LONG"
                                        else (_mts["last_px"] - _pxmt) >= 0.6 * _avmt)
                                _nowmt = time.time()
                                if (_mts["adds"] < 3 and _ext
                                        and _nowmt - _mts["last_add"] >= 90.0):
                                    _slm = (_pxmt - 0.75 * _avmt
                                            if _mts["ride"] == "LONG"
                                            else _pxmt + 0.75 * _avmt)
                                    _tpm = (_pxmt + 0.6 * _avmt
                                            if _mts["ride"] == "LONG"
                                            else _pxmt - 0.6 * _avmt)
                                    _gotm = send_brain(
                                        _mts["ride"], _slm, _tpm, "macd-trix-add",
                                        lots=0.03,
                                        extra={"macd-trix": True,
                                               "born": _nowmt,
                                               "soft_sl": None})
                                    if _gotm:
                                        _mts["adds"] += 1
                                        _mts["last_add"] = _nowmt
                                        _mts["last_px"] = _pxmt
                                        print(f"[macd] scale-in "
                                              f"#{_mts['adds']} {_mts['ride']} "
                                              f"0.03 @ {_pxmt:.2f}")
                    # fresh momentum cross + a ride cleared -> arm new entry
                    if (_nbm is not None and _mts["ride"] is None
                            and _mtm and _mtm["fresh"]):
                        _lvm2 = [p for p in (mt5.positions_get(symbol=cfg["symbol"]) or [])
                                 if "macd-trix" in live_ctx.get(p.ticket, {})]
                        if not _lvm2:
                            _avm = av3 if (av3 and av3 > 0) else 1.0
                            _sl2 = (cur_close - 0.9 * _avm if _nbm == "LONG"
                                    else cur_close + 0.9 * _avm)
                            _tp2 = (cur_close + 0.6 * _avm if _nbm == "LONG"
                                    else cur_close - 0.6 * _avm)
                            _nice2 = (cur_close + 0.15 * _avm if _nbm == "SHORT"
                                      else cur_close - 0.15 * _avm)
                            arm_entry("macd-trix", _nbm, _nice2, sl=_sl2,
                                      tp=_tp2, lots=0.03,
                                      extra={"macd-trix": True})
                            _mts["ride"] = _nbm
                            _mts["adds"] = 0
                            _mts["last_add"] = 0.0
                            _mts["last_px"] = cur_close
                            _mtcur = mt5.symbol_info_tick(cfg["symbol"])
                            _pxn = ((_mtcur.bid + _mtcur.ask) / 2.0
                                    if _mtcur else cur_close)
                            print(f"[macd] {_nbm} fresh cross (hist "
                                  f"{_mtm['hist']:+.3f} accel "
                                  f"{_mtm['accel']:+.3f} trix "
                                  f"{_mtm['trix']:+.4f}) armed 0.03 @ "
                                  f"near {_pxn:.2f}")
                except Exception as _me:
                    print(f"[macd] error: {_me}")

                # BRAIN 7: double-arrow (SEDUCE-contra only; the +EV
                # slice from the backtest) - 0.04 lots, TP +$4, 1.0 ATR
                # stop, re-learned every 2 candles; the exit engine and
                # the bounce guard handle every market-out.
                if fade_ok and av3 is not None and current_bar != last_arrow_bar \
                        and (arrow_n % ARROW["relearn_every"] == 0):
                    last_arrow_bar = current_bar
                    fphA = fomo_phase(df_h1)
                    if fphA is not None and fphA["phase"] == ARROW["phase"] \
                            and fphA.get("dir"):
                        a_dir = "SHORT" if fphA["dir"] == "UP" else "LONG"
                        midc = (float(bars["high"][-1])
                                + float(bars["low"][-1])) / 2.0
                        a_sl = (midc + ARROW["sl_atr"] * av3
                                if a_dir == "SHORT"
                                else midc - ARROW["sl_atr"] * av3)
                        a_tp = (midc - ARROW["tp_dist"] if a_dir == "SHORT"
                                else midc + ARROW["tp_dist"])
                        sym_a = mt5.symbol_info(cfg["symbol"])
                        if ARROW["lots"] >= sym_a.volume_min:
                            arm_entry("arrow", a_dir, midc, sl=a_sl,
                                      tp=a_tp)
                            arrow_n += 1
                            print(f"[arrow] SEDUCE {fphA['dir']} "
                                  f"contra -> {a_dir} ARMED "
                                  f"@ {midc:.2f} SL {a_sl:.2f} "
                                  f"TP {a_tp:.2f}")
                            write_arrow_row({
                                "ticket": "", "phase_dir": fphA["dir"],
                                "side": a_dir, "lots": ARROW["lots"],
                                "entry": round(midc, 2),
                                "sl": round(a_sl, 2),
                                "tp": round(a_tp, 2),
                                "run_dir": fphA["dir"]})

                # BRAIN 8: 10+10 whipsaw channel fades - the top-sweep
                # slice (100%/0.1h/94%WR). Both sides, puffer pegged at
                # trough/crest, only while the wobble holds, capped per
                # side; re-learned every relearn_every bars.
                if fade_ok and av3 is not None \
                        and current_bar != last_whip_bar \
                        and (whip_n % WHIPSAW["relearn_every"] == 0):
                    wch = whip_channel(df_h1)
                    if wch is not None:
                        last_whip_bar = current_bar
                        whip_n += 1
                        sym_w = mt5.symbol_info(cfg["symbol"])
                        used_b = used_s = 0
                        for p2 in positions:
                            if (live_ctx.get(p2.ticket) or {}).get("whip-buy"):
                                used_b += 1
                            if (live_ctx.get(p2.ticket) or {}).get("whip-sell"):
                                used_s += 1
                        if used_b < WHIPSAW["side_cap"] \
                                and cur_close > wch["trough"]:
                            b_lvl = wch["trough"] + WHIPSAW["tie_in_atr"] * av3
                            b_sl = wch["trough"] - WHIPSAW["sl_atr"] * av3
                            b_tp = b_lvl + WHIPSAW["tp_atr"] * av3
                            arm_entry("whip-buy", "LONG", b_lvl,
                                      sl=b_sl, tp=b_tp,
                                      lots=min(WHIPSAW["lots_cap"], 0.5))
                            print(f"[whip] wobble buy wall {b_lvl:.2f} "
                                  f"-> LONG ARMED SL {b_sl:.2f} "
                                  f"TP {b_tp:.2f}")
                        if used_s < WHIPSAW["side_cap"] \
                                and cur_close < wch["crest"]:
                            s_lvl = wch["crest"] - WHIPSAW["tie_in_atr"] * av3
                            s_sl = wch["crest"] + WHIPSAW["sl_atr"] * av3
                            s_tp = s_lvl - WHIPSAW["tp_atr"] * av3
                            arm_entry("whip-sell", "SHORT", s_lvl,
                                      sl=s_sl, tp=s_tp,
                                      lots=min(WHIPSAW["lots_cap"], 0.5))
                            print(f"[whip] wobble sell wall {s_lvl:.2f} "
                                  f"-> SHORT ARMED SL {s_sl:.2f} "
                                  f"TP {s_tp:.2f}")

                # CANDLE-LAP (user spec 2026-10-08 11:0x): scalp EVERY
                # candle. Entries are TIMED: react only inside the first 60s
                # of the new candle, enter at a retest of the just-closed
                # candle's extreme (precision fill, LIMIT-in market-out).
                # Exits stay default (1-candle cut + exit-engine) - entries
                # matter more than exits.
                if "lap_state" not in globals():
                    globals()["lap_state"] = {"last_bar": None, "n": 0}
                _lst = globals()["lap_state"]
                try:
                    if current_bar != _lst["last_bar"]:
                        _lst["last_bar"] = current_bar
                        _age_lp = _sr_now - (current_bar + 60)
                        if _age_lp >= 0 and _age_lp <= 60 and av3 and av3 > 0:
                            _o1 = df_h1["open"].astype(float).iloc[-1]
                            _c1 = df_h1["close"].astype(float).iloc[-1]
                            _h1 = df_h1["high"].astype(float).iloc[-1]
                            _l1 = df_h1["low"].astype(float).iloc[-1]
                            _rng1 = _h1 - _l1
                            _body1 = abs(_c1 - _o1)
                            if _rng1 > 0 and _body1 >= 0.2 * av3:
                                # FADE doctrine (user 2026-10-08): a
                                # GREEN 1-min candle -> SHORT it at ITS HIGH;
                                # RED 1-min candle -> LONG it at ITS LOW.
                                # HEARTBEAT (user 2026-10-08): no HTF
                                # side-lock - both directions always.
                                _up1 = _c1 >= _o1
                                _wants_lap = "SHORT" if _up1 else "LONG"
                                if _wants_lap:
                                    _tag1 = ("candle-lap-buy"
                                             if _wants_lap == "LONG"
                                             else "candle-lap-sell")
                                    _used1 = 0
                                    for _p1 in positions:
                                        if (live_ctx.get(_p1.ticket) or {}
                                                ).get(_tag1):
                                            _used1 += 1
                                    if _used1 < 2:
                                        _lvl1 = (_l1 if _wants_lap == "LONG"
                                                 else _h1)
                                        _sl1 = (_lvl1 - 0.6 * av3
                                                if _wants_lap == "LONG"
                                                else _lvl1 + 0.6 * av3)
                                        _tp1 = (_lvl1 + 0.5 * av3
                                                if _wants_lap == "LONG"
                                                else _lvl1 - 0.5 * av3)
                                        _armed_l = arm_entry(
                                            _tag1, _wants_lap, _lvl1,
                                            sl=_sl1, tp=_tp1, lots=0.03,
                                            extra={_tag1: True,
                                                   "born": _sr_now,
                                                   "soft_sl": None})
                                        if _armed_l:
                                            _lst["n"] += 1
                                            print(f"[lap] fade {_wants_lap} "
                                                  f"@ {_lvl1:.2f} "
                                                  f"({'UP' if _up1
                                                     else 'DOWN'} candle, "
                                                  f"body {_body1 / av3:.2f}A, "
                                                  f"age {_age_lp:.0f}s) -> "
                                                  f"ARMED 0.03 "
                                                  f"SL {_sl1:.2f} "
                                                  f"TP {_tp1:.2f}")
                except Exception as _lpe:
                    print(f"[lap] error: {_lpe}")

                # ULTRA-FAST IN/OUT (user 2026-10-08 ~16:0x: "OPPOSITE of
                # fomorian - MACD regular moving average 144 and 30,
                # timing lows and highs, literally in and out non-stop on
                # the 1min chart, precaution on the 1h and 30min too").
                # The anti-martingale: ONE ride at a time, 0.5 ATR stop,
                # 0.75 ATR target, histogram-flip exit NOW, 8s re-fire
                # cooldown. Never adds to a loser, never lets one breathe.
                # HTF precaution: 1h AND 30m must both sit on the same
                # side of their own SMA144 or nothing fires at all.
                if ultra_active and av3 and av3 > 0:
                    if "ultra_state" not in globals():
                        globals()["ultra_state"] = {
                            "bar": None, "allow_l": False,
                            "allow_s": False, "hist": 0.0, "hist0": 0.0,
                            "s144": 0.0, "px": 0.0, "hi20": 0.0,
                            "lo20": 0.0, "last": 0.0, "n": 0,
                            "flip_ts": 0.0, "htf_note": ""}
                    _us = globals()["ultra_state"]
                    try:
                        if _us["bar"] != current_bar:
                            # recompute ONCE per closed 1min bar: MACD
                            # = SMA30 - SMA144, signal = SMA9 of MACD
                            _cu = df_h1["close"].astype(float)
                            _hu = df_h1["high"].astype(float)
                            _lu = df_h1["low"].astype(float)
                            _mu = (_cu.rolling(30).mean()
                                   - _cu.rolling(144).mean())
                            _hsu = _mu - _mu.rolling(9).mean()
                            _h_now = float(_hsu.iloc[-1])
                            _h_0 = float(_hsu.iloc[-2])
                            if _h_now != _h_now or _h_0 != _h_0:
                                _h_now = _h_0 = 0.0
                                _us["allow_l"] = _us["allow_s"] = False
                            _s144 = float(_cu.rolling(144).mean().iloc[-1])
                            _hi = float(_hu.tail(20).max())
                            _lo = float(_lu.tail(20).min())
                            # HTF precaution: 1h + 30m vs their SMA144
                            _r60 = mt5.copy_rates_from_pos(
                                cfg["symbol"], mt5.TIMEFRAME_H1, 0, 160)
                            _up60 = False
                            _up30 = False
                            if _r60 is not None and len(_r60) > 150:
                                _c60 = _r60["close"][:-1]
                                _up60 = bool(
                                    float(_c60[-1])
                                    > float(_c60[-144:].mean()))
                            if df30 is not None and len(df30) > 150:
                                _c30 = df30["close"].astype(float)
                                _up30 = bool(
                                    float(_c30.iloc[-1])
                                    > float(_c30.iloc[-144:].mean()))
                            # HEARTBEAT: 1h+30m agree -> take that side;
                            # mixed -> S-shape range: BOTH directions armed
                            _mixed = _up60 != _up30
                            _us.update({
                                "bar": current_bar,
                                "hist": _h_now, "hist0": _h_0,
                                "s144": _s144,
                                "px": float(_cu.iloc[-1]),
                                "hi20": _hi, "lo20": _lo,
                                "allow_l": (_up60 and _up30) or _mixed,
                                "allow_s": ((not _up60 and not _up30)
                                            or _mixed)})
                            _note_u = "mixed" if _mixed else "agree"
                            if (_note_u == "mixed"
                                    and _us["htf_note"] != "mixed"):
                                print("[ultra] HTF 1h/30m mixed - "
                                      "heartbeat range: BOTH directions "
                                      "armed")
                            _us["htf_note"] = _note_u
                        # FLIP EXIT: histogram crossed against an open
                        # ride -> out NOW (in and out means OUT)
                        for _pu in [p for p in (mt5.positions_get(
                                symbol=cfg["symbol"]) or [])
                                if "ultra" in (live_ctx.get(p.ticket)
                                               or {})]:
                            _ctxu = live_ctx.get(_pu.ticket) or {}
                            _sgn = float(_ctxu.get("hist_sign") or 0)
                            if _sgn and _us["hist"] * _sgn < 0 \
                                    and time.time() - _us["flip_ts"] > 3:
                                _ctp_u = (mt5.ORDER_TYPE_SELL
                                          if _pu.type == mt5.POSITION_TYPE_BUY
                                          else mt5.ORDER_TYPE_BUY)
                                _tk2 = mt5.symbol_info_tick(cfg["symbol"])
                                if _tk2 is None:
                                    continue
                                _px2u = (_tk2.bid
                                         if _ctp_u == mt5.ORDER_TYPE_SELL
                                         else _tk2.ask)
                                _rs2 = mt5.order_send({
                                    "action": mt5.TRADE_ACTION_DEAL,
                                    "symbol": cfg["symbol"],
                                    "volume": float(_pu.volume),
                                    "type": _ctp_u,
                                    "position": _pu.ticket,
                                    "price": _px2u, "deviation": 30,
                                    "magic": magic,
                                    "comment": "ultra-flip",
                                    "type_time": mt5.ORDER_TIME_GTC,
                                    "type_filling": fill_mode})
                                _us["flip_ts"] = time.time()
                                print(f"[ultra] hist flip -> OUT "
                                      f"#{_pu.ticket} "
                                      f"{_pu.profit:+.2f} retcode="
                                      f"{_rs2.retcode if _rs2 else 'None'}")
                                live_ctx.pop(_pu.ticket, None)
                        # ENTRY: one ride at a time, timed at a local
                        # LOW (LONG) or HIGH (SHORT) - never chase
                        _open_u = [p for p in (mt5.positions_get(
                            symbol=cfg["symbol"]) or [])
                            if "ultra" in (live_ctx.get(p.ticket) or {})]
                        _ready_u = (time.time() - _us["last"]
                                    >= float(
                                        cfg.get("ultra_cooldown", 8) or 8))
                        if (not _open_u and _ready_u
                                and _us["hi20"] > _us["lo20"]):
                            _rngu = _us["hi20"] - _us["lo20"]
                            _want_u = None
                            if (_us["allow_l"]
                                    and _us["px"] > _us["s144"]
                                    and _us["px"]
                                    <= _us["lo20"] + 0.30 * _rngu
                                    and _us["hist"] > 0
                                    and _us["hist"] >= _us["hist0"]):
                                _want_u = "LONG"
                            elif (_us["allow_s"]
                                  and _us["px"] < _us["s144"]
                                  and _us["px"]
                                  >= _us["hi20"] - 0.30 * _rngu
                                  and _us["hist"] < 0
                                  and _us["hist"] <= _us["hist0"]):
                                _want_u = "SHORT"
                            if _want_u:
                                _slu = (_us["px"] - 0.5 * av3
                                        if _want_u == "LONG"
                                        else _us["px"] + 0.5 * av3)
                                _tpu = (_us["px"] + 0.75 * av3
                                        if _want_u == "LONG"
                                        else _us["px"] - 0.75 * av3)
                                _gotu = send_brain(
                                    _want_u, _slu, _tpu, "ultra",
                                    extra={"ultra": True,
                                           "born": _sr_now,
                                           "soft_sl": None,
                                           "hist_sign": (1
                                                         if _want_u == "LONG"
                                                         else -1)})
                                if _gotu:
                                    _us["last"] = time.time()
                                    _us["n"] += 1
                                    print(f"[ultra] {_want_u} in/out "
                                          f"#{_us['n']} @ "
                                          f"{_gotu[1]:.2f} SL "
                                          f"{_slu:.2f} TP {_tpu:.2f} "
                                          f"({_us['hist']:+.3f} MACD "
                                          f"hist, low/high timed, "
                                          f"HTF ok)")
                    except Exception as _ue:
                        print(f"[ultra] error: {_ue}")

                # FOMORIAN MARTINGALE DEMO (user spec 2026-10-08 13:4x):
                # gambling style - cut a loss then DOUBLE/TRIPLE the next
                # trade, a WIN resets to the initial size, a loss at the
                # CAP step resets too (the classic ruin return). All other
                # brains are drained/paused while MODE ON so the spiral is
                # pure and visible. Deliberately ruinous - demo money only.
                if cfg.get("fomorian_mode"):
                    if "fomoria" not in globals():
                        globals()["fomoria"] = {"streak": 0, "ticket": None,
                                                "lots": None, "n": 0,
                                                "last_print": ""}
                    _fm = globals()["fomoria"]
                    try:
                        _fmult = float(cfg.get("fomorian_mult", 2.0) or 2.0)
                        _finit = float(cfg.get("fomorian_initial", 0.02)
                                       or 0.02)
                        _fcap = float(cfg.get("fomorian_cap", 0.5) or 0.5)
                        # drain every NON-fomorian position so the martin-
                        # gale cycle is unmistakable (MUTED in manual mode:
                        # nothing auto-closes, the user is in control)
                        if auto_close:
                            for _p9 in (mt5.positions_get(symbol=cfg["symbol"])
                                        or []):
                                if "fomorian" not in (live_ctx.get(_p9.ticket)
                                                      or {}):
                                    _c9 = (mt5.ORDER_TYPE_SELL
                                           if _p9.type == mt5.POSITION_TYPE_BUY
                                           else mt5.ORDER_TYPE_BUY)
                                    mt5.order_send({"action":
                                                    mt5.TRADE_ACTION_DEAL,
                                                    "symbol": cfg["symbol"],
                                                    "volume":
                                                    float(_p9.volume),
                                                    "type": _c9,
                                                    "position": _p9.ticket,
                                                    "deviation": 30,
                                                    "magic": magic,
                                                    "comment":
                                                    "fomorian-drain",
                                                    "type_time":
                                                    mt5.ORDER_TIME_GTC,
                                                    "type_filling":
                                                    fill_mode})
                        _open_fom = [p for p in
                                     (mt5.positions_get(
                                         symbol=cfg["symbol"]) or [])
                                     if "fomorian" in
                                     (live_ctx.get(p.ticket) or {})]
                        if _fm["ticket"] and not _open_fom:
                            # the tracked trade just closed - autopsy it
                            _dl = (mt5.history_deals_get(
                                position=_fm["ticket"]) or [])
                            _pnl = sum(float(d.profit or 0) for d in _dl)
                            if _fm.get("decimate"):
                                _fm["streak"] = 0
                                _fm["decimate"] = False
                                print(f"[fomorian] DECIMATE STEP "
                                      f"{_pnl:+.2f} -> streak RESET "
                                      f"(margin-wall cap step)")
                            elif _pnl > 0:
                                _fm["streak"] = 0
                                print(f"[fomorian] WIN {_pnl:+.2f} at "
                                      f"{_fm['lots']} lots -> streak "
                                      f"RESET to {_finit}")
                            else:
                                _fm["streak"] += 1
                                _nxt = min(_fcap,
                                           _finit * (_fmult ** _fm["streak"]))
                                print(f"[fomorian] CUT LOSS {_pnl:+.2f} "
                                      f"at {_fm['lots']} lots -> streak "
                                      f"{_fm['streak']} -> NEXT {_nxt:.2f} "
                                      f"({_fmult:g}x down)")
                                if (_finit * (_fmult ** _fm["streak"])
                                        > _fcap):
                                    _fm["streak"] = 0
                            _fm["ticket"] = None
                        if not _open_fom and not _fm["ticket"] \
                                and time.time() >= _fm.get("wait_until", 0):
                            _o9 = df_h1["close"].astype(float).iloc[-1]
                            _o90 = df_h1["open"].astype(float).iloc[-1]
                            _fdir = "LONG" if _o9 >= _o90 else "SHORT"
                            _lots9 = min(_fcap,
                                         _finit * (_fmult ** _fm["streak"]))
                            _tk9 = mt5.symbol_info_tick(cfg["symbol"])
                            # MARGIN WALL (2026-10-08): the doubling can
                            # price itself out of the account (0.32 lots
                            # needs ~$1321 > ~$1296 free) -> stuck in a
                            # 10019 'No money' rejection loop forever. Fix:
                            # clamp to what the account can actually fund
                            # and treat the play as a CAP step (decimate:
                            # resets the streak once it closes).
                            _dec9 = False
                            try:
                                _si9 = mt5.symbol_info(cfg["symbol"])
                                _px9a = _tk9.bid if _tk9 else cur_close
                                _perlot9 = mt5.order_calc_margin(
                                    mt5.ORDER_TYPE_BUY, cfg["symbol"],
                                    1.0, float(_px9a))
                                if _perlot9 and _perlot9 > 0:
                                    _acc9 = mt5.account_info()
                                    if _acc9 is not None:
                                        _max9 = (_acc9.margin_free
                                                 / _perlot9)
                                        _budget9 = _max9 * 0.80
                                        if _lots9 > _budget9:
                                            _dec9 = True
                                            _lots9 = max(
                                                0.01, round(_budget9, 2))
                                            print(f"[fomorian] MARGIN WALL: "
                                                  f"streak "
                                                  f"{_fm['streak']} size "
                                                  f"{_finit * _fmult ** _fm['streak']:.2f}"
                                                  f" > fundable "
                                                  f"{_budget9:.2f} -> "
                                                  f"DECIMATE to "
                                                  f"{_lots9:.2f} (streak "
                                                  f"resets after it closes)")
                                        if _lots9 < _finit:
                                            _lots9 = round(_budget9, 2)
                                        if _lots9 < 0.01:
                                            _lots9 = 0.0
                                            _fm["wait_until"] = (
                                                time.time() + 15)
                                            print("[fomorian] cannot even "
                                                  "fund one step - backing "
                                                  "off 15s")
                            except Exception:
                                pass
                            _av9 = av3 if (av3 and av3 > 0) else 1.0
                            _mid9 = ((_tk9.bid + _tk9.ask) / 2.0
                                     if _tk9 else cur_close)
                            _sl9 = (_mid9 - 0.7 * _av9 if _fdir == "LONG"
                                    else _mid9 + 0.7 * _av9)
                            _tp9 = (_mid9 + 0.5 * _av9 if _fdir == "LONG"
                                    else _mid9 - 0.5 * _av9)
                            _got9 = None
                            if _lots9 > 0:
                                _got9 = send_brain(
                                    _fdir, _sl9, _tp9, "fomorian",
                                    lots=_lots9,
                                    extra={"fomorian": True,
                                           "streak": _fm["streak"],
                                           "born": _sr_now,
                                           "soft_sl": None})
                            if _lots9 <= 0:
                                pass
                            elif _got9:
                                _known9 = {p.ticket for p in _open_fom}
                                for _p9b in (mt5.positions_get(
                                        symbol=cfg["symbol"]) or []):
                                    if (_p9b.magic == magic
                                            and _p9b.ticket not in _known9):
                                        _fm["ticket"] = _p9b.ticket
                                        break
                                _fm["lots"] = _lots9
                                _fm["decimate"] = bool(_dec9)
                                _fm["n"] += 1
                                print(f"[fomorian] step {_fm['streak']} "
                                      f"{_fdir} {_lots9:.2f} lots @ "
                                      f"{_mid9:.2f} (SL {_sl9:.2f} "
                                      f"TP {_tp9:.2f}) - streak "
                                      f"{_fm['streak']}")
                            else:
                                _fm["wait_until"] = time.time() + 8
                                print("[fomorian] entry rejected - "
                                      "retry in 8s")
                    except Exception as _fe:
                        print(f"[fomorian] error: {_fe}")

            class _F:  # small shim so write_god_data works for live too
                spread = live_spread
            ti_now = mt5.terminal_info()
            acc_now = mt5.account_info()
            _thought = narrate(cfg, reading, positions, live_ctx, df_h1,
                               acc_now)
            print(f"[think] {_thought}")
            if gate_note and gate_note != last_gate_note:
                print(f"[think] setup NOT qualified -> {gate_note}")
            if pending is not None and pending.get("tries", 0) > 0:
                print(f"[think] signal HELD: {pending['sig']} from bar "
                      f"{time.strftime('%H:%M:%S', time.localtime(pending['bar']))} "
                      f"is still valid - holding it for the broker "
                      f"(rejected {pending['tries']}x so far)")
            # trap fades are FAST IN / FAST OUT: a fade that has gone
            # nowhere for the time-stop window exits at market
            for p in positions:
                c = live_ctx.get(p.ticket) or {}
                if auto_close and (any(c.get(k) for k in ("trap-fade", "psych-fade",
                                           "army-ride", "re-ride"))
                        and (time.time() - float(c.get("born", 0))
                             ) > TRAP_CFG["time_stop_min"] * 60):
                    ctype = (mt5.ORDER_TYPE_SELL
                             if p.type == mt5.POSITION_TYPE_BUY
                             else mt5.ORDER_TYPE_BUY)
                    tick3 = mt5.symbol_info_tick(cfg["symbol"])
                    if tick3 is None:
                        continue
                    px3 = (tick3.bid if ctype == mt5.ORDER_TYPE_SELL
                           else tick3.ask)
                    req3 = {"action": mt5.TRADE_ACTION_DEAL,
                            "symbol": cfg["symbol"], "volume": p.volume,
                            "type": ctype, "position": p.ticket,
                            "price": px3, "deviation": 30, "magic": p.magic,
                            "comment": "trap-time-stop",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": fill_mode}
                    res3 = mt5.order_send(req3)
                    rc3 = res3.retcode if res3 else "None"
                    print(f"[trap] TIME-STOP #{p.ticket} {p.profit:+.2f} "
                          f"(dead fade > {TRAP_CFG['time_stop_min']}min) "
                          f"-> retcode={rc3}")
                    live_ctx.pop(p.ticket, None)
            if (auto_close and profit_target > 0 and acc_now is not None
                    and acc_now.equity >= guard_target):
                banked = bank_block(cfg, float(acc_now.equity),
                                    float(acc_now.balance))
                run_command(cfg, mt5, fill_mode, magic,
                            {"action": "close_all"}, live_ctx)
                if cfg.get("bank_mode"):
                    # aggressive vault mode: bank the $100 block, start the
                    # next cycle immediately, keep hunting
                    print("=" * 56)
                    print(f"[bank] BLOCK banked +$100 (today "
                          f"{banked['today']} blocks, vault "
                          f"${banked['total']:.2f}). New cycle: "
                          f"target ${banked['balance'] + profit_target:.2f}.")
                    print("=" * 56)
                    guard_baseline = float(banked["balance"])
                    guard_target = guard_baseline + profit_target
                    try:
                        with open(os.path.join(BASE, "profit_guard.json"),
                                  "w") as f:
                            json.dump({"baseline": guard_baseline}, f)
                    except OSError:
                        pass
                else:
                    print("=" * 56)
                    print(f"PROFIT TARGET HIT: equity "
                          f"${acc_now.equity:.2f} >= target "
                          f"${guard_target:.2f}")
                    print("closing all positions, then stopping the bot")
                    print("=" * 56)
                    try:
                        with open(os.path.join(BASE,
                                               "watchdog-disable.flag"),
                                  "w") as f:
                            f.write(datetime.now(
                                timezone.utc).isoformat())
                    except OSError:
                        pass
                    print("kill switch engaged - the mission is done "
                          "(START BOT on the dashboard begins a new one)")
                    try:
                        os.remove(mission_file)   # deadline served / done
                    except OSError:
                        pass
                    break
            # EQUITY FLOOR (user 2026-10-12): downhill twin of the profit
            # guard - a fuse, not a strategy. Equity at/below the floor =
            # close everything, halt, wait for the human. A later START
            # re-halts until the floor is lowered or funds are added
            # (by design - the floor must be re-armed deliberately).
            _floor = float(cfg.get("equity_floor_usd", 0) or 0)
            if auto_close and _floor > 0 and acc_now is not None \
                    and acc_now.equity <= _floor:
                print("=" * 56)
                print(f"[floor] EQUITY FLOOR HIT: equity "
                      f"${acc_now.equity:.2f} <= floor ${_floor:.2f}")
                print("closing all positions, then stopping the bot")
                print("=" * 56)
                run_command(cfg, mt5, fill_mode, magic,
                            {"action": "close_all"}, live_ctx)
                try:
                    with open(os.path.join(BASE,
                                           "watchdog-disable.flag"),
                              "w") as f:
                        f.write(datetime.now(
                            timezone.utc).isoformat())
                except OSError:
                    pass
                print("[floor] kill switch engaged - add funds or lower "
                      "equity_floor_usd, then START BOT on the dashboard")
                break
            if (auto_close and mission_hours > 0
                    and time.time() - mission_t0 > mission_hours * 3600):
                print("=" * 56)
                print(f"[mission] {mission_hours:.0f}h DEADLINE reached "
                      f"(equity ${acc_now.equity if acc_now else 0:.2f}) - "
                      f"closing all positions, then stopping the bot")
                print("=" * 56)
                run_command(cfg, mt5, fill_mode, magic,
                            {"action": "close_all"}, live_ctx)
                try:
                    with open(os.path.join(BASE, "watchdog-disable.flag"),
                              "w") as f:
                        f.write(datetime.now(timezone.utc).isoformat())
                except OSError:
                    pass
                try:
                    os.remove(mission_file)   # deadline served
                except OSError:
                    pass
                print("mission deadline kill switch engaged")
                break
            # LIVE METER truth from the broker (last 30 days): win rate,
            # today's P&L, streaks, lot range - restart-proof (engine.closed
            # is per-session; deals history is permanent).
            _cd = all_closed or closed_by_magic or []
            _win = [d for d in _cd if d.profit > 0]
            _loss = [d for d in _cd if d.profit <= 0]
            _gw = sum(d.profit for d in _win)
            _gl = abs(sum(d.profit for d in _loss))
            _st = sorted(((d.time, d.profit) for d in _cd),
                         key=lambda x: x[0])
            _cur = 0
            if _st:
                _sign = 1 if _st[-1][1] > 0 else -1
                for _t, _pf in reversed(_st):
                    if (1 if _pf > 0 else -1) != _sign or _pf == 0:
                        break
                    _cur += 1
            _today = datetime.now().date()
            _td = [d for d in _cd
                   if datetime.fromtimestamp(d.time).date() == _today]
            live_stats_30 = {
                "trades": len(_cd), "wins": len(_win), "losses": len(_loss),
                "win_rate": round(len(_win) / len(_cd) * 100, 1) if _cd else 0.0,
                "profit_factor": round(_gw / _gl, 2) if _gl else
                                 (None if not _win else 999.0),
                "avg_win": round(_gw / len(_win), 2) if _win else 0.0,
                "avg_loss": round(-_gl / len(_loss), 2) if _loss else 0.0,
                "best_win": round(max((d.profit for d in _win), default=0.0), 2),
                "worst_loss": round(min((d.profit for d in _loss),
                                        default=0.0), 2),
                "streak_current": _cur,
                "streak_win": _streak_best(_st, 1),
                "streak_loss": _streak_best(_st, -1),
                "lot_min": round(min((d.volume for d in _cd), default=0.0), 2),
                "lot_max": round(max((d.volume for d in _cd), default=0.0), 2),
                "today": _today.isoformat(),
                "today_trades": len(_td),
                "today_wins": sum(1 for d in _td if d.profit > 0),
                "today_losses": sum(1 for d in _td if d.profit <= 0),
                "today_pnl": round(sum(d.profit for d in _td), 2),
            }
            cost_meter = {
                **(LIVE_METER.get("last") or {}),
                "slip": {k: round(v, 3)
                         for k, v in (LIVE_METER.get("slip") or {}).items()},
                "market_fills": LIVE_METER.get("market", 0),
                "limit_falls": LIVE_METER.get("rises", 0),
            }
            ladder_meter = {
                "on": bool(LADDER_CFG.get("on")),
                "layers": int(LADDER_CFG.get("layers", 3)),
                "active": {
                    str(ot3): {
                        "dir": ld3["dir"], "layer": ld3["layer"],
                        "level": round(ld3["level"], 2),
                        "sl": round(ld3["sl"], 2),
                        "tp": round(ld3["tp"], 2),
                        "age_s": round(time.time() - ld3["born"], 1),
                    }
                    for ot3, ld3 in LADDER_STATE["active"].items()},
                "fills": LADDER_STATE["fills"],
                "cancels": LADDER_STATE["cancels"],
            }
            write_god_data(cfg, engine, reading, _F, "live",
                           {"open_positions_broker": len(positions),
                            "recent_stop_outs": recent_stops,
                            "broker_balance": round(acc_now.balance, 2)
                            if acc_now else None,
                            "broker_equity": round(acc_now.equity, 2)
                            if acc_now else None,
                            "autotrading": bool(ti_now.trade_allowed)
                            if ti_now else False,
                            "profit_guard": {
                                "baseline": round(guard_baseline, 2),
                                "target": round(guard_target, 2)}
                            if profit_target > 0 else None,
                            "main_wallet": load_wallet()
                            if profit_target > 0 else None},
                           live_positions=positions,
                           live_account=acc_now,
                           live_stats=live_stats_30,
                           cost_meter=cost_meter,
                           ladder_meter=ladder_meter,
                           tactics=build_tactics(cfg, df_h1, reading,
                                                 brain_orders, engine))
            print(f"[{datetime.now():%H:%M:%S}] radar={reading.mode:<10} "
                  f"score={reading.score:<3} equity=${getattr(acc_now, 'equity', float('nan')):>9.2f} "
                  f"pos={len(positions)}")
            time.sleep(cfg["cycle_seconds"])
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        save_trades_csv(engine.closed)
        mt5.shutdown()


if __name__ == "__main__":
    cfg = load_config()
    if "--paper" in sys.argv:
        run_paper(cfg)
    else:
        run_live(cfg)





