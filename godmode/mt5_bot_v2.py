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
    "base_lots": 0.01,
    "max_concurrent": 2,
    "max_leverage": 10.0,
    "wick_proof_stop": False,
    "disaster_stop_mult": 1.5,
    "profit_target_usd": 0,
    "cycle_seconds": 60,
    "cooldown_checks": 3,
    "start_balance_paper": 500.0,
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
    if len(df) < cfg["ema_slow"] + 2:
        return None
    fast = ema(df["close"], cfg["ema_fast"])
    slow = ema(df["close"], cfg["ema_slow"])
    f_now, f_prev = fast.iloc[-1] > slow.iloc[-1], fast.iloc[-2] > slow.iloc[-2]
    s_now, s_prev = fast.iloc[-1] < slow.iloc[-1], fast.iloc[-2] < slow.iloc[-2]
    if f_now and not f_prev:
        return "LONG"
    if s_now and not s_prev:
        return "SHORT"
    return None


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
                   live_positions=None, live_account=None):
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
        "stats": engine.stats(),
        "learning": engine.learner.summary() if engine.learner else None,
        "cooldown": extra or {},
    }
    tmp = DATA_PATH + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, DATA_PATH)
    except (PermissionError, OSError):
        # transient Windows file lock (AV / dashboard read / racing process):
        # never crash the trading loop over a status-file write
        print("[warn] god_data.json write skipped (file locked)")


def run_paper(cfg):
    print("=" * 66)
    print("  GOD-MODE BOT - PAPER MODE (synthetic feed, zero risk)")
    print("  dashboard: run bridge.py then open http://localhost:8765")
    print("=" * 66)
    feed = PaperFeed(n_seed=1100, seed=42)
    engine = Engine(cfg["start_balance_paper"])
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
    profit_target = float(cfg.get("profit_target_usd", 0) or 0)
    expected_login = getattr(acc, "login", None)
    guard_baseline = load_profit_baseline(acc.balance) if profit_target > 0 else None
    guard_target = (guard_baseline + profit_target) if profit_target > 0 else None
    if profit_target > 0:
        print(f"[profit-guard] ON: stop everything at equity "
              f"${guard_target:.2f} (baseline ${guard_baseline:.2f} "
              f"+ ${profit_target:.2f})")
    REASON_SL = getattr(mt5, "DEAL_REASON_SL", 4)
    REASON_TP = getattr(mt5, "DEAL_REASON_TP", 5)
    DEAL_OUT = getattr(mt5, "DEAL_ENTRY_OUT", 1)
    try:
        while True:
            cfg = load_config()
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
                         cfg.get("profit_target_usd"))
            if strat_sig != last_strat:
                print(f"[config] strategy: TF {cfg.get('timeframe')} "
                      f"EMA {cfg.get('ema_fast')}/{cfg.get('ema_slow')} "
                      f"stop x{cfg.get('atr_stop_mult')} "
                      f"RR {cfg.get('rr_ratio')} "
                      f"risk {cfg.get('risk_pct')}% "
                      f"lots {cfg.get('base_lots')} "
                      f"max {cfg.get('max_concurrent')} "
                      f"target ${cfg.get('profit_target_usd')}")
                last_strat = strat_sig
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
            bars = mt5.copy_rates_from_pos(cfg["symbol"], tf, 0, 900)
            h4 = mt5.copy_rates_from_pos(cfg["symbol"], mt5.TIMEFRAME_H4, 0, 260)
            if bars is None or len(bars) < 60 or h4 is None or len(h4) < 60:
                time.sleep(cfg["cycle_seconds"]); continue
            # use CLOSED bars only: last row from copy_rates_from_pos is the
            # forming candle - its close flickers with every tick
            df_h1 = pd.DataFrame(bars).iloc[:-1].reset_index(drop=True)
            df_h4 = pd.DataFrame(h4).iloc[:-1].reset_index(drop=True)
            engine._last_h1 = df_h1
            sym = mt5.symbol_info(cfg["symbol"])
            live_spread = float(sym.spread) or 0.30
            tick = mt5.symbol_info_tick(cfg["symbol"])
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

            if reading.should_flatten:
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

            elif reading.can_open_new and len(positions) < cfg["max_concurrent"]:
                # signal latch: fire on the bar close, then KEEP retrying
                # while that same closed bar is still the signal bar. A
                # transient broker condition (Algo Trading off, network
                # blip, requote) used to throw the whole setup away - the
                # bot then idled until the next cross, minutes or an hour
                # later, with the setup silently lost.
                sig_now = signal(df_h1, cfg)
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
                    risk_amt = acc_now.balance * cfg["risk_pct"] / 100.0
                    lots = size_lots(acc_now.balance, risk_amt, stop_dist, cfg)
                    ctx = build_entry_ctx(sig, reading, df_h1, cfg,
                                          live_spread,
                                          datetime.now().hour, price)
                    if lots < sym.volume_min:
                        pending = None   # sizing impossible - do not latch
                    if sig and lots >= sym.volume_min:
                        if engine.learner is not None:
                            ok_gate, why = engine.learner.gate(ctx)
                            if not ok_gate:
                                print(f"  [learner] FILTERED {sig} entry -> {why}")
                                sig = None
                                pending = None   # a veto is final, not a retry
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
            if (profit_target > 0 and acc_now is not None
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
                    print("kill switch engaged - run RESUME-TRADING.bat "
                          "to start a new $100 cycle")
                    break
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
                           live_account=acc_now)
            print(f"[{datetime.now():%H:%M:%S}] radar={reading.mode:<10} "
                  f"score={reading.score:<3} equity=${acc_now.equity:>9.2f} "
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
