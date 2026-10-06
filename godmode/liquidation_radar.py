"""
Liquidation Radar
=================
Detects mass-liquidation environments before they eat the account.

The pattern you observed is real and well documented:
  - Once every few weeks / month-end, gold makes a violent sweep
    (often into the H4 200 MA), stops out a huge cluster of trades
    and bots, then reverses. Winning streaks die in one candle.

Radar signals (each adds to a danger score 0-100):
  1. RETEST ZONE    price is within 1.0 ATR of the H4 200 MA
  2. VOL EXPANSION  H1 ATR > 2.0x its own 50-period average
  3. SPREAD STRESS  live spread > 3x normal
  4. VIOLENT SWEEP  range of last 10 H1 candles > 6.0 ATR
  5. STOP CLUSTER   bot itself got stopped 3x within last 10 candles
  6. MONTH-END      last 2 trading days of the month (statistically wild)

Danger >= 70  -> STAND DOWN (no new trades)
Danger >= 90  -> FLATTEN    (close everything, hide)

Usage:
    from liquidation_radar import LiquidationRadar, RadarConfig
    radar = LiquidationRadar(RadarConfig())
    reading = radar.assess(df_h1, df_h4, live_spread, normal_spread,
                            recent_stop_outs, now)
    if reading.should_flatten: ...
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, timezone
import numpy as np
import pandas as pd


@dataclass
class RadarConfig:
    ma_period: int = 200                # the H4 200 MA level
    retest_atr_dist: float = 1.0        # danger if price within 1.0 ATR of MA
    vol_expand_mult: float = 2.0        # ATR > 2x its own baseline
    spread_stress_mult: float = 3.0    # live spread > 3x normal
    sweep_candles: int = 10             # look-back for violent sweep
    sweep_atr_mult: float = 6.0         # 10-candle range > 6 ATR = violent
    stop_cluster_count: int = 3         # own stops within window
    stand_down_score: int = 70
    flatten_score: int = 90
    weights: dict = field(default_factory=lambda: {
        "RETEST_ZONE": 30,
        "VOL_EXPANSION": 25,
        "SPREAD_STRESS": 15,
        "VIOLENT_SWEEP": 30,
        "STOP_CLUSTER": 25,
        "MONTH_END": 10,
    })


@dataclass
class RadarReading:
    score: int = 0
    mode: str = "GREEN"                 # GREEN / STAND_DOWN / FLATTEN
    triggers: List[str] = field(default_factory=list)
    ma_distance_atr: float = float("nan")
    atr_current: float = 0.0
    atr_baseline: float = 0.0
    atr_ratio: float = 0.0
    sweep_range: float = 0.0
    sweep_ratio: float = 0.0

    @property
    def can_open_new(self) -> bool:
        return self.mode == "GREEN"

    @property
    def should_flatten(self) -> bool:
        return self.mode == "FLATTEN"


def _true_range(df: pd.DataFrame) -> pd.Series:
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def rolling_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _true_range(df).rolling(period).mean()


def _is_month_end_window(now: datetime) -> bool:
    today = now.date()
    days_in_month = (
        pd.Timestamp(today.year, today.month, 1) + pd.offsets.MonthEnd(0)
    ).date().day
    return (days_in_month - today.day) <= 2


class LiquidationRadar:
    def __init__(self, config: RadarConfig = None):
        self.cfg = config or RadarConfig()
        self.events: List[dict] = []     # history of danger-mode entries
        self.last_mode = "GREEN"

    def assess(
        self,
        df_h1: pd.DataFrame,
        df_h4: pd.DataFrame,
        live_spread: float,
        normal_spread: float,
        recent_stop_outs: int = 0,
        now: Optional[datetime] = None,
    ) -> RadarReading:
        cfg = self.cfg
        now = now or datetime.now(timezone.utc)
        triggers: List[str] = []
        score = 0

        price = float(df_h1["close"].iloc[-1])

        atr_series = rolling_atr(df_h1)
        atr = float(atr_series.iloc[-1])
        if np.isnan(atr) or atr <= 0:
            atr = 0.0

        # 1) RETEST ZONE: distance to H4 200 MA in H1-ATR units
        ma_distance_atr = float("nan")
        if len(df_h4) >= cfg.ma_period and atr > 0:
            ma = float(df_h4["close"].rolling(cfg.ma_period).mean().iloc[-1])
            ma_distance_atr = abs(price - ma) / atr
            if ma_distance_atr <= cfg.retest_atr_dist:
                triggers.append("RETEST_ZONE")
                score += cfg.weights["RETEST_ZONE"]

        # 2) VOL EXPANSION: current ATR vs its own 50-period baseline
        if len(atr_series) >= 50:
            atr_baseline = float(atr_series.rolling(50).mean().iloc[-1])
        else:
            atr_baseline = atr
        atr_ratio = (atr / atr_baseline) if atr_baseline > 0 else 0.0
        if atr_ratio >= cfg.vol_expand_mult:
            triggers.append("VOL_EXPANSION")
            score += cfg.weights["VOL_EXPANSION"]

        # 3) SPREAD STRESS
        if normal_spread and normal_spread > 0:
            if live_spread >= cfg.spread_stress_mult * normal_spread:
                triggers.append("SPREAD_STRESS")
                score += cfg.weights["SPREAD_STRESS"]

        # 4) VIOLENT SWEEP: total range of last N candles vs ATR
        sweep_range = 0.0
        sweep_ratio = 0.0
        if atr > 0 and len(df_h1) >= cfg.sweep_candles:
            tail = df_h1.tail(cfg.sweep_candles)
            sweep_range = float(tail["high"].max() - tail["low"].min())
            sweep_ratio = sweep_range / atr
            if sweep_ratio >= cfg.sweep_atr_mult:
                triggers.append("VIOLENT_SWEEP")
                score += cfg.weights["VIOLENT_SWEEP"]

        # 5) STOP CLUSTER: bot's own recent stops
        if recent_stop_outs >= cfg.stop_cluster_count:
            triggers.append("STOP_CLUSTER")
            score += cfg.weights["STOP_CLUSTER"]

        # 6) MONTH-END calendar
        if _is_month_end_window(now):
            triggers.append("MONTH_END")
            score += cfg.weights["MONTH_END"]

        score = int(min(score, 100))

        if score >= cfg.flatten_score:
            mode = "FLATTEN"
        elif score >= cfg.stand_down_score:
            mode = "STAND_DOWN"
        else:
            mode = "GREEN"

        # log danger-mode entries as liquidation events
        if mode != "GREEN" and self.last_mode == "GREEN":
            self.events.append({
                "time": now.isoformat(),
                "score": score,
                "mode": mode,
                "triggers": list(triggers),
                "price": price,
            })
        self.last_mode = mode

        return RadarReading(
            score=score,
            mode=mode,
            triggers=triggers,
            ma_distance_atr=ma_distance_atr,
            atr_current=atr,
            atr_baseline=atr_baseline,
            atr_ratio=atr_ratio,
            sweep_range=sweep_range,
            sweep_ratio=sweep_ratio,
        )


def synthetic_df(n: int = 1200, seed: int = 7, atr_base: float = 1.2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 2350.0
    rows = []
    for i in range(n):
        ret = rng.normal(0, atr_base) / 100
        o = price
        price = max(1800, min(3200, price * (1 + ret)))
        c = price
        wick = abs(rng.normal(0, atr_base * 0.4)) / 100 * price
        h = max(o, c) + wick
        l = min(o, c) - wick
        rows.append({"open": o, "high": h, "low": l, "close": c})
    return pd.DataFrame(rows)


def selftest():
    print("=" * 66)
    print("  LIQUIDATION RADAR SELF-TEST")
    print("=" * 66)

    full_h1 = synthetic_df(n=1200, seed=7)

    # ---- calm market ------------------------------------------------
    calm_h1 = full_h1.iloc[:1000].copy().reset_index(drop=True)
    calm_h4 = calm_h1.iloc[::4].reset_index(drop=True)   # 250 bars -> MA exists
    radar = LiquidationRadar()
    r = radar.assess(calm_h1, calm_h4,
                     live_spread=0.30, normal_spread=0.30,
                     now=datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc))
    print("\n[Calm market - normal spread, mid-month]")
    print(f"  score={r.score:<3} mode={r.mode:<10} triggers={r.triggers}")
    assert r.mode == "GREEN", f"calm market should be GREEN, got {r.mode}"

    # ---- violent month-end sweep into the H4 200 MA -----------------
    viol_h1 = full_h1.iloc[:1000].copy().reset_index(drop=True)
    viol_h4 = viol_h1.iloc[::4].reset_index(drop=True)
    ma = float(viol_h4["close"].rolling(200).mean().iloc[-1])
    atr0 = float(rolling_atr(viol_h1).iloc[-20])
    start = float(viol_h1["close"].iloc[-1])
    step = (ma - start) / 10.0

    # rewrite last 10 candles: sustained one-directional liquidation
    # sweep that lands exactly on the H4 200 MA
    idx = viol_h1.index[-10:]
    cur = start
    for j, i in enumerate(idx):
        o = cur
        c = start + step * (j + 1)
        wick = atr0 * 0.3
        viol_h1.loc[i, "open"] = o
        viol_h1.loc[i, "close"] = c
        viol_h1.loc[i, "high"] = max(o, c) + wick
        viol_h1.loc[i, "low"] = min(o, c) - wick
        cur = c

    r = radar.assess(viol_h1, viol_h4,
                     live_spread=1.20, normal_spread=0.30,
                     recent_stop_outs=3,
                     now=datetime(2026, 10, 30, 21, 0, tzinfo=timezone.utc))
    print("\n[Violent sweep - month-end, spread blown out, 3 stops]")
    print(f"  score={r.score:<3} mode={r.mode:<10} triggers={r.triggers}")
    print(f"  distance to H4 200MA: {r.ma_distance_atr:.2f} ATR")
    print(f"  ATR expansion: {r.atr_ratio:.2f}x baseline")
    print(f"  sweep range: {r.sweep_ratio:.1f} ATR over 10 candles")
    assert r.mode in ("STAND_DOWN", "FLATTEN"), "violent sweep must trigger"
    assert "RETEST_ZONE" in r.triggers
    assert "VIOLENT_SWEEP" in r.triggers

    # ---- events log --------------------------------------------------
    print("\n[Logged liquidation events]")
    for e in radar.events:
        print(f"  {e['time'][:19]}  score={e['score']}  mode={e['mode']}"
              f"  price=${e['price']:.2f}  triggers={','.join(e['triggers'])}")

    print("\n  PASS - radar flips GREEN -> FLATTEN on the month-end sweep.")
    print("  Live: the bot refuses new trades at STAND_DOWN and closes")
    print("  everything at FLATTEN. That is what protects a 3-month")
    print("  winning streak from one month-end candle.")
    print("=" * 66)


if __name__ == "__main__":
    selftest()
