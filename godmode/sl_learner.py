"""
SL Learner — autopsy every stop-loss, relearn the no-trade zones
================================================================
The feedback loop the user asked for:

  run -> gather SL trades with their ENTRY CONTEXT
       -> compare losing contexts vs winning contexts
       -> find zones where SL rate is statistically deadly
       -> turn those zones into hard filters for the NEXT run
       -> persist everything in learning_db.json

A zone becomes a NO-TRADE rule only if it earned it:
    SL rate >= 60%  AND  sample count >= 5
No data, no rules. The system learns from real pain,
not from imagination. That is the honest version of "AI".

Usage:
    learner = SLLearner()
    ctx = learner.market_ctx(...)          # at entry
    ok, why = learner.gate(ctx)            # before opening
    learner.record_completed(ctx, pnl, reason)   # at exit
    report = learner.relearn()             # after N closes
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "learning_db.json")

# feature -> list of bins  (label, min inclusive, max exclusive)
BINS: Dict[str, List[tuple]] = {
    "atr_ratio":   [("calm <1.0", 0.0, 1.0), ("warm 1.0-1.5", 1.0, 1.5),
                    ("hot 1.5-2.0", 1.5, 2.0), ("extreme >=2.0", 2.0, 1e9)],
    "radar_score": [("calm <30", 0, 30), ("elevated 30-60", 30, 60),
                    ("danger >=60", 60, 101)],
    "ma_dist_atr": [("on_the_ma <1", 0.0, 1.0), ("near 1-3", 1.0, 3.0),
                    ("far >=3", 3.0, 1e9)],
    "sweep_ratio": [("quiet <3", 0.0, 3.0), ("rough 3-6", 3.0, 6.0),
                    ("violent >=6", 6.0, 1e9)],
    "hour4":       [("h0-3", 0, 4), ("h4-7", 4, 8), ("h8-11", 8, 12),
                    ("h12-15", 12, 16), ("h16-19", 16, 20), ("h20-23", 20, 24)],
    "ema_gap_pct": [("weak <0.05", 0.0, 0.05), ("ok 0.05-0.15", 0.05, 0.15),
                    ("strong >=0.15", 0.15, 1e9)],
    "direction":   [("LONG", None, None), ("SHORT", None, None)],
}


@dataclass
class SLTradeRecord:
    ts: str
    direction: str
    lots: float
    entry: float
    exit: Optional[float]
    pnl: float
    reason: str
    atr_ratio: float
    radar_score: int
    ma_dist_atr: float
    sweep_ratio: float
    spread: float
    hour: int
    ema_gap_pct: float


@dataclass
class Rule:
    feature: str
    label: str
    sl_rate: float
    n: int
    damage: float
    created: str


class SLLearner:
    def __init__(self, min_samples: int = 5, sl_rate_trigger: float = 0.60,
                 db_path: str = DB_PATH):
        self.min_samples = min_samples
        self.sl_rate_trigger = sl_rate_trigger
        self.db_path = db_path
        self.trades: List[SLTradeRecord] = []
        self.rules: List[Rule] = []
        self.history: List[dict] = []
        self._load()

    # ------------------------------------------------------------------ io
    def _load(self):
        if not os.path.exists(self.db_path):
            return
        try:
            with open(self.db_path, "r") as f:
                raw = json.load(f)
            self.trades = [SLTradeRecord(**t) for t in raw.get("trades", [])]
            self.rules = [Rule(**r) for r in raw.get("rules", [])]
            self.history = raw.get("history", [])
        except Exception:
            self.trades, self.rules, self.history = [], [], []

    def save(self):
        tmp = self.db_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "trades": [asdict(t) for t in self.trades],
                "rules": [asdict(r) for r in self.rules],
                "history": self.history[-100:],
            }, f, indent=2)
        os.replace(tmp, self.db_path)

    # ------------------------------------------------------------- context
    @staticmethod
    def market_ctx(direction: str, atr_ratio: float, radar_score: int,
                   ma_dist_atr: float, sweep_ratio: float, spread: float,
                   hour: int, ema_gap_pct: float) -> Dict[str, Any]:
        return {
            "direction": direction,
            "atr_ratio": round(float(atr_ratio), 3),
            "radar_score": int(radar_score),
            "ma_dist_atr": round(float(ma_dist_atr), 3) if ma_dist_atr == ma_dist_atr else 99.0,
            "sweep_ratio": round(float(sweep_ratio), 3),
            "spread": round(float(spread), 3),
            "hour": int(hour),
            "ema_gap_pct": round(float(ema_gap_pct), 5),
        }

    # ---------------------------------------------------------------- gate
    def gate(self, ctx: Dict[str, Any]) -> (bool, str):
        """Check ctx against learned no-trade zones BEFORE opening."""
        for r in self.rules:
            hit = self._in_bin(ctx, r.feature, r.label)
            if hit:
                return False, f"filter[{r.feature}:{r.label} sl{r.sl_rate:.0%}/{r.n}]"
        return True, ""

    @staticmethod
    def _bin_label(feature: str, value) -> Optional[str]:
        if feature == "direction":
            return value if value in ("LONG", "SHORT") else None
        if feature == "hour4":
            for label, lo, hi in BINS["hour4"]:
                if lo <= value < hi:
                    return label
            return None
        for label, lo, hi in BINS.get(feature, []):
            try:
                if lo <= value < hi:
                    return label
            except TypeError:
                return None
        return None

    def _in_bin(self, ctx: Dict[str, Any], feature: str, label: str) -> bool:
        if feature not in ctx:
            return False
        if feature == "hour4":
            return self._bin_label("hour4", ctx.get("hour", 0)) == label
        if feature == "direction":
            return ctx.get("direction") == label
        return self._bin_label(feature, ctx.get(feature)) == label

    # --------------------------------------------------------------- record
    def record_completed(self, ctx: Dict[str, Any], pnl: float, reason: str,
                         lots: float, entry: float, exit_price: Optional[float]):
        rec = SLTradeRecord(
            ts=datetime.now(timezone.utc).isoformat(),
            direction=ctx.get("direction", "?"),
            lots=lots,
            entry=entry,
            exit=exit_price,
            pnl=round(float(pnl), 2),
            reason=reason,
            atr_ratio=ctx.get("atr_ratio", 0.0),
            radar_score=ctx.get("radar_score", 0),
            ma_dist_atr=ctx.get("ma_dist_atr", 99.0),
            sweep_ratio=ctx.get("sweep_ratio", 0.0),
            spread=ctx.get("spread", 0.0),
            hour=ctx.get("hour", 0),
            ema_gap_pct=ctx.get("ema_gap_pct", 0.0),
        )
        self.trades.append(rec)
        # persist immediately - a crash must never lose the autopsy history
        self.save()

    # --------------------------------------------------------------- relearn
    def relearn(self) -> dict:
        """Rebuild rules from ALL accumulated trade records."""
        now = datetime.now(timezone.utc).isoformat()
        buckets: Dict[tuple, dict] = defaultdict(
            lambda: {"n": 0, "sl": 0, "damage": 0.0})

        for t in self.trades:
            for feature in BINS:
                value = getattr(t, "hour", 0) if feature == "hour4" else \
                        (t.direction if feature == "direction" else getattr(t, feature))
                label = self._bin_label(feature, value)
                if label is None:
                    continue
                b = buckets[(feature, label)]
                b["n"] += 1
                if t.reason == "SL":
                    b["sl"] += 1
                    b["damage"] += abs(t.pnl)

        new_rules: List[Rule] = []
        zones = []
        for (feature, label), b in buckets.items():
            sl_rate = b["sl"] / b["n"] if b["n"] else 0.0
            zones.append({
                "feature": feature, "label": label, "n": b["n"],
                "sl_rate": round(sl_rate, 3), "damage": round(b["damage"], 2),
            })
            if b["n"] >= self.min_samples and sl_rate >= self.sl_rate_trigger:
                new_rules.append(Rule(
                    feature=feature, label=label,
                    sl_rate=round(sl_rate, 3), n=b["n"],
                    damage=round(b["damage"], 2), created=now))

        # worst zones first (most damage)
        new_rules.sort(key=lambda r: -r.damage)
        self.rules = new_rules

        total = len(self.trades)
        sl_n = sum(1 for t in self.trades if t.reason == "SL")
        entry_hist = {
            "ts": now, "trades_in_db": total,
            "overall_sl_rate": round(sl_n / total, 3) if total else 0.0,
            "rules_active": len(self.rules),
            "worst_zone": (zones and max(zones, key=lambda z: z["damage"]) or None),
        }
        self.history.append(entry_hist)
        self.save()
        return entry_hist

    # --------------------------------------------------------------- summary
    def summary(self) -> dict:
        sl_trades = [t for t in self.trades if t.reason == "SL"]
        wins = [t for t in self.trades if t.pnl > 0]
        zones_by_damage = []
        buckets: Dict[tuple, dict] = defaultdict(
            lambda: {"n": 0, "sl": 0, "damage": 0.0})
        for t in self.trades:
            for feature in BINS:
                value = getattr(t, "hour", 0) if feature == "hour4" else \
                        (t.direction if feature == "direction" else getattr(t, feature))
                label = self._bin_label(feature, value)
                if label is None:
                    continue
                b = buckets[(feature, label)]
                b["n"] += 1
                if t.reason == "SL":
                    b["sl"] += 1
                    b["damage"] += abs(t.pnl)
        for (feature, label), b in buckets.items():
            zones_by_damage.append({
                "feature": feature, "label": label, "n": b["n"],
                "sl_rate": round(b["sl"] / b["n"], 3) if b["n"] else 0.0,
                "damage": round(b["damage"], 2),
            })
        zones_by_damage.sort(key=lambda z: -z["damage"])

        return {
            "trades_in_db": len(self.trades),
            "sl_count": len(sl_trades),
            "wins": len(wins),
            "overall_sl_rate": round(len(sl_trades) / len(self.trades), 3)
                                if self.trades else 0.0,
            "sl_damage_total": round(sum(abs(t.pnl) for t in sl_trades), 2),
            "rules": [asdict(r) for r in self.rules],
            "top_loss_zones": zones_by_damage[:8],
            "history": self.history[-8:],
        }


def selftest():
    print("=" * 66)
    print("  SL LEARNER SELF-TEST")
    print("  (synthetic history: hot-ATR entries lose 80% of the time)")
    print("=" * 66)
    test_db = os.path.join(BASE, "_learner_selftest.json")
    if os.path.exists(test_db):
        os.remove(test_db)
    learner = SLLearner(min_samples=5, sl_rate_trigger=0.60,
                        db_path=test_db)

    import random
    random.seed(11)
    # 25 calm-market trades: 50% SL   -> not enough SL rate to ban
    for i in range(25):
        ctx = SLLearner.market_ctx("LONG" if i % 2 else "SHORT",
                                   atr_ratio=0.9, radar_score=10,
                                   ma_dist_atr=5.0, sweep_ratio=2.0,
                                   spread=0.3, hour=10, ema_gap_pct=0.10)
        lost = (i % 2 == 0)
        learner.record_completed(ctx, pnl=-5.0 if lost else 10.0,
                                 reason="SL" if lost else "TP",
                                 lots=0.01, entry=2350.0, exit_price=2345.0)
    # 10 hot-market trades: 9 of 10 stop out -> zone must be banned
    for i in range(10):
        ctx = SLLearner.market_ctx("LONG" if i % 2 else "SHORT",
                                   atr_ratio=1.8, radar_score=55,
                                   ma_dist_atr=0.8, sweep_ratio=4.0,
                                   spread=0.9, hour=21, ema_gap_pct=0.03)
        lost = (i < 9)
        learner.record_completed(ctx, pnl=-8.0 if lost else 16.0,
                                 reason="SL" if lost else "TP",
                                 lots=0.01, entry=2350.0, exit_price=2342.0)

    report = learner.relearn()
    print(f"\n  trades analyzed : {report['trades_in_db']}")
    print(f"  overall SL rate: {report['overall_sl_rate']:.0%}")
    print(f"  rules activated : {report['rules_active']}")
    for r in learner.rules:
        print(f"    - {r.feature} [{r.label}]  SL {r.sl_rate:.0%} of {r.n}"
              f"  damage ${r.damage:.2f}")

    hot_ctx = SLLearner.market_ctx("LONG", atr_ratio=1.9, radar_score=58,
                                   ma_dist_atr=0.8, sweep_ratio=4.5,
                                   spread=0.9, hour=21, ema_gap_pct=0.03)
    ok, why = learner.gate(hot_ctx)
    print(f"\n  gate(hot ATR entry)   -> {'BLOCKED' if not ok else 'allowed'}  ({why})")
    assert not ok, "hot-ATR entry must be blocked after learning"

    calm_ctx = SLLearner.market_ctx("LONG", atr_ratio=0.9, radar_score=10,
                                    ma_dist_atr=5.0, sweep_ratio=2.0,
                                    spread=0.3, hour=10, ema_gap_pct=0.10)
    ok2, why2 = learner.gate(calm_ctx)
    print(f"  gate(calm ATR entry)  -> {'BLOCKED' if not ok2 else 'allowed'}")
    assert ok2, "calm entry must stay allowed"

    # persistence check
    learner2 = SLLearner(db_path=test_db)
    ok3, _ = learner2.gate(hot_ctx)
    print(f"  reload from disk      -> rules survived: {not ok3}")
    assert not ok3, "rules must persist across runs"

    os.remove(test_db)
    print("\n  PASS - SL trades were autopsied, the deadly ATR zone became")
    print("  a no-trade filter, and it survives restarts.")
    print("=" * 66)


if __name__ == "__main__":
    selftest()
