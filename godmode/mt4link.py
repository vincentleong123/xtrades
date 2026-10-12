"""MetaTrader5-compatible shim over the xtbridge MT4 file protocol.

install_backend(cfg) is called from run_live() BEFORE ``import
MetaTrader5``; when strategy_config.json says ``"backend": "mt4"`` it
registers this module as ``sys.metaTrader5``-equivalent
(``sys.modules["MetaTrader5"]``), so every ``import MetaTrader5 as mt5``
in the bot transparently talks to the MT4 terminal through files in
MQL4\\Files\\.

protocol (each side owns its files, no shared renames):
  python -> EA : xtb_req_<pid>_<seq>.txt   (write tmp, atomic rename)
  EA -> python : xtb_r<pid>_<seq>.txt      (written whole, then deleted
                                            by python after reading)
  EA heartbeat : xtb_alive.txt             (mtime = freshness)
"""

import json
import os
import sys
import threading
import time
import types

import numpy as np

# ---------------------------------------------------------------- paths
_HERE = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.environ.get("XTB_FILES_DIR") or (
    r"C:\Users\User\AppData\Roaming\MetaQuotes\Terminal"
    r"\50CA3DFB510CC5A8F28B48D1BF2A5702\MQL4\Files")

# ------------------------------------------------------------ constants
TIMEFRAME_M1 = 1
TIMEFRAME_M2 = 2
TIMEFRAME_M3 = 3
TIMEFRAME_M4 = 4
TIMEFRAME_M5 = 5
TIMEFRAME_M6 = 6
TIMEFRAME_M10 = 10
TIMEFRAME_M12 = 12
TIMEFRAME_M15 = 15
TIMEFRAME_M20 = 20
TIMEFRAME_M30 = 30
TIMEFRAME_H1 = 16385
TIMEFRAME_H2 = 16386
TIMEFRAME_H3 = 16387
TIMEFRAME_H4 = 16388
TIMEFRAME_H6 = 16390
TIMEFRAME_H8 = 16392
TIMEFRAME_H12 = 16396
TIMEFRAME_D1 = 16408
TIMEFRAME_W1 = 32769
TIMEFRAME_MN1 = 49153

ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TYPE_BUY_LIMIT = 2
ORDER_TYPE_SELL_LIMIT = 3
ORDER_TYPE_BUY_STOP = 4
ORDER_TYPE_SELL_STOP = 5
POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1

TRADE_ACTION_DEAL = 1
TRADE_ACTION_PENDING = 2
TRADE_ACTION_SLTP = 3
TRADE_ACTION_MODIFY = 4
TRADE_ACTION_REMOVE = 5

TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_REQUOTE = 10010
TRADE_RETCODE_REJECT = 10011
TRADE_RETCODE_INVALID = 10013
TRADE_RETCODE_INVALID_VOLUME = 10014
TRADE_RETCODE_INVALID_PRICE = 10015
TRADE_RETCODE_INVALID_STOPS = 10016
TRADE_RETCODE_TRADE_DISABLED = 10018
TRADE_RETCODE_NO_MONEY = 10019
TRADE_RETCODE_POSITION_CLOSED = 10034

ORDER_TIME_GTC = 0
ORDER_TIME_DAY = 1
ORDER_FILLING_FOK = 0
ORDER_FILLING_IOC = 1
ORDER_FILLING_RETURN = 2

DEAL_ENTRY_IN = 0
DEAL_ENTRY_OUT = 1
DEAL_ENTRY_INOUT = 2
DEAL_ENTRY_OUT_BY = 3
DEAL_REASON_CLIENT = 0
DEAL_REASON_MOBILE = 1
DEAL_REASON_WEB = 2
DEAL_REASON_EXPERT = 3
DEAL_REASON_SL = 4
DEAL_REASON_TP = 5
DEAL_REASON_SO = 6

ACCOUNT_TRADE_MODE_DEMO = 0
ACCOUNT_TRADE_MODE_CONTEST = 1
ACCOUNT_TRADE_MODE_REAL = 2

_RATE_DTYPE = np.dtype([
    ("time", "<u8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"),
    ("close", "<f8"), ("tick_volume", "<i8"), ("spread", "<i4"),
    ("real_volume", "<i8")])


def _tf_minutes(tf):
    tf = int(tf)
    if 1 <= tf <= 44640:            # plain minutes (M1..H12 as minutes too)
        if tf <= 30 or tf in (60, 120, 180, 240, 360, 480, 720, 1440,
                              10080, 43200):
            return tf
    if 16385 <= tf <= 16396:        # H1..H12 hourly codes
        return (tf & 0x3FFF) * 60
    if tf == TIMEFRAME_D1:
        return 1440
    if tf == TIMEFRAME_W1:
        return 10080
    if tf == TIMEFRAME_MN1:
        return 43200
    raise ValueError("unsupported timeframe code %s" % tf)


# --------------------------------------------------------------- bridge
class _Bridge(object):
    def __init__(self):
        self.lock = threading.RLock()
        self.seq = 0
        self.err = (0, "OK")
        self.pid = os.getpid()
        self._cache = {}

    # ---- cache helpers (TTL) ----
    def cached(self, key, ttl):
        hit = self._cache.get(key)
        if hit and hit[1] > time.time():
            return hit[0]
        return None

    def put(self, key, value, ttl):
        self._cache[key] = (value, time.time() + ttl)
        return value

    # ---- core call ----
    def call(self, lines, timeout=4.0):
        with self.lock:
            self.seq += 1
            seq = self.seq
            body = "SEQ|%d\n%s\n" % (seq, "\n".join(lines))
            base = FILES_DIR
            final = os.path.join(base, "xtb_req_%d_%d.txt" % (self.pid, seq))
            part = final + ".part"
            try:
                if not os.path.isdir(base):
                    os.makedirs(base, exist_ok=True)
                with open(part, "w", encoding="ascii", errors="replace") as f:
                    f.write(body)
                    f.flush()
                    os.fsync(f.fileno())
                for _ in range(12):
                    try:
                        os.replace(part, final)
                        break
                    except PermissionError:
                        time.sleep(0.05)
                else:
                    raise PermissionError("req replace refused")
            except OSError as e:
                self.err = (-1, "bridge write failed: %s" % e)
                _silent(part)
                return None

            resp_path = os.path.join(base, "xtb_r%d_%d.txt" % (self.pid, seq))
            t0 = time.time()
            while time.time() - t0 < timeout:
                if os.path.exists(resp_path):
                    txt = None
                    try:
                        with open(resp_path, "r", encoding="ascii",
                                  errors="replace") as f:
                            txt = f.read()
                    except OSError:
                        time.sleep(0.012)
                        continue
                    out = [l for l in txt.replace("\r", "").split("\n")]
                    while out and out[-1] == "":
                        out.pop()
                    if out and out[0] == "SEQ|%d" % seq and len(out) > 1:
                        _silent(resp_path)
                        self.err = (0, "OK")
                        return out[1:]
                    # partial write - wait for more
                time.sleep(0.012)

            # timeout: our order may or may not have been executed
            for _ in range(6):
                if not os.path.exists(final):
                    break
                try:
                    os.remove(final)
                    break
                except OSError:
                    time.sleep(0.05)
            self.err = (-1, "xtbridge timeout after %.1fs (EA busy/not "
                            "attached?) - side effects UNKNOWN" % timeout)
            return None

    def sweep(self, max_age=600.0):
        """drop orphaned bridge files (crashed runs)"""
        now = time.time()
        try:
            for fn in os.listdir(FILES_DIR):
                if not (fn.startswith("xtb_r") or fn.startswith("xtb_req_")):
                    continue
                p = os.path.join(FILES_DIR, fn)
                try:
                    if now - os.path.getmtime(p) > max_age:
                        os.remove(p)
                except OSError:
                    pass
        except OSError:
            pass


_B = _Bridge()
_swept = False


def _silent(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _lines_to_rows(lines, head_prefix, row_prefix):
    """generic parser: header line + data lines"""
    if not lines or not lines[0].startswith(head_prefix):
        return None
    rows = []
    for ln in lines[1:]:
        if ln.startswith(row_prefix):
            rows.append(ln.split("|", 40))
    return rows


def _alive_info():
    p = os.path.join(FILES_DIR, "xtb_alive.txt")
    try:
        age = time.time() - os.path.getmtime(p)
        txt = ""
        with open(p, "r", encoding="ascii", errors="replace") as f:
            txt = f.read().strip()
        return {"age": round(age, 1), "raw": txt}
    except OSError:
        return {"age": None, "raw": None}


# ------------------------------------------------------------ API types
class _Obj(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def _asdict(self):
        return dict(self.__dict__)

    def __repr__(self):
        kv = ", ".join("%s=%r" % kv for kv in sorted(self.__dict__.items()))
        return "%s(%s)" % (type(self).__name__, kv)


def _parse_acc(line):
    p = line.split("|", 12)
    return _Obj(
        login=int(p[1]), balance=float(p[2]), equity=float(p[3]),
        margin_free=float(p[4]), margin=float(p[5]), profit=float(p[6]),
        server=p[7], trade_mode=int(p[8]), leverage=int(p[9]),
        currency=p[10], name="60453165", company="HFMarketsSV",
        margin_level=0.0, margin_initial=0.0, margin_maintenance=0.0,
        last_update_date=0)


def _parse_ti(line):
    p = line.split("|")
    return _Obj(trade_allowed=bool(int(p[1])), expert_allowed=bool(int(p[2])),
                connected=bool(int(p[3])), language="en", cpu_cores=4,
                ram=8, trade_allowed_disabled=False)


def _parse_sym(symbol, line):
    p = line.split("|", 22)
    return _Obj(
        name=symbol, path=symbol, visible=True, session_deals=0,
        session_buy=0, session_sell=0, volume=0, time=0,
        digits=int(p[2]), point=float(p[3]), spread=float(p[4]),
        volume_min=float(p[5]), volume_max=float(p[6]),
        volume_step=float(p[7]), trade_contract_size=float(p[8]),
        stops_level=int(p[9]), freeze_level=int(p[10]),
        tick_value=float(p[11]), tick_size=float(p[12]),
        swap_long=float(p[13]), swap_short=float(p[14]),
        trade_mode=int(p[15]), bid=float(p[16]), ask=float(p[17]),
        filling_mode=int(p[18]) if len(p) > 18 else 2,
        trade_execution_mode=0, exchange=0, swap_mode=0,
        profit_calc_mode=0, margin_calc_mode=0, trade_stop_level=0,
        tradeContracts=0)


def _tick(line):
    # TICK|1|bid|ask|time|vol
    p = line.split("|")
    if len(p) < 5 or p[1] == "0":
        return None
    return _Obj(time=int(float(p[4])), bid=float(p[2]), ask=float(p[3]),
                last=0.0, volume=0, volume_real=0, flags=0)


def _parse_pos(row, symbol=None):
    # P|ticket|type|open|vol|profit|sl|tp|magic|time|swap|symbol
    return _Obj(
        ticket=int(row[1]), type=int(row[2]), price_open=float(row[3]),
        volume=float(row[4]), profit=float(row[5]), sl=float(row[6]),
        tp=float(row[7]), magic=int(row[8]), time=int(row[9]),
        swap=float(row[10]), symbol=row[11] if len(row) > 11 else (symbol or ""),
        identifier=int(row[1]), comment="", reason=0, time_update=int(row[9]),
        price_current=0.0, sl_price_current=0.0, tp_price_current=0.0)


def _parse_ord(row, symbol=None):
    # O|ticket|type|price|vol|sl|tp|magic|time|symbol
    return _Obj(
        ticket=int(row[1]), type=int(row[2]), price_open=float(row[3]),
        volume_initial=float(row[4]), volume_current=float(row[4]),
        sl=float(row[5]), tp=float(row[6]), magic=int(row[7]),
        time_setup=int(row[8]),
        symbol=row[9] if len(row) > 9 else (symbol or ""),
        comment="", time_expiration=0, price_current=0.0,
        type_filling=0, type_time=0, position_id=0)


def _reason_for(pos_type, sl, tp, close_px):
    tol = 0.02
    if pos_type == ORDER_TYPE_BUY:
        if sl > 0 and close_px <= sl + tol:
            return DEAL_REASON_SL
        if tp > 0 and close_px >= tp - tol:
            return DEAL_REASON_TP
    else:
        if sl > 0 and close_px >= sl - tol:
            return DEAL_REASON_SL
        if tp > 0 and close_px <= tp - tol:
            return DEAL_REASON_TP
    return DEAL_REASON_EXPERT


def _row_to_deals(row, from_ts=None, to_ts=None):
    # H|ticket|type|opentime|closetime|openpx|closepx|vol|comm|swap|
    #   profit|sl|tp|magic|symbol|comment
    if not row or row[0] != "H":
        return []
    ticket = int(row[1])
    pos_type = int(row[2])
    t_open = int(row[3])
    t_close = int(row[4])
    open_px = float(row[5])
    close_px = float(row[6])
    vol = float(row[7])
    comm = float(row[8])
    swap = float(row[9])
    profit = float(row[10])
    sl = float(row[11])
    tp = float(row[12])
    magic = int(row[13])
    symbol = row[14]
    comment = row[15] if len(row) > 15 else ""
    deals = []
    if from_ts is None or (from_ts <= t_open <= (to_ts or from_ts)):
        deals.append(_Obj(
            ticket=ticket, order=ticket, position_id=ticket,
            symbol=symbol, magic=magic, time=t_open, volume=vol,
            profit=0.0, commission=0.0, swap=0.0, price=open_px,
            comment="in " + comment, entry=DEAL_ENTRY_IN, type=pos_type,
            reason=DEAL_REASON_EXPERT, position=ticket))
    deals.append(_Obj(
        ticket=ticket, order=ticket, position_id=ticket, symbol=symbol,
        magic=magic, time=t_close, volume=vol, profit=profit,
        commission=comm, swap=swap, price=close_px, comment=comment,
        entry=DEAL_ENTRY_OUT, type=pos_type,
        reason=_reason_for(pos_type, sl, tp, close_px), position=ticket))
    return deals


# ---------------------------------------------------------- API section
def initialize(path=None, login=0, password="", server="", portable=False,
               *args, **kwargs):
    global _swept
    if not _swept:
        _B.sweep()
        _swept = True
    lines = _B.call(["ACC"], timeout=3.0)
    if not lines or not lines[0].startswith("ACC|"):
        _B.err = (-1, "xtbridge EA not running - attach xtbridge.ex4 to a "
                      "chart in the MT4 terminal (AutoTrading ON); alive=%r"
                      % _alive_info())
        return False
    return True


def shutdown():
    _B._cache.clear()
    return True


def last_error():
    return tuple(_B.err)


def account_info():
    hit = _B.cached("acc", 0.4)
    if hit is not None:
        return hit
    lines = _B.call(["ACC"], timeout=3.0)
    if not lines or not lines[0].startswith("ACC|"):
        return None
    return _B.put("acc", _parse_acc(lines[0]), 0.4)


def terminal_info():
    hit = _B.cached("ti", 0.4)
    if hit is not None:
        return hit
    lines = _B.call(["TI"], timeout=3.0)
    if not lines or not lines[0].startswith("TI|"):
        return None
    return _B.put("ti", _parse_ti(lines[0]), 0.4)


def symbol_info(symbol):
    key = "sym:" + str(symbol)
    hit = _B.cached(key, 0.5)
    if hit is not None:
        return hit if hit else None
    lines = _B.call(["SYM|" + str(symbol)], timeout=3.0)
    if not lines or lines[0] == "SYM|0" or not lines[0].startswith("SYM|1|"):
        _B.put(key, False, 2.0)
        return None
    return _B.put(key, _parse_sym(str(symbol), lines[0]), 0.5)


def symbol_select(symbol, add=True):
    key = "sel:" + str(symbol)
    hit = _B.cached(key, 5.0)
    if hit is not None:
        return hit
    lines = _B.call(["SEL|" + str(symbol)], timeout=3.0)
    ok = bool(lines and lines[0] == "SEL|1")
    return _B.put(key, ok, 5.0)


def symbol_info_tick(symbol):
    lines = _B.call(["TICK|" + str(symbol)], timeout=3.0)
    if not lines:
        return None
    return _tick(lines[0])


def _resample_m1(rows, factor):
    """Aggregate M1 bridge rows into factor-minute bars, grouped by epoch
    bucket (gap-proof: session breaks never merge into franken-bars).
    Partial edge groups are emitted, never dropped (the forming bar matters
    most). Returns row tuples in the same (t,o,h,l,c,tickvol,spread,0) shape.
    Leeway: shortfalls pass through as fewer bars; callers already handle
    short arrays and None."""
    span = 60 * factor
    out = []
    cur_key = None
    acc = None
    for r in rows:
        key = int(r[0]) // span
        if key != cur_key:
            if acc is not None:
                out.append(tuple(acc))
            cur_key = key
            acc = [key * span, r[1], r[2], r[3], r[4], r[5], r[6], 0]
        else:
            if r[2] > acc[2]:
                acc[2] = r[2]
            if r[3] < acc[3]:
                acc[3] = r[3]
            acc[4] = r[4]
            acc[5] = acc[5] + r[5]
            acc[6] = r[6]
    if acc is not None:
        out.append(tuple(acc))
    return out


def copy_rates_from_pos(symbol, timeframe, start_pos, count):
    minutes = _tf_minutes(timeframe)
    want = int(count) + max(0, int(start_pos))
    # MQL4 has no M2/M3/... timeframe: fetch M1 and aggregate locally so
    # every minute-based TF works without touching the EA.
    _MQL4_NATIVE = {1, 5, 15, 30, 60, 240, 1440, 10080, 43200}
    fetch_minutes, factor = (1, minutes) if minutes not in _MQL4_NATIVE else (minutes, 1)
    lines = _B.call(["BARS|%s|%d|%d" % (symbol, fetch_minutes, min(want * factor, 20000))], timeout=6.0)
    if not lines or not lines[0].startswith("BARS|1|"):
        return None
    try:
        n = int(lines[0].split("|")[2])
    except (IndexError, ValueError):
        return None
    rows = []
    for ln in lines[1:1 + n]:
        p = ln.split("|")
        if len(p) < 7 or p[0] != "B":
            continue
        rows.append((int(p[1]), float(p[2]), float(p[3]), float(p[4]),
                     float(p[5]), int(p[6]), int(p[7]), 0))
    if not rows:
        return None
    if factor != 1:
        rows = _resample_m1(rows, factor)
        if not rows:
            return None
    arr = np.array(rows, dtype=_RATE_DTYPE)
    start = max(0, int(start_pos))
    if start:
        if len(arr) <= start:
            return None
        arr = arr[len(arr) - int(count):] if len(arr) > int(count) \
            else arr[start:]
    return arr


def copy_rates_range(symbol, timeframe, date_from, date_to):
    """not supported by the bridge - use copy_rates_from_pos"""
    _B.err = (-1, "copy_rates_range unsupported on mt4 bridge")
    return None


def copy_ticks_range(symbol, date_from, date_to, flags=None):
    _B.err = (-1, "copy_ticks_range unsupported on mt4 bridge")
    return None


def positions_get(symbol=None, ticket=None, magic=None, *args, **kwargs):
    sym = str(symbol) if symbol else "*"
    lines = _B.call(["POS|" + sym], timeout=3.0)
    if lines is None:
        return None
    out = []
    for ln in lines:
        if not ln.startswith("P|"):
            continue
        p = _parse_pos(ln.split("|"))
        if ticket is not None and p.ticket != int(ticket):
            continue
        if magic is not None and p.magic != int(magic):
            continue
        out.append(p)
    if ticket is not None and not out:
        return None
    return tuple(out)


def orders_get(symbol=None, *args, **kwargs):
    sym = str(symbol) if symbol else "*"
    lines = _B.call(["ORD|" + sym], timeout=3.0)
    if lines is None:
        return None
    out = []
    for ln in lines:
        if ln.startswith("O|"):
            out.append(_parse_ord(ln.split("|")))
    return tuple(out)


def order_calc_margin(order_type, symbol, volume, price=None, *args, **kwargs):
    lines = _B.call(["MAR|%s|%s" % (symbol, float(volume))], timeout=3.0)
    if not lines or not lines[0].startswith("MAR|"):
        return 0.0
    try:
        return float(lines[0].split("|")[1])
    except (IndexError, ValueError):
        return 0.0


def _ret(retcode, order=0, price=0.0, comment="", volume=0.0):
    return _Obj(retcode=int(retcode), deal=int(order), order=int(order),
                volume=float(volume), price=float(price),
                comment=str(comment), request_id=0, external_return=0)


def order_send(req, timeout=6.0):
    action = req.get("action")
    sym = str(req.get("symbol", "") or "")
    vol = float(req.get("volume", 0) or 0)
    typ = int(req.get("type", 0) or 0)
    price = float(req.get("price", 0) or 0)
    sl = float(req.get("sl", 0) or 0)
    tp = float(req.get("tp", 0) or 0)
    dev = int(req.get("deviation", 30) or 0)
    magic = int(req.get("magic", 0) or 0)
    cmt = str(req.get("comment", "") or "").replace("|", "/")

    if action == TRADE_ACTION_DEAL:
        if req.get("position") is not None:
            cmd = "SPC|%s|%d|%s|%d|%s|%d|%d|%s" % (
                sym, typ, vol, int(req["position"]), price, dev, magic, cmt)
        else:
            cmd = "SPO|%s|%d|%s|%s|%s|%s|%d|%d|%s" % (
                sym, typ, vol, price, sl, tp, dev, magic, cmt)
    elif action == TRADE_ACTION_PENDING:
        cmd = "SPP|%s|%d|%s|%s|%s|%s|%d|%s" % (
            sym, typ, vol, price, sl, tp, magic, cmt)
    elif action == TRADE_ACTION_SLTP:
        pos = req.get("position", req.get("order", 0))
        cmd = "MOD|%d|%s|%s" % (int(pos), sl, tp)
    elif action == TRADE_ACTION_REMOVE:
        cmd = "RMV|%d" % int(req.get("order", 0))
    else:
        return _ret(TRADE_RETCODE_REJECT, comment="unsupported action %r"
                    % action)

    lines = _B.call([cmd], timeout=timeout)
    if not lines or not lines[0].startswith("R|"):
        return _ret(TRADE_RETCODE_REJECT, volume=vol,
                    comment="xtbridge timeout - order state UNKNOWN; err=%s"
                            % (_B.err[1],))
    p = lines[0].split("|", 6)
    try:
        retcode = int(p[1])
        ticket = int(p[2])
        done_px = float(p[3]) if p[3] else 0.0
        comment = p[4] if len(p) > 4 else ""
    except (IndexError, ValueError):
        return _ret(TRADE_RETCODE_REJECT, comment="bad bridge reply: %r"
                    % lines[:1])
    return _ret(retcode, order=ticket, price=done_px, comment=comment,
                volume=vol)


def history_deals_get(date_from=None, date_to=None, position=None,
                      *args, **kwargs):
    if position is not None:
        lines = _B.call(["HPH|%d" % int(position)], timeout=4.0)
        if not lines or lines[0] == "HPH|0" or not lines[0].startswith("H|"):
            return tuple()
        return tuple(_row_to_deals(lines[0].split("|", 16)))
    if date_from is None or date_to is None:
        _B.err = (-1, "history_deals_get needs dates or position=")
        return tuple()
    try:
        a = int(date_from.timestamp())
        b = int(date_to.timestamp())
    except AttributeError:
        a = int(date_from)
        b = int(date_to)
    lines = _B.call(["HSYN|%d|%d" % (a, b)], timeout=6.0)
    if not lines or not lines[0].startswith("HSYN|"):
        return tuple()
    try:
        n = int(lines[0].split("|")[1])
    except (IndexError, ValueError):
        n = 0
    if n == 0:
        return tuple()
    deals = []
    for ln in lines[1:1 + n]:
        deals.extend(_row_to_deals(ln.split("|", 16), a, b))
    deals.sort(key=lambda d: (d.time, d.entry))
    return tuple(deals)


def history_orders_get(*args, **kwargs):
    _B.err = (-1, "history_orders_get unsupported on mt4 bridge")
    return None


def orders_total():
    lines = _B.call(["ORD|*"], timeout=3.0)
    if lines is None:
        return 0
    return sum(1 for ln in lines if ln.startswith("O|"))


def positions_total():
    lines = _B.call(["POS|*"], timeout=3.0)
    if lines is None:
        return 0
    return sum(1 for ln in lines if ln.startswith("P|"))


def market_book_add(symbol):
    return False


def market_book_get(symbol):
    return None


def symbol_info_tick_fresh():
    """diagnostic helper (probe): alive-file info"""
    return _alive_info()


# ----------------------------------------------------------- installers
def _build_module():
    m = types.ModuleType("MetaTrader5")
    m._xtb_shim = True
    m.__doc__ = "xtbridge MT4 shim pretending to be MetaTrader5"
    api = {
        "initialize": initialize, "shutdown": shutdown,
        "last_error": last_error, "account_info": account_info,
        "terminal_info": terminal_info, "symbol_info": symbol_info,
        "symbol_select": symbol_select, "symbol_info_tick": symbol_info_tick,
        "copy_rates_from_pos": copy_rates_from_pos,
        "copy_rates_range": copy_rates_range,
        "copy_ticks_range": copy_ticks_range,
        "positions_get": positions_get, "orders_get": orders_get,
        "order_calc_margin": order_calc_margin, "order_send": order_send,
        "history_deals_get": history_deals_get,
        "history_orders_get": history_orders_get,
        "orders_total": orders_total, "positions_total": positions_total,
        "market_book_add": market_book_add, "market_book_get": market_book_get,
        "alive_info": symbol_info_tick_fresh,
    }
    for k, v in api.items():
        setattr(m, k, v)
    for k, v in list(globals().items()):
        if k.startswith("TIMEFRAME_") or k.startswith("ORDER_TYPE_") \
                or k.startswith("POSITION_TYPE_") \
                or k.startswith("TRADE_ACTION_") \
                or k.startswith("TRADE_RETCODE_") \
                or k.startswith("ORDER_TIME_") \
                or k.startswith("ORDER_FILLING_") \
                or k.startswith("DEAL_ENTRY_") \
                or k.startswith("DEAL_REASON_") \
                or k.startswith("ACCOUNT_TRADE_MODE_"):
            setattr(m, k, v)
    return m


def install_backend(cfg=None):
    """Return True if the MetaTrader5 module was replaced by the shim.

    Called from run_live() before ``import MetaTrader5``. Reads
    strategy_config.json when cfg is None; installs only when
    cfg["backend"] == "mt4".
    """
    if cfg is None:
        try:
            with open(os.path.join(_HERE, "strategy_config.json")) as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    if str((cfg or {}).get("backend", "mt5")).lower() != "mt4":
        return False
    return force_install()


def force_install():
    if "MetaTrader5" in sys.modules and \
            getattr(sys.modules["MetaTrader5"], "_xtb_shim", False):
        return True
    sys.modules["MetaTrader5"] = _build_module()
    return True
