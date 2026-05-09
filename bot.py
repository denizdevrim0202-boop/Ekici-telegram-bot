"""
SMC Crypto Futures Signal Bot — tek dosya, Railway deploy için
GitHub'a yükle → Railway'e bağla → TELEGRAM_TOKEN ve TELEGRAM_CHAT_ID env var ekle → Deploy
"""

import os
import sys
import io
import json
import time
import threading
import logging
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from datetime import datetime, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import schedule
from flask import Flask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "8412174563:AAFOlWXJLnLTDkrNUXALIKJc46DIAk_iEwI")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "1497161616")

MIN_VOLUME_USDT       = 5_000_000
MIN_SCORE             = 7
MAX_ACTIVE_SIGNALS    = 5
MIN_RR                = 3.0
SIGNAL_EXPIRY_HOURS   = 36
COIN_COOLDOWN_HOURS   = 4
SL_BUFFER_PCT         = 0.006
NEAR_ZONE_PCT         = 0.030
ATR_PERIOD            = 14
ATR_MIN_PCT           = 0.003
CHART_CANDLES         = 80
BTC_MOVE_LIMIT_PCT    = 2.0
FUNDING_LONG_MAX      = 0.001
FUNDING_SHORT_MIN     = -0.001
QUIET_HOURS_START     = 0
QUIET_HOURS_END       = 6
VOLUME_SPIKE_MULTIPLIER = 1.5
SPREAD_MAX_PCT        = 0.005

DATA_DIR  = os.environ.get("DATA_DIR", "./data")
DATA_FILE = os.path.join(DATA_DIR, "signals.json")
os.makedirs(DATA_DIR, exist_ok=True)

SMC_MIN_SCORE      = 7
SMC_MAX_SCORE      = 12
SMT_BONUS_CAP      = 13
SWING_LOOKBACK     = 5
SCORING_WINDOW     = 100
SWING_HISTORY      = 200

OB_VOLUME_MULTIPLIER  = 1.0
OB_BODY_MIN_PCT       = 0.60
SWEEP_CONFIRM_CANDLES = 2

TIMEFRAMES    = ["15m", "1h", "2h", "4h"]
MTF_MIN_AGREE = 3

UNIVERSE_CACHE_TTL = 60
BINANCE_CACHE_TTL  = 86400
SESSION_POOL_SIZE  = 50

MEXC_ENDPOINTS = [
    "https://contract.mexc.com",
    "https://futures.mexc.com",
    "https://api.mexc.com",
]

# ─────────────────────────────────────────────
# KEEP-ALIVE  (Flask for Railway / Render)
# ─────────────────────────────────────────────

flask_app = Flask(__name__)

@flask_app.route("/")
def _index():
    return "Signal bot is running.", 200

@flask_app.route("/health")
def _health():
    return {"status": "ok"}, 200

def start_keep_alive():
    port = int(os.environ.get("PORT", 8080))
    def _run():
        logger.info(f"Keep-alive server on port {port}")
        flask_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
    t = threading.Thread(target=_run, daemon=True, name="keep-alive")
    t.start()

# ─────────────────────────────────────────────
# BINANCE LISTING CHECK
# ─────────────────────────────────────────────

_binance_symbols: set = set()
_binance_last_fetch: float = 0.0
_binance_lock = threading.Lock()

BINANCE_FAPI_URL        = "https://fapi.binance.com/fapi/v1/exchangeInfo"
COINGECKO_DERIVATIVES_URL = "https://api.coingecko.com/api/v3/derivatives/exchanges/binance_futures"


def _fetch_binance_symbols() -> set:
    try:
        resp = requests.get(BINANCE_FAPI_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        syms = set()
        for s in data.get("symbols", []):
            if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING":
                syms.add(s["baseAsset"].upper())
        if syms:
            logger.info(f"Binance fapi: {len(syms)} symbols")
            return syms
    except Exception as e:
        logger.warning(f"Binance fapi failed: {e}")
    return set()


def _fetch_coingecko_symbols() -> set:
    try:
        resp = requests.get(COINGECKO_DERIVATIVES_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        syms = set()
        tickers = data if isinstance(data, list) else data.get("tickers", [])
        for item in tickers:
            base   = item.get("base", "").upper()
            target = item.get("target", "").upper()
            if target == "USDT" and base:
                syms.add(base)
        if syms:
            logger.info(f"CoinGecko fallback: {len(syms)} symbols")
            return syms
    except Exception as e:
        logger.warning(f"CoinGecko fallback failed: {e}")
    return set()


def get_binance_listed_symbols() -> set:
    global _binance_symbols, _binance_last_fetch
    now = time.time()
    with _binance_lock:
        if _binance_symbols and (now - _binance_last_fetch) < BINANCE_CACHE_TTL:
            return set(_binance_symbols)
        syms = _fetch_binance_symbols() or _fetch_coingecko_symbols()
        if syms:
            _binance_symbols = syms
            _binance_last_fetch = now
        return set(_binance_symbols)


def is_binance_listed(base_asset: str) -> bool:
    return base_asset.upper() in get_binance_listed_symbols()

# ─────────────────────────────────────────────
# STRUCTURE / INDICATOR HELPERS
# ─────────────────────────────────────────────

def find_swings(df: pd.DataFrame, lookback: int = SWING_LOOKBACK):
    highs, lows = [], []
    n = len(df)
    for i in range(lookback, n - lookback):
        if all(df["high"].iloc[i] > df["high"].iloc[i - j] for j in range(1, lookback + 1)) and \
           all(df["high"].iloc[i] > df["high"].iloc[i + j] for j in range(1, lookback + 1)):
            highs.append((i, df["high"].iloc[i]))
        if all(df["low"].iloc[i] < df["low"].iloc[i - j] for j in range(1, lookback + 1)) and \
           all(df["low"].iloc[i] < df["low"].iloc[i + j] for j in range(1, lookback + 1)):
            lows.append((i, df["low"].iloc[i]))
    return highs, lows


def detect_choch(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> dict:
    highs, lows = find_swings(df, lookback)
    result = {"detected": False, "direction": None, "level": None, "index": None}
    if len(highs) < 2 or len(lows) < 2:
        return result
    last_lh_idx, last_lh = min(highs[-4:], key=lambda x: x[1]) if len(highs) >= 4 else highs[-1]
    last_hl_idx, last_hl = max(lows[-4:],  key=lambda x: x[1]) if len(lows)  >= 4 else lows[-1]
    curr = df["close"].iloc[-1]
    if curr > last_lh:
        result = {"detected": True, "direction": "long",  "level": last_lh, "index": last_lh_idx}
    elif curr < last_hl:
        result = {"detected": True, "direction": "short", "level": last_hl, "index": last_hl_idx}
    return result


def detect_bos(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> dict:
    highs, lows = find_swings(df, lookback)
    result = {"detected": False, "direction": None, "level": None, "index": None}
    if not highs or not lows:
        return result
    curr = df["close"].iloc[-1]
    rh = max(highs[-3:], key=lambda x: x[1]) if len(highs) >= 3 else highs[-1]
    rl = min(lows[-3:],  key=lambda x: x[1]) if len(lows)  >= 3 else lows[-1]
    if curr > rh[1]:
        result = {"detected": True, "direction": "long",  "level": rh[1], "index": rh[0]}
    elif curr < rl[1]:
        result = {"detected": True, "direction": "short", "level": rl[1], "index": rl[0]}
    return result


def detect_order_blocks(df: pd.DataFrame, vol_multiplier: float = OB_VOLUME_MULTIPLIER) -> list:
    obs = []
    avg_vol  = df["volume"].rolling(20).mean()
    curr     = df["close"].iloc[-1]
    n        = len(df)
    for i in range(1, n - 1):
        c = df.iloc[i]
        total_range = c["high"] - c["low"]
        if total_range == 0:
            continue
        body = abs(c["close"] - c["open"])
        if body / total_range < OB_BODY_MIN_PCT:
            continue
        if avg_vol.iloc[i] > 0 and c["volume"] < avg_vol.iloc[i] * vol_multiplier:
            continue
        nxt = df.iloc[i + 1]
        if c["close"] < c["open"] and nxt["close"] > nxt["open"]:
            ob_high = max(c["open"], c["close"])
            ob_low  = min(c["open"], c["close"])
            if curr >= ob_low:
                obs.append({"type": "demand", "high": ob_high, "low": ob_low, "index": i, "mitigated": False})
        elif c["close"] > c["open"] and nxt["close"] < nxt["open"]:
            ob_high = max(c["open"], c["close"])
            ob_low  = min(c["open"], c["close"])
            if curr <= ob_high:
                obs.append({"type": "supply", "high": ob_high, "low": ob_low, "index": i, "mitigated": False})
    return obs


def detect_fvg(df: pd.DataFrame) -> list:
    fvgs = []
    n = len(df)
    for i in range(2, n):
        prev = df.iloc[i - 2]
        curr = df.iloc[i]
        if curr["low"] > prev["high"]:
            fvgs.append({"type": "bullish", "high": curr["low"], "low": prev["high"],
                         "index": i - 1, "midpoint": (curr["low"] + prev["high"]) / 2})
        elif curr["high"] < prev["low"]:
            fvgs.append({"type": "bearish", "high": prev["low"], "low": curr["high"],
                         "index": i - 1, "midpoint": (prev["low"] + curr["high"]) / 2})
    return fvgs


def detect_eqh_eql(df: pd.DataFrame, lookback: int = SWING_LOOKBACK, tolerance: float = 0.003) -> dict:
    highs, lows = find_swings(df, lookback)
    eqh, eql = [], []
    for i in range(len(highs)):
        for j in range(i + 1, len(highs)):
            h1, h2 = highs[i][1], highs[j][1]
            if abs(h1 - h2) / max(h1, h2) <= tolerance:
                eqh.append((highs[i][0], highs[j][0], (h1 + h2) / 2))
    for i in range(len(lows)):
        for j in range(i + 1, len(lows)):
            l1, l2 = lows[i][1], lows[j][1]
            if abs(l1 - l2) / max(l1, l2) <= tolerance:
                eql.append((lows[i][0], lows[j][0], (l1 + l2) / 2))
    return {"eqh": eqh, "eql": eql}


def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = SWING_LOOKBACK, confirm: int = SWEEP_CONFIRM_CANDLES) -> dict:
    highs, lows = find_swings(df, lookback)
    result = {"detected": False, "direction": None, "level": None}
    n = len(df)
    for idx, level in reversed(highs[-5:]):
        if idx + confirm + 1 >= n:
            continue
        if df["high"].iloc[idx + 1] > level:
            closes_below = sum(1 for k in range(idx + 2, min(idx + 2 + confirm, n)) if df["close"].iloc[k] < level)
            if closes_below >= confirm:
                return {"detected": True, "direction": "short", "level": level}
    for idx, level in reversed(lows[-5:]):
        if idx + confirm + 1 >= n:
            continue
        if df["low"].iloc[idx + 1] < level:
            closes_above = sum(1 for k in range(idx + 2, min(idx + 2 + confirm, n)) if df["close"].iloc[k] > level)
            if closes_above >= confirm:
                return {"detected": True, "direction": "long", "level": level}
    return result


def detect_premium_discount(df: pd.DataFrame) -> dict:
    highs, lows = find_swings(df)
    if not highs or not lows:
        return {"zone": "neutral", "equilibrium": None}
    high_val = max(h[1] for h in highs[-5:]) if len(highs) >= 5 else highs[-1][1]
    low_val  = min(l[1] for l in lows[-5:])  if len(lows)  >= 5 else lows[-1][1]
    eq   = (high_val + low_val) / 2
    curr = df["close"].iloc[-1]
    zone = "premium" if curr > eq else ("discount" if curr < eq else "equilibrium")
    return {"zone": zone, "equilibrium": eq}


def detect_rsi_divergence(df: pd.DataFrame, period: int = 14) -> dict:
    result = {"detected": False, "direction": None}
    if len(df) < period + 5:
        return result
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    prices, rsi_vals = df["close"].values, rsi.values
    if len(prices) < 10:
        return result
    if prices[-1] > prices[-5] and rsi_vals[-1] < rsi_vals[-5]:
        return {"detected": True, "direction": "short"}
    if prices[-1] < prices[-5] and rsi_vals[-1] > rsi_vals[-5]:
        return {"detected": True, "direction": "long"}
    return result


def get_trend_direction(df: pd.DataFrame) -> str:
    if len(df) < 20:
        return "neutral"
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    curr  = df["close"].iloc[-1]
    if ema20.iloc[-1] > ema50.iloc[-1] and curr > ema20.iloc[-1]:
        return "long"
    if ema20.iloc[-1] < ema50.iloc[-1] and curr < ema20.iloc[-1]:
        return "short"
    return "neutral"


def calc_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    high  = df["high"]
    low   = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat([(high - low), (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

# ─────────────────────────────────────────────
# EXCHANGE
# ─────────────────────────────────────────────

_session      = None
_session_lock = threading.Lock()

_universe_cache: list  = []
_universe_cache_time: float = 0.0
_universe_lock = threading.Lock()

TF_MAP = {"1m": "Min1", "5m": "Min5", "15m": "Min15", "30m": "Min30",
          "1h": "Hour1", "4h": "Hour4", "1d": "Day1"}
TF_MAP_SPOT = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
               "1h": "1h", "4h": "4h", "1d": "1d"}


def _make_session() -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=SESSION_POOL_SIZE, pool_maxsize=SESSION_POOL_SIZE,
        max_retries=Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504]),
    )
    s.mount("https://", adapter)
    s.mount("http://",  adapter)
    return s


def get_session() -> requests.Session:
    global _session
    with _session_lock:
        if _session is None:
            _session = _make_session()
        return _session


def _get_futures_klines(symbol, interval, limit=200, base="https://contract.mexc.com"):
    tf  = TF_MAP.get(interval, "Hour1")
    url = f"{base}/api/v1/contract/kline/{symbol}"
    r   = get_session().get(url, params={"interval": tf, "limit": limit}, timeout=10)
    r.raise_for_status()
    rows = r.json().get("data", {})
    if not rows:
        raise ValueError("Empty klines")
    df = pd.DataFrame({
        "timestamp": [t * 1000 for t in rows.get("time", [])],
        "open":  list(map(float, rows.get("open",  []))),
        "high":  list(map(float, rows.get("high",  []))),
        "low":   list(map(float, rows.get("low",   []))),
        "close": list(map(float, rows.get("close", []))),
        "volume":list(map(float, rows.get("vol",   []))),
    })
    return df.sort_values("timestamp").reset_index(drop=True)


def _get_spot_klines(symbol, interval, limit=200):
    tf  = TF_MAP_SPOT.get(interval, "1h")
    url = "https://api.mexc.com/api/v3/klines"
    sym = symbol.replace("_USDT", "USDT").replace("_", "")
    r   = get_session().get(url, params={"symbol": sym, "interval": tf, "limit": limit}, timeout=10)
    r.raise_for_status()
    df  = pd.DataFrame(r.json(), columns=["timestamp","open","high","low","close","volume",
                                          "close_time","qav","trades","tbav","tqav","ignore"])
    df  = df[["timestamp","open","high","low","close","volume"]].astype(float)
    return df.sort_values("timestamp").reset_index(drop=True)


def _build_2h(df_1h: pd.DataFrame) -> pd.DataFrame:
    df = df_1h.copy()
    df["ts_2h"] = (df["timestamp"] // (7200 * 1000)) * (7200 * 1000)
    g = df.groupby("ts_2h").agg(
        timestamp=("timestamp","first"), open=("open","first"),
        high=("high","max"), low=("low","min"),
        close=("close","last"), volume=("volume","sum"),
    ).reset_index(drop=True)
    return g.sort_values("timestamp").reset_index(drop=True)


def get_klines(symbol, interval, limit=200) -> pd.DataFrame:
    if interval == "2h":
        return _build_2h(get_klines(symbol, "1h", limit=limit * 2 + 10))

    fut_sym = symbol.replace("USDT", "_USDT") if "_" not in symbol else symbol

    for ep in MEXC_ENDPOINTS[:2]:
        try:
            df = _get_futures_klines(fut_sym, interval, limit, ep)
            if len(df) > 10:
                return df
        except Exception as e:
            logger.debug(f"Futures klines {ep}: {e}")

    try:
        df = _get_spot_klines(symbol, interval, limit)
        if len(df) > 10:
            return df
    except Exception as e:
        logger.debug(f"Spot klines fallback: {e}")

    try:
        import ccxt
        ex    = ccxt.mexc()
        clean = symbol.replace("_USDT","").replace("USDT","")
        raw   = ex.fetch_ohlcv(f"{clean}/USDT", interval, limit=limit)
        return pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    except Exception as e:
        logger.error(f"ccxt fallback failed for {symbol}: {e}")
        raise


def get_current_price(symbol) -> float:
    fut_sym = symbol.replace("USDT", "_USDT") if "_" not in symbol else symbol
    for ep in MEXC_ENDPOINTS[:2]:
        try:
            r = get_session().get(f"{ep}/api/v1/contract/ticker",
                                  params={"symbol": fut_sym}, timeout=5)
            r.raise_for_status()
            p = float(r.json().get("data", {}).get("lastPrice", 0))
            if p > 0:
                return p
        except Exception:
            pass
    try:
        r = get_session().get("https://api.mexc.com/api/v3/ticker/price",
                              params={"symbol": symbol.replace("_","")}, timeout=5)
        r.raise_for_status()
        return float(r.json().get("price", 0))
    except Exception:
        pass
    return 0.0


def get_funding_rate(symbol) -> float:
    fut_sym = symbol.replace("USDT", "_USDT") if "_" not in symbol else symbol
    for ep in MEXC_ENDPOINTS[:2]:
        try:
            r = get_session().get(f"{ep}/api/v1/contract/funding_rate/{fut_sym}", timeout=5)
            r.raise_for_status()
            return float(r.json().get("data", {}).get("fundingRate", 0))
        except Exception:
            pass
    return 0.0


def get_universe(min_volume=MIN_VOLUME_USDT) -> list:
    global _universe_cache, _universe_cache_time
    now = time.time()
    with _universe_lock:
        if _universe_cache and (now - _universe_cache_time) < UNIVERSE_CACHE_TTL:
            return list(_universe_cache)

    symbols = []
    for ep in MEXC_ENDPOINTS[:2]:
        try:
            r = get_session().get(f"{ep}/api/v1/contract/ticker", timeout=15)
            r.raise_for_status()
            for t in r.json().get("data", []):
                sym = t.get("symbol","")
                if not sym.endswith("_USDT"):
                    continue
                try:
                    vol = float(t.get("amount24", 0))
                except Exception:
                    vol = 0
                if vol >= min_volume:
                    symbols.append({"symbol": sym, "volume_24h": vol,
                                    "last_price": float(t.get("lastPrice", 0))})
            if symbols:
                break
        except Exception as e:
            logger.warning(f"Universe fetch {ep}: {e}")

    if not symbols:
        try:
            r = get_session().get("https://api.mexc.com/api/v3/ticker/24hr", timeout=15)
            r.raise_for_status()
            for t in r.json():
                sym = t.get("symbol","")
                if not sym.endswith("USDT"):
                    continue
                try:
                    vol = float(t.get("quoteVolume", 0))
                except Exception:
                    vol = 0
                if vol >= min_volume:
                    symbols.append({"symbol": sym, "volume_24h": vol,
                                    "last_price": float(t.get("lastPrice", 0))})
        except Exception as e:
            logger.warning(f"Universe spot fallback: {e}")

    symbols.sort(key=lambda x: x["volume_24h"], reverse=True)
    with _universe_lock:
        _universe_cache      = symbols
        _universe_cache_time = time.time()
    return symbols


def get_btc_change_pct() -> float:
    try:
        df = get_klines("BTCUSDT", "1h", limit=3)
        if len(df) < 2:
            return 0.0
        return (df.iloc[-1]["close"] - df.iloc[-2]["close"]) / df.iloc[-2]["close"] * 100
    except Exception:
        return 0.0


def get_volume_spike(symbol, multiplier=VOLUME_SPIKE_MULTIPLIER) -> bool:
    try:
        df = get_klines(symbol, "1h", limit=25)
        if len(df) < 22:
            return False
        avg = df["volume"].iloc[-21:-1].mean()
        return df["volume"].iloc[-1] >= avg * multiplier
    except Exception:
        return False

# ─────────────────────────────────────────────
# FILTERS
# ─────────────────────────────────────────────

def filter_atr(df, price):
    atr_pct = calc_atr(df) / price
    if atr_pct < ATR_MIN_PCT:
        return False, f"ATR {atr_pct:.4f} < min {ATR_MIN_PCT:.4f}"
    return True, f"ATR OK ({atr_pct:.4f})"


def filter_mtf_confluence(symbol, direction):
    agree, results = 0, []
    for tf in ["15m","1h","2h","4h"]:
        try:
            trend = get_trend_direction(get_klines(symbol, tf, limit=100))
            results.append(f"{tf}:{trend}")
            if trend == direction:
                agree += 1
        except Exception:
            results.append(f"{tf}:err")
    if agree >= MTF_MIN_AGREE:
        return True, f"MTF {agree}/4 ({', '.join(results)})"
    return False, f"MTF only {agree}/4 ({', '.join(results)})"


def filter_daily_bias(symbol, direction):
    try:
        bias = get_trend_direction(get_klines(symbol, "1d", limit=50))
        if bias == "long"  and direction == "short": return False, "Daily bias UP blocks shorts"
        if bias == "short" and direction == "long":  return False, "Daily bias DOWN blocks longs"
        return True, f"Daily bias {bias} allows {direction}"
    except Exception:
        return True, "Daily bias check skipped"


def filter_btc_correlation(direction):
    try:
        chg = get_btc_change_pct()
        if chg <= -BTC_MOVE_LIMIT_PCT and direction == "long":
            return False, f"BTC dropped {chg:.2f}% blocks longs"
        if chg >= BTC_MOVE_LIMIT_PCT  and direction == "short":
            return False, f"BTC rose {chg:.2f}% blocks shorts"
        return True, f"BTC change {chg:.2f}% OK"
    except Exception:
        return True, "BTC correlation skipped"


def filter_funding_rate(symbol, direction):
    try:
        rate = get_funding_rate(symbol)
        if direction == "long"  and rate > FUNDING_LONG_MAX:
            return False, f"Funding {rate:.5f} too high for long"
        if direction == "short" and rate < FUNDING_SHORT_MIN:
            return False, f"Funding {rate:.5f} too low for short"
        return True, f"Funding {rate:.5f} OK"
    except Exception:
        return True, "Funding rate skipped"


def filter_quiet_hours():
    hour = datetime.now(timezone.utc).hour
    if QUIET_HOURS_START <= hour < QUIET_HOURS_END:
        return False, f"Quiet hours ({hour:02d}:xx UTC)"
    return True, f"Active hours ({hour:02d}:xx UTC)"


def filter_spread(entry, current_price):
    if entry <= 0 or current_price <= 0:
        return True, "Spread check skipped"
    diff = abs(current_price - entry) / entry
    if diff > SPREAD_MAX_PCT:
        return False, f"Entry passed: spread {diff:.4f}"
    return True, f"Spread OK ({diff:.4f})"


def filter_htf_bias(symbol, direction):
    try:
        trend_4h = get_trend_direction(get_klines(symbol, "4h", limit=100))
        if trend_4h != "neutral" and trend_4h != direction:
            return False, f"4H trend {trend_4h} conflicts with {direction}"
        return True, f"4H trend {trend_4h} OK"
    except Exception:
        return True, "HTF bias check skipped"


def run_all_filters(symbol, direction, entry, current_price, df_1h):
    checks = []
    for name, fn, args in [
        ("Quiet Hours",      filter_quiet_hours,     []),
        ("ATR",              filter_atr,              [df_1h, current_price]),
        ("Spread",           filter_spread,           [entry, current_price]),
        ("BTC Correlation",  filter_btc_correlation,  [direction]),
        ("Funding Rate",     filter_funding_rate,     [symbol, direction]),
        ("Daily Bias",       filter_daily_bias,       [symbol, direction]),
        ("HTF Bias (4H)",    filter_htf_bias,         [symbol, direction]),
        ("MTF Confluence",   filter_mtf_confluence,   [symbol, direction]),
    ]:
        ok, msg = fn(*args)
        checks.append((name, ok, msg))
        if not ok:
            return False, checks
    return True, checks


def get_filter_summary(symbol, direction, entry, current_price, df_1h) -> str:
    _, checks = run_all_filters(symbol, direction, entry, current_price, df_1h)
    return "\n".join(f"{'✅' if ok else '❌'} {n}: {m}" for n, ok, m in checks)

# ─────────────────────────────────────────────
# SMC SCORING
# ─────────────────────────────────────────────

def score_smc(df: pd.DataFrame) -> dict:
    df_score = df.tail(SCORING_WINDOW).reset_index(drop=True)
    df_swing = df.tail(SWING_HISTORY).reset_index(drop=True)

    sl, ss = 0, 0
    criteria = {}
    curr = df_score["close"].iloc[-1]

    # CHoCH
    choch = detect_choch(df_score, SWING_LOOKBACK)
    if choch["detected"]:
        criteria["CHoCH"] = choch["direction"]
        if choch["direction"] == "long": sl += 1
        else: ss += 1
    else:
        criteria["CHoCH"] = None

    # BOS
    bos = detect_bos(df_score, SWING_LOOKBACK)
    if bos["detected"]:
        criteria["BOS"] = bos["direction"]
        if bos["direction"] == "long": sl += 1
        else: ss += 1
    else:
        criteria["BOS"] = None

    # Order Block
    obs = detect_order_blocks(df_score)
    ob_demand = [o for o in obs if o["type"] == "demand"]
    ob_supply = [o for o in obs if o["type"] == "supply"]
    ob_l = [o for o in ob_demand if o["low"] * 0.995 <= curr <= o["high"] * 1.005]
    ob_s = [o for o in ob_supply if o["low"] * 0.995 <= curr <= o["high"] * 1.005]
    if ob_l:   sl += 1; criteria["Order Block"] = "long"
    elif ob_s: ss += 1; criteria["Order Block"] = "short"
    else:      criteria["Order Block"] = None

    # FVG
    fvgs    = detect_fvg(df_score)
    fvg_bull= [f for f in fvgs if f["type"] == "bullish" and f["low"] <= curr <= f["high"]]
    fvg_bear= [f for f in fvgs if f["type"] == "bearish" and f["low"] <= curr <= f["high"]]
    if fvg_bull:   sl += 1; criteria["FVG"] = "long"
    elif fvg_bear: ss += 1; criteria["FVG"] = "short"
    else:          criteria["FVG"] = None

    # EQH/EQL
    eq       = detect_eqh_eql(df_swing, SWING_LOOKBACK)
    eqh_lvls = [e[2] for e in eq["eqh"]]
    eql_lvls = [e[2] for e in eq["eql"]]
    near_eqh = any(abs(curr - l) / curr < 0.005 for l in eqh_lvls)
    near_eql = any(abs(curr - l) / curr < 0.005 for l in eql_lvls)
    if near_eql:   sl += 1; criteria["EQH/EQL"] = "long"
    elif near_eqh: ss += 1; criteria["EQH/EQL"] = "short"
    else:          criteria["EQH/EQL"] = None

    # Premium/Discount
    pd_r = detect_premium_discount(df_score)
    zone = pd_r["zone"]
    if zone == "discount":     sl += 1; criteria["Premium/Discount"] = "long"
    elif zone == "premium":    ss += 1; criteria["Premium/Discount"] = "short"
    else:                      criteria["Premium/Discount"] = None

    # Liquidity Sweep
    sweep = detect_liquidity_sweep(df_score, SWING_LOOKBACK)
    if sweep["detected"]:
        criteria["Liquidity Sweep"] = sweep["direction"]
        if sweep["direction"] == "long": sl += 1
        else: ss += 1
    else:
        criteria["Liquidity Sweep"] = None

    # 2H Bias
    try:
        sym = df.attrs.get("symbol", "")
        bias_2h = get_trend_direction(get_klines(sym, "2h", limit=50)) if sym else "neutral"
    except Exception:
        bias_2h = "neutral"
    if bias_2h == "long":   sl += 1; criteria["2H Bias"] = "long"
    elif bias_2h == "short":ss += 1; criteria["2H Bias"] = "short"
    else:                   criteria["2H Bias"] = None

    # Zone Confluence (OB + FVG same direction)
    if ob_l and fvg_bull: sl += 2; criteria["Zone Confluence"] = "long"
    elif ob_s and fvg_bear:ss+= 2; criteria["Zone Confluence"] = "short"
    else:                  criteria["Zone Confluence"] = None

    # Momentum
    ema20 = df_score["close"].ewm(span=20, adjust=False).mean()
    ema50 = df_score["close"].ewm(span=50, adjust=False).mean()
    if ema20.iloc[-1] > ema50.iloc[-1]:   sl += 1; criteria["Momentum"] = "long"
    elif ema20.iloc[-1] < ema50.iloc[-1]: ss += 1; criteria["Momentum"] = "short"
    else:                                 criteria["Momentum"] = None

    # RSI Divergence
    rsi_div = detect_rsi_divergence(df_score)
    if rsi_div["detected"]:
        criteria["RSI Divergence"] = rsi_div["direction"]
        if rsi_div["direction"] == "long": sl += 1
        else: ss += 1
    else:
        criteria["RSI Divergence"] = None

    # Volume Confirmation
    avg_vol = df_score["volume"].iloc[-21:-1].mean() if len(df_score) >= 21 else df_score["volume"].mean()
    curr_vol = df_score["volume"].iloc[-1]
    if curr_vol >= avg_vol * VOLUME_SPIKE_MULTIPLIER and avg_vol > 0:
        hint = "long" if sl >= ss else "short"
        criteria["Volume Confirmation"] = hint
        if hint == "long": sl += 1
        else: ss += 1
    else:
        criteria["Volume Confirmation"] = None

    # SMT Trap bonus
    smt = (near_eqh and sweep.get("direction") == "short") or (near_eql and sweep.get("direction") == "long")
    direction = "long" if sl >= ss else "short"
    score = sl if direction == "long" else ss
    if smt:
        score = min(score + 1, SMT_BONUS_CAP)

    return {
        "direction": direction, "score": score, "score_long": sl, "score_short": ss,
        "criteria": criteria, "smt_bonus": smt,
        "obs_demand": ob_demand, "obs_supply": ob_supply,
        "fvgs_bull": [f for f in fvgs if f["type"] == "bullish"],
        "fvgs_bear": [f for f in fvgs if f["type"] == "bearish"],
        "eqh": eqh_lvls, "eql": eql_lvls,
        "choch": choch, "bos": bos, "sweep": sweep, "pd_zone": zone,
    }


def _find_tp_long(entry, sl, smc_result, highs):
    risk   = entry - sl
    min_tp = entry + risk * MIN_RR
    cands  = []
    for ob in smc_result.get("obs_supply", []):
        if ob["low"] > entry: cands.append(ob["low"])
    for fvg in smc_result.get("fvgs_bear", []):
        if fvg["low"] > entry: cands.append(fvg["low"])
    for lvl in smc_result.get("eqh", []):
        if lvl > entry: cands.append(lvl)
    above = [h[1] for h in highs if h[1] > entry]
    if above: cands.append(min(above))
    valid = [c for c in cands if c >= min_tp]
    return min(valid) if valid else min_tp


def _find_tp_short(entry, sl, smc_result, lows):
    risk   = sl - entry
    min_tp = entry - risk * MIN_RR
    cands  = []
    for ob in smc_result.get("obs_demand", []):
        if ob["high"] < entry: cands.append(ob["high"])
    for fvg in smc_result.get("fvgs_bull", []):
        if fvg["high"] < entry: cands.append(fvg["high"])
    for lvl in smc_result.get("eql", []):
        if lvl < entry: cands.append(lvl)
    below = [l[1] for l in lows if l[1] < entry]
    if below: cands.append(max(below))
    valid = [c for c in cands if c <= min_tp]
    return max(valid) if valid else min_tp


def find_entry_sl_tp(df, direction, smc_result) -> dict:
    curr   = df["close"].iloc[-1]
    highs, lows = find_swings(df.tail(SWING_HISTORY).reset_index(drop=True), SWING_LOOKBACK)

    if direction == "long":
        ob_c  = [o for o in smc_result["obs_demand"] if o["low"] <= curr <= o["high"] * 1.02]
        fvg_c = [f for f in smc_result["fvgs_bull"]  if f["low"] <= curr <= f["high"] * 1.02]
        if ob_c:
            ob    = min(ob_c, key=lambda x: abs(curr - x["low"]))
            entry = ob["low"] + (ob["high"] - ob["low"]) * 0.10
        elif fvg_c:
            entry = fvg_c[0]["low"]
        else:
            entry = curr * (1 - SL_BUFFER_PCT)

        sl_cands = []
        if ob_c:  sl_cands.append(ob_c[0]["low"]  * (1 - SL_BUFFER_PCT))
        if fvg_c: sl_cands.append(fvg_c[0]["low"] * (1 - SL_BUFFER_PCT))
        near = [l[1] for l in lows if l[1] < curr]
        if near: sl_cands.append(max(near) * (1 - SL_BUFFER_PCT))
        sl_min = curr * (1 - 0.03)
        sl = min(min(sl_cands), sl_min) if sl_cands else sl_min

        risk = max(entry - sl, curr * 0.03)
        tp   = _find_tp_long(entry, sl, smc_result, highs)

    else:
        ob_c  = [o for o in smc_result["obs_supply"] if o["low"] * 0.98 <= curr <= o["high"]]
        fvg_c = [f for f in smc_result["fvgs_bear"]  if f["low"] * 0.98 <= curr <= f["high"]]
        if ob_c:
            ob    = min(ob_c, key=lambda x: abs(curr - x["high"]))
            entry = ob["high"] - (ob["high"] - ob["low"]) * 0.10
        elif fvg_c:
            entry = fvg_c[0]["high"]
        else:
            entry = curr * (1 + SL_BUFFER_PCT)

        sl_cands = []
        if ob_c:  sl_cands.append(ob_c[0]["high"]  * (1 + SL_BUFFER_PCT))
        if fvg_c: sl_cands.append(fvg_c[0]["high"] * (1 + SL_BUFFER_PCT))
        near = [h[1] for h in highs if h[1] > curr]
        if near: sl_cands.append(min(near) * (1 + SL_BUFFER_PCT))
        sl_max = curr * (1 + 0.03)
        sl = max(max(sl_cands), sl_max) if sl_cands else sl_max

        risk = max(sl - entry, curr * 0.03)
        tp   = _find_tp_short(entry, sl, smc_result, lows)

    rr = abs(tp - entry) / abs(sl - entry) if abs(sl - entry) > 0 else 0
    return {"entry": entry, "sl": sl, "tp": tp, "rr": round(rr, 2),
            "risk_pct": abs(sl - entry) / entry * 100}


def calc_leverage(sl_pct) -> int:
    if sl_pct < 2.0: return 20
    if sl_pct < 4.0: return 10
    if sl_pct < 7.0: return 5
    return 3

# ─────────────────────────────────────────────
# TRACKER
# ─────────────────────────────────────────────

_tracker_lock = threading.Lock()


def _load() -> dict:
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Tracker load error: {e}")
    return {"active": [], "history": [], "cooldowns": {}}


def _save(data: dict):
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Tracker save error: {e}")


def add_signal(signal: dict) -> bool:
    with _tracker_lock:
        data    = _load()
        active  = data.get("active", [])
        cds     = data.get("cooldowns", {})
        symbol  = signal["symbol"]
        now     = time.time()
        if len(active) >= MAX_ACTIVE_SIGNALS:
            return False
        if symbol in cds and (now - cds[symbol]) < COIN_COOLDOWN_HOURS * 3600:
            return False
        if any(s["symbol"] == symbol for s in active):
            return False
        signal["created_at"] = now
        signal["status"]     = "active"
        active.append(signal)
        data["active"] = active
        _save(data)
        return True


def get_active_signals() -> list:
    with _tracker_lock:
        return _load().get("active", [])


def close_signal(symbol, result, close_price=None):
    with _tracker_lock:
        data    = _load()
        updated = []
        for s in data.get("active", []):
            if s["symbol"] == symbol:
                s["status"] = result; s["closed_at"] = time.time()
                if close_price: s["close_price"] = close_price
                data.setdefault("history", []).append(s)
                data.setdefault("cooldowns", {})[symbol] = time.time()
            else:
                updated.append(s)
        data["active"]  = updated
        data["history"] = data.get("history", [])[-200:]
        _save(data)


def get_history(limit=10) -> list:
    with _tracker_lock:
        h = _load().get("history", [])
        return [x for x in h if x.get("status") in ("win","loss","expired")][-limit:]


def check_cooldown(symbol) -> bool:
    with _tracker_lock:
        cds = _load().get("cooldowns", {})
        if symbol in cds:
            return (time.time() - cds[symbol]) < COIN_COOLDOWN_HOURS * 3600
        return False


def active_count() -> int:
    with _tracker_lock:
        return len(_load().get("active", []))


def expire_old_signals() -> list:
    with _tracker_lock:
        data = _load(); now = time.time(); expired = []
        still = []
        for s in data.get("active", []):
            if (now - s.get("created_at", now)) / 3600 >= SIGNAL_EXPIRY_HOURS:
                s["status"] = "expired"; s["closed_at"] = now
                data.setdefault("history", []).append(s)
                data.setdefault("cooldowns", {})[s["symbol"]] = now
                expired.append(s)
            else:
                still.append(s)
        data["active"]  = still
        data["history"] = data.get("history", [])[-200:]
        _save(data)
        return expired


def update_signal_hit_sl(symbol, candle_low, candle_high) -> bool:
    with _tracker_lock:
        for s in _load().get("active", []):
            if s["symbol"] != symbol: continue
            sl = s.get("sl")
            if sl is None: continue
            if s.get("direction") == "long"  and candle_low  <= sl: return True
            if s.get("direction") == "short" and candle_high >= sl: return True
        return False


def update_signal_hit_tp(symbol, candle_close) -> bool:
    with _tracker_lock:
        for s in _load().get("active", []):
            if s["symbol"] != symbol: continue
            tp = s.get("tp")
            if tp is None: continue
            if s.get("direction") == "long"  and candle_close >= tp: return True
            if s.get("direction") == "short" and candle_close <= tp: return True
        return False


def save_scan_results(results):
    with _tracker_lock:
        data = _load()
        data["last_scan"]      = results
        data["last_scan_time"] = time.time()
        _save(data)


def get_last_scan_results() -> list:
    with _tracker_lock:
        return _load().get("last_scan", [])


def get_stats() -> dict:
    with _tracker_lock:
        data    = _load()
        history = data.get("history", [])
        wins    = sum(1 for h in history if h.get("status") == "win")
        losses  = sum(1 for h in history if h.get("status") == "loss")
        expired = sum(1 for h in history if h.get("status") == "expired")
        total   = wins + losses
        return {"active": len(data.get("active",[])), "wins": wins, "losses": losses,
                "expired": expired, "total_closed": total,
                "win_rate": wins / total * 100 if total > 0 else 0}

# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────

BG_COLOR          = "#0b0e14"
TEXT_COLOR        = "#e0e0e0"
CANDLE_BULL_COLOR = "#2979ff"
CANDLE_BEAR_COLOR = "#0b0e14"
CANDLE_BEAR_OUT   = "white"
VOL_BULL_COLOR    = "#2979ff"
VOL_BEAR_COLOR    = "#1a1a2e"
TP_COLOR          = "#00e676"
ENTRY_COLOR       = "#2979ff"
SL_COLOR          = "#ff1744"
CHOCH_COLOR       = "#ce93d8"
BOS_COLOR         = "#ff9800"
OB_DEMAND_COLOR   = "#00e676"
OB_SUPPLY_COLOR   = "#ff1744"
EQ_COLOR          = "#757575"


def _fp(price):
    if price >= 1000: return f"{price:,.1f}"
    if price >= 1:    return f"{price:.4f}"
    return f"{price:.6f}"


def _draw_candle(ax, idx, o, h, l, c, w=0.6):
    bull  = c >= o
    color = CANDLE_BULL_COLOR if bull else CANDLE_BEAR_COLOR
    edge  = CANDLE_BULL_COLOR if bull else CANDLE_BEAR_OUT
    bl, bh = min(o, c), max(o, c)
    ax.plot([idx, idx], [l, bl], color=edge, linewidth=0.8, zorder=2)
    ax.plot([idx, idx], [bh, h], color=edge, linewidth=0.8, zorder=2)
    ax.add_patch(patches.Rectangle(
        (idx - w / 2, bl), w, bh - bl,
        linewidth=0.8, edgecolor=edge, facecolor=color, zorder=3))


def generate_chart(df, symbol, timeframe, direction, entry, sl, tp, rr, smc_result, chart_candles=CHART_CANDLES) -> bytes:
    df_p  = df.tail(chart_candles).reset_index(drop=True)
    n     = len(df_p)
    extra = 16

    fig    = plt.figure(figsize=(14, 9), facecolor=BG_COLOR)
    ax     = fig.add_axes([0.05, 0.22, 0.82, 0.72], facecolor=BG_COLOR)
    ax_vol = fig.add_axes([0.05, 0.05, 0.82, 0.15], facecolor=BG_COLOR, sharex=ax)

    for a in [ax, ax_vol]:
        a.set_facecolor(BG_COLOR)
        a.tick_params(colors=TEXT_COLOR, labelsize=7)
        for sp in a.spines.values():
            sp.set_color("#1f2433")

    for i in range(n):
        r = df_p.iloc[i]
        _draw_candle(ax, i, r["open"], r["high"], r["low"], r["close"])
        vc = VOL_BULL_COLOR if r["close"] >= r["open"] else VOL_BEAR_COLOR
        ax_vol.bar(i, r["volume"], width=0.6, color=vc, edgecolor=None)

    choch = smc_result.get("choch", {})
    if choch.get("detected"):
        ci = choch.get("index")
        if ci is not None and 0 <= ci < n:
            ax.axvline(x=ci, color=CHOCH_COLOR, linewidth=1.2, alpha=0.8)

    bos = smc_result.get("bos", {})
    if bos.get("detected") and bos.get("level"):
        ax.axhline(y=bos["level"], color=BOS_COLOR, linewidth=1.0, linestyle="--", alpha=0.8)

    for lvl in smc_result.get("eqh", []):
        ax.axhline(y=lvl, color=EQ_COLOR, linewidth=0.8, linestyle=":", alpha=0.6)
    for lvl in smc_result.get("eql", []):
        ax.axhline(y=lvl, color=EQ_COLOR, linewidth=0.8, linestyle=":", alpha=0.6)

    if direction == "long":
        obs  = [o for o in smc_result.get("obs_demand",[]) if o["low"] <= entry <= o["high"] * 1.02]
        fvgs = [f for f in smc_result.get("fvgs_bull", []) if f["low"] <= entry <= f["high"] * 1.02]
    else:
        obs  = [o for o in smc_result.get("obs_supply",[]) if o["low"] * 0.98 <= entry <= o["high"]]
        fvgs = [f for f in smc_result.get("fvgs_bear", []) if f["low"] * 0.98 <= entry <= f["high"]]

    for ob in obs[:1]:
        clr = OB_DEMAND_COLOR if ob["type"] == "demand" else OB_SUPPLY_COLOR
        ax.add_patch(patches.Rectangle(
            (max(0, ob.get("index", 0)), ob["low"]),
            n + extra - 2 - max(0, ob.get("index", 0)), ob["high"] - ob["low"],
            linewidth=0.8, edgecolor=clr, facecolor=clr, alpha=0.12, zorder=2))

    for fvg in fvgs[:1]:
        clr = OB_DEMAND_COLOR if fvg["type"] == "bullish" else OB_SUPPLY_COLOR
        ax.add_patch(patches.Rectangle(
            (max(0, fvg.get("index", 0)), fvg["low"]),
            n + extra - 2 - max(0, fvg.get("index", 0)), fvg["high"] - fvg["low"],
            linewidth=0.8, edgecolor=clr, facecolor=clr, alpha=0.10, zorder=2, linestyle="--"))

    y_min  = df_p["low"].min()
    y_max  = df_p["high"].max()
    y_range= max(y_max - y_min, y_min * 0.01)
    levels = []

    def _label(price, color, text):
        for lv in levels:
            while abs(price - lv) / (y_range + 1e-10) < 0.015:
                price += y_range * 0.015
        levels.append(price)
        ax.axhline(y=price, color=color, linewidth=1.0, linestyle="--", alpha=0.85, zorder=6)
        ax.text(n + 0.5, price, f" {text}: {_fp(price)}", color=color, fontsize=7.5,
                fontweight="bold", va="center", ha="left", zorder=7,
                bbox=dict(facecolor=BG_COLOR, edgecolor="none", alpha=0.7, pad=1))

    _label(entry, ENTRY_COLOR, "Entry")
    _label(sl,    SL_COLOR,    "SL")
    _label(tp,    TP_COLOR,    "TP")

    ax.set_xlim(-1, n + extra)
    ax.set_ylim(min(y_min, sl) * 0.998, max(y_max, tp) * 1.002)
    ax_vol.set_xlim(-1, n + extra)

    dir_lbl  = "LONG" if direction == "long" else "SHORT"
    dir_clr  = TP_COLOR if direction == "long" else SL_COLOR
    ax.text(0.01, 0.98, f"{symbol}/USDT  {timeframe.upper()}  {dir_lbl}  RR: {rr:.2f}x",
            transform=ax.transAxes, color=dir_clr, fontsize=11, fontweight="bold",
            va="top", ha="left",
            bbox=dict(facecolor=BG_COLOR, edgecolor="#1f2433", boxstyle="round,pad=0.3", alpha=0.9))

    legend = [
        Line2D([0],[0], color=CHOCH_COLOR, linewidth=1.2, label="CHoCH"),
        Line2D([0],[0], color=BOS_COLOR,   linewidth=1.0, linestyle="--", label="BOS"),
        Line2D([0],[0], color=EQ_COLOR,    linewidth=0.8, linestyle=":",  label="EQH/EQL"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=7,
              facecolor=BG_COLOR, edgecolor="#1f2433", labelcolor=TEXT_COLOR)

    ax.set_ylabel("Price",  color=TEXT_COLOR, fontsize=8)
    ax_vol.set_ylabel("Vol",color=TEXT_COLOR, fontsize=8)
    plt.setp(ax.get_xticklabels(), visible=False)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────
# TELEGRAM BOT
# ─────────────────────────────────────────────

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


def send_message(text, chat_id=None, parse_mode="HTML") -> bool:
    cid = chat_id or TELEGRAM_CHAT_ID
    try:
        r = requests.post(f"{BASE_URL}/sendMessage",
                          json={"chat_id": cid, "text": text, "parse_mode": parse_mode},
                          timeout=15)
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"sendMessage failed: {e}")
        return False


def send_photo(image_bytes, caption="", chat_id=None) -> bool:
    cid = chat_id or TELEGRAM_CHAT_ID
    try:
        r = requests.post(f"{BASE_URL}/sendPhoto",
                          data={"chat_id": cid, "caption": caption, "parse_mode": "HTML"},
                          files={"photo": ("chart.png", image_bytes, "image/png")},
                          timeout=30)
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"sendPhoto failed: {e}")
        return False


def _fmt_price(p):
    if p >= 1000: return f"{p:,.2f}"
    if p >= 1:    return f"{p:.4f}"
    return f"{p:.6f}"


def format_signal_message(symbol, direction, timeframe, entry, sl, tp, rr, score, max_score=12, leverage=10) -> str:
    base  = symbol.replace("_USDT","").replace("USDT","")
    arrow = "📈" if direction == "long" else "📉"
    lbl   = "Long Setup" if direction == "long" else "Short Setup"
    return (
        f"{arrow} #{base}USDT – {lbl}\n"
        f"🕒 Timeframe: {timeframe.upper()}\n"
        f"⚙️ Leverage: {leverage}x\n"
        f"Entry - {_fmt_price(entry)}\n"
        f"StopLoss - {_fmt_price(sl)}\n"
        f"🎯 TAKE PROFIT 👇\n"
        f"TP: {_fmt_price(tp)}\n"
        f"📊 Risk-Reward Ratio: Approx. {rr:.2f}x\n"
        f"📋 SMC Score: {score}/{max_score}\n"
        f"⚠️ Key Notes:\n"
        f"▪️Use Recommended Leverage\n"
        f"🔔 Use Only 2-3% Of Total Funds"
    )


def send_signal(symbol, direction, timeframe, entry, sl, tp, rr, score, leverage, chart_bytes=None) -> bool:
    msg = format_signal_message(symbol, direction, timeframe, entry, sl, tp, rr, score, leverage=leverage)
    return send_photo(chart_bytes, caption=msg) if chart_bytes else send_message(msg)


def send_expired_message(symbol, direction, entry):
    base = symbol.replace("_USDT","").replace("USDT","")
    send_message(f"⏰ Signal Expired\n#{base}USDT {direction.upper()} @ {entry:.6g}\n"
                 f"Closed after 36h without hitting TP or SL.")


def send_sl_hit_message(symbol, direction, entry, sl):
    base = symbol.replace("_USDT","").replace("USDT","")
    send_message(f"🛑 Stop Loss Hit\n#{base}USDT {direction.upper()}\n"
                 f"Entry: {entry:.6g} | SL: {sl:.6g}")


def send_tp_hit_message(symbol, direction, entry, tp):
    base = symbol.replace("_USDT","").replace("USDT","")
    send_message(f"✅ Take Profit Hit! 🎉\n#{base}USDT {direction.upper()}\n"
                 f"Entry: {entry:.6g} | TP: {tp:.6g}")

# ─────────────────────────────────────────────
# COMMAND BOT (Telegram polling)
# ─────────────────────────────────────────────

_scan_fn_ref    = None
_analyze_fn_ref = None


def _get_updates(offset=0) -> list:
    try:
        r = requests.get(f"{BASE_URL}/getUpdates",
                         params={"timeout": 30, "offset": offset}, timeout=35)
        r.raise_for_status()
        return r.json().get("result", [])
    except Exception as e:
        logger.error(f"getUpdates failed: {e}")
        return []


def _cmd_start(chat_id):
    send_message(
        "🤖 <b>Crypto Futures Signal Bot</b>\n\n"
        "/signals — Active signals\n"
        "/stats — Win/loss statistics\n"
        "/symbols — Tracked symbols\n"
        "/analyze SYMBOL — Analyze a symbol\n"
        "/filters SYMBOL — Show filter results\n"
        "/scan 1h|2h — Run manual scan\n"
        "/test — Send test message\n"
        "/topsetups — Top 5 setups\n"
        "/history — Last 10 closed signals\n"
        "/ping — Bot health check",
        chat_id=chat_id)


def _cmd_signals(chat_id):
    sigs = get_active_signals()
    if not sigs:
        send_message("📭 No active signals.", chat_id=chat_id)
        return
    lines = [f"📡 <b>Active Signals ({len(sigs)})</b>\n"]
    for s in sigs:
        age = (time.time() - s.get("created_at", time.time())) / 3600
        lines.append(
            f"• {s.get('symbol','?')} {s.get('direction','?').upper()} | {s.get('timeframe','?').upper()} | Score: {s.get('score',0)}/12\n"
            f"  Entry: {s.get('entry',0):.6g} | SL: {s.get('sl',0):.6g} | TP: {s.get('tp',0):.6g} | RR: {s.get('rr',0):.2f}x\n"
            f"  Age: {age:.1f}h\n")
    send_message("\n".join(lines), chat_id=chat_id)


def _cmd_stats(chat_id):
    s = get_stats()
    send_message(
        f"📊 <b>Signal Statistics</b>\n\n"
        f"Active: {s['active']}\nTotal Closed: {s['total_closed']}\n"
        f"Wins: {s['wins']} ✅\nLosses: {s['losses']} ❌\n"
        f"Expired: {s['expired']} ⏰\nWin Rate: {s['win_rate']:.1f}%",
        chat_id=chat_id)


def _cmd_history(chat_id):
    hist = get_history(10)
    if not hist:
        send_message("📭 No closed signals yet.", chat_id=chat_id)
        return
    lines = ["📜 <b>Last 10 Closed Signals</b>\n"]
    for s in reversed(hist):
        icon = {"win": "✅", "loss": "❌", "expired": "⏰"}.get(s.get("status",""), "•")
        lines.append(f"{icon} {s.get('symbol','?')} {s.get('direction','?').upper()} "
                     f"| RR: {s.get('rr',0):.2f}x | {s.get('status','?')}")
    send_message("\n".join(lines), chat_id=chat_id)


def _cmd_topsetups(chat_id):
    results = get_last_scan_results()
    if not results:
        send_message("📭 No scan results yet. Run /scan 1h first.", chat_id=chat_id)
        return
    top = sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:5]
    lines = ["🏆 <b>Top 5 Setups</b>\n"]
    for i, r in enumerate(top, 1):
        lines.append(f"{i}. {r.get('symbol','?')} {r.get('direction','?').upper()} "
                     f"| Score: {r.get('score',0)}/12 | RR: {r.get('rr',0):.2f}x")
    send_message("\n".join(lines), chat_id=chat_id)


def _cmd_symbols(chat_id):
    try:
        universe = get_universe()
        top = [s["symbol"] for s in universe[:30]]
        send_message("📋 <b>Top 30 symbols by volume:</b>\n" + ", ".join(top), chat_id=chat_id)
    except Exception as e:
        send_message(f"Error: {e}", chat_id=chat_id)


def _cmd_test(chat_id):
    send_message("✅ Bot is alive and responding!", chat_id=chat_id)


def _cmd_ping(chat_id):
    lines = ["🏓 <b>Ping Results</b>\n"]
    for ep in MEXC_ENDPOINTS:
        try:
            r = requests.get(f"{ep}/api/v1/contract/ping", timeout=5)
            lines.append(f"✅ {ep} — {r.status_code}")
        except Exception as e:
            lines.append(f"❌ {ep} — {e}")
    send_message("\n".join(lines), chat_id=chat_id)


def _cmd_filters(chat_id, args):
    if not args:
        send_message("Usage: /filters SYMBOL (e.g. /filters BTCUSDT)", chat_id=chat_id)
        return
    symbol = args[0].upper()
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    sym_mexc = symbol.replace("USDT","_USDT")
    send_message(f"⏳ Checking filters for {symbol}...", chat_id=chat_id)
    try:
        df   = get_klines(sym_mexc, "1h", limit=200)
        smc  = score_smc(df)
        lvls = find_entry_sl_tp(df, smc["direction"], smc)
        curr = get_current_price(sym_mexc) or df["close"].iloc[-1]
        summary = get_filter_summary(sym_mexc, smc["direction"], lvls["entry"], curr, df)
        send_message(f"🔍 <b>{symbol} Filter Results</b>\nDirection: {smc['direction'].upper()}\n\n{summary}", chat_id=chat_id)
    except Exception as e:
        send_message(f"❌ Error: {e}", chat_id=chat_id)


def _cmd_analyze(chat_id, args):
    if not args:
        send_message("Usage: /analyze SYMBOL (e.g. /analyze BTCUSDT)", chat_id=chat_id)
        return
    symbol = args[0].upper()
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    sym_mexc = symbol.replace("USDT","_USDT")
    send_message(f"⏳ Analyzing {symbol}...", chat_id=chat_id)
    try:
        df = get_klines(sym_mexc, "1h", limit=250)
        df.attrs["symbol"] = sym_mexc
        smc  = score_smc(df)
        lvls = find_entry_sl_tp(df, smc["direction"], smc)
        curr = get_current_price(sym_mexc) or df["close"].iloc[-1]

        # Position status
        e, tp_, sl_ = lvls["entry"], lvls["tp"], lvls["sl"]
        rr_    = lvls["rr"]
        d      = smc["direction"]
        pct_from_entry = abs(curr - e) / e * 100
        halfway = (e + tp_) / 2 if d == "long" else (e + tp_) / 2

        if d == "long":
            if curr > e * 1.02:      status = "❌ Entry passed, do not enter"
            elif curr >= tp_ * 0.99: status = "❌ Near TP, do not enter"
            elif curr >= halfway:    status = "⚠️ Going to TP — late entry"
            else:                    status = "✅ Still valid — inside entry zone"
        else:
            if curr < e * 0.98:      status = "❌ Entry passed, do not enter"
            elif curr <= tp_ * 1.01: status = "❌ Near TP, do not enter"
            elif curr <= halfway:    status = "⚠️ Going to TP — late entry"
            else:                    status = "✅ Still valid — inside entry zone"

        passing_criteria = [k for k, v in smc["criteria"].items() if v == d]
        lev = calc_leverage(lvls["risk_pct"])

        msg = (
            f"🔍 <b>{symbol} Analysis</b>\n"
            f"Direction: <b>{d.upper()}</b>\n"
            f"Score: {smc['score']}/12\n"
            f"Criteria: {', '.join(passing_criteria)}\n\n"
            f"Entry:  {_fmt_price(e)}\n"
            f"SL:     {_fmt_price(sl_)}\n"
            f"TP:     {_fmt_price(tp_)}\n"
            f"RR:     {rr_:.2f}x\n"
            f"Leverage: {lev}x\n\n"
            f"Position: {status}"
        )

        try:
            base_sym = symbol.replace("USDT","")
            chart = generate_chart(df, base_sym, "1h", d, e, sl_, tp_, rr_, smc)
            send_photo(chart, caption=msg, chat_id=chat_id)
        except Exception:
            send_message(msg, chat_id=chat_id)
    except Exception as e:
        send_message(f"❌ Error analyzing {symbol}: {e}", chat_id=chat_id)


def _cmd_scan(chat_id, args):
    tf = args[0].lower() if args and args[0].lower() in ("1h","2h") else "1h"
    send_message(f"⏳ Running {tf.upper()} scan...", chat_id=chat_id)
    if _scan_fn_ref:
        threading.Thread(target=_scan_fn_ref, args=(tf,), daemon=True).start()
        send_message(f"✅ {tf.upper()} scan started in background.", chat_id=chat_id)
    else:
        send_message("❌ Scan function not registered.", chat_id=chat_id)


def dispatch_command(text, chat_id):
    parts = text.strip().split()
    cmd   = parts[0].lower().split("@")[0]
    args  = parts[1:]
    dispatch = {
        "/start":     lambda: _cmd_start(chat_id),
        "/signals":   lambda: _cmd_signals(chat_id),
        "/stats":     lambda: _cmd_stats(chat_id),
        "/history":   lambda: _cmd_history(chat_id),
        "/topsetups": lambda: _cmd_topsetups(chat_id),
        "/symbols":   lambda: _cmd_symbols(chat_id),
        "/test":      lambda: _cmd_test(chat_id),
        "/ping":      lambda: _cmd_ping(chat_id),
        "/filters":   lambda: _cmd_filters(chat_id, args),
        "/analyze":   lambda: _cmd_analyze(chat_id, args),
        "/scan":      lambda: _cmd_scan(chat_id, args),
    }
    fn = dispatch.get(cmd)
    if fn:
        fn()
    else:
        send_message("❓ Unknown command. Use /start for help.", chat_id=chat_id)


def start_polling():
    logger.info("Starting Telegram polling...")
    try:
        requests.get(f"{BASE_URL}/deleteWebhook",
                     params={"drop_pending_updates": True}, timeout=10)
        time.sleep(2)
    except Exception:
        pass
    offset = 0
    while True:
        updates = _get_updates(offset)
        for upd in updates:
            offset = upd["update_id"] + 1
            msg    = upd.get("message") or upd.get("edited_message")
            if not msg:
                continue
            text = msg.get("text","")
            if not text.startswith("/"):
                continue
            chat_id = str(msg["chat"]["id"])
            try:
                dispatch_command(text, chat_id)
            except Exception as e:
                logger.error(f"Command dispatch error: {e}")

# ─────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────

def analyze_symbol(symbol, timeframe="1h"):
    try:
        df = get_klines(symbol, timeframe, limit=250)
        if df is None or len(df) < 50:
            return None
        df.attrs["symbol"] = symbol
        smc = score_smc(df)
        if smc["score"] < MIN_SCORE:
            return None
        direction = smc["direction"]
        levels    = find_entry_sl_tp(df, direction, smc)
        if levels["rr"] < MIN_RR:
            return None
        curr = get_current_price(symbol) or df["close"].iloc[-1]
        passed, _ = run_all_filters(symbol, direction, levels["entry"], curr, df)
        if not passed:
            return None
        return {"symbol": symbol, "direction": direction, "timeframe": timeframe,
                "entry": levels["entry"], "sl": levels["sl"], "tp": levels["tp"],
                "rr": levels["rr"], "score": smc["score"], "sl_pct": levels["risk_pct"],
                "smc_result": smc, "df": df}
    except Exception as e:
        logger.error(f"analyze_symbol {symbol} {timeframe}: {e}")
        return None


def run_scan(timeframe="1h"):
    logger.info(f"Starting {timeframe} scan...")
    try:
        universe = get_universe(MIN_VOLUME_USDT)
        logger.info(f"Universe: {len(universe)} symbols")
    except Exception as e:
        logger.error(f"Universe failed: {e}")
        return

    scan_results = []
    for sym_info in universe:
        symbol = sym_info["symbol"]
        base   = symbol.replace("_USDT","").replace("USDT","")
        if not is_binance_listed(base):   continue
        if check_cooldown(symbol):        continue
        if active_count() >= MAX_ACTIVE_SIGNALS:
            break

        result = analyze_symbol(symbol, timeframe)
        if not result:
            continue

        scan_results.append({"symbol": result["symbol"], "direction": result["direction"],
                              "score": result["score"], "rr": result["rr"], "entry": result["entry"]})

        leverage = calc_leverage(result["sl_pct"])
        base_sym = symbol.replace("_USDT","").replace("USDT","")

        try:
            chart_bytes = generate_chart(
                result["df"], base_sym, timeframe, result["direction"],
                result["entry"], result["sl"], result["tp"], result["rr"],
                result["smc_result"], CHART_CANDLES)
        except Exception as e:
            logger.warning(f"Chart failed {symbol}: {e}")
            chart_bytes = None

        signal_data = {"symbol": symbol, "direction": result["direction"], "timeframe": timeframe,
                       "entry": result["entry"], "sl": result["sl"], "tp": result["tp"],
                       "rr": result["rr"], "score": result["score"], "leverage": leverage}

        if add_signal(signal_data):
            send_signal(symbol=symbol, direction=result["direction"], timeframe=timeframe,
                        entry=result["entry"], sl=result["sl"], tp=result["tp"],
                        rr=result["rr"], score=result["score"], leverage=leverage,
                        chart_bytes=chart_bytes)
            logger.info(f"Signal: {symbol} {result['direction'].upper()} RR:{result['rr']:.2f}")

    save_scan_results(scan_results)
    logger.info(f"{timeframe} scan done. {len(universe)} symbols checked.")


def run_exit_monitor():
    for sig in get_active_signals():
        symbol = sig.get("symbol")
        if not symbol: continue
        try:
            df_1m = get_klines(symbol, "1m", limit=5)
            if df_1m is None or len(df_1m) == 0: continue
            last = df_1m.iloc[-1]
            if update_signal_hit_sl(symbol, last["low"], last["high"]):
                close_signal(symbol, "loss", sig.get("sl"))
                send_sl_hit_message(symbol, sig["direction"], sig["entry"], sig["sl"])
                continue
            if update_signal_hit_tp(symbol, last["close"]):
                close_signal(symbol, "win", sig.get("tp"))
                send_tp_hit_message(symbol, sig["direction"], sig["entry"], sig["tp"])
        except Exception as e:
            logger.error(f"Exit monitor {symbol}: {e}")


def run_expiry_check():
    for sig in expire_old_signals():
        try:
            send_expired_message(sig["symbol"], sig["direction"], sig["entry"])
        except Exception as e:
            logger.error(f"Expiry notify {sig.get('symbol')}: {e}")


def run_volume_spike_check():
    try:
        for sym_info in get_universe(MIN_VOLUME_USDT)[:50]:
            symbol = sym_info["symbol"]
            base   = symbol.replace("_USDT","").replace("USDT","")
            if not is_binance_listed(base): continue
            if check_cooldown(symbol):      continue
            if active_count() >= MAX_ACTIVE_SIGNALS: break
            if not get_volume_spike(symbol): continue
            logger.info(f"Volume spike: {symbol}")
            result = analyze_symbol(symbol, "1h")
            if not result: continue
            leverage = calc_leverage(result["sl_pct"])
            signal_data = {"symbol": symbol, "direction": result["direction"], "timeframe": "1h",
                           "entry": result["entry"], "sl": result["sl"], "tp": result["tp"],
                           "rr": result["rr"], "score": result["score"], "leverage": leverage}
            if add_signal(signal_data):
                try:
                    chart_bytes = generate_chart(
                        result["df"], base, "1h", result["direction"],
                        result["entry"], result["sl"], result["tp"],
                        result["rr"], result["smc_result"], CHART_CANDLES)
                except Exception:
                    chart_bytes = None
                send_signal(symbol=symbol, direction=result["direction"], timeframe="1h",
                            entry=result["entry"], sl=result["sl"], tp=result["tp"],
                            rr=result["rr"], score=result["score"], leverage=leverage,
                            chart_bytes=chart_bytes)
    except Exception as e:
        logger.error(f"Volume spike check: {e}")


def _safe(fn):
    def wrapper(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"Scheduled job {fn.__name__}: {e}")
    return wrapper


def setup_scheduler():
    schedule.every(30).minutes.do(_safe(lambda: run_scan("1h")))
    schedule.every(60).minutes.do(_safe(lambda: run_scan("2h")))
    schedule.every(5).minutes.do(_safe(run_volume_spike_check))
    schedule.every(1).minutes.do(_safe(run_exit_monitor))
    schedule.every(10).minutes.do(_safe(run_expiry_check))
    logger.info("Scheduler configured.")


def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(10)


def main():
    logger.info("=== Crypto Futures Signal Bot Starting ===")
    start_keep_alive()

    global _scan_fn_ref
    _scan_fn_ref = run_scan

    setup_scheduler()

    sched_t = threading.Thread(target=run_scheduler, daemon=True, name="scheduler")
    sched_t.start()

    poll_t = threading.Thread(target=start_polling, daemon=True, name="telegram-polling")
    poll_t.start()

    logger.info("Running initial 1H scan...")
    threading.Thread(target=_safe(lambda: run_scan("1h")), daemon=True, name="initial-scan").start()

    try:
        while True:
            time.sleep(60)
            if not sched_t.is_alive():
                logger.error("Scheduler died, restarting...")
                sched_t = threading.Thread(target=run_scheduler, daemon=True, name="scheduler")
                sched_t.start()
            if not poll_t.is_alive():
                logger.error("Polling thread died, restarting...")
                poll_t = threading.Thread(target=start_polling, daemon=True, name="telegram-polling")
                poll_t.start()
    except KeyboardInterrupt:
        logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
