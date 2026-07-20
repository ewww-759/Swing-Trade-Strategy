"""
Swing Pullback Strategy — scored signals + risk-based sizing.

Python port of a Pine Script v6 daily-timeframe system.

Architecture: hard gates -> layered score -> graded entries -> risk sizing.

Layer 1 (hard gates, binary):
    - liquidity: 10-day average dollar volume >= $50M
    - uptrend:   close > SMA200, EMA50 > SMA200, SMA200 rising

Layers 2-5 (weighted score, 0-100):
    - position (30): price pulled back into the EMA20-EMA50 zone,
      holding above EMA50, structure not broken
    - momentum (25): RSI reset toward 45, with an accumulation/
      distribution (close-location-value x volume) penalty multiplier
    - trigger  (30): bullish reversal candle (full credit) or higher-low
      bull bar (partial credit)
    - volume   (15): at/above 20-day average volume (partial band below)

A damping factor (trend separation in ATRs) scales the score in weak
trends; a "spring" wick through EMA50 that closes back above adds a
small bonus.

Entries:
    A-grade: score >= 80 -> full size
    B-grade: 60 <= score < 80, green candle, once per pullback episode -> half size

Exits:
    structural stop: swing low minus 0.5 ATR
    premium sell:    close stretched > 2 ATR above EMA20 with RSI > 70,
                     confirmed by a bearish reversal candle

Position sizing:
    qty = (capital x risk%) / stop_distance, capped by a max-notional
    limit — risk-defined sizing rather than fixed dollar amounts.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SwingConfig:
    trend_len: int = 200
    ema_fast_len: int = 20
    ema_slow_len: int = 50
    rsi_len: int = 14
    rsi_premium: int = 70
    atr_len: int = 14
    stretch_mult: float = 2.0
    lookback: int = 3
    zone_touch_win: int = 3
    vol_len: int = 20

    # scoring weights
    w_pos: float = 30.0
    w_rsi: float = 25.0
    w_trig: float = 30.0
    w_vol: float = 15.0
    a_threshold: float = 80.0
    b_threshold: float = 60.0

    # tolerance bands
    rsi_full: int = 45
    rsi_zero: int = 52
    trig_partial: float = 0.6
    vol_part_frac: float = 0.8

    # structure & quality
    sep_full_atr: float = 1.0
    damp_floor: float = 0.7
    wick_bonus: float = 5.0
    rec_pct: float = 50.0
    rec_bars: int = 3
    avp_win: int = 10
    avp_bad: float = -0.2
    avp_penalty: float = 0.3

    # liquidity gate
    min_dollar_vol_m: float = 50.0
    dollar_vol_len: int = 10

    # risk sizing
    capital: float = 100_000.0
    risk_pct: float = 1.0
    notional_cap_pct: float = 25.0
    whole_shares: bool = True

    # stop loss
    sl_swing_len: int = 5
    sl_atr_buff: float = 0.5


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, min_periods=length).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length).mean()


def compute_signals(df: pd.DataFrame, cfg: SwingConfig | None = None) -> pd.DataFrame:
    """Annotate an OHLCV DataFrame with scores, entries, exits, and sizing."""
    cfg = cfg or SwingConfig()
    df = df.reset_index(drop=True).copy()
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    # ---- core series ----
    ema_fast = close.ewm(span=cfg.ema_fast_len, min_periods=cfg.ema_fast_len).mean()
    ema_slow = close.ewm(span=cfg.ema_slow_len, min_periods=cfg.ema_slow_len).mean()
    sma_trend = close.rolling(cfg.trend_len).mean()
    rsi = _rsi(close, cfg.rsi_len)
    atr = _atr(df, cfg.atr_len)
    vol_avg = vol.rolling(cfg.vol_len).mean()

    # ---- hard gates ----
    avg_dollar_vol = (vol * close).rolling(cfg.dollar_vol_len).mean()
    liquid_ok = avg_dollar_vol >= cfg.min_dollar_vol_m * 1e6
    trend_rising = sma_trend > sma_trend.shift(5)
    uptrend = (close > sma_trend) & (ema_slow > sma_trend) & trend_rising
    gates_ok = liquid_ok & uptrend

    sep = ((ema_slow - sma_trend) / atr).replace([np.inf, -np.inf], 0).fillna(0)
    damp = np.maximum(cfg.damp_floor, np.minimum(sep / cfg.sep_full_atr, 1.0))

    # ---- structure state machine (break below EMA50 -> recovery count) ----
    recovery_level = ema_slow + cfg.rec_pct / 100 * np.maximum(ema_fast - ema_slow, 0)
    struct_broken = np.zeros(len(df), dtype=bool)
    broken, recov = False, 0
    for i in range(len(df)):
        if close.iat[i] < ema_slow.iat[i]:
            broken, recov = True, 0
        elif broken:
            if close.iat[i] >= recovery_level.iat[i]:
                recov += 1
                if recov >= cfg.rec_bars:
                    broken, recov = False, 0
            else:
                recov = 0
        struct_broken[i] = broken
    struct_broken = pd.Series(struct_broken, index=df.index)

    spring_wick = (low < ema_slow) & (close >= ema_slow)
    spring_recent = spring_wick.rolling(cfg.zone_touch_win, min_periods=1).max().astype(bool)

    # ---- layer 2: position ----
    touched_zone = low <= ema_fast
    zone_recent = touched_zone.rolling(cfg.zone_touch_win, min_periods=1).max().astype(bool)
    pos_factor = (zone_recent & (close >= ema_slow) & ~struct_broken).astype(float)

    # ---- layer 3: momentum (RSI reset + AVP absorption penalty) ----
    reset_win = max(cfg.lookback, cfg.zone_touch_win + 1)
    rsi_low = rsi.rolling(reset_win, min_periods=1).min()
    rsi_factor = np.where(
        rsi_low <= cfg.rsi_full, 1.0,
        np.where(rsi_low < cfg.rsi_zero,
                 (cfg.rsi_zero - rsi_low) / (cfg.rsi_zero - cfg.rsi_full), 0.0))

    rng = (high - low).replace(0, np.nan)
    clv = (((close - low) - (high - close)) / rng).fillna(0)
    avp_ratio = ((clv * vol).rolling(cfg.avp_win).sum()
                 / vol.rolling(cfg.avp_win).sum().replace(0, np.nan)).fillna(0)
    avp_mult = np.where(
        avp_ratio >= 0, 1.0,
        np.where(avp_ratio <= cfg.avp_bad, cfg.avp_penalty,
                 cfg.avp_penalty + (1 - cfg.avp_penalty)
                 * (avp_ratio - cfg.avp_bad) / (0 - cfg.avp_bad)))
    rsi_factor_adj = rsi_factor * avp_mult

    # ---- layer 4: trigger candle ----
    green = close > df["open"]
    reversal_up = green & (close > high.shift())
    higher_low = green & (low > low.shift()) & (close > close.shift())
    trig_factor = np.where(reversal_up, 1.0,
                           np.where(higher_low, cfg.trig_partial,
                                    np.where(green, cfg.trig_partial * 0.5, 0.0)))

    # ---- layer 5: volume ----
    vratio = (vol / vol_avg.replace(0, np.nan)).fillna(0)
    vol_factor = np.where(
        vratio >= 1, 1.0,
        np.where(vratio >= cfg.vol_part_frac,
                 (vratio - cfg.vol_part_frac) / (1 - cfg.vol_part_frac), 0.0))

    # ---- score ----
    total_w = cfg.w_pos + cfg.w_rsi + cfg.w_trig + cfg.w_vol
    raw = 100 * (cfg.w_pos * pos_factor + cfg.w_rsi * rsi_factor_adj
                 + cfg.w_trig * trig_factor + cfg.w_vol * vol_factor) / total_w
    bonus = np.where(spring_recent & (pos_factor > 0), cfg.wick_bonus, 0.0)
    score = np.minimum(100, raw * damp + bonus)

    # ---- graded entries (B-grade fires once per pullback episode) ----
    buy_a_raw = gates_ok & (pos_factor > 0) & (score >= cfg.a_threshold)
    buy_b_raw = (gates_ok & (pos_factor > 0) & (score >= cfg.b_threshold)
                 & (score < cfg.a_threshold) & green)
    buy_a = np.zeros(len(df), dtype=bool)
    buy_b = np.zeros(len(df), dtype=bool)
    episode_used = False
    for i in range(len(df)):
        if buy_a_raw.iat[i]:
            buy_a[i] = True
            episode_used = True
        elif buy_b_raw.iat[i] and not episode_used:
            buy_b[i] = True
            episode_used = True
        if close.iat[i] > ema_fast.iat[i] and not touched_zone.iat[i]:
            episode_used = False

    # ---- premium sell ----
    stretch = (close - ema_fast) / atr
    was_premium = ((stretch.rolling(cfg.lookback, min_periods=1).max() > cfg.stretch_mult)
                   & (rsi.rolling(cfg.lookback, min_periods=1).max() > cfg.rsi_premium))
    reversal_down = (close < df["open"]) & (close < low.shift())
    sell = liquid_ok & was_premium & reversal_down

    # ---- structural stop + risk-based sizing ----
    struct_sl = low.rolling(cfg.sl_swing_len).min() - cfg.sl_atr_buff * atr
    stop_dist = close - struct_sl
    risk_budget = cfg.capital * cfg.risk_pct / 100
    qty_risk = np.where(stop_dist > 0, risk_budget / stop_dist, 0.0)
    qty_cap = cfg.capital * cfg.notional_cap_pct / 100 / close
    qty = np.minimum(qty_risk, qty_cap)
    if cfg.whole_shares:
        qty = np.floor(qty)

    out = df.copy()
    out["score"] = score
    out["buy_a"] = buy_a
    out["buy_b"] = buy_b
    out["sell"] = sell.values
    out["struct_sl"] = struct_sl
    out["qty"] = qty
    out["uptrend"] = uptrend.values
    out["struct_broken"] = struct_broken.values
    return out
