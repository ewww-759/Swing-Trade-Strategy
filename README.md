# Swing Pullback Strategy — Scored Signals + Risk-Based Sizing

A daily-timeframe trend-pullback system, ported to Python from a Pine Script v6 indicator I run for live swing trade alerts. The design separates **hard gates** (binary go/no-go) from a **weighted quality score**, then sizes positions by risk rather than dollars.

## Architecture

```
hard gates ──> layered score (0-100) ──> graded entry ──> risk-based sizing
```

**Layer 1 — hard gates (binary).** No score can rescue a failed gate:
- Liquidity: 10-day average dollar volume ≥ $50M
- Trend: close > SMA200, EMA50 > SMA200, SMA200 rising

**Layers 2–5 — weighted score:**

| Layer | Weight | What it measures |
|---|---|---|
| Position | 30 | Pullback into the EMA20–EMA50 zone, holding above EMA50, structure intact |
| Momentum | 25 | RSI reset toward 45, penalized by an accumulation/distribution (CLV×volume) ratio when sellers dominate |
| Trigger | 30 | Bullish reversal candle (full) or higher-low bull bar (partial) |
| Volume | 15 | At/above 20-day average (partial credit band below) |

The score is damped by trend separation (EMA50 vs SMA200 in ATRs) in weak trends, and a "spring" — a wick through EMA50 that closes back above — adds a small bonus.

**Structure state machine.** A close below EMA50 marks structure broken; recovery requires N consecutive closes back above a percentile of the pullback zone before the position layer can score again.

**Entries and exits:**
- A-grade (score ≥ 80): full risk unit
- B-grade (60–79, green candle, once per pullback episode): half risk unit
- Exit: structural stop (swing low − 0.5 ATR) or premium sell (close > 2 ATR above EMA20 with RSI > 70, confirmed by a bearish reversal candle)

**Risk-based sizing:**

```
qty = (capital × risk%) / stop_distance    capped by max-notional %
```

Every trade risks the same fraction of capital regardless of stop width — wide stops get small size, tight stops get large size.

## Files

```
strategy.py   # gates, scoring layers, state machine, sizing
backtest.py   # trade simulation: compounding equity, Sharpe, drawdown
```

## Quick start

```bash
pip install numpy pandas
python backtest.py                # synthetic demo data
python backtest.py daily.csv      # columns: open,high,low,close,volume
```

## Notes vs the live version

The live Pine version adds a sum-of-the-parts (SOTP) valuation modulator that scales size by fundamental floor bands, alert routing, and an on-chart dashboard. This port keeps the signal engine and sizing so behavior is reproducible and testable.
