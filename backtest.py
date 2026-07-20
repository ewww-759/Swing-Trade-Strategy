"""
Backtest for the swing pullback strategy.

Entry:  A-grade -> full risk unit, B-grade -> half risk unit
Exit:   structural stop hit, or premium-sell signal
Sizing: risk-defined — qty = (capital x risk%) / stop distance,
        capped by max notional. Equity compounds per trade.

Usage
-----
    python backtest.py                 # synthetic demo data
    python backtest.py path/to/daily_ohlcv.csv
"""

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy import SwingConfig, compute_signals


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    grade: str
    entry: float
    exit: float
    qty: float
    reason: str

    @property
    def pnl(self) -> float:
        return (self.exit - self.entry) * self.qty


def backtest(df: pd.DataFrame, cfg: SwingConfig | None = None) -> dict:
    cfg = cfg or SwingConfig()
    a = compute_signals(df, cfg)

    closes, lows = a["close"].values, a["low"].values
    n = len(a)
    trades: list[Trade] = []
    equity = cfg.capital
    curve = [equity]

    in_pos = False
    entry_i = sl = qty = 0.0
    grade = ""

    for i in range(n):
        if in_pos:
            exited, reason, px = False, "", closes[i]
            if lows[i] <= sl:
                exited, reason, px = True, "stop", sl
            elif a["sell"].iat[i]:
                exited, reason = True, "premium_sell"
            if exited:
                t = Trade(int(entry_i), i, grade, closes[int(entry_i)], px, qty, reason)
                trades.append(t)
                equity += t.pnl
                curve.append(equity)
                in_pos = False
            continue

        if (a["buy_a"].iat[i] or a["buy_b"].iat[i]) and i < n - 1:
            grade = "A" if a["buy_a"].iat[i] else "B"
            scale = 1.0 if grade == "A" else 0.5
            qty = a["qty"].iat[i] * scale * (equity / cfg.capital)
            if cfg.whole_shares:
                qty = np.floor(qty)
            if qty < 1:
                continue
            entry_i, sl, in_pos = i, a["struct_sl"].iat[i], True

    curve = np.array(curve)
    rets = np.diff(curve) / curve[:-1] if len(curve) > 1 else np.array([])
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak
    pnls = np.array([t.pnl for t in trades])

    stats = {
        "trades": len(trades),
        "win_rate": float((pnls > 0).mean()) if len(pnls) else 0.0,
        "total_return": float(curve[-1] / cfg.capital - 1),
        "sharpe": float(rets.mean() / rets.std() * np.sqrt(252))
        if len(rets) > 1 and rets.std() > 0 else 0.0,
        "max_drawdown": float(dd.min()),
        "final_equity": float(curve[-1]),
    }
    return {"stats": stats, "trades": trades, "annotated": a}


def make_demo_data(n: int = 1500, seed: int = 3) -> pd.DataFrame:
    """Synthetic daily data with a trending regime and pullbacks."""
    rng = np.random.default_rng(seed)
    trend = np.linspace(0, 60, n)
    cycles = 8 * np.sin(np.linspace(0, 12 * np.pi, n))
    noise = np.cumsum(rng.normal(0, 0.8, n))
    close = 100 + trend + cycles + noise
    close = np.maximum(close, 5)
    open_ = np.roll(close, 1) + rng.normal(0, 0.3, n)
    open_[0] = 100
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.6, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.6, n))
    volume = rng.lognormal(15.5, 0.4, n)  # ~$50M+ dollar volume
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume})


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data = pd.read_csv(sys.argv[1])
        data.columns = [c.lower() for c in data.columns]
    else:
        print("No CSV supplied — running on synthetic demo data.\n")
        data = make_demo_data()

    result = backtest(data)
    s = result["stats"]
    print(f"Trades:        {s['trades']}")
    print(f"Win rate:      {s['win_rate']:.1%}")
    print(f"Total return:  {s['total_return']:.2%}")
    print(f"Sharpe:        {s['sharpe']:.2f}")
    print(f"Max drawdown:  {s['max_drawdown']:.2%}")
    print(f"Final equity:  ${s['final_equity']:,.0f}")
