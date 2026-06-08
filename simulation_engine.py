"""
simulation_engine.py
Concentrated AI Value Chain Portfolio – Backtest Engine
========================================================
Tracks a high-conviction basket of companies spanning every layer of the
modern AI value chain (silicon → cloud → software) and back-tests portfolio
performance against SPY and QQQ benchmarks.

Layers covered
--------------
  L1  Silicon / Compute   NVDA  TSM  AMD  AVGO
  L2  Cloud Hyperscalers  MSFT  GOOGL  AMZN
  L3  AI Software / Apps  META  PLTR  ANET

Outputs
-------
  • Console summary table (metrics + per-holding contribution)
  • backtest_daily.csv   – daily portfolio NAV & benchmark values
  • backtest_metrics.csv – headline performance metrics
  • backtest_holdings.csv – per-holding attribution
"""

from __future__ import annotations

import sys
import warnings
import datetime as dt
import textwrap

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO: dict[str, dict] = {
    # ticker : {weight, layer, description}
    "NVDA":  {"weight": 0.25, "layer": "L1 Silicon/Compute",   "desc": "GPU / AI Accelerators"},
    "TSM":   {"weight": 0.10, "layer": "L1 Silicon/Compute",   "desc": "Leading-edge Foundry"},
    "AMD":   {"weight": 0.08, "layer": "L1 Silicon/Compute",   "desc": "CPU + GPU / MI-series AI"},
    "AVGO":  {"weight": 0.07, "layer": "L1 Silicon/Compute",   "desc": "Custom ASICs + Networking ICs"},
    "MSFT":  {"weight": 0.15, "layer": "L2 Cloud Hyperscaler", "desc": "Azure + OpenAI partnership"},
    "GOOGL": {"weight": 0.12, "layer": "L2 Cloud Hyperscaler", "desc": "GCP + DeepMind + TPUs"},
    "AMZN":  {"weight": 0.06, "layer": "L2 Cloud Hyperscaler", "desc": "AWS + Bedrock + Trainium"},
    "META":  {"weight": 0.08, "layer": "L3 AI Software/Apps",  "desc": "Llama models + AI Infra"},
    "ANET":  {"weight": 0.05, "layer": "L3 AI Software/Apps",  "desc": "AI Datacenter Networking"},
    "PLTR":  {"weight": 0.04, "layer": "L3 AI Software/Apps",  "desc": "AIP / Enterprise AI Analytics"},
}

BENCHMARKS     = ["SPY", "QQQ"]
START_DATE     = "2022-01-01"
END_DATE       = dt.date.today().isoformat()
INITIAL_CAPITAL = 1_000_000.00   # USD
REBAL_FREQ     = "QE"            # calendar-quarter end rebalance
RISK_FREE_RATE = 0.0525          # approx 10-yr treasury yield for Sharpe/Sortino

assert abs(sum(v["weight"] for v in PORTFOLIO.values()) - 1.0) < 1e-9, "Weights must sum to 1."

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted-close prices; fall back to 'Close' if adj not available."""
    print(f"  Fetching {len(tickers)} tickers from {start} to {end} …")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    prices = prices.dropna(how="all")
    missing = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    if missing:
        print(f"  WARNING – could not fetch data for: {missing}. They will be excluded.")
        for t in missing:
            if t in prices.columns:
                prices.drop(columns=[t], inplace=True)
    return prices.ffill().dropna(how="all")


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def build_target_weights(portfolio: dict, available: list[str]) -> pd.Series:
    """Return target weights renormalised to available tickers."""
    w = {t: portfolio[t]["weight"] for t in portfolio if t in available}
    total = sum(w.values())
    return pd.Series({t: v / total for t, v in w.items()})


def run_backtest(
    prices: pd.DataFrame,
    bm_prices: pd.DataFrame,
    portfolio: dict,
    initial_capital: float,
    rebal_freq: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate a rebalanced long-only portfolio.

    Returns
    -------
    daily : DataFrame with columns [portfolio_nav, *benchmark_navs, *ticker_values]
    trades : DataFrame of rebalancing events
    """
    port_tickers  = [t for t in portfolio if t in prices.columns]
    target_w      = build_target_weights(portfolio, port_tickers)
    common_index  = prices.index.intersection(bm_prices.index)
    prices        = prices.loc[common_index, port_tickers]
    bm_prices     = bm_prices.loc[common_index]

    # Rebalancing dates (quarter-end calendar dates)
    rebal_dates = set(
        prices.resample(rebal_freq).last().index.tolist()
    )

    # ── Initialise state ──────────────────────────────────────────────────────
    shares    = pd.Series(0.0, index=port_tickers)
    cash      = 0.0
    nav_rows  = []
    trade_log = []

    first_day      = prices.index[0]
    opening_prices = prices.loc[first_day]
    shares         = (target_w * initial_capital) / opening_prices
    cash           = 0.0   # fully invested from day 1

    # Benchmark: buy & hold from day 1
    bm_shares: dict[str, float] = {}
    for bm in bm_prices.columns:
        bm_shares[bm] = initial_capital / bm_prices[bm].iloc[0]

    trade_log.append({"date": first_day, "event": "INITIAL_ALLOC", **shares.to_dict()})

    for date, row in prices.iterrows():
        port_nav = (shares * row).sum() + cash

        nav_entry: dict = {"date": date, "portfolio_nav": port_nav}
        for bm in bm_prices.columns:
            nav_entry[f"{bm}_nav"] = bm_shares[bm] * bm_prices[bm].loc[date]
        for t in port_tickers:
            nav_entry[f"{t}_value"] = shares[t] * row[t]
        nav_rows.append(nav_entry)

        # Rebalance if this is a rebalance date (and not the first day)
        if date in rebal_dates and date != first_day:
            new_shares = (target_w * port_nav) / row
            trade_log.append({
                "date": date,
                "event": "REBALANCE",
                **new_shares.to_dict(),
            })
            shares = new_shares
            cash   = 0.0

    daily  = pd.DataFrame(nav_rows).set_index("date")
    trades = pd.DataFrame(trade_log).set_index("date")
    return daily, trades


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

TRADING_DAYS = 252


def cagr(start_val: float, end_val: float, days: int) -> float:
    years = days / TRADING_DAYS
    return (end_val / start_val) ** (1 / years) - 1


def annualised_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe(returns: pd.Series, rfr: float) -> float:
    daily_rf = (1 + rfr) ** (1 / TRADING_DAYS) - 1
    excess   = returns - daily_rf
    vol      = excess.std()
    return (excess.mean() / vol) * np.sqrt(TRADING_DAYS) if vol else np.nan


def sortino(returns: pd.Series, rfr: float) -> float:
    daily_rf    = (1 + rfr) ** (1 / TRADING_DAYS) - 1
    excess      = returns - daily_rf
    downside    = excess[excess < 0].std()
    return (excess.mean() / downside) * np.sqrt(TRADING_DAYS) if downside else np.nan


def max_drawdown(nav: pd.Series) -> float:
    roll_max = nav.cummax()
    dd       = (nav - roll_max) / roll_max
    return dd.min()


def calmar(annual_ret: float, mdd: float) -> float:
    return annual_ret / abs(mdd) if mdd else np.nan


def beta_alpha(port_ret: pd.Series, bm_ret: pd.Series, rfr: float) -> tuple[float, float]:
    daily_rf = (1 + rfr) ** (1 / TRADING_DAYS) - 1
    p_excess = port_ret - daily_rf
    b_excess = bm_ret   - daily_rf
    cov      = np.cov(p_excess, b_excess)
    beta_val = cov[0, 1] / cov[1, 1] if cov[1, 1] else np.nan
    alpha    = (p_excess.mean() - beta_val * b_excess.mean()) * TRADING_DAYS
    return beta_val, alpha


def compute_metrics(
    daily: pd.DataFrame,
    portfolio: dict,
    rfr: float,
    initial_capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary_metrics_df, holding_attribution_df)."""

    port_nav  = daily["portfolio_nav"]
    port_ret  = port_nav.pct_change().dropna()
    n_days    = len(port_nav)

    metrics: list[dict] = []

    def _row(label, series_nav, series_ret):
        ann_ret  = cagr(series_nav.iloc[0], series_nav.iloc[-1], n_days)
        vol      = annualised_vol(series_ret)
        sr       = sharpe(series_ret, rfr)
        so       = sortino(series_ret, rfr)
        mdd      = max_drawdown(series_nav)
        cal      = calmar(ann_ret, mdd)
        tot_ret  = series_nav.iloc[-1] / series_nav.iloc[0] - 1
        return {
            "Series":           label,
            "Start NAV ($)":    f"{series_nav.iloc[0]:,.0f}",
            "End NAV ($)":      f"{series_nav.iloc[-1]:,.0f}",
            "Total Return":     f"{tot_ret:+.1%}",
            "CAGR":             f"{ann_ret:+.1%}",
            "Ann. Volatility":  f"{vol:.1%}",
            "Sharpe Ratio":     f"{sr:.2f}",
            "Sortino Ratio":    f"{so:.2f}",
            "Max Drawdown":     f"{mdd:.1%}",
            "Calmar Ratio":     f"{cal:.2f}",
        }

    metrics.append(_row("AI Value Chain Portfolio", port_nav, port_ret))

    for bm in [c.replace("_nav", "") for c in daily.columns if c.endswith("_nav")]:
        bm_nav = daily[f"{bm}_nav"]
        bm_ret = bm_nav.pct_change().dropna()
        row    = _row(bm, bm_nav, bm_ret)
        b, a   = beta_alpha(port_ret, bm_ret, rfr)
        row["Beta (vs port)"]  = f"{b:.2f}"
        row["Alpha (annual)"]  = f"{a:+.1%}"
        metrics.append(row)

    metrics_df = pd.DataFrame(metrics).set_index("Series")

    # ── Per-holding attribution ───────────────────────────────────────────────
    holding_rows = []
    for t, cfg in portfolio.items():
        col = f"{t}_value"
        if col not in daily.columns:
            continue
        h_nav    = daily[col]
        h_start  = h_nav.iloc[0]
        h_end    = h_nav.iloc[-1]
        h_ret    = h_end / h_start - 1
        h_final  = h_end
        port_end = port_nav.iloc[-1]
        contrib  = (h_end - h_start) / initial_capital  # contribution to total return

        h_price_ret = (
            daily[col] / daily[col].shift(1) - 1
        ).dropna()

        holding_rows.append({
            "Ticker":           t,
            "Layer":            cfg["layer"],
            "Description":      cfg["desc"],
            "Target Weight":    f"{cfg['weight']:.0%}",
            "End Value ($)":    f"{h_final:,.0f}",
            "Holding Return":   f"{h_ret:+.1%}",
            "Return Contrib.":  f"{contrib:+.1%}",
            "End Port. Weight": f"{h_final/port_end:.1%}",
            "Sharpe":           f"{sharpe(h_price_ret, rfr):.2f}",
            "Max DD":           f"{max_drawdown(daily[col]):.1%}",
        })

    holding_df = pd.DataFrame(holding_rows).set_index("Ticker")
    return metrics_df, holding_df


# ─────────────────────────────────────────────────────────────────────────────
# REPORT PRINTING
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "═" * 80
SEP2 = "─" * 80


def print_report(
    metrics_df: pd.DataFrame,
    holding_df: pd.DataFrame,
    daily: pd.DataFrame,
    start: str,
    end: str,
    initial_capital: float,
) -> None:
    print()
    print(SEP)
    print(" CONCENTRATED AI VALUE CHAIN PORTFOLIO  –  BACKTEST REPORT")
    print(f" Period : {start}  →  {end}  |  Initial Capital : ${initial_capital:,.0f}")
    print(SEP)

    print("\n▸ PERFORMANCE SUMMARY")
    print(SEP2)
    # Transpose for readability
    print(metrics_df.T.to_string())
    print()

    print("\n▸ HOLDING-LEVEL ATTRIBUTION")
    print(SEP2)
    print(holding_df.to_string())
    print()

    # Layer-level aggregation
    print("\n▸ LAYER AGGREGATION (End Value)")
    print(SEP2)
    layer_map = {t: v["layer"] for t, v in PORTFOLIO.items()}
    holding_copy = holding_df.copy()
    holding_copy["_layer"] = holding_copy["Layer"]
    val_col = holding_copy["End Value ($)"].str.replace(",", "").str.replace("$", "").astype(float)
    holding_copy["_end_val"] = val_col
    layer_agg = holding_copy.groupby("_layer")["_end_val"].sum()
    layer_pct = layer_agg / layer_agg.sum()
    layer_out = pd.DataFrame({"End Value ($)": layer_agg.map("${:,.0f}".format),
                               "Portfolio %":   layer_pct.map("{:.1%}".format)})
    print(layer_out.to_string())
    print()

    # Quick drawdown summary
    port_nav = daily["portfolio_nav"]
    roll_max = port_nav.cummax()
    dd_series = (port_nav - roll_max) / roll_max
    worst_dd_date = dd_series.idxmin()
    print("\n▸ DRAWDOWN DETAIL")
    print(SEP2)
    print(f"  Worst drawdown of {dd_series.min():.1%} reached on {worst_dd_date.date()}")
    below_10 = (dd_series < -0.10).sum()
    below_20 = (dd_series < -0.20).sum()
    print(f"  Days below –10 % : {below_10}")
    print(f"  Days below –20 % : {below_20}")

    print()
    print(SEP)
    print(" Output files:")
    print("   backtest_daily.csv    – daily NAV for portfolio + benchmarks")
    print("   backtest_metrics.csv  – headline performance metrics")
    print("   backtest_holdings.csv – per-holding attribution table")
    print(SEP)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("━" * 80)
    print("  SIMULATION ENGINE  –  Concentrated AI Value Chain Portfolio Backtest")
    print("━" * 80)

    all_tickers = list(PORTFOLIO.keys()) + BENCHMARKS
    print(f"\n[1/4] Downloading price data ({START_DATE} → {END_DATE}) …")
    all_prices  = fetch_prices(all_tickers, START_DATE, END_DATE)

    port_prices = all_prices[[t for t in PORTFOLIO if t in all_prices.columns]]
    bm_prices   = all_prices[[b for b in BENCHMARKS if b in all_prices.columns]]

    print(f"  ✓  {len(port_prices.columns)} portfolio tickers | "
          f"{len(bm_prices.columns)} benchmarks | "
          f"{len(port_prices)} trading days\n")

    print("[2/4] Running backtest …")
    daily, trades = run_backtest(
        port_prices,
        bm_prices,
        PORTFOLIO,
        INITIAL_CAPITAL,
        REBAL_FREQ,
    )
    print(f"  ✓  {len(trades)} rebalancing events\n")

    print("[3/4] Computing performance metrics …")
    metrics_df, holding_df = compute_metrics(daily, PORTFOLIO, RISK_FREE_RATE, INITIAL_CAPITAL)
    print("  ✓  Done\n")

    print("[4/4] Writing output files …")
    daily.reset_index().rename(columns={"date": "Date"}).to_csv(
        "backtest_daily.csv", index=False, float_format="%.4f"
    )
    metrics_df.to_csv("backtest_metrics.csv")
    holding_df.to_csv("backtest_holdings.csv")
    print("  ✓  Files written\n")

    print_report(metrics_df, holding_df, daily, START_DATE, END_DATE, INITIAL_CAPITAL)


if __name__ == "__main__":
    main()
