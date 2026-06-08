"""
simulation_engine.py  –  v2
Concentrated AI Infrastructure Bottleneck Portfolio
$1,000 Capital  |  $50 Max Risk Per Position (5 % Rule)
=========================================================

Asset universe  (low share-price AI infrastructure bottleneck plays)
--------------------------------------------------------------------
  INOD  Innodata Inc         L1 Data Engineering   LLM training-data prep
  DUOT  Duos Technologies    L1 Edge AI Infra       Modular edge AI datacenters
  BABA  Alibaba Group        L2 AI Cloud            Discounted AI cloud infra
  AI    C3.ai Inc            L3 Enterprise AI       Enterprise AI application layer

Execution logic
---------------
  EQUITY PATH   : buy shares sized so a 20 % adverse move = exactly $50 max loss
                  → position size = $50 / 0.20 = $250 per holding
                  → 4 × $250 = $1,000 — fully invested, quarterly-rebalanced

  OPTIONS PATH  : 1-contract ATM vertical call debit spread scaled to $50 cost
                  → monthly roll; remaining $800 earns risk-free rate in cash
                  → max loss per position = debit paid = $50

Both execution paths are back-tested side-by-side.

Outputs
-------
  backtest_daily.csv    – daily equity + options NAV + benchmarks
  backtest_metrics.csv  – headline performance metrics for all series
  backtest_holdings.csv – per-holding equity attribution
"""

from __future__ import annotations

import subprocess
import sys
import warnings
import datetime as dt
from math import log, sqrt, exp

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO & RISK PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO: dict[str, dict] = {
    "INOD": {
        "layer": "L1 Data Engineering",
        "desc":  "LLM Training Data Prep (Innodata)",
    },
    "DUOT": {
        "layer": "L1 Edge AI Infra",
        "desc":  "Modular Edge AI Data Centers (Duos Technologies)",
    },
    "BABA": {
        "layer": "L2 AI Cloud",
        "desc":  "Discounted AI Cloud Infrastructure (Alibaba)",
    },
    "AI": {
        "layer": "L3 Enterprise AI",
        "desc":  "Enterprise AI Application Layer (C3.ai)",
    },
}

BENCHMARKS        = ["SPY", "QQQ"]
START_DATE        = "2021-06-01"          # post C3.ai IPO (Dec 2020)
END_DATE          = dt.date.today().isoformat()
INITIAL_CAPITAL   = 1_000.00             # strict $1,000 cap
MAX_RISK_PER_POS  = 50.00               # 5 % of $1,000 per trade
EQUITY_STOP_PCT   = 0.20               # 20 % stop defines equity position size
REBAL_FREQ        = "QE"               # quarterly calendar-end rebalance
RISK_FREE_RATE    = 0.0525
TRADING_DAYS      = 252

N_POSITIONS       = len(PORTFOLIO)
EQUITY_POS_SIZE   = MAX_RISK_PER_POS / EQUITY_STOP_PCT   # = $250 per position
OPT_ROLL_DAYS     = 30                   # roll options every 30 calendar days
OPT_SPREAD_WIDTH  = 0.10                 # short strike = long strike × 1.10
OPT_IV_PREMIUM    = 1.20                 # realised vol × 1.20 ≈ implied vol proxy
OPT_BUDGET        = MAX_RISK_PER_POS * N_POSITIONS   # $200 total premium at risk
OPT_CASH_RESERVE  = INITIAL_CAPITAL - OPT_BUDGET    # $800 held in cash

GIT_COMMIT_MSG = (
    "feat: pivot rocky3 to low-premium AI bottleneck tickers under $1k cap"
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted-close prices for all tickers."""
    print(f"  Fetching {len(tickers)} tickers ({start} → {end}) …")
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

    prices = prices.ffill().dropna(how="all")

    missing = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    if missing:
        print(f"  WARNING  No data for: {missing} — they will be dropped.")
        prices = prices.drop(columns=[t for t in missing if t in prices.columns],
                             errors="ignore")
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European call price (Black-Scholes)."""
    if T < 1e-6:
        return max(S - K, 0.0)
    sigma = max(sigma, 1e-4)
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return float(S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2))


def call_spread_value(
    S: float, K1: float, K2: float, T: float, r: float, sigma: float
) -> float:
    """Value of long K1-call / short K2-call vertical debit spread (per share)."""
    return bs_call(S, K1, T, r, sigma) - bs_call(S, K2, T, r, sigma)


def rolling_hist_vol(prices: pd.Series, window: int = 21) -> pd.Series:
    """Annualised realised volatility (rolling window)."""
    return prices.pct_change().rolling(window).std() * sqrt(TRADING_DAYS)


# ─────────────────────────────────────────────────────────────────────────────
# EQUITY BACKTEST  (quarterly rebalance, $250 / position)
# ─────────────────────────────────────────────────────────────────────────────

def run_equity_backtest(
    port_prices: pd.DataFrame,
    bm_prices: pd.DataFrame,
    initial_capital: float,
    pos_size: float,
    rebal_freq: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Equity portfolio backtest.
    Each ticker is initially sized to pos_size dollars; quarterly rebalanced
    back to equal-weight on current NAV.
    Returns (daily_df, trades_df).
    """
    tickers     = port_prices.columns.tolist()
    common_idx  = port_prices.index.intersection(bm_prices.index)
    port_prices = port_prices.loc[common_idx]
    bm_prices   = bm_prices.loc[common_idx]

    rebal_dates = set(port_prices.resample(rebal_freq).last().index)
    n_pos       = len(tickers)

    first_day   = port_prices.index[0]
    p0          = port_prices.loc[first_day]
    shares      = pd.Series({t: pos_size / p0[t] for t in tickers})
    cash        = initial_capital - (shares * p0).sum()

    bm_shares = {bm: initial_capital / bm_prices[bm].iloc[0]
                 for bm in bm_prices.columns}

    rows, trade_log = [], []
    trade_log.append({"date": first_day, "event": "INITIAL_ALLOC",
                      **shares.round(4).to_dict()})

    for date, row in port_prices.iterrows():
        nav   = float((shares * row).sum()) + cash
        entry: dict = {"date": date, "equity_nav": nav}
        for bm in bm_prices.columns:
            entry[f"{bm}_nav"] = bm_shares[bm] * float(bm_prices[bm][date])
        for t in tickers:
            entry[f"{t}_value"] = float(shares[t]) * float(row[t])
        rows.append(entry)

        if date in rebal_dates and date != first_day:
            alloc   = nav / n_pos
            shares  = pd.Series({t: alloc / float(row[t]) for t in tickers})
            cash    = 0.0
            trade_log.append({"date": date, "event": "REBALANCE",
                              **shares.round(4).to_dict()})

    daily  = pd.DataFrame(rows).set_index("date")
    trades = pd.DataFrame(trade_log).set_index("date")
    return daily, trades


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONS BACKTEST  (monthly ATM call debit spreads, $50 debit each)
# ─────────────────────────────────────────────────────────────────────────────

def run_options_backtest(
    port_prices: pd.DataFrame,
    initial_capital: float,
    max_risk: float,
    spread_width: float,
    roll_days: int,
    iv_premium: float,
    rfr: float,
) -> pd.DataFrame:
    """
    Monthly-roll vertical call debit spread simulation.

    For each ticker every roll_days calendar days:
      - Open an ATM call / (ATM × (1+spread_width)) call spread
      - Scale fractional contracts so total debit = max_risk (= $50)
      - Remaining capital earns risk-free rate in cash

    At each expiry, settle at intrinsic value and roll to a fresh spread.
    Returns a DataFrame with columns [options_nav, options_cash, <ticker>_opt_val].
    """
    tickers   = port_prices.columns.tolist()
    hist_vols = {t: rolling_hist_vol(port_prices[t]) for t in tickers}

    # State per ticker
    positions: dict[str, dict | None] = {t: None for t in tickers}
    last_roll:  dict[str, pd.Timestamp | None] = {t: None for t in tickers}
    opt_vals:   dict[str, float] = {t: 0.0 for t in tickers}

    cash_nav  = float(initial_capital)      # full $1,000 at start; deducted at first roll
    daily_rfr = (1 + rfr) ** (1 / TRADING_DAYS) - 1

    rows: list[dict] = []

    for date, row in port_prices.iterrows():

        for t in tickers:
            S      = float(row[t])
            iv_raw = hist_vols[t].get(date, np.nan)
            sigma  = float(iv_raw) * iv_premium if pd.notna(iv_raw) else 0.60

            should_roll = (
                last_roll[t] is None
                or (date - last_roll[t]).days >= roll_days
            )

            if should_roll:
                # ── Settle expiring spread at intrinsic ─────────────────────
                if positions[t] is not None:
                    p = positions[t]
                    intrinsic_per_share = max(
                        0.0, min(S - p["K1"], p["K2"] - p["K1"])
                    )
                    cash_nav   += intrinsic_per_share * 100.0 * p["contracts"]
                    opt_vals[t] = 0.0

                # ── Open fresh ATM spread ────────────────────────────────────
                K1    = S
                K2    = S * (1.0 + spread_width)
                T_new = roll_days / 365.0
                debit_ps = call_spread_value(K1, K1, K2, T_new, rfr,
                                             max(sigma, 0.05))
                debit_ps = max(debit_ps, 1e-4)

                contracts     = max_risk / (debit_ps * 100.0)
                actual_debit  = contracts * debit_ps * 100.0  # ≈ $50

                cash_nav   -= actual_debit
                opt_vals[t] = actual_debit

                positions[t] = {
                    "K1": K1, "K2": K2,
                    "contracts": contracts,
                    "entry_date": date,
                }
                last_roll[t] = date

            else:
                # ── Daily mark-to-market ─────────────────────────────────────
                p           = positions[t]
                days_elapsed = (date - last_roll[t]).days
                T_rem        = max((roll_days - days_elapsed) / 365.0, 1e-6)
                mtm_ps       = call_spread_value(S, p["K1"], p["K2"],
                                                 T_rem, rfr, max(sigma, 0.05))
                opt_vals[t]  = mtm_ps * 100.0 * p["contracts"]

        # Accrue interest on cash balance
        cash_nav *= (1 + daily_rfr)
        options_nav = cash_nav + sum(opt_vals.values())

        rows.append({
            "date":         date,
            "options_nav":  options_nav,
            "options_cash": cash_nav,
            **{f"{t}_opt_val": opt_vals[t] for t in tickers},
        })

    return pd.DataFrame(rows).set_index("date")


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

def _cagr(start: float, end: float, days: int) -> float:
    return (end / start) ** (TRADING_DAYS / days) - 1

def _ann_vol(ret: pd.Series) -> float:
    return float(ret.std() * sqrt(TRADING_DAYS))

def _sharpe(ret: pd.Series, rfr: float) -> float:
    daily_rf = (1 + rfr) ** (1 / TRADING_DAYS) - 1
    ex = ret - daily_rf
    return float(ex.mean() / ex.std() * sqrt(TRADING_DAYS)) if ex.std() else float("nan")

def _sortino(ret: pd.Series, rfr: float) -> float:
    daily_rf = (1 + rfr) ** (1 / TRADING_DAYS) - 1
    ex   = ret - daily_rf
    down = ex[ex < 0].std()
    return float(ex.mean() / down * sqrt(TRADING_DAYS)) if down else float("nan")

def _max_dd(nav: pd.Series) -> float:
    dd = (nav - nav.cummax()) / nav.cummax()
    return float(dd.min())

def _calmar(cagr_val: float, mdd: float) -> float:
    return cagr_val / abs(mdd) if mdd else float("nan")


def compute_metrics(daily: pd.DataFrame, rfr: float) -> pd.DataFrame:
    """Build headline metrics for every *_nav column in daily."""
    rows = []
    nav_cols = [c for c in daily.columns if c.endswith("_nav")]

    for col in nav_cols:
        nav  = daily[col].dropna()
        ret  = nav.pct_change().dropna()
        n    = len(nav)

        label_raw = col.replace("_nav", "")
        label = {
            "equity":  "Equity Path  ($250/pos, 20% stop)",
            "options": "Options Path ($50 debit spread/pos)",
            "SPY":     "SPY Benchmark",
            "QQQ":     "QQQ Benchmark",
        }.get(label_raw, label_raw.upper())

        cagr_val = _cagr(nav.iloc[0], nav.iloc[-1], n)
        mdd_val  = _max_dd(nav)

        rows.append({
            "Series":         label,
            "Start ($)":      f"{nav.iloc[0]:,.2f}",
            "End ($)":        f"{nav.iloc[-1]:,.2f}",
            "Total Return":   f"{nav.iloc[-1] / nav.iloc[0] - 1:+.1%}",
            "CAGR":           f"{cagr_val:+.1%}",
            "Ann. Vol":       f"{_ann_vol(ret):.1%}",
            "Sharpe":         f"{_sharpe(ret, rfr):.2f}",
            "Sortino":        f"{_sortino(ret, rfr):.2f}",
            "Max DD":         f"{mdd_val:.1%}",
            "Calmar":         f"{_calmar(cagr_val, mdd_val):.2f}",
        })

    return pd.DataFrame(rows).set_index("Series")


def compute_holdings(
    daily: pd.DataFrame,
    portfolio: dict,
    rfr: float,
    initial_capital: float,
) -> pd.DataFrame:
    """Per-holding attribution for the equity path."""
    eq_end = daily["equity_nav"].iloc[-1]
    rows = []
    for t, cfg in portfolio.items():
        col = f"{t}_value"
        if col not in daily.columns:
            continue
        h   = daily[col].dropna()
        ret = h.pct_change().dropna()
        rows.append({
            "Ticker":       t,
            "Layer":        cfg["layer"],
            "Description":  cfg["desc"],
            "Start ($)":    f"{h.iloc[0]:,.2f}",
            "End ($)":      f"{h.iloc[-1]:,.2f}",
            "Hold. Return": f"{h.iloc[-1] / h.iloc[0] - 1:+.1%}",
            "Contrib.":     f"{(h.iloc[-1] - h.iloc[0]) / initial_capital:+.1%}",
            "Port. Wt":     f"{h.iloc[-1] / eq_end:.1%}",
            "Sharpe":       f"{_sharpe(ret, rfr):.2f}",
            "Max DD":       f"{_max_dd(h):.1%}",
        })
    return pd.DataFrame(rows).set_index("Ticker")


# ─────────────────────────────────────────────────────────────────────────────
# POSITION SIZING TABLE  (printed for transparency)
# ─────────────────────────────────────────────────────────────────────────────

def print_sizing_table(prices: pd.DataFrame) -> None:
    """Show entry-day sizing for each ticker under both execution paths."""
    SEP2 = "─" * 80
    print("\n▸ POSITION SIZING AT ENTRY")
    print(SEP2)
    header = (
        f"{'Ticker':<6}  {'Price':>8}  "
        f"{'Eq. Shares':>10}  {'Eq. $':>8}  {'Eq. Risk':>9}  {'Stop @':>9}  │  "
        f"{'Opt K1':>8}  {'Opt K2':>8}  {'Opt Debit':>10}  {'Opt Risk':>9}"
    )
    print(header)
    print(SEP2)
    first_row = prices.iloc[0]
    for t in prices.columns:
        if t not in PORTFOLIO:
            continue
        S       = float(first_row[t])
        # Equity
        eq_shs  = EQUITY_POS_SIZE / S
        eq_risk = eq_shs * S * EQUITY_STOP_PCT
        stop_px = S * (1 - EQUITY_STOP_PCT)
        # Options: illustrative at-entry pricing
        iv_est  = 0.65
        T30     = 30 / 365.0
        K1      = S
        K2      = S * (1 + OPT_SPREAD_WIDTH)
        debit   = call_spread_value(S, K1, K2, T30, RISK_FREE_RATE, iv_est)
        contracts = MAX_RISK_PER_POS / (max(debit, 1e-4) * 100)
        print(
            f"{t:<6}  {S:>8.2f}  "
            f"{eq_shs:>10.3f}  {EQUITY_POS_SIZE:>8.2f}  {eq_risk:>9.2f}  {stop_px:>9.2f}  │  "
            f"{K1:>8.2f}  {K2:>8.2f}  {debit:>10.4f}  {MAX_RISK_PER_POS:>9.2f}"
        )
    print()
    print(
        f"  Equity   : {N_POSITIONS} × ${EQUITY_POS_SIZE:.0f} = "
        f"${EQUITY_POS_SIZE * N_POSITIONS:.0f} deployed  "
        f"| max loss per pos = ${MAX_RISK_PER_POS:.0f} if stop hit"
    )
    print(
        f"  Options  : {N_POSITIONS} × ${MAX_RISK_PER_POS:.0f} debit = "
        f"${OPT_BUDGET:.0f} at risk  "
        f"| ${OPT_CASH_RESERVE:.0f} cash earns {RISK_FREE_RATE:.1%} risk-free"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE REPORT
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "═" * 80
SEP2 = "─" * 80


def print_report(
    metrics: pd.DataFrame,
    holdings: pd.DataFrame,
    daily: pd.DataFrame,
    port_prices: pd.DataFrame,
    start: str,
    end: str,
) -> None:
    print()
    print(SEP)
    print("  CONCENTRATED AI BOTTLENECK PORTFOLIO  –  BACKTEST REPORT v2")
    print(f"  Period  : {start}  →  {end}")
    print(f"  Capital : ${INITIAL_CAPITAL:,.2f}  |  Max Risk / Position : ${MAX_RISK_PER_POS:.2f}  |  N = {N_POSITIONS}")
    print(SEP)

    print_sizing_table(port_prices)

    print("\n▸ PERFORMANCE COMPARISON")
    print(SEP2)
    print(metrics.T.to_string())

    print("\n▸ EQUITY PATH — HOLDING-LEVEL ATTRIBUTION")
    print(SEP2)
    print(holdings.to_string())

    # Drawdown detail
    eq_nav  = daily["equity_nav"]
    opt_nav = daily["options_nav"]
    dd_eq   = ((eq_nav - eq_nav.cummax()) / eq_nav.cummax())
    dd_opt  = ((opt_nav - opt_nav.cummax()) / opt_nav.cummax())

    print("\n▸ DRAWDOWN DETAIL")
    print(SEP2)
    print(f"  Equity  worst DD : {dd_eq.min():.1%}  on {dd_eq.idxmin().date()}")
    print(f"  Options worst DD : {dd_opt.min():.1%}  on {dd_opt.idxmin().date()}")
    print(f"  Equity  days below –10 % : {(dd_eq < -0.10).sum()}")
    print(f"  Options days below –10 % : {(dd_opt < -0.10).sum()}")

    print()
    print(SEP)
    print("  Output files written:")
    print("    backtest_daily.csv    – daily equity + options NAV + benchmarks")
    print("    backtest_metrics.csv  – performance metrics for all series")
    print("    backtest_holdings.csv – per-holding equity attribution")
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# GIT AUTOMATION  (stage → commit → push active branch)
# ─────────────────────────────────────────────────────────────────────────────

def git_push(commit_msg: str) -> bool:
    """Run git add / commit / push in the current working directory."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        print(f"\n[GIT] Active branch : {branch}")
    except Exception as exc:
        print(f"[GIT] Could not determine branch: {exc}")
        branch = "HEAD"

    success = True
    steps = [
        (["git", "add", "."],                           "git add ."),
        (["git", "commit", "-m", commit_msg],           f"git commit -m '…'"),
        (["git", "push", "-u", "origin", branch],       f"git push origin {branch}"),
    ]

    for cmd, desc in steps:
        result = subprocess.run(cmd, capture_output=True, text=True)
        ok     = result.returncode == 0
        symbol = "✓" if ok else "✗"
        print(f"  {symbol}  {desc}")
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines()[:6]:
                print(f"       {line}")
        if not ok:
            success = False
            if result.stderr.strip():
                for line in result.stderr.strip().splitlines()[:4]:
                    print(f"     STDERR: {line}")
            # "nothing to commit" is not a real failure
            if "nothing to commit" in result.stderr + result.stdout:
                success = True

    return success


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("━" * 80)
    print("  SIMULATION ENGINE  v2  –  Concentrated AI Bottleneck Portfolio")
    print(f"  Capital: ${INITIAL_CAPITAL:,.2f}  |  Risk/Pos: ${MAX_RISK_PER_POS:.2f}  |  Tickers: {list(PORTFOLIO)}")
    print("━" * 80)

    # ── 1. Download prices ────────────────────────────────────────────────────
    all_tickers = list(PORTFOLIO.keys()) + BENCHMARKS
    print(f"\n[1/5] Downloading price data ({START_DATE} → {END_DATE}) …")
    all_prices  = fetch_prices(all_tickers, START_DATE, END_DATE)

    port_tickers_avail = [t for t in PORTFOLIO if t in all_prices.columns]
    bm_tickers_avail   = [b for b in BENCHMARKS if b in all_prices.columns]
    port_prices = all_prices[port_tickers_avail].copy()
    bm_prices   = all_prices[bm_tickers_avail].copy()

    # Drop tickers with >20 % missing data after forward-fill
    coverage = port_prices.notna().mean()
    good = coverage[coverage > 0.80].index.tolist()
    dropped = [t for t in port_prices.columns if t not in good]
    if dropped:
        print(f"  WARNING  Sparse data — dropping: {dropped}")
        port_prices = port_prices[good]

    print(
        f"  ✓  {len(port_prices.columns)} portfolio tickers | "
        f"{len(bm_prices.columns)} benchmarks | "
        f"{len(port_prices)} trading days\n"
    )

    # ── 2. Equity backtest ────────────────────────────────────────────────────
    print("[2/5] Running equity backtest (quarterly rebalance, $250/position) …")
    equity_daily, trades = run_equity_backtest(
        port_prices, bm_prices, INITIAL_CAPITAL, EQUITY_POS_SIZE, REBAL_FREQ
    )
    print(f"  ✓  {len(trades)} rebalancing events\n")

    # ── 3. Options backtest ───────────────────────────────────────────────────
    print("[3/5] Running options backtest (monthly-roll ATM call spreads, $50/spread) …")
    opts_daily = run_options_backtest(
        port_prices,
        INITIAL_CAPITAL,
        MAX_RISK_PER_POS,
        OPT_SPREAD_WIDTH,
        OPT_ROLL_DAYS,
        OPT_IV_PREMIUM,
        RISK_FREE_RATE,
    )
    print("  ✓  Done\n")

    # ── 4. Metrics & CSV output ───────────────────────────────────────────────
    print("[4/5] Computing metrics and writing CSVs …")

    opt_cols = ["options_nav", "options_cash"] + \
               [c for c in opts_daily.columns if c.endswith("_opt_val")]
    daily = equity_daily.join(opts_daily[opt_cols], how="left")

    metrics  = compute_metrics(daily, RISK_FREE_RATE)
    holdings = compute_holdings(daily, PORTFOLIO, RISK_FREE_RATE, INITIAL_CAPITAL)

    daily.reset_index().rename(columns={"date": "Date"}).to_csv(
        "backtest_daily.csv", index=False, float_format="%.4f"
    )
    metrics.to_csv("backtest_metrics.csv")
    holdings.to_csv("backtest_holdings.csv")
    print("  ✓  CSVs written\n")

    print_report(metrics, holdings, daily, port_prices, START_DATE, END_DATE)

    # ── 5. Git commit & push ──────────────────────────────────────────────────
    print("[5/5] Committing and pushing to GitHub …")
    ok = git_push(GIT_COMMIT_MSG)

    # ── Final success log ─────────────────────────────────────────────────────
    eq_end  = daily["equity_nav"].iloc[-1]
    opt_end = daily["options_nav"].iloc[-1]
    print()
    print("━" * 80)
    if ok:
        print("  ✅  SUCCESS  –  All 5 steps complete.")
    else:
        print("  ⚠️   PARTIAL SUCCESS  –  Backtest complete; git step had warnings.")
    print()
    print(f"  📊  Backtest period  : {START_DATE}  →  {END_DATE}")
    print(f"  💵  Starting capital : ${INITIAL_CAPITAL:,.2f}")
    print(f"  📈  Equity path end  : ${eq_end:,.2f}  "
          f"({eq_end / INITIAL_CAPITAL - 1:+.1%} total return)")
    print(f"  📉  Options path end : ${opt_end:,.2f}  "
          f"({opt_end / INITIAL_CAPITAL - 1:+.1%} total return)")
    print()
    print("  Refresh your repo to view:")
    print("    backtest_daily.csv  |  backtest_metrics.csv  |  backtest_holdings.csv")
    print("━" * 80)
    print()


if __name__ == "__main__":
    main()
