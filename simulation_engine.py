"""
simulation_engine.py  –  v3
Dynamic Capital Layering Portfolio
$1,000 Capital  |  Momentum-Weighted Rotation  |  200-day MA Cash Reservation
==============================================================================

Framework
---------
  Every quarter the engine applies a three-step decision process:

  STEP 1  200-day MA filter  (cash reservation rule)
          Any ticker trading below its 200-day MA is dropped from the rotation;
          its capital share stays as cash earning the risk-free rate.

  STEP 2  60-day & 63-trading-day momentum ranking
          Surviving tickers are ranked by their 63-trading-day return.
          Profits are NEVER cut to fund laggards.

  STEP 3  50 / 30 / 20 momentum weighting
          Rank-1 → 50 %, Rank-2 → 30 %, Rank-3 → 20 %.
          Any capital without a qualifying ticker accumulates as cash.

  INTRADAY DEFENSE  20 % trailing stop-loss per position
          Each position tracks its post-entry high-water mark.
          If price falls 20 % below that mark, the position is closed
          immediately and held as cash until the next quarterly rebalance.

Asset universe  (AI infrastructure bottleneck plays)
----------------------------------------------------
  INOD  Innodata Inc         L1 Data Engineering   LLM training-data prep
  DUOT  Duos Technologies    L1 Edge AI Infra       Modular edge AI datacenters
  BABA  Alibaba Group        L2 AI Cloud            Discounted AI cloud infra
  AI    C3.ai Inc            L3 Enterprise AI       Enterprise AI application layer

Output files
------------
  backtest_daily.csv       – daily NAV (dynamic equity + options + benchmarks)
  backtest_metrics.csv     – headline performance metrics for all series
  backtest_holdings.csv    – end-state per-holding equity attribution
  backtest_allocations.csv – quarterly allocation decisions + momentum scores
"""

from __future__ import annotations

import subprocess
import warnings
import datetime as dt
from math import log, sqrt, exp

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO & GLOBAL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO: dict[str, dict] = {
    "INOD": {"layer": "L1 Data Engineering", "desc": "LLM Training Data Prep (Innodata)"},
    "DUOT": {"layer": "L1 Edge AI Infra",    "desc": "Modular Edge AI Data Centers (Duos Tech)"},
    "BABA": {"layer": "L2 AI Cloud",         "desc": "Discounted AI Cloud Infrastructure (Alibaba)"},
    "AI":   {"layer": "L3 Enterprise AI",    "desc": "Enterprise AI Application Layer (C3.ai)"},
}

BENCHMARKS       = ["SPY", "QQQ"]
START_DATE       = "2021-06-01"
END_DATE         = dt.date.today().isoformat()
INITIAL_CAPITAL  = 1_000.00
RISK_FREE_RATE   = 0.0525
TRADING_DAYS     = 252
REBAL_FREQ       = "QE"            # quarterly calendar-end

# Dynamic Capital Layering
MA_LONG          = 200             # cash reservation filter
MA_SHORT         = 60              # momentum reference window
LOOKBACK_DAYS    = 63              # ≈ 1 calendar quarter of trading days
WEIGHTS_SCHEDULE = [0.50, 0.30, 0.20]   # rank-1 / rank-2 / rank-3
TRAIL_STOP_PCT   = 0.20            # 20 % trailing stop per position

# Options simulation (unchanged from v2)
MAX_RISK_PER_POS = 50.00
OPT_ROLL_DAYS    = 30
OPT_SPREAD_WIDTH = 0.10
OPT_IV_PREMIUM   = 1.20
N_POSITIONS      = len(PORTFOLIO)

GIT_COMMIT_MSG   = "feat: refactor to Dynamic Capital Layering framework"

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    print(f"  Fetching {len(tickers)} tickers ({start} → {end}) …")
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, threads=True,
    )
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = prices.ffill().dropna(how="all")
    missing = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    if missing:
        print(f"  WARNING  No data for: {missing}")
        prices = prices.drop(columns=[t for t in missing if t in prices.columns], errors="ignore")
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC CAPITAL LAYERING BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

def run_dynamic_backtest(
    port_prices: pd.DataFrame,
    bm_prices: pd.DataFrame,
    initial_capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dynamic Capital Layering backtest.

    Quarterly rebalance logic
    -------------------------
    1. Liquidate all open positions → compute total NAV.
    2. Filter by 200-day MA:  price < MA200 → ticker excluded (cash reservation).
    3. Rank survivors by 63-trading-day return (highest = rank 1).
    4. Allocate NAV:  50% rank-1 | 30% rank-2 | 20% rank-3.
       Any unallocated fraction stays as cash and earns the risk-free rate.

    Intraday stop logic (every trading day between rebalances)
    ----------------------------------------------------------
    Each open position tracks a high-water mark (HWM).  When:
        current_price < HWM × (1 − TRAIL_STOP_PCT)
    the position is liquidated immediately and converted to cash.
    The stop is only re-armed after the next quarterly rebalance.

    Returns
    -------
    daily       : DatetimeIndex DataFrame with equity_nav, bm navs, per-holding values
    alloc_log   : DataFrame of every rebalance and stop-trigger event
    """
    tickers    = port_prices.columns.tolist()
    common_idx = port_prices.index.intersection(bm_prices.index)
    port_prices = port_prices.loc[common_idx]
    bm_prices   = bm_prices.loc[common_idx]

    ma200      = port_prices.rolling(MA_LONG,  min_periods=MA_LONG).mean()
    ma60       = port_prices.rolling(MA_SHORT, min_periods=MA_SHORT).mean()

    rebal_dates = set(port_prices.resample(REBAL_FREQ).last().index)
    daily_rfr   = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1

    shares: dict[str, float] = {t: 0.0        for t in tickers}
    hwm:    dict[str, float] = {t: float("nan") for t in tickers}
    cash    = float(initial_capital)

    bm_shares = {
        bm: initial_capital / float(bm_prices[bm].iloc[0])
        for bm in bm_prices.columns
    }

    price_index = port_prices.index.tolist()
    rows: list[dict]       = []
    alloc_rows: list[dict] = []

    for date, row in port_prices.iterrows():
        today = {t: float(row[t]) for t in tickers}

        # ── REBALANCE ─────────────────────────────────────────────────────────
        if date in rebal_dates:
            # Accrue interest and compute total NAV before trading
            cash_with_interest = cash * (1 + daily_rfr)
            nav = cash_with_interest + sum(
                shares[t] * today[t] for t in tickers if shares[t] > 0
            )

            # Liquidate all positions
            for t in tickers:
                shares[t] = 0.0
                hwm[t]    = float("nan")
            cash = nav  # fully in cash

            # 200-day MA filter
            eligible: list[str] = []
            ma200_snap: dict[str, float] = {}
            ma60_snap:  dict[str, float] = {}

            for t in tickers:
                m200 = float(ma200[t].get(date, np.nan))
                m60  = float(ma60[t].get(date, np.nan))
                ma200_snap[t] = m200
                ma60_snap[t]  = m60
                if not np.isnan(m200) and today[t] > m200:
                    eligible.append(t)

            # 63-trading-day momentum ranking
            idx_pos    = price_index.index(date)
            lb_pos     = max(0, idx_pos - LOOKBACK_DAYS)
            momentum:  dict[str, float] = {}

            for t in eligible:
                past = float(port_prices[t].iloc[lb_pos])
                momentum[t] = today[t] / past - 1 if past > 0 else 0.0

            ranked = sorted(eligible, key=lambda t: momentum[t], reverse=True)

            # Assign weights: top-3 only
            alloc_w: dict[str, float] = {}
            for i, t in enumerate(ranked[: len(WEIGHTS_SCHEDULE)]):
                alloc_w[t] = WEIGHTS_SCHEDULE[i]

            # Open new positions
            for t, w in alloc_w.items():
                shares[t] = (nav * w) / today[t]
                hwm[t]    = today[t]          # arm trailing stop at entry price
            cash = nav - sum(shares[t] * today[t] for t in tickers)

            deployed_pct = 1.0 - cash / nav if nav > 0 else 0.0

            alloc_rows.append({
                "Date":              date.date(),
                "Event":             "REBALANCE",
                "NAV ($)":           round(nav, 2),
                "Eligible (>MA200)": ", ".join(eligible) if eligible else "NONE (all cash)",
                **{f"{t}_MA200":     round(ma200_snap[t], 2) if not np.isnan(ma200_snap[t]) else "N/A"
                   for t in tickers},
                **{f"{t}_MA60":      round(ma60_snap[t], 2) if not np.isnan(ma60_snap[t]) else "N/A"
                   for t in tickers},
                **{f"{t}_Momentum":  f"{momentum.get(t, float('nan')):+.1%}"
                   for t in tickers},
                "Rank1 (50%)":       ranked[0] if len(ranked) > 0 else "—",
                "Rank2 (30%)":       ranked[1] if len(ranked) > 1 else "—",
                "Rank3 (20%)":       ranked[2] if len(ranked) > 2 else "—",
                "Cash Reserve ($)":  round(cash, 2),
                "Deployed (%)":      f"{deployed_pct:.1%}",
            })

        # ── INTRADAY TRAILING STOP (non-rebalance days) ───────────────────────
        else:
            # Accrue daily interest on idle cash
            cash *= (1 + daily_rfr)

            for t in tickers:
                if shares[t] > 0:
                    p     = today[t]
                    hwm[t] = max(hwm[t] if not np.isnan(hwm[t]) else p, p)
                    drop_from_hwm = 1.0 - p / hwm[t]

                    if drop_from_hwm >= TRAIL_STOP_PCT:
                        proceeds = shares[t] * p
                        cash    += proceeds
                        alloc_rows.append({
                            "Date":              date.date(),
                            "Event":             f"TRAILING STOP — {t}",
                            "NAV ($)":           round(
                                cash + sum(shares[tt] * today[tt] for tt in tickers), 2
                            ),
                            "Eligible (>MA200)": f"Stop: {t} closed @ ${p:.2f}  "
                                                 f"(HWM=${hwm[t]:.2f}, "
                                                 f"drawdown={drop_from_hwm:.1%})",
                            **{f"{t}_MA200":     "—" for t in tickers},
                            **{f"{t}_MA60":      "—" for t in tickers},
                            **{f"{t}_Momentum":  "—" for t in tickers},
                            "Rank1 (50%)": "—", "Rank2 (30%)": "—", "Rank3 (20%)": "—",
                            "Cash Reserve ($)":  round(cash, 2),
                            "Deployed (%)":      "—",
                        })
                        shares[t] = 0.0
                        hwm[t]    = float("nan")

        # ── RECORD DAILY STATE ────────────────────────────────────────────────
        nav = sum(shares[t] * today[t] for t in tickers) + cash
        entry: dict = {"date": date, "equity_nav": nav}
        for bm in bm_prices.columns:
            entry[f"{bm}_nav"] = bm_shares[bm] * float(bm_prices[bm][date])
        for t in tickers:
            entry[f"{t}_value"] = shares[t] * today[t]
        rows.append(entry)

    daily    = pd.DataFrame(rows).set_index("date")
    alloc_df = pd.DataFrame(alloc_rows)
    return daily, alloc_df


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONS BACKTEST  (monthly ATM call debit spreads — unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T < 1e-6:
        return max(S - K, 0.0)
    sigma = max(sigma, 1e-4)
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return float(S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2))


def _spread_value(S: float, K1: float, K2: float, T: float, r: float, sigma: float) -> float:
    return _bs_call(S, K1, T, r, sigma) - _bs_call(S, K2, T, r, sigma)


def _rolling_vol(prices: pd.Series, window: int = 21) -> pd.Series:
    return prices.pct_change().rolling(window).std() * sqrt(TRADING_DAYS)


def run_options_backtest(
    port_prices: pd.DataFrame,
    initial_capital: float,
) -> pd.DataFrame:
    """Monthly-roll ATM call debit spread simulation (max $50 risk per ticker)."""
    tickers   = port_prices.columns.tolist()
    hist_vols = {t: _rolling_vol(port_prices[t]) for t in tickers}

    positions: dict[str, dict | None] = {t: None for t in tickers}
    last_roll:  dict[str, pd.Timestamp | None] = {t: None for t in tickers}
    opt_vals:   dict[str, float] = {t: 0.0 for t in tickers}

    cash_nav  = float(initial_capital)
    daily_rfr = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
    rows: list[dict] = []

    for date, row in port_prices.iterrows():
        for t in tickers:
            S     = float(row[t])
            iv    = float(hist_vols[t].get(date, np.nan))
            sigma = iv * OPT_IV_PREMIUM if not np.isnan(iv) else 0.60

            should_roll = (
                last_roll[t] is None
                or (date - last_roll[t]).days >= OPT_ROLL_DAYS
            )

            if should_roll:
                if positions[t] is not None:
                    p         = positions[t]
                    intrinsic = max(0.0, min(S - p["K1"], p["K2"] - p["K1"]))
                    cash_nav += intrinsic * 100.0 * p["contracts"]
                    opt_vals[t] = 0.0

                K1 = S
                K2 = S * (1.0 + OPT_SPREAD_WIDTH)
                T_new    = OPT_ROLL_DAYS / 365.0
                debit_ps = _spread_value(K1, K1, K2, T_new, RISK_FREE_RATE,
                                         max(sigma, 0.05))
                debit_ps  = max(debit_ps, 1e-4)
                contracts = MAX_RISK_PER_POS / (debit_ps * 100.0)

                cash_nav   -= contracts * debit_ps * 100.0
                opt_vals[t] = contracts * debit_ps * 100.0
                positions[t] = {"K1": K1, "K2": K2, "contracts": contracts}
                last_roll[t] = date
            else:
                p          = positions[t]
                days_el    = (date - last_roll[t]).days
                T_rem      = max((OPT_ROLL_DAYS - days_el) / 365.0, 1e-6)
                opt_vals[t] = _spread_value(
                    S, p["K1"], p["K2"], T_rem, RISK_FREE_RATE, max(sigma, 0.05)
                ) * 100.0 * p["contracts"]

        cash_nav *= (1 + daily_rfr)
        rows.append({
            "date":         date,
            "options_nav":  cash_nav + sum(opt_vals.values()),
            "options_cash": cash_nav,
            **{f"{t}_opt_val": opt_vals[t] for t in tickers},
        })

    return pd.DataFrame(rows).set_index("date")


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

def _cagr(start: float, end: float, n_days: int) -> float:
    return (end / start) ** (TRADING_DAYS / n_days) - 1

def _ann_vol(ret: pd.Series) -> float:
    return float(ret.std() * sqrt(TRADING_DAYS))

def _sharpe(ret: pd.Series) -> float:
    rf   = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
    ex   = ret - rf
    s    = ex.std()
    return float(ex.mean() / s * sqrt(TRADING_DAYS)) if s else float("nan")

def _sortino(ret: pd.Series) -> float:
    rf   = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
    ex   = ret - rf
    down = ex[ex < 0].std()
    return float(ex.mean() / down * sqrt(TRADING_DAYS)) if down else float("nan")

def _max_dd(nav: pd.Series) -> float:
    dd = (nav - nav.cummax()) / nav.cummax()
    return float(dd.min())

def _calmar(cagr_val: float, mdd: float) -> float:
    return cagr_val / abs(mdd) if mdd else float("nan")


def compute_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "equity":  "Dynamic Layering  (50/30/20 momentum)",
        "options": "Options Path      ($50 debit spreads)",
        "SPY":     "SPY  Benchmark",
        "QQQ":     "QQQ  Benchmark",
    }
    rows = []
    for col in [c for c in daily.columns if c.endswith("_nav")]:
        nav  = daily[col].dropna()
        ret  = nav.pct_change().dropna()
        n    = len(nav)
        key  = col.replace("_nav", "")
        label = labels.get(key, key.upper())

        cagr_v = _cagr(nav.iloc[0], nav.iloc[-1], n)
        mdd_v  = _max_dd(nav)
        rows.append({
            "Series":        label,
            "Start ($)":     f"{nav.iloc[0]:,.2f}",
            "End ($)":       f"{nav.iloc[-1]:,.2f}",
            "Total Return":  f"{nav.iloc[-1] / nav.iloc[0] - 1:+.1%}",
            "CAGR":          f"{cagr_v:+.1%}",
            "Ann. Vol":      f"{_ann_vol(ret):.1%}",
            "Sharpe":        f"{_sharpe(ret):.2f}",
            "Sortino":       f"{_sortino(ret):.2f}",
            "Max DD":        f"{mdd_v:.1%}",
            "Calmar":        f"{_calmar(cagr_v, mdd_v):.2f}",
        })
    return pd.DataFrame(rows).set_index("Series")


def compute_holdings(daily: pd.DataFrame) -> pd.DataFrame:
    eq_end = daily["equity_nav"].iloc[-1]
    rows   = []
    for t, cfg in PORTFOLIO.items():
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
            "Contrib.":     f"{(h.iloc[-1] - h.iloc[0]) / INITIAL_CAPITAL:+.1%}",
            "Port. Wt":     f"{h.iloc[-1] / eq_end:.1%}",
            "Sharpe":       f"{_sharpe(ret):.2f}",
            "Max DD":       f"{_max_dd(h):.1%}",
        })
    return pd.DataFrame(rows).set_index("Ticker")


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE REPORT
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "═" * 80
SEP2 = "─" * 80


def print_report(
    metrics: pd.DataFrame,
    holdings: pd.DataFrame,
    alloc_df: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    print()
    print(SEP)
    print("  DYNAMIC CAPITAL LAYERING PORTFOLIO  –  BACKTEST REPORT v3")
    print(f"  Period  : {START_DATE}  →  {END_DATE}")
    print(f"  Capital : ${INITIAL_CAPITAL:,.2f}  |  Weights : 50/30/20  |  "
          f"Stop : {TRAIL_STOP_PCT:.0%} trailing  |  MA filter : {MA_LONG}-day")
    print(SEP)

    print("\n▸ PERFORMANCE COMPARISON  (Dynamic Layering vs Options vs Benchmarks)")
    print(SEP2)
    print(metrics.T.to_string())

    print("\n▸ QUARTERLY ALLOCATION LOG  (rebalances + trailing-stop events)")
    print(SEP2)
    rebal_only = alloc_df[alloc_df["Event"] == "REBALANCE"].copy()
    if not rebal_only.empty:
        cols_to_show = [
            "Date", "NAV ($)", "Eligible (>MA200)",
            "Rank1 (50%)", "Rank2 (30%)", "Rank3 (20%)",
            "Cash Reserve ($)", "Deployed (%)",
        ]
        # Add momentum cols for tickers present
        for t in PORTFOLIO:
            mc = f"{t}_Momentum"
            if mc in rebal_only.columns:
                cols_to_show.insert(3, mc)
        existing = [c for c in cols_to_show if c in rebal_only.columns]
        print(rebal_only[existing].to_string(index=False))

    stop_events = alloc_df[alloc_df["Event"].str.startswith("TRAILING")].copy()
    if not stop_events.empty:
        print(f"\n  Trailing-stop events: {len(stop_events)}")
        for _, ev in stop_events.iterrows():
            print(f"  {ev['Date']}  {ev['Eligible (>MA200)']}")

    print("\n▸ END-STATE HOLDING ATTRIBUTION  (equity path)")
    print(SEP2)
    print(holdings.to_string())

    # Drawdown detail
    eq_nav  = daily["equity_nav"]
    opt_nav = daily["options_nav"]
    dd_eq   = (eq_nav - eq_nav.cummax()) / eq_nav.cummax()
    dd_opt  = (opt_nav - opt_nav.cummax()) / opt_nav.cummax()

    print("\n▸ DRAWDOWN DETAIL")
    print(SEP2)
    print(f"  Dynamic Layering  worst DD : {dd_eq.min():.1%}  on {dd_eq.idxmin().date()}")
    print(f"  Options path      worst DD : {dd_opt.min():.1%}  on {dd_opt.idxmin().date()}")
    print(f"  Days equity below –20 %    : {(dd_eq < -0.20).sum()}")

    print()
    print(SEP)
    print("  Output files written:")
    print("    backtest_daily.csv       – daily equity + options NAV + benchmarks")
    print("    backtest_metrics.csv     – performance metrics for all series")
    print("    backtest_holdings.csv    – per-holding equity attribution")
    print("    backtest_allocations.csv – quarterly allocation decisions")
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# GIT AUTOMATION
# ─────────────────────────────────────────────────────────────────────────────

def git_push(commit_msg: str) -> bool:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        print(f"\n[GIT] Active branch : {branch}")
    except Exception as exc:
        print(f"[GIT] Could not determine branch: {exc}")
        branch = "HEAD"

    ok = True
    for cmd, label in [
        (["git", "add", "."],                         "git add ."),
        (["git", "commit", "-m", commit_msg],         "git commit"),
        (["git", "push", "-u", "origin", branch],     f"git push → {branch}"),
    ]:
        res = subprocess.run(cmd, capture_output=True, text=True)
        symbol = "✓" if res.returncode == 0 else "✗"
        print(f"  {symbol}  {label}")
        for line in (res.stdout + res.stderr).strip().splitlines()[:5]:
            print(f"       {line}")
        if res.returncode != 0:
            if "nothing to commit" in res.stdout + res.stderr:
                print("       (nothing to commit — files already up to date)")
            else:
                ok = False
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("━" * 80)
    print("  SIMULATION ENGINE  v3  –  Dynamic Capital Layering Portfolio")
    print(f"  Framework : 200-MA filter → 63-day momentum rank → 50/30/20 weights")
    print(f"  Defense   : {TRAIL_STOP_PCT:.0%} trailing stop per position")
    print(f"  Capital   : ${INITIAL_CAPITAL:,.2f}  |  Period : {START_DATE} → {END_DATE}")
    print("━" * 80)

    # ── 1. Download data ──────────────────────────────────────────────────────
    all_tickers = list(PORTFOLIO.keys()) + BENCHMARKS
    print(f"\n[1/5] Downloading price data …")
    all_prices = fetch_prices(all_tickers, START_DATE, END_DATE)

    port_prices = all_prices[[t for t in PORTFOLIO if t in all_prices.columns]].copy()
    bm_prices   = all_prices[[b for b in BENCHMARKS if b in all_prices.columns]].copy()

    coverage = port_prices.notna().mean()
    sparse   = coverage[coverage < 0.80].index.tolist()
    if sparse:
        print(f"  WARNING  Dropping sparse tickers: {sparse}")
        port_prices = port_prices.drop(columns=sparse)

    print(
        f"  ✓  {len(port_prices.columns)} portfolio tickers | "
        f"{len(bm_prices.columns)} benchmarks | "
        f"{len(port_prices)} trading days\n"
    )

    # ── 2. Dynamic Capital Layering backtest ──────────────────────────────────
    print("[2/5] Running Dynamic Capital Layering backtest …")
    print(f"       MA filter : {MA_LONG}-day  |  Momentum window : {LOOKBACK_DAYS}-day  "
          f"|  Stop : {TRAIL_STOP_PCT:.0%} trailing")
    eq_daily, alloc_df = run_dynamic_backtest(port_prices, bm_prices, INITIAL_CAPITAL)
    rebal_count = len(alloc_df[alloc_df["Event"] == "REBALANCE"])
    stop_count  = len(alloc_df[alloc_df["Event"].str.startswith("TRAILING", na=False)])
    print(f"  ✓  {rebal_count} quarterly rebalances | {stop_count} trailing-stop triggers\n")

    # ── 3. Options backtest ───────────────────────────────────────────────────
    print("[3/5] Running options backtest (monthly ATM call spreads, $50 max risk) …")
    opts_daily = run_options_backtest(port_prices, INITIAL_CAPITAL)
    print("  ✓  Done\n")

    # ── 4. Metrics & CSV output ───────────────────────────────────────────────
    print("[4/5] Computing metrics and writing CSVs …")
    opt_cols = ["options_nav", "options_cash"] + \
               [c for c in opts_daily.columns if c.endswith("_opt_val")]
    daily    = eq_daily.join(opts_daily[opt_cols], how="left")

    metrics  = compute_metrics(daily)
    holdings = compute_holdings(daily)

    daily.reset_index().rename(columns={"date": "Date"}).to_csv(
        "backtest_daily.csv", index=False, float_format="%.4f"
    )
    metrics.to_csv("backtest_metrics.csv")
    holdings.to_csv("backtest_holdings.csv")
    alloc_df.to_csv("backtest_allocations.csv", index=False)
    print("  ✓  All 4 CSVs written\n")

    print_report(metrics, holdings, alloc_df, daily)

    # ── 5. Git commit & push ──────────────────────────────────────────────────
    print("[5/5] Committing and pushing to GitHub …")
    ok = git_push(GIT_COMMIT_MSG)

    # ── Final success log ─────────────────────────────────────────────────────
    eq_end  = daily["equity_nav"].iloc[-1]
    opt_end = daily["options_nav"].iloc[-1]
    spy_end = daily["SPY_nav"].iloc[-1] if "SPY_nav" in daily.columns else float("nan")

    print()
    print("━" * 80)
    print("  ✅  SUCCESS" if ok else "  ⚠️   PARTIAL SUCCESS (git step had warnings)")
    print()
    print(f"  📊  Period            : {START_DATE}  →  {END_DATE}")
    print(f"  💵  Starting capital  : ${INITIAL_CAPITAL:,.2f}")
    print(f"  📈  Dynamic Layering  : ${eq_end:,.2f}  "
          f"({eq_end / INITIAL_CAPITAL - 1:+.1%} total return)")
    print(f"  📉  Options path      : ${opt_end:,.2f}  "
          f"({opt_end / INITIAL_CAPITAL - 1:+.1%} total return)")
    print(f"  📊  SPY benchmark     : ${spy_end:,.2f}  "
          f"({spy_end / INITIAL_CAPITAL - 1:+.1%} total return)")
    print()
    print(f"  Rebalances : {rebal_count}  |  Stop triggers : {stop_count}")
    print()
    print("  Refresh your repo to view:")
    print("    backtest_daily.csv  |  backtest_metrics.csv  |  "
          "backtest_holdings.csv  |  backtest_allocations.csv")
    print("━" * 80)
    print()


if __name__ == "__main__":
    main()
