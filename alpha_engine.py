"""
alpha_engine.py – AlphaEngine v1.0
Modular ticker evaluation engine for the Rocky3 portfolio framework.

Fetches fundamental quality metrics [Gross Margin, Revenue Growth,
Net Debt / EBITDA] via Yahoo Finance and runs them through the
`is_economically_elite` gatekeeper to produce a standardised JSON payload.

Usage
-----
    from alpha_engine import evaluateTicker
    result = evaluateTicker("MSFT")
    # {
    #   "ticker": "MSFT",
    #   "status": "ELITE",        # "ELITE" | "REJECTED" | "INSUFFICIENT_DATA"
    #   "score": 81.25,           # composite quality score 0–100
    #   "stop_loss_price": 371.84 # 20% trailing stop below current price
    # }

Constraint
----------
This file is intentionally standalone — it does NOT modify any existing
UI logic in index.html / splash.html or simulation_engine.py.
"""
from __future__ import annotations

import json
import warnings
from typing import Any

import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# ELITE GATE THRESHOLDS  (adjust here to tighten or loosen the filter)
# ─────────────────────────────────────────────────────────────────────────────
GROSS_MARGIN_FLOOR: float = 0.40        # ≥ 40 %  — capital-light moat proxy
REVENUE_GROWTH_FLOOR: float = 0.10     # ≥ 10 % YoY — compounding trajectory
NET_DEBT_EBITDA_CEILING: float = 3.0   # ≤ 3×   — controlled leverage

# Mirrors TRAIL_STOP_PCT from simulation_engine.py so stop calculations align
_TRAIL_STOP_PCT: float = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# DATA PROVIDER  (Yahoo Finance via yfinance)
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_fundamentals(ticker: str) -> dict[str, float | None]:
    """
    Pull quality metrics for `ticker` from Yahoo Finance.

    Returns a dict with four keys (any may be None if data is unavailable):
      gross_margin      – trailing gross profit / revenue  (0–1 float)
      revenue_growth    – most-recent YoY revenue growth   (decimal)
      net_debt_ebitda   – Net Debt / EBITDA                (negative = net-cash)
      current_price     – latest closing price
    """
    info: dict[str, Any] = {}
    ticker_obj = yf.Ticker(ticker)

    try:
        info = ticker_obj.info or {}
    except Exception:
        pass

    # ── Gross Margin ──────────────────────────────────────────────────────────
    gross_margin: float | None = info.get("grossMargins")

    if gross_margin is None:
        try:
            fin = ticker_obj.financials
            if fin is not None and not fin.empty:
                rev_key = next((k for k in ("Total Revenue",) if k in fin.index), None)
                gp_key  = next((k for k in ("Gross Profit",)  if k in fin.index), None)
                if rev_key and gp_key:
                    rev = float(fin.loc[rev_key].iloc[0])
                    gp  = float(fin.loc[gp_key].iloc[0])
                    if rev != 0:
                        gross_margin = gp / rev
        except Exception:
            gross_margin = None

    # ── Revenue Growth ────────────────────────────────────────────────────────
    revenue_growth: float | None = info.get("revenueGrowth")

    if revenue_growth is None:
        try:
            fin = ticker_obj.financials
            if fin is not None and not fin.empty and "Total Revenue" in fin.index:
                rev_row = fin.loc["Total Revenue"]
                if len(rev_row) >= 2:
                    r0 = float(rev_row.iloc[0])   # most recent period
                    r1 = float(rev_row.iloc[1])   # prior period
                    if r1 != 0:
                        revenue_growth = (r0 - r1) / abs(r1)
        except Exception:
            revenue_growth = None

    # ── Net Debt / EBITDA ─────────────────────────────────────────────────────
    net_debt_ebitda: float | None = None
    try:
        bs  = ticker_obj.balance_sheet
        fin = ticker_obj.financials

        total_debt: float | None = None
        cash_equiv: float | None = None
        ebitda: float | None     = info.get("ebitda")

        # Balance sheet: total debt
        if bs is not None and not bs.empty:
            for debt_key in (
                "Total Debt",
                "Long Term Debt And Capital Lease Obligation",
                "Long Term Debt",
                "Short Long Term Debt",
            ):
                if debt_key in bs.index:
                    total_debt = float(bs.loc[debt_key].iloc[0])
                    break

        # Balance sheet: cash & equivalents
        if bs is not None and not bs.empty:
            for cash_key in (
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Short Term Investments",
            ):
                if cash_key in bs.index:
                    cash_equiv = float(bs.loc[cash_key].iloc[0])
                    break

        # Income statement: build EBITDA when not in info
        if ebitda is None and fin is not None and not fin.empty:
            ebit: float | None = None
            da:   float | None = None
            for ebit_key in ("EBIT", "Operating Income"):
                if ebit_key in fin.index:
                    ebit = float(fin.loc[ebit_key].iloc[0])
                    break
            for da_key in (
                "Reconciled Depreciation",
                "Depreciation And Amortization",
                "Depreciation",
                "Depreciation Amortization Depletion",
            ):
                if da_key in fin.index:
                    da = float(fin.loc[da_key].iloc[0])
                    break
            if ebit is not None and da is not None:
                ebitda = ebit + abs(da)

        if total_debt is not None and cash_equiv is not None and ebitda and ebitda != 0:
            net_debt = total_debt - cash_equiv
            net_debt_ebitda = net_debt / abs(float(ebitda))

    except Exception:
        net_debt_ebitda = None

    # ── Current Price ─────────────────────────────────────────────────────────
    current_price: float | None = (
        info.get("currentPrice") or info.get("regularMarketPrice")
    )
    if current_price is None:
        try:
            hist = ticker_obj.history(period="2d")
            if not hist.empty:
                current_price = float(hist["Close"].iloc[-1])
        except Exception:
            current_price = None

    return {
        "gross_margin":    gross_margin,
        "revenue_growth":  revenue_growth,
        "net_debt_ebitda": net_debt_ebitda,
        "current_price":   current_price,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GATEKEEPER
# ─────────────────────────────────────────────────────────────────────────────
def is_economically_elite(
    gross_margin: float | None,
    revenue_growth: float | None,
    net_debt_ebitda: float | None,
) -> bool:
    """
    Return True when ALL three economic quality gates pass:
      1. Gross Margin       ≥ GROSS_MARGIN_FLOOR     (40 %)
      2. Revenue Growth     ≥ REVENUE_GROWTH_FLOOR   (10 % YoY)
      3. Net Debt / EBITDA  ≤ NET_DEBT_EBITDA_CEILING (3×)
         Net-cash companies (ratio ≤ 0) automatically clear the leverage gate.

    A None metric is treated as a fail — unknown data is not elite.
    """
    if gross_margin is None or revenue_growth is None or net_debt_ebitda is None:
        return False
    return (
        gross_margin >= GROSS_MARGIN_FLOOR
        and revenue_growth >= REVENUE_GROWTH_FLOOR
        and net_debt_ebitda <= NET_DEBT_EBITDA_CEILING
    )


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE QUALITY SCORE  (0–100)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_score(
    gross_margin: float | None,
    revenue_growth: float | None,
    net_debt_ebitda: float | None,
) -> float:
    """
    Three equally weighted sub-scores, each worth up to 33.3 points.

    Gross Margin sub-score
        0 %   → 0 pts  |  40 % → 16.7 pts  |  80 %+ → 33.3 pts  (capped)

    Revenue Growth sub-score
        0 %   → 0 pts  |  15 % → 16.7 pts  |  30 %+ → 33.3 pts  (capped)

    Leverage sub-score (inverted Net Debt / EBITDA)
        ≤ 0 (net cash) → 33.3 pts
        Linear from 33.3 (ratio=0) down to 0 (ratio≥6)
    """
    THIRD = 100.0 / 3.0

    margin_score = (
        min(float(gross_margin) / 0.80, 1.0) * THIRD
        if gross_margin is not None
        else 0.0
    )

    growth_score = (
        min(max(float(revenue_growth), 0.0) / 0.30, 1.0) * THIRD
        if revenue_growth is not None
        else 0.0
    )

    if net_debt_ebitda is None:
        leverage_score = 0.0
    elif float(net_debt_ebitda) <= 0:
        leverage_score = THIRD
    else:
        leverage_score = max(1.0 - float(net_debt_ebitda) / 6.0, 0.0) * THIRD

    return round(margin_score + growth_score + leverage_score, 2)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def evaluateTicker(ticker: str) -> dict:
    """
    Evaluate a single equity ticker through the AlphaEngine quality gate.

    Parameters
    ----------
    ticker : str
        Equity symbol, e.g. ``"AAPL"``.  Case-insensitive.

    Returns
    -------
    dict
        .. code-block:: json

            {
                "ticker":           "AAPL",
                "status":           "ELITE",
                "score":            78.54,
                "stop_loss_price":  154.32
            }

        status values:
          * ``"ELITE"``             – passes all three quality gates
          * ``"REJECTED"``          – fails one or more quality gates
          * ``"INSUFFICIENT_DATA"`` – no fundamental data available for any metric

        stop_loss_price is ``null`` when the current price cannot be fetched.
    """
    ticker = ticker.strip().upper()
    fundamentals = _fetch_fundamentals(ticker)

    gm  = fundamentals["gross_margin"]
    rg  = fundamentals["revenue_growth"]
    nde = fundamentals["net_debt_ebitda"]
    px  = fundamentals["current_price"]

    has_any_data = any(v is not None for v in (gm, rg, nde))

    if not has_any_data:
        status = "INSUFFICIENT_DATA"
    elif is_economically_elite(gm, rg, nde):
        status = "ELITE"
    else:
        status = "REJECTED"

    score = _compute_score(gm, rg, nde)

    stop_loss_price = (
        round(float(px) * (1.0 - _TRAIL_STOP_PCT), 2)
        if px is not None
        else None
    )

    return {
        "ticker":          ticker,
        "status":          status,
        "score":           score,
        "stop_loss_price": stop_loss_price,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI HELPER  (python alpha_engine.py AAPL MSFT TSLA)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    symbols = sys.argv[1:] or ["AAPL", "MSFT"]
    for sym in symbols:
        result = evaluateTicker(sym)
        print(json.dumps(result, indent=2))
