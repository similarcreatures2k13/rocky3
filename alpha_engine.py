"""
alpha_engine.py – Rocky3 Fundamental Gatekeeper
Evaluates tickers based on Gross Margin, Revenue Growth, and Net Debt/EBITDA.
"""
import yfinance as yf
import json
import warnings

warnings.filterwarnings("ignore")

# --- ELITE GATE THRESHOLDS ---
GROSS_MARGIN_FLOOR = 0.40        # ≥ 40% (Capital-light moat)
REVENUE_GROWTH_FLOOR = 0.10      # ≥ 10% YoY (Compounding trajectory)
NET_DEBT_EBITDA_CEILING = 3.0    # ≤ 3.0x (Controlled leverage)
_TRAIL_STOP_PCT = 0.20           # 20% stop loss

def evaluateTicker(ticker: str) -> dict:
    ticker = ticker.strip().upper()
    t = yf.Ticker(ticker)
    
    try:
        info = t.info or {}
    except Exception:
        info = {}

    # 1. Fetch Core Metrics
    gm = info.get("grossMargins") or 0
    rg = info.get("revenueGrowth") or 0
    px = info.get("currentPrice") or info.get("regularMarketPrice")
    
    # 2. Calculate TRUE Net Debt / EBITDA (Debt minus Cash on hand)
    total_debt = info.get("totalDebt") or 0
    cash = info.get("totalCash") or 0
    ebitda = info.get("ebitda") or 1  # Avoid division by zero
    net_debt = max(0, total_debt - cash)
    nde = net_debt / ebitda if ebitda > 0 else 99.0
    
    # 3. Gatekeeper Logic
    is_elite = (
        gm >= GROSS_MARGIN_FLOOR and 
        rg >= REVENUE_GROWTH_FLOOR and 
        nde <= NET_DEBT_EBITDA_CEILING
    )
    status = "ELITE" if is_elite else "REJECTED"
    
    # 4. Composite Score (0-100)
    margin_score = min(gm / 0.80, 1.0) * 33.3
    growth_score = min(max(rg, 0) / 0.30, 1.0) * 33.3
    leverage_score = max(0, (6 - nde) / 6) * 33.3
    score = round(margin_score + growth_score + leverage_score, 2)
    
    # 5. Risk Parameters
    stop_loss = round(px * (1 - _TRAIL_STOP_PCT), 2) if px else None
    
    return {
        "ticker": ticker,
        "status": status,
        "score": score,
        "stop_loss_price": stop_loss,
        "metrics": {
            "price": round(px, 2) if px else None,
            "gross_margin": round(gm * 100, 1),
            "rev_growth": round(rg * 100, 1),
            "net_debt_ebitda": round(nde, 2)
        }
    }

if __name__ == "__main__":
    # Quick CLI test
    result = evaluateTicker("ANET")
    print(json.dumps(result, indent=2))
