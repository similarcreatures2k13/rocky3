"""
options_scanner.py – Rocky3 Asymmetric Options Engine
Scans for ELITE mid-caps and calculates OTM Call Debit Spread parameters.
"""
from alpha_engine import evaluateTicker
import yfinance as yf
import datetime as dt

# Universe of liquid mid/large-cap AI/Tech infrastructure tickers
UNIVERSE = ["ANET", "AVGO", "VRT", "AMD", "ARM", "MRVL", "CRDO"]

def scan_for_asymmetry(universe: list, risk_per_trade: float = 500.0):
    print(f"🔍 Scanning {len(universe)} tickers for ELITE status and options setups...\n")
    
    opportunities = []
    
    for ticker in universe:
        eval_data = evaluateTicker(ticker)
        
        # ONLY look at options for tickers that pass the fundamental gate
        if eval_data["status"] == "ELITE":
            print(f"✅ {ticker} is ELITE (Score: {eval_data['score']}). Checking momentum...")
            
            # Check 63-day momentum (Simple price change)
            hist = yf.Ticker(ticker).history(period="3mo")
            if len(hist) >= 63:
                mom_63 = (hist['Close'].iloc[-1] / hist['Close'].iloc[-63]) - 1
                
                if mom_63 > 0.05: # Only buy calls if it's already in an uptrend
                    px = eval_data["metrics"]["price"]
                    
                    # Calculate OTM Spread Parameters
                    # Buy Call 5% OTM, Sell Call 15% OTM (Width = 10%)
                    long_strike = round(px * 1.05, 0) 
                    short_strike = round(px * 1.15, 0)
                    
                    opportunities.append({
                        "ticker": ticker,
                        "score": eval_data["score"],
                        "momentum_63d": f"{mom_63:.1%}",
                        "price": px,
                        "spread_structure": f"Buy {long_strike}C / Sell {short_strike}C",
                        "max_risk": risk_per_trade,
                        "target_reward": risk_per_trade * 4 # 1:4 Risk/Reward minimum
                    })
            else:
                print(f"⚠️ {ticker} is ELITE but lacks 63-day history.")
        else:
            print(f"❌ {ticker} REJECTED (Score: {eval_data['score']}).")

    print("\n" + "═"*60)
    print("🎯 ASYMMETRIC OPTIONS TARGETS (Ready for IBKR execution)")
    print("═"*60)
    for opp in opportunities:
        print(f"\n> {opp['ticker']} | Score: {opp['score']} | 63d Mom: {opp['momentum_63d']}")
        print(f"  Structure: {opp['spread_structure']}")
        print(f"  Risk: ${opp['max_risk']} | Target Reward: ${opp['target_reward']}+")

if __name__ == "__main__":
    scan_for_asymmetry(UNIVERSE)
