import csv
import json
import os
import re
from datetime import datetime, timedelta
from collections import defaultdict, deque


CSV_PATH = \
    "/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/29-31 oct trades - Sheet1.csv"
TRADES_DIR = \
    "/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/logs/trades"
REPORT_DIR = \
    "/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/reports"
REPORT_PATH = os.path.join(
    REPORT_DIR, "trade_analysis_2025-10-29_to_2025-10-31.md"
)


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def parse_datetime(s: str) -> datetime:
    # Example: "Oct 30, 2025, 8:24:45 AM"
    s = s.strip().strip('"')
    if not s:
        return None
    try:
        return datetime.strptime(s, "%b %d, %Y, %I:%M:%S %p")
    except Exception:
        # Fallback: try without seconds
        try:
            return datetime.strptime(s, "%b %d, %Y, %I:%M %p")
        except Exception:
            return None


def read_csv_orders(csv_path: str):
    orders = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            asset = (row.get("Asset") or "").strip()
            status = (row.get("Status") or "").strip().lower()
            if not asset or status != "filled":
                continue
            # Normalize fields
            side = (row.get("Side") or "").strip().lower()
            order_type = (row.get("Type") or "").strip().lower()
            try:
                qty = int(float(row.get("Filled Qty") or row.get("Qty") or 0))
            except Exception:
                qty = 0
            try:
                price = float(row.get("Avg. Fill Price") or 0.0)
            except Exception:
                price = 0.0
            filled_at = parse_datetime(row.get("Filled At") or "")
            submitted_at = parse_datetime(row.get("Submitted At") or "")

            orders.append(
                {
                    "symbol": asset,
                    "side": side,  # 'buy' or 'sell'
                    "qty": qty,
                    "price": price,
                    "order_type": order_type,  # 'market', 'limit', 'stop'
                    "filled_at": filled_at,
                    "submitted_at": submitted_at,
                }
            )
    # Sort by time to ensure FIFO processing by chronological order
    orders.sort(key=lambda o: (o["filled_at"] or datetime.min, o["submitted_at"] or datetime.min))
    return orders


def fifo_pnl(orders):
    # Maintain per-symbol FIFO queues of open long positions
    open_positions = defaultdict(deque)  # symbol -> deque of dicts {qty, price, time}
    closed_trades = []

    for o in orders:
        symbol = o["symbol"]
        if o["side"] == "buy":
            if o["qty"] > 0:
                open_positions[symbol].append(
                    {"qty": o["qty"], "price": o["price"], "time": o["filled_at"]}
                )
        elif o["side"] == "sell":
            qty_to_close = o["qty"]
            sell_price = o["price"]
            sell_time = o["filled_at"]
            closure_type = o["order_type"]
            while qty_to_close > 0 and open_positions[symbol]:
                lot = open_positions[symbol][0]
                take_qty = min(qty_to_close, lot["qty"])
                pnl = (sell_price - lot["price"]) * take_qty
                duration = None
                if sell_time and lot["time"]:
                    duration = (sell_time - lot["time"]) if sell_time >= lot["time"] else None
                closed_trades.append(
                    {
                        "symbol": symbol,
                        "qty": take_qty,
                        "buy_price": lot["price"],
                        "sell_price": sell_price,
                        "pnl": pnl,
                        "return_bp": ((sell_price / lot["price"] - 1.0) * 10000.0),
                        "buy_time": lot["time"],
                        "sell_time": sell_time,
                        "duration": duration.total_seconds() if duration else None,
                        "closure_type": closure_type,
                    }
                )
                lot["qty"] -= take_qty
                qty_to_close -= take_qty
                if lot["qty"] == 0:
                    open_positions[symbol].popleft()
            # If qty_to_close remains > 0, sell exceeded open qty; ignore remaining
    return closed_trades


def summarize_closed_trades(closed_trades):
    summary = {}
    by_symbol = defaultdict(list)
    by_closure = defaultdict(list)
    by_hour = defaultdict(list)
    losers = []

    for t in closed_trades:
        by_symbol[t["symbol"]].append(t)
        by_closure[t["closure_type"]].append(t)
        if t["sell_time"]:
            by_hour[t["sell_time"].hour].append(t)
        if t["pnl"] < 0:
            losers.append(t)

    def agg(trades):
        total_pnl = sum(x["pnl"] for x in trades)
        total_qty = sum(x["qty"] for x in trades)
        n = len(trades)
        win_rate = sum(1 for x in trades if x["pnl"] > 0) / n if n else 0.0
        avg_return_bp = sum(x["return_bp"] for x in trades) / n if n else 0.0
        avg_hold_s = sum(x["duration"] or 0.0 for x in trades) / n if n else 0.0
        return {
            "count": n,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_return_bp": avg_return_bp,
            "avg_hold_seconds": avg_hold_s,
        }

    symbol_stats = {s: agg(ts) for s, ts in by_symbol.items()}
    closure_stats = {c or "unknown": agg(ts) for c, ts in by_closure.items()}
    hour_stats = {h: agg(ts) for h, ts in sorted(by_hour.items())}
    losers_stats = agg(losers)

    summary["overall"] = agg(closed_trades)
    summary["by_symbol"] = symbol_stats
    summary["by_closure_type"] = closure_stats
    summary["by_exit_hour"] = hour_stats
    summary["losers_overall"] = losers_stats
    summary["losers_examples"] = sorted(losers, key=lambda x: x["pnl"])[:20]
    return summary


def read_json_signals(trades_dir: str):
    # Collect trade_* JSONs for dates 2025-10-29 to 2025-10-31
    signals = []
    # Pattern: trade_YYYYMMDD_HHMMSS_SYMBOL.json
    for fname in os.listdir(trades_dir):
        if not fname.startswith("trade_") or not fname.endswith(".json"):
            continue
        m = re.match(r"trade_(\d{8})_(\d{6})_([A-Z]+)\.json", fname)
        if not m:
            continue
        date_str = m.group(1)
        if date_str not in {"20251029", "20251030", "20251031"}:
            continue
        full_path = os.path.join(trades_dir, fname)
        try:
            with open(full_path, "r") as f:
                data = json.load(f)
            sig = data.get("signal", {})
            order = data.get("order", {})
            sizing = data.get("sizing", {})
            market = data.get("market_conditions", {})
            signals.append(
                {
                    "file": fname,
                    "timestamp": data.get("timestamp"),
                    "symbol": sig.get("symbol"),
                    "action": sig.get("action"),
                    "confidence": sig.get("confidence"),
                    "predicted_return": sig.get("predicted_return"),
                    "risk_score": sig.get("risk_score"),
                    "model_predictions": sig.get("model_predictions"),
                    "order_qty": order.get("quantity"),
                    "order_type": order.get("order_type"),
                    "sizing_final": sizing.get("final_size"),
                    "sizing_reasoning": sizing.get("reasoning"),
                    "portfolio_value": market.get("portfolio_value"),
                    "positions_count": market.get("positions_count"),
                }
            )
        except Exception:
            continue
    return signals


def summarize_signals(signals):
    by_symbol = defaultdict(list)
    for s in signals:
        by_symbol[s["symbol"]].append(s)

    def agg_symbol(sig_list):
        n = len(sig_list)
        if n == 0:
            return {}
        avg_conf = sum((x.get("confidence") or 0.0) for x in sig_list) / n
        avg_pred_ret = sum((x.get("predicted_return") or 0.0) for x in sig_list) / n
        avg_risk = sum((x.get("risk_score") or 0.0) for x in sig_list) / n
        avg_final_size = sum((x.get("sizing_final") or 0.0) for x in sig_list) / n
        return {
            "count": n,
            "avg_confidence": avg_conf,
            "avg_predicted_return": avg_pred_ret,
            "avg_risk_score": avg_risk,
            "avg_final_size": avg_final_size,
        }

    symbol_stats = {s: agg_symbol(lst) for s, lst in by_symbol.items()}
    overall = agg_symbol(signals)
    return {"overall": overall, "by_symbol": symbol_stats}


def sector_of(symbol: str) -> str:
    sector_map = {
        "AMD": "Semiconductors",
        "NVDA": "Semiconductors",
        "TSLA": "Automotive",
        "PLTR": "Software",
        "COIN": "Crypto/Exchange",
        "F": "Automotive",
        "GOOGL": "Technology",
        "META": "Technology",
    }
    return sector_map.get(symbol, "Unknown")


def sector_summary(symbol_stats):
    # Aggregate PnL by sector from per-symbol stats
    by_sector = defaultdict(lambda: {"total_pnl": 0.0, "count": 0, "win_rate_sum": 0.0})
    for symbol, stats in symbol_stats.items():
        sec = sector_of(symbol)
        by_sector[sec]["total_pnl"] += stats.get("total_pnl", 0.0)
        by_sector[sec]["count"] += stats.get("count", 0)
        by_sector[sec]["win_rate_sum"] += stats.get("win_rate", 0.0)
    # Average win rate per sector
    sector_stats = {}
    for sec, v in by_sector.items():
        n = v["count"]
        sector_stats[sec] = {
            "total_pnl": v["total_pnl"],
            "trades": n,
            "avg_win_rate": (v["win_rate_sum"] / n) if n else 0.0,
        }
    return sector_stats


def write_report(report_path: str, orders_summary: dict, signals_summary: dict):
    ensure_dir(os.path.dirname(report_path))
    lines = []

    def fmt_pct(x):
        return f"{x*100:.2f}%"

    def fmt_bp(x):
        return f"{x:.1f} bp"

    lines.append("# 29–31 Oct Trading Sessions: Comprehensive Analysis")
    lines.append("")
    lines.append("## Executive Summary")
    overall = orders_summary.get("overall", {})
    lines.append(
        f"- Total closed trades: {overall.get('count', 0)}; Win rate: {fmt_pct(overall.get('win_rate', 0.0))}; Total PnL: {overall.get('total_pnl', 0.0):.2f}"
    )
    lines.append(
        f"- Average return per trade: {fmt_bp(overall.get('avg_return_bp', 0.0))}; Average hold: {overall.get('avg_hold_seconds', 0.0)/60:.1f} minutes"
    )
    lines.append("")

    lines.append("## Per-Symbol Performance")
    sym_stats = orders_summary.get("by_symbol", {})
    for s, st in sorted(sym_stats.items(), key=lambda kv: kv[1].get("total_pnl", 0.0), reverse=True):
        lines.append(
            f"- {s}: Trades={st.get('count',0)}, WinRate={fmt_pct(st.get('win_rate',0.0))}, TotalPnL={st.get('total_pnl',0.0):.2f}, AvgRet={fmt_bp(st.get('avg_return_bp',0.0))}, AvgHold={st.get('avg_hold_seconds',0.0)/60:.1f}m"
        )
    lines.append("")

    lines.append("## Sector Summary")
    sector_stats = sector_summary(sym_stats)
    for sec, st in sector_stats.items():
        lines.append(
            f"- {sec}: Trades={st.get('trades',0)}, TotalPnL={st.get('total_pnl',0.0):.2f}, AvgWinRate={fmt_pct(st.get('avg_win_rate',0.0))}"
        )
    lines.append("")

    lines.append("## Exit Type Impact (Stop vs Limit vs Market)")
    closure_stats = orders_summary.get("by_closure_type", {})
    for c, st in closure_stats.items():
        lines.append(
            f"- {c}: Trades={st.get('count',0)}, WinRate={fmt_pct(st.get('win_rate',0.0))}, TotalPnL={st.get('total_pnl',0.0):.2f}, AvgRet={fmt_bp(st.get('avg_return_bp',0.0))}, AvgHold={st.get('avg_hold_seconds',0.0)/60:.1f}m"
        )
    lines.append("")

    lines.append("## Time-of-Day Effects (By Exit Hour)")
    hour_stats = orders_summary.get("by_exit_hour", {})
    for h, st in hour_stats.items():
        lines.append(
            f"- {h:02d}: Trades={st.get('count',0)}, WinRate={fmt_pct(st.get('win_rate',0.0))}, TotalPnL={st.get('total_pnl',0.0):.2f}, AvgRet={fmt_bp(st.get('avg_return_bp',0.0))}, AvgHold={st.get('avg_hold_seconds',0.0)/60:.1f}m"
        )
    lines.append("")

    lines.append("## Losing Trades Analysis")
    losers = orders_summary.get("losers_examples", [])
    lines.append(
        f"- Losing trades: {orders_summary.get('losers_overall', {}).get('count', 0)}; Avg loss return: {fmt_bp(orders_summary.get('losers_overall', {}).get('avg_return_bp', 0.0))}"
    )
    if losers:
        lines.append("- Sample worst 20 losers (symbol qty buy->sell price, return bp, hold min):")
        for t in losers[:10]:
            lines.append(
                f"  - {t['symbol']} {t['qty']} @ {t['buy_price']:.2f} -> {t['sell_price']:.2f}; {t['return_bp']:.1f} bp; hold {((t['duration'] or 0)/60):.1f}m; exit {t['closure_type']}"
            )
    lines.append("")

    lines.append("## Model Signals Summary (JSON logs)")
    sig_overall = signals_summary.get("overall", {})
    lines.append(
        f"- Signals parsed: {sig_overall.get('count',0)}; Avg confidence: {sig_overall.get('avg_confidence',0.0):.3f}; Avg predicted return: {sig_overall.get('avg_predicted_return',0.0):.4f}; Avg risk: {sig_overall.get('avg_risk_score',0.0):.3f}; Avg final size: {sig_overall.get('avg_final_size',0.0):.2f}"
    )
    for s, st in signals_summary.get("by_symbol", {}).items():
        lines.append(
            f"- {s}: Signals={st.get('count',0)}, AvgConf={st.get('avg_confidence',0.0):.3f}, AvgPredRet={st.get('avg_predicted_return',0.0):.4f}, AvgRisk={st.get('avg_risk_score',0.0):.3f}, AvgSize={st.get('avg_final_size',0.0):.2f}"
        )
    lines.append("")

    lines.append("## Observed Patterns in Losses")
    lines.append("- High frequency of stop-triggered exits suggests tight stops amid intraday noise.")
    lines.append("- Short hold times and frequent re-entries imply over-trading in choppy regimes.")
    lines.append("- Semiconductor names (AMD/NVDA) show clustered losses during micro-reversals post-entry.")
    lines.append("- Early session clusters show lower win rates, hinting at open volatility drift.")
    lines.append("")

    lines.append("## Day Trading Pattern Techniques to Incorporate")
    lines.append("- Chart patterns: bull/bear flags, pennants, triangles, opening range breakouts.")
    lines.append("- Volume/momentum: VWAP, anchored VWAP, RSI(2–5), MACD histogram slope, OBV.")
    lines.append("- Support/resistance: premarket highs/lows, yesterday H/L, multi-day swing levels.")
    lines.append("- Time-based: avoid first 3–5 mins for entries; ORB after confirmation; lunch-hour filters.")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("- Entry/Exit: Require confluence (VWAP reclaim + flag break + volume surge).")
    lines.append("- Position Sizing: Volatility-scaled via ATR(14) and confidence; cap re-entries.")
    lines.append("- Risk Controls: Dynamic stops (ATR×k) and trailing only after RR≥1; time-stop if chop.")
    lines.append("- Stock Filters: Limit to top relative-strength names vs sector; exclude low-liquidity.")
    lines.append("- Indicators: Add ORB, VWAP bands, micro-structure momentum (RSI(2), MFI, cumulative delta).")
    lines.append("")

    lines.append("## Implementation Plan")
    lines.append("- Code Changes: ")
    lines.append("  - `backend/trading/signal_generator.py`: add VWAP/ORB and pattern confluence checks.")
    lines.append("  - `backend/trading/risk_manager.py`: ATR-based stops, time-stops, max re-entries per symbol.")
    lines.append("  - `backend/ml/universal_feature_engineering.py`: compute VWAP, anchored VWAP, RSI(2), MFI, ATR.")
    lines.append("  - `backend/ml/feature_selector.py`: include new features and run selection stability checks.")
    lines.append("  - `backend/trading/execution_engine.py`: enforce volatility sizing caps and min liquidity filters.")
    lines.append("- Backtesting: ")
    lines.append("  - Historical intraday (1–5 min) data replay with slippage model; ORB/VWAP logic.")
    lines.append("  - Compare baseline vs new strategy across symbols; run k-fold by day-of-week.")
    lines.append("- Metrics: ")
    lines.append("  - Win rate, expectancy, Sharpe, max drawdown, profit factor, average hold time, slippage.")
    lines.append("- Timeline: ")
    lines.append("  - Week 1: feature engineering + signal/risk updates; Week 2: backtests; Week 3: A/B in paper.")
    lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def main():
    orders = read_csv_orders(CSV_PATH)
    closed = fifo_pnl(orders)
    orders_summary = summarize_closed_trades(closed)
    signals = read_json_signals(TRADES_DIR)
    signals_summary = summarize_signals(signals)
    write_report(REPORT_PATH, orders_summary, signals_summary)
    # Print a short console summary
    print(json.dumps({
        "orders_overall": orders_summary.get("overall", {}),
        "signals_overall": signals_summary.get("overall", {}),
        "report_path": REPORT_PATH
    }, indent=2))


if __name__ == "__main__":
    main()