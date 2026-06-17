import argparse
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from backtester.data import fetch_yfinance_data, fetch_earnings_weeks
from backtester.engine import BacktestEngine
from backtester.analysis import calculate_metrics, print_performance_summary, generate_plots
from strategies.wheel_strategy import WheelStrategy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Options Backtester CLI")
    
    parser.add_argument("--ticker", type=str, default="AAPL:2,MSFT:3", 
                        help="Underlying ticker config: e.g. 'AAPL:2,MSFT:3' or 'AAPL,MSFT' (default: 'AAPL:2,MSFT:3')")
    parser.add_argument("--start", type=str, default="2022-01-01", help="Start date YYYY-MM-DD (default: 2022-01-01)")
    parser.add_argument("--end", type=str, default="2023-12-31", help="End date YYYY-MM-DD (default: 2023-12-31)")
    parser.add_argument("--capital", type=float, default=300000.0, help="Initial portfolio capital (default: $300,000)")
    parser.add_argument("--allocation-tolerance", type=float, default=0.05, help="Wiggle room capital leverage (default: 0.05)")
    parser.add_argument("--delta-target", type=float, default=0.30, help="Target option delta (default: 0.30)")
    parser.add_argument("--avoid-calls-below-cost", action="store_true", help="Avoid covered calls below stock purchase basis")
    
    parser.add_argument("--commission", type=float, default=0.65, help="Option commission per contract (default: $0.65)")
    parser.add_argument("--slippage-pct", type=float, default=0.01, help="Slippage percentage of premium (default: 1%)")
    parser.add_argument("--slippage-flat", type=float, default=0.01, help="Flat slippage premium adjustment per share (default: $0.01)")

    args = parser.parse_args()

    # Parse tickers configuration
    tickers_config = {}
    try:
        for item in args.ticker.split(','):
            item = item.strip()
            if not item:
                continue
            if ':' in item:
                t, q = item.split(':')
                tickers_config[t.strip().upper()] = int(q.strip())
            else:
                tickers_config[item.strip().upper()] = None
    except Exception as e:
        logger.error(f"Failed to parse ticker config string '{args.ticker}'. Error: {e}")
        return

    num_tickers = len(tickers_config)
    capital_per_ticker = args.capital / num_tickers
    
    all_histories = {}
    all_tx_logs = []
    
    logger.info(f"Starting multi-ticker backtest for config: {tickers_config}")
    
    # 1. Run backtest for each ticker
    for ticker_symbol, qty in tickers_config.items():
        try:
            # Download stock data
            df = fetch_yfinance_data(
                ticker=ticker_symbol,
                start_date=args.start,
                end_date=args.end
            )
            
            # Fetch earnings weeks
            earnings_weeks = fetch_earnings_weeks(ticker_symbol)
            
            # Initialize engine and strategy
            engine = BacktestEngine(
                initial_capital=capital_per_ticker,
                commission_per_contract=args.commission,
                slippage_pct=args.slippage_pct,
                slippage_flat=args.slippage_flat,
                allocation_tolerance=args.allocation_tolerance
            )
            
            strategy = WheelStrategy(
                ticker=ticker_symbol,
                delta_target=args.delta_target,
                earnings_weeks=earnings_weeks,
                avoid_calls_below_cost=args.avoid_calls_below_cost,
                fixed_qty=qty
            )
            
            # Run simulation
            history_df = engine.run(df, strategy)
            all_histories[ticker_symbol] = history_df
            
            # Accumulate transactions
            for tx in engine.transaction_log:
                tx_copy = tx.copy()
                tx_copy['ticker'] = ticker_symbol
                all_tx_logs.append(tx_copy)
                
        except Exception as e:
            logger.error(f"Error backtesting {ticker_symbol}: {e}")
            import traceback
            traceback.print_exc()
            return

    # 2. Aggregate histories
    date_indices = [df.index for df in all_histories.values()]
    common_dates = date_indices[0]
    for idx in date_indices[1:]:
        common_dates = common_dates.intersection(idx)
        
    combined_history = pd.DataFrame(index=common_dates)
    combined_history['PortfolioValue'] = 0.0
    combined_history['Cash'] = 0.0
    combined_history['StockValue'] = 0.0
    combined_history['OptionsValue'] = 0.0
    combined_history['Rate'] = 0.0
    
    for ticker_symbol, hist in all_histories.items():
        aligned_hist = hist.reindex(common_dates).ffill().bfill()
        combined_history['PortfolioValue'] += aligned_hist['PortfolioValue']
        combined_history['Cash'] += aligned_hist['Cash']
        combined_history['StockValue'] += aligned_hist['StockValue']
        combined_history['OptionsValue'] += aligned_hist['OptionsValue']
        combined_history['Rate'] = aligned_hist['Rate']
        
    # Compute combined stock benchmark baseline
    combined_history['Benchmark'] = 0.0
    for ticker_symbol, hist in all_histories.items():
        aligned_hist = hist.reindex(common_dates).ffill().bfill()
        normalized_stock = (aligned_hist['Spot'] / aligned_hist['Spot'].iloc[0]) * capital_per_ticker
        combined_history['Benchmark'] += normalized_stock
        
    # Map Benchmark to Spot so generate_plots plotting functions work seamlessly
    combined_history['Spot'] = combined_history['Benchmark']
    
    # Sort combined transactions
    all_tx_logs.sort(key=lambda x: x['date'])

    # 3. Calculate and output performance metrics
    metrics = calculate_metrics(combined_history, all_tx_logs, args.capital)
    
    if metrics:
        print_performance_summary(metrics)
        
        # Save plots to current directory
        generate_plots(
            history_df=combined_history,
            ticker="PORTFOLIO",
            save_equity_path="equity_curve.png",
            save_dd_path="drawdown.png"
        )
        
        # Output transaction log summary details of the last 15 trades
        if len(all_tx_logs) > 0:
            print("\nRecent Transactions:")
            tx_df = pd.DataFrame(all_tx_logs)
            if 'expiry' in tx_df.columns:
                tx_df['expiry'] = pd.to_datetime(tx_df['expiry']).dt.strftime('%Y-%m-%d')
            tx_df['date'] = pd.to_datetime(tx_df['date']).dt.strftime('%Y-%m-%d')
            print(tx_df.tail(15)[['date', 'ticker', 'type', 'symbol', 'quantity', 'price', 'cash_flow', 'notes']])
        else:
            print("\nNo trades executed during the backtest.")
            
if __name__ == "__main__":
    main()
