import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# Apply style settings for professional charts
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (12, 6)

def calculate_metrics(history_df: pd.DataFrame, transaction_log: List[dict], initial_capital: float) -> dict:
    """
    Computes portfolio performance metrics.
    """
    if history_df.empty:
        return {}

    start_val = initial_capital
    end_val = history_df['PortfolioValue'].iloc[-1]
    total_return = (end_val - start_val) / start_val

    # CAGR calculation
    start_date = history_df.index[0]
    end_date = history_df.index[-1]
    days = (end_date - start_date).days
    years = max(days / 365.25, 0.001)
    cagr = (end_val / start_val) ** (1 / years) - 1 if end_val > 0 else -1.0

    # Daily returns
    daily_returns = history_df['PortfolioValue'].pct_change().dropna()
    
    # Sharpe Ratio (assuming risk-free rate is daily mean of the actual rates in the data)
    avg_rf_daily = history_df['Rate'].mean() / 252.0
    excess_returns = daily_returns - avg_rf_daily
    
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino Ratio (downside risk only)
    downside_returns = daily_returns[daily_returns < avg_rf_daily]
    if len(downside_returns) > 1 and downside_returns.std() > 0:
        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
    else:
        sortino = 0.0

    # Max Drawdown
    peak = history_df['PortfolioValue'].cummax()
    drawdown = (history_df['PortfolioValue'] - peak) / peak
    max_dd = drawdown.min()

    # Trade stats from transaction log
    trade_stats = analyze_trades(transaction_log)

    return {
        'initial_capital': start_val,
        'final_value': end_val,
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'days_in_backtest': days,
        **trade_stats
    }

def analyze_trades(transaction_log: List[dict]) -> dict:
    """
    Pairs open and close transactions to calculate trade-level statistics.
    """
    trades = []
    # Track open positions by symbol name (e.g. "100_C_105_012023")
    # Store list of dicts: {'qty': int, 'cash_flow': float, 'commission': float}
    open_positions = {}

    for tx in transaction_log:
        tx_type = tx['type']
        symbol = tx['symbol']
        qty = tx['quantity']
        price = tx['price']
        commission = tx['commission']
        cash_flow = tx['cash_flow']
        date = tx['date']

        if tx_type.startswith('OPEN_'):
            # Opening a position
            if symbol not in open_positions:
                open_positions[symbol] = []
            open_positions[symbol].append({
                'qty': qty,
                'entry_price': price,
                'entry_date': date,
                'cash_flow': cash_flow,
                'commission': commission,
                'ticker': tx.get('ticker'),
                'delta': tx.get('delta', 0.0),
                'entry_spot': tx.get('spot', 0.0)
            })
        elif tx_type.startswith('CLOSE_') or tx_type.startswith('EXPIRY_'):
            # Closing/settling a position
            if symbol in open_positions and len(open_positions[symbol]) > 0:
                # Pair with the oldest opening transaction (FIFO)
                open_tx = open_positions[symbol].pop(0)
                
                total_comm = open_tx['commission'] + commission
                
                if tx_type.startswith('EXPIRY_'):
                    strike = tx.get('strike')
                    spot = tx.get('spot', price) # Fallback to price if spot is missing
                    opt_type = tx.get('option_type', 'put' if '_P_' in symbol or 'P' in symbol.split('_') else 'call')
                    side = tx.get('side', 'short')
                    
                    if opt_type.lower() == 'put':
                        intrinsic = max(strike - spot, 0.0)
                    else:
                        intrinsic = max(spot - strike, 0.0)
                        
                    # Cash settled option value:
                    # Short: pay intrinsic to close, Long: receive intrinsic to close
                    if side.lower() == 'short':
                        cash_flow_settled = -intrinsic * 100.0 * qty - commission
                    else:
                        cash_flow_settled = intrinsic * 100.0 * qty - commission
                        
                    net_pnl = open_tx['cash_flow'] + cash_flow_settled
                    exit_price = intrinsic
                else:
                    # For early closes, it's just the cash flow sum (standard premium capture)
                    net_pnl = open_tx['cash_flow'] + cash_flow
                    exit_price = price
                
                trades.append({
                    'symbol': symbol,
                    'entry_date': open_tx['entry_date'],
                    'exit_date': date,
                    'qty': qty,
                    'entry_price': open_tx['entry_price'],
                    'exit_price': exit_price,
                    'net_pnl': net_pnl,
                    'commission': total_comm,
                    'ticker': open_tx.get('ticker'),
                    'delta': open_tx.get('delta', 0.0),
                    'entry_spot': open_tx.get('entry_spot', 0.0)
                })
                
                if len(open_positions[symbol]) == 0:
                    del open_positions[symbol]
            else:
                # Stock or unmatched option closure
                pass
        elif tx_type in ('BUY_STOCK', 'SELL_STOCK'):
            # For simplicity, we track stock trades separately or log cash directly
            pass

    if len(trades) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_trade_pnl': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }

    df_trades = pd.DataFrame(trades)
    wins = df_trades[df_trades['net_pnl'] > 0]
    losses = df_trades[df_trades['net_pnl'] <= 0]
    
    total_trades = len(df_trades)
    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
    
    total_gain = wins['net_pnl'].sum()
    total_loss = abs(losses['net_pnl'].sum())
    profit_factor = total_gain / total_loss if total_loss > 0 else float('inf') if total_gain > 0 else 1.0
    
    avg_trade_pnl = df_trades['net_pnl'].mean()
    avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0.0
    avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0.0

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_trade_pnl': avg_trade_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'raw_trades': trades
    }

def print_performance_summary(metrics: dict):
    """
    Prints a formatted performance report.
    """
    print("=" * 50)
    print("           OPTIONS BACKTEST PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Initial Capital      : ${metrics['initial_capital']:,.2f}")
    print(f"Final Portfolio Value: ${metrics['final_value']:,.2f}")
    print(f"Total Net Return     : {metrics['total_return']*100:.2f}%")
    print(f"Annualized Return    : {metrics['cagr']*100:.2f}%")
    print(f"Sharpe Ratio         : {metrics['sharpe']:.2f}")
    print(f"Sortino Ratio        : {metrics['sortino']:.2f}")
    print(f"Max Drawdown         : {metrics['max_drawdown']*100:.2f}%")
    print(f"Duration (Days)      : {metrics['days_in_backtest']}")
    print("-" * 50)
    print(f"Total Closed Trades  : {metrics['total_trades']}")
    print(f"Win Rate             : {metrics['win_rate']*100:.2f}%")
    print(f"Profit Factor        : {metrics['profit_factor']:.2f}")
    print(f"Avg Trade PnL        : ${metrics['avg_trade_pnl']:,.2f}")
    print(f"Avg Win Trade        : ${metrics['avg_win']:,.2f}")
    print(f"Avg Loss Trade       : ${metrics['avg_loss']:,.2f}")
    print("=" * 50)

def generate_plots(history_df: pd.DataFrame, ticker: str, save_equity_path: str = "equity_curve.png", save_dd_path: str = "drawdown.png"):
    """
    Generates and saves performance charts.
    """
    # 1. Equity Curve Chart
    plt.figure(figsize=(12, 6))
    
    # Portfolio equity curve
    portfolio_curve = history_df['PortfolioValue']
    initial_val = portfolio_curve.iloc[0]
    
    # Stock buy-and-hold curve normalized to start at portfolio initial value
    stock_curve = history_df['Spot']
    stock_curve_normalized = (stock_curve / stock_curve.iloc[0]) * initial_val
    
    plt.plot(portfolio_curve.index, portfolio_curve, label='Options Strategy Portfolio', color='#3b82f6', linewidth=2.5)
    plt.plot(stock_curve_normalized.index, stock_curve_normalized, label=f'Buy & Hold {ticker}', color='#94a3b8', linestyle='--', linewidth=1.5)
    
    plt.title(f'Strategy Equity Curve vs. Buy & Hold {ticker}', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    plt.tight_layout()
    plt.savefig(save_equity_path, dpi=300)
    plt.close()
    logger.info(f"Saved equity curve plot to {save_equity_path}")

    # 2. Drawdown Chart
    plt.figure(figsize=(12, 4))
    peak = portfolio_curve.cummax()
    drawdown = (portfolio_curve - peak) / peak
    
    plt.fill_between(drawdown.index, drawdown * 100, 0, color='#ef4444', alpha=0.3)
    plt.plot(drawdown.index, drawdown * 100, color='#ef4444', linewidth=1.5)
    
    plt.title('Portfolio Drawdown (%)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown %', fontsize=12)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))
    plt.tight_layout()
    plt.savefig(save_dd_path, dpi=300)
    plt.close()
    logger.info(f"Saved drawdown plot to {save_dd_path}")
