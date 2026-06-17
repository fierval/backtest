import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

import importlib
import backtester.data
import backtester.engine
import backtester.analysis
import strategies.wheel_strategy

importlib.reload(backtester.data)
importlib.reload(backtester.engine)
importlib.reload(backtester.analysis)
importlib.reload(strategies.wheel_strategy)

from backtester.data import fetch_yfinance_data, fetch_earnings_weeks
from backtester.engine import BacktestEngine
from backtester.analysis import calculate_metrics

# Set up page config
st.set_page_config(
    page_title="Multi-Ticker Wheel Backtesting Dashboard",
    page_icon="💫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom dark/glassmorphic CSS style to Streamlit
st.markdown("""
<style>
    .reportview-container {
        background: #0f172a;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        border: 1px solid #334155;
    }
    .metric-card b {
        color: #cbd5e1;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    
    /* Tooltip container */
    .metric-tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
        margin-left: 6px;
        vertical-align: middle;
    }
    
    /* Tooltip icon badge */
    .tooltip-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background-color: #475569;
        color: #cbd5e1;
        font-size: 9px;
        font-family: monospace;
        font-weight: bold;
    }
    
    /* Tooltip text box */
    .metric-tooltip .tooltiptext {
        visibility: hidden;
        width: 220px;
        background-color: #0f172a;
        color: #e2e8f0;
        text-align: center;
        border-radius: 6px;
        padding: 8px 12px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.2s;
        border: 1px solid #475569;
        font-size: 11px;
        font-weight: normal;
        text-transform: none;
        letter-spacing: normal;
        line-height: 1.4;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    /* Show tooltip text on hover */
    .metric-tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("💫 Options Wheel Backtesting Dashboard")
st.markdown("Backtest a **Weekly Cash-Secured Put & Covered Call (Wheel) Strategy** on a portfolio of stocks with custom contract sizes and earnings week filters.")

# Sidebar Configuration for advanced settings
st.sidebar.header("Execution Parameters")
avoid_calls_below_cost = st.sidebar.checkbox(
    "Avoid Calls Below Cost Basis", 
    value=True,
    help="If checked, strike selection for Covered Calls will be adjusted to be at least equal to the average stock assignment cost basis."
)
commission = st.sidebar.number_input("Commission per Contract ($)", value=0.65, step=0.05, min_value=0.0)
slippage_pct = st.sidebar.slider("Slippage (% Premium)", min_value=0.0, max_value=5.0, value=1.0, step=0.5) / 100.0
slippage_flat = st.sidebar.number_input("Flat Slippage per Share ($)", value=0.01, step=0.005, min_value=0.0)

# Main Page configuration layout
st.markdown("---")
config_container = st.container()

with config_container:
    st.markdown("### ⚙️ Backtest Parameters")
    
    col_config_left, col_config_right = st.columns([1, 1.2])
    
    with col_config_left:
        st.markdown("**1. Configure Tickers & Options Contracts**")
        st.caption("Add/edit/remove tickers in the table below. Specify contract size per ticker.")
        
        # Prepopulate default tickers DataFrame
        if 'tickers_df' not in st.session_state:
            st.session_state['tickers_df'] = pd.DataFrame([
                {"Ticker": "AAPL", "Contracts": 2},
                {"Ticker": "MSFT", "Contracts": 3},
                {"Ticker": "AMD", "Contracts": 1}
            ])
            
        edited_df = st.data_editor(
            st.session_state['tickers_df'],
            num_rows="dynamic",
            column_config={
                "Ticker": st.column_config.TextColumn(
                    "Ticker Symbol",
                    help="Stock ticker symbol (e.g., AAPL, TSLA, NVDA)",
                    max_chars=5,
                    required=True,
                ),
                "Contracts": st.column_config.NumberColumn(
                    "Contracts",
                    help="Number of options contracts to write (min 1, default 1)",
                    min_value=1,
                    step=1,
                    required=True,
                    default=1
                )
            },
            use_container_width=True,
            key="tickers_editor"
        )
        # Store back to session state
        st.session_state['tickers_df'] = edited_df
        
        # Parse tickers configuration
        tickers_config = {}
        parse_error = False
        
        for idx, row in edited_df.iterrows():
            t = str(row.get("Ticker", "")).strip().upper()
            q = row.get("Contracts")
            if not t:
                continue
            if not t.isalpha():
                st.error(f"Invalid Ticker '{t}': Symbols must contain alphabetical letters only.")
                parse_error = True
                break
            
            qty = int(q) if pd.notna(q) else 1
            tickers_config[t] = qty

        if not tickers_config and not parse_error:
            st.warning("Please specify at least one ticker symbol in the editor.")
            parse_error = True

    with col_config_right:
        st.markdown("**2. Portfolio Capital & Strategy Constraints**")
        
        # Wide capital entry with comma separators
        capital_input = st.text_input(
            "Starting Portfolio Capital ($)", 
            value="300,000",
            help="Enter starting capital. Commas are allowed as thousands separators (e.g. 300,000)."
        )
        
        # Parse capital input
        initial_capital = 300000.0
        try:
            clean_cap = capital_input.replace(",", "").replace("$", "").strip()
            if clean_cap:
                initial_capital = float(clean_cap)
                if initial_capital <= 0:
                    st.error("Starting capital must be a positive number.")
                    parse_error = True
                else:
                    st.markdown(f"<span style='color:#10b981; font-size:13px;'>Parsed Capital: <b>${initial_capital:,.2f}</b></span>", unsafe_allow_html=True)
            else:
                st.error("Starting capital cannot be blank.")
                parse_error = True
        except ValueError:
            st.error("Format Error. Please enter a valid number (e.g. 300,000)")
            parse_error = True

        # Columns for wiggle room, delta and dates
        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            allocation_tolerance_pct = st.slider(
                "Wiggle Room (%)", 
                min_value=0, 
                max_value=20, 
                value=5, 
                step=1,
                help="Extra leverage tolerance buffer allowed to open a position when cash is tight."
            )
            allocation_tolerance = allocation_tolerance_pct / 100.0
            
            delta_target = st.slider("Target Option Delta", min_value=0.15, max_value=0.45, value=0.30, step=0.01)
            
        with col_sub2:
            start_date = st.date_input("Start Date", value=datetime(2022, 1, 1))
            end_date = st.date_input("End Date", value=datetime(2023, 12, 31))
            
            auto_size = st.checkbox(
                "Auto-Size Contracts by Capital",
                value=False,
                help="If checked, the number of contracts for each stock will be automatically computed based on its share of capital (Starting Capital / Number of Tickers) and the put strike price. Manual contract counts in the table will be ignored."
            )
            
            if start_date >= end_date:
                st.error("Start Date must be before End Date.")
                parse_error = True

    # Large prominent run button
    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button("🚀 Run Portfolio Backtest", type="primary", use_container_width=True, disabled=parse_error)
st.markdown("---")

# Helper function to resample data and calculate periodic returns
def get_periodic_returns(history_df, initial_capital):
    portfolio = history_df['PortfolioValue']
    
    # 1. Monthly returns
    monthly_series = portfolio.resample('ME').last()
    monthly_series = pd.concat([pd.Series([initial_capital], index=[portfolio.index[0] - timedelta(days=1)]), monthly_series])
    monthly_returns = monthly_series.pct_change().dropna()
    
    # 2. Annual returns
    annual_series = portfolio.resample('YE').last()
    annual_series = pd.concat([pd.Series([initial_capital], index=[portfolio.index[0] - timedelta(days=1)]), annual_series])
    annual_returns = annual_series.pct_change().dropna()
    
    return monthly_returns, annual_returns

if run_clicked:
    if not tickers_config:
        st.error("Please enter at least one ticker symbol in the configuration.")
    else:
        num_tickers = len(tickers_config)
        capital_per_ticker = initial_capital / num_tickers
        
        # Dictionary to store individual ticker histories and transactions
        all_histories = {}
        all_tx_logs = []
        all_earnings_weeks = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        success_count = 0
        
        # Import strategy here to keep namespace clean
        from strategies.wheel_strategy import WheelStrategy
        
        for i, (ticker, qty) in enumerate(tickers_config.items()):
            status_text.text(f"Processing ticker {i+1} of {num_tickers}: {ticker}...")
            try:
                # 1. Fetch data
                df = fetch_yfinance_data(
                    ticker=ticker,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )
                
                # Fetch earnings calendar
                earnings_weeks = fetch_earnings_weeks(ticker)
                all_earnings_weeks[ticker] = earnings_weeks
                
                # 2. Initialize engine and strategy for this ticker
                engine = BacktestEngine(
                    initial_capital=capital_per_ticker,
                    commission_per_contract=commission,
                    slippage_pct=slippage_pct,
                    slippage_flat=slippage_flat,
                    allocation_tolerance=allocation_tolerance
                )
                
                strategy = WheelStrategy(
                    ticker=ticker,
                    delta_target=delta_target,
                    earnings_weeks=earnings_weeks,
                    avoid_calls_below_cost=avoid_calls_below_cost,
                    fixed_qty=None if auto_size else qty
                )
                
                # 3. Run simulation
                history_df = engine.run(df, strategy)
                all_histories[ticker] = history_df
                
                # Format transactions to include ticker
                for tx in engine.transaction_log:
                    tx_copy = tx.copy()
                    tx_copy['ticker'] = ticker
                    all_tx_logs.append(tx_copy)
                    
                success_count += 1
            except Exception as e:
                st.error(f"Error backtesting {ticker}: {e}")
                logger.error(f"Error for {ticker}: {e}", exc_info=True)
                
            progress_bar.progress((i + 1) / num_tickers)
            
        status_text.empty()
        
        if success_count > 0:
            # Aggregate histories to create a combined portfolio history
            # Find the common date index
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
            
            for ticker, hist in all_histories.items():
                aligned_hist = hist.reindex(common_dates).ffill().bfill()
                combined_history['PortfolioValue'] += aligned_hist['PortfolioValue']
                combined_history['Cash'] += aligned_hist['Cash']
                combined_history['StockValue'] += aligned_hist['StockValue']
                combined_history['OptionsValue'] += aligned_hist['OptionsValue']
                combined_history['Rate'] = aligned_hist['Rate']
                
            # Compute a benchmark baseline (average Buy & Hold of all tickers)
            combined_history['Benchmark'] = 0.0
            for ticker, hist in all_histories.items():
                aligned_hist = hist.reindex(common_dates).ffill().bfill()
                normalized_stock = (aligned_hist['Spot'] / aligned_hist['Spot'].iloc[0]) * (capital_per_ticker)
                combined_history['Benchmark'] += normalized_stock
                
            # Sort transaction log
            all_tx_logs.sort(key=lambda x: x['date'])
            
            # Store in session state
            st.session_state['portfolio_history'] = combined_history
            st.session_state['portfolio_tx_log'] = all_tx_logs
            st.session_state['portfolio_earnings'] = all_earnings_weeks
            st.session_state['portfolio_config'] = tickers_config
            st.session_state['portfolio_run_info'] = {
                'initial_capital': initial_capital,
                'start_date': start_date,
                'end_date': end_date,
                'auto_size': auto_size
            }
            st.success(f"Portfolio backtest complete! Verified {success_count} of {num_tickers} tickers.")

# Render Portfolio Results
if 'portfolio_history' in st.session_state:
    history_df = st.session_state['portfolio_history']
    tx_log = st.session_state['portfolio_tx_log']
    earnings_weeks_dict = st.session_state['portfolio_earnings']
    tickers_config = st.session_state['portfolio_config']
    run_info = st.session_state['portfolio_run_info']
    
    # Calculate performance metrics
    metrics = calculate_metrics(history_df, tx_log, run_info['initial_capital'])
    
    # Render KPIs Dashboard Row with Tooltips
    st.markdown("### 📊 Combined Portfolio Performance")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
    
    with kpi_col1:
        st.markdown(
            f"<div class='metric-card'><b>Net Return <span class='metric-tooltip'><span class='tooltip-icon'>?</span><span class='tooltiptext'>The total net return generated by the portfolio over the backtest period relative to the starting capital.</span></span></b>"
            f"<h2 style='color:#10b981;margin:5px 0;'>{metrics['total_return']*100:.2f}%</h2></div>", 
            unsafe_allow_html=True
        )
    with kpi_col2:
        st.markdown(
            f"<div class='metric-card'><b>CAGR <span class='metric-tooltip'><span class='tooltip-icon'>?</span><span class='tooltiptext'>Compound Annual Growth Rate—the annualized rate of portfolio growth over the backtest period.</span></span></b>"
            f"<h2 style='color:#3b82f6;margin:5px 0;'>{metrics['cagr']*100:.2f}%</h2></div>", 
            unsafe_allow_html=True
        )
    with kpi_col3:
        st.markdown(
            f"<div class='metric-card'><b>Sharpe Ratio <span class='metric-tooltip'><span class='tooltip-icon'>?</span><span class='tooltiptext'>Risk-adjusted return ratio showing the excess return per unit of portfolio standard deviation (volatility).</span></span></b>"
            f"<h2 style='color:#eab308;margin:5px 0;'>{metrics['sharpe']:.2f}</h2></div>", 
            unsafe_allow_html=True
        )
    with kpi_col4:
        st.markdown(
            f"<div class='metric-card'><b>Max Drawdown <span class='metric-tooltip'><span class='tooltip-icon'>?</span><span class='tooltiptext'>The largest peak-to-trough percentage decline in portfolio value during the backtest period.</span></span></b>"
            f"<h2 style='color:#ef4444;margin:5px 0;'>{metrics['max_drawdown']*100:.2f}%</h2></div>", 
            unsafe_allow_html=True
        )
    with kpi_col5:
        st.markdown(
            f"<div class='metric-card'><b>Win Rate <span class='metric-tooltip'><span class='tooltip-icon'>?</span><span class='tooltiptext'>The percentage of closed options trades that resulted in a positive net profit.</span></span></b>"
            f"<h2 style='color:#10b981;margin:5px 0;'>{metrics['win_rate']*100:.1f}%</h2></div>", 
            unsafe_allow_html=True
        )
    with kpi_col6:
        st.markdown(
            f"<div class='metric-card'><b>Profit Factor <span class='metric-tooltip'><span class='tooltip-icon'>?</span><span class='tooltiptext'>The ratio of gross profits to gross losses (gross profits / absolute gross losses).</span></span></b>"
            f"<h2 style='color:#3b82f6;margin:5px 0;'>{metrics['profit_factor']:.2f}</h2></div>", 
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for different charts and data
    tab_weekly, tab_monthly_annual, tab_heatmap, tab_trades_analysis, tab_ledger, tab_earnings = st.tabs([
        "📈 Weekly/Daily P/L", 
        "📊 Monthly & Annual Returns", 
        "🗓️ Monthly Return Heatmap",
        "🎯 Closed Trades Analysis",
        "📝 Combined Transaction Ledger", 
        "🔔 Skipped Earnings Weeks"
    ], key="results_tabs")
    
    # TAB 1: Weekly / Daily P/L (Equity Curve)
    with tab_weekly:
        st.subheader("Weekly Combined Portfolio Equity Curve")
        
        # Interactive Plotly Equity Curve
        fig_equity = go.Figure()
        
        fig_equity.add_trace(go.Scatter(
            x=history_df.index,
            y=history_df['PortfolioValue'],
            mode='lines',
            name='Option Wheel Portfolio',
            line=dict(color='#3b82f6', width=2.5),
            hovertemplate='Date: %{x}<br>Portfolio Value: $%{y:,.2f}<extra></extra>'
        ))
        
        fig_equity.add_trace(go.Scatter(
            x=history_df.index,
            y=history_df['Benchmark'],
            mode='lines',
            name='Combined Tickers Buy & Hold',
            line=dict(color='#94a3b8', width=1.5, dash='dash'),
            hovertemplate='Date: %{x}<br>Buy & Hold: $%{y:,.2f}<extra></extra>'
        ))
        
        fig_equity.update_layout(
            xaxis=dict(
                title='Date',
                gridcolor='#334155',
                showgrid=True
            ),
            yaxis=dict(
                title='Portfolio Value ($)',
                tickformat='$,.0f',
                gridcolor='#334155',
                showgrid=True
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(15, 23, 42, 0.8)',
                bordercolor='#334155',
                borderwidth=1
            ),
            margin=dict(l=40, r=40, t=20, b=40),
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # Interactive Plotly Drawdowns
        st.subheader("Portfolio Drawdown (%)")
        
        fig_drawdown = go.Figure()
        
        peak = history_df['PortfolioValue'].cummax()
        drawdown = (history_df['PortfolioValue'] - peak) / peak
        
        fig_drawdown.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            mode='lines',
            name='Drawdown',
            line=dict(color='#ef4444', width=1.2),
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.15)',
            hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
        ))
        
        fig_drawdown.update_layout(
            xaxis=dict(
                title='Date',
                gridcolor='#334155',
                showgrid=True
            ),
            yaxis=dict(
                title='Drawdown (%)',
                tickformat='.1f',
                ticksuffix='%',
                gridcolor='#334155',
                showgrid=True
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=20, b=40),
            height=250
        )
        st.plotly_chart(fig_drawdown, use_container_width=True)

    # TAB 2: Monthly & Annual Returns (Periodic P/L)
    with tab_monthly_annual:
        monthly_ret, annual_ret = get_periodic_returns(history_df, run_info['initial_capital'])
        
        col_m, col_a = st.columns(2)
        
        with col_m:
            st.subheader("Monthly Net Returns (%)")
            x_labels_monthly = monthly_ret.index.strftime('%y-%b')
            colors_monthly = ['#10b981' if r >= 0 else '#ef4444' for r in monthly_ret]
            
            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Bar(
                x=x_labels_monthly,
                y=monthly_ret * 100,
                marker_color=colors_monthly,
                hovertemplate='Month: %{x}<br>Return: %{y:.2f}%<extra></extra>'
            ))
            
            fig_monthly.update_layout(
                xaxis=dict(title='Month'),
                yaxis=dict(
                    title='Return (%)',
                    tickformat='.2f',
                    ticksuffix='%',
                    gridcolor='#334155',
                    showgrid=True
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=20, b=40),
                height=350
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
            
        with col_a:
            st.subheader("Annual Net Returns (%)")
            x_labels_annual = annual_ret.index.strftime('%Y')
            colors_annual = ['#10b981' if r >= 0 else '#ef4444' for r in annual_ret]
            
            fig_annual = go.Figure()
            fig_annual.add_trace(go.Bar(
                x=x_labels_annual,
                y=annual_ret * 100,
                marker_color=colors_annual,
                width=0.4 if len(annual_ret) > 1 else 0.2,
                hovertemplate='Year: %{x}<br>Return: %{y:.2f}%<extra></extra>'
            ))
            
            fig_annual.update_layout(
                xaxis=dict(title='Year', type='category'),
                yaxis=dict(
                    title='Return (%)',
                    tickformat='.2f',
                    ticksuffix='%',
                    gridcolor='#334155',
                    showgrid=True
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=20, b=40),
                height=350
            )
            st.plotly_chart(fig_annual, use_container_width=True)

    # TAB 3: Monthly Return Heatmap
    with tab_heatmap:
        st.subheader("Monthly Returns Matrix (%)")
        monthly_ret, _ = get_periodic_returns(history_df, run_info['initial_capital'])
        
        # Build Heatmap DataFrame
        heatmap_df = pd.DataFrame(monthly_ret)
        heatmap_df.columns = ['Return']
        heatmap_df['Year'] = heatmap_df.index.year
        heatmap_df['Month'] = heatmap_df.index.strftime('%b')
        
        pivot_df = heatmap_df.pivot(index='Year', columns='Month', values='Return')
        months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_df = pivot_df.reindex(columns=[m for m in months_order if m in pivot_df.columns])
        
        pivot_df_pct = pivot_df * 100
        
        st.dataframe(
            pivot_df_pct.style.format("{:+.2f}%", na_rep="-")
            .background_gradient(cmap="RdYlGn", axis=None, vmin=-5, vmax=5),
            use_container_width=True
        )

    # TAB 4: Closed Trades Analysis (Trade-by-trade portfolio change and charts)
    with tab_trades_analysis:
        st.subheader("Closed Options Trades & Portfolio Value Impact")
        
        # Get raw trades list
        raw_trades = metrics.get('raw_trades', [])
        
        if len(raw_trades) > 0:
            # Enrich trades with portfolio value at exit and percentage contribution
            enriched_trades = []
            for t_idx, trade in enumerate(raw_trades):
                exit_dt = pd.to_datetime(trade['exit_date'])
                
                # Retrieve portfolio value at exit
                try:
                    if exit_dt in history_df.index:
                        port_val_exit = history_df.loc[exit_dt, 'PortfolioValue']
                    else:
                        closest_idx = history_df.index.get_indexer([exit_dt], method='pad')[0]
                        if closest_idx != -1:
                            port_val_exit = history_df['PortfolioValue'].iloc[closest_idx]
                        else:
                            port_val_exit = history_df['PortfolioValue'].iloc[0]
                except Exception:
                    port_val_exit = history_df['PortfolioValue'].iloc[-1]
                
                if isinstance(port_val_exit, pd.Series):
                    port_val_exit = port_val_exit.iloc[-1]
                    
                port_val_exit = float(port_val_exit)
                
                trade_copy = trade.copy()
                trade_copy['port_val_exit'] = port_val_exit
                trade_copy['port_pct_change'] = (trade['net_pnl'] / port_val_exit) * 100.0 if port_val_exit > 0 else 0.0
                enriched_trades.append(trade_copy)
                
            df_enriched = pd.DataFrame(enriched_trades)
            df_enriched['cum_closed_pnl'] = df_enriched['net_pnl'].cumsum()
            df_enriched['trade_number'] = range(1, len(df_enriched) + 1)
            
            # Displays closed trade level metric summary
            col_t1, col_t2, col_t3, col_t4 = st.columns(4)
            with col_t1:
                st.metric("Total Closed Trades", f"{len(df_enriched)}")
            with col_t2:
                avg_pnl = df_enriched['net_pnl'].mean()
                st.metric("Avg Trade P&L", f"${avg_pnl:+,.2f}")
            with col_t3:
                max_win = df_enriched['net_pnl'].max()
                st.metric("Largest Win", f"${max_win:+,.2f}")
            with col_t4:
                max_loss = df_enriched['net_pnl'].min()
                st.metric("Largest Loss", f"${max_loss:+,.2f}")
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Interactive Trade P&L plots
            col_ch1, col_ch2 = st.columns(2)
            
            with col_ch1:
                st.write("**Sequential Trade P&L ($)**")
                colors_seq = ['#10b981' if p >= 0 else '#ef4444' for p in df_enriched['net_pnl']]
                fig_seq = go.Figure()
                fig_seq.add_trace(go.Bar(
                    x=df_enriched['trade_number'],
                    y=df_enriched['net_pnl'],
                    marker_color=colors_seq,
                    hovertemplate='Trade #%{x}<br>Symbol: %{customdata[0]}<br>PnL: $%{y:+,.2f}<extra></extra>',
                    customdata=df_enriched[['symbol']].values
                ))
                fig_seq.update_layout(
                    xaxis=dict(title='Trade Sequence #', dtick=max(1, len(df_enriched)//10)),
                    yaxis=dict(
                        title='Net P&L ($)',
                        tickformat='$,.0f',
                        gridcolor='#334155',
                        showgrid=True
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=40, t=20, b=40),
                    height=300
                )
                st.plotly_chart(fig_seq, use_container_width=True)
                
            with col_ch2:
                st.write("**Cumulative Closed Trades P&L ($)**")
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=df_enriched['trade_number'],
                    y=df_enriched['cum_closed_pnl'],
                    mode='lines+markers',
                    line=dict(color='#3b82f6', width=2),
                    marker=dict(size=5),
                    hovertemplate='Trade #%{x}<br>Cum PnL: $%{y:,.2f}<extra></extra>'
                ))
                fig_cum.update_layout(
                    xaxis=dict(title='Trade Sequence #', dtick=max(1, len(df_enriched)//10)),
                    yaxis=dict(
                        title='Cumulative P&L ($)',
                        tickformat='$,.0f',
                        gridcolor='#334155',
                        showgrid=True
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=40, t=20, b=40),
                    height=300
                )
                st.plotly_chart(fig_cum, use_container_width=True)
                
            # Displays closed trades ledger table
            st.write("**Closed Options Trades Ledger**")
            
            df_disp = df_enriched.copy()
            if 'ticker' not in df_disp.columns:
                df_disp['ticker'] = 'N/A'
            if 'delta' not in df_disp.columns:
                df_disp['delta'] = 0.0
            if 'entry_spot' not in df_disp.columns:
                df_disp['entry_spot'] = 0.0
                
            df_disp['entry_date'] = pd.to_datetime(df_disp['entry_date']).dt.strftime('%Y-%m-%d')
            df_disp['exit_date'] = pd.to_datetime(df_disp['exit_date']).dt.strftime('%Y-%m-%d')
            
            # Determine Phase type based on symbol name
            df_disp['Phase'] = df_disp['symbol'].apply(lambda s: "Covered Call (CC)" if "_C_" in str(s) or "C" in str(s).split('_') else "Cash-Secured Put (CSP)")
            
            df_disp.rename(columns={
                'trade_number': 'Trade #',
                'ticker': 'Ticker',
                'delta': 'Entry Delta',
                'entry_spot': 'Stock Price at Entry',
                'entry_date': 'Open Date',
                'exit_date': 'Close Date',
                'symbol': 'Option Symbol',
                'qty': 'Contracts',
                'entry_price': 'Entry Price',
                'exit_price': 'Exit Price',
                'net_pnl': 'Net P&L ($)',
                'port_val_exit': 'Portfolio Value at Close ($)',
                'port_pct_change': 'Portfolio Change (%)'
            }, inplace=True)
            
            st.dataframe(
                df_disp[['Trade #', 'Ticker', 'Open Date', 'Close Date', 'Phase', 'Option Symbol', 'Stock Price at Entry', 'Entry Delta', 'Contracts', 'Entry Price', 'Exit Price', 'Net P&L ($)', 'Portfolio Value at Close ($)', 'Portfolio Change (%)']]
                .sort_values(by='Trade #', ascending=False)
                .style.format({
                    'Stock Price at Entry': "${:,.2f}",
                    'Entry Delta': "{:.2f}",
                    'Entry Price': "${:,.2f}",
                    'Exit Price': "${:,.2f}",
                    'Net P&L ($)': "${:+,.2f}",
                    'Portfolio Value at Close ($)': "${:,.2f}",
                    'Portfolio Change (%)': "{:+.2f}%"
                }),
                use_container_width=True,
                height=350
            )
        else:
            st.info("No closed trades available to analyze. Run backtest first.")

    # TAB 5: Combined Transaction Ledger
    with tab_ledger:
        st.subheader("Combined Transaction History Log")
        if len(tx_log) > 0:
            tx_df = pd.DataFrame(tx_log)
            display_df = tx_df.copy()
            
            # Defensive check for missing fields in transaction logs
            if 'delta' not in display_df.columns:
                display_df['delta'] = np.nan
            if 'spot' not in display_df.columns:
                display_df['spot'] = np.nan
                
            display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
            if 'expiry' in display_df.columns:
                display_df['expiry'] = pd.to_datetime(display_df['expiry']).dt.strftime('%Y-%m-%d')
            
            display_df.rename(columns={
                'date': 'Trade Date',
                'ticker': 'Ticker',
                'type': 'Action',
                'symbol': 'Option Symbol',
                'strike': 'Strike',
                'expiry': 'Expiration',
                'quantity': 'Contracts',
                'price': 'Execution Price',
                'commission': 'Commissions',
                'cash_flow': 'Net Cash Flow',
                'delta': 'Entry Delta',
                'spot': 'Underlying Price',
                'notes': 'Description'
            }, inplace=True)
            
            # Filters
            col_filt1, col_filt2 = st.columns(2)
            with col_filt1:
                selected_ticker = st.selectbox("Filter by Ticker", ["All"] + list(tickers_config.keys()), key="ledger_ticker_select")
            with col_filt2:
                search_query = st.text_input("Search description", "", key="ledger_search_query")
                
            if selected_ticker != "All":
                display_df = display_df[display_df['Ticker'] == selected_ticker]
            if search_query:
                display_df = display_df[display_df['Description'].str.contains(search_query, case=False, na=False)]
                
            st.dataframe(
                display_df[['Trade Date', 'Ticker', 'Action', 'Option Symbol', 'Underlying Price', 'Entry Delta', 'Contracts', 'Execution Price', 'Commissions', 'Net Cash Flow', 'Description']]
                .style.format({
                    'Underlying Price': lambda x: f"${x:.2f}" if pd.notna(x) else "-",
                    'Entry Delta': lambda x: f"{x:.2f}" if pd.notna(x) else "-",
                    'Execution Price': "${:,.2f}",
                    'Commissions': "${:,.2f}",
                    'Net Cash Flow': "${:+,.2f}"
                }),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No trades executed during this backtest period.")

    # TAB 6: Skipped Earnings Weeks
    with tab_earnings:
        st.subheader("Skipped Weeks (Earnings Calendar)")
        
        selected_earn_ticker = st.selectbox("Select Ticker for Earnings skipped weeks", list(tickers_config.keys()), key="earn_ticker_select")
        
        if selected_earn_ticker in earnings_weeks_dict:
            weeks = earnings_weeks_dict[selected_earn_ticker]
            st.caption(f"Found **{len(weeks)}** earnings weeks in yfinance database for **{selected_earn_ticker}**.")
            if len(weeks) > 0:
                sorted_weeks = sorted(list(weeks))
                start_date_obj = pd.to_datetime(run_info['start_date']).date()
                end_date_obj = pd.to_datetime(run_info['end_date']).date()
                filtered_weeks = [w for w in sorted_weeks if start_date_obj <= w <= end_date_obj]
                
                if len(filtered_weeks) > 0:
                    earn_df = pd.DataFrame({
                        "Monday of Earnings Week": filtered_weeks,
                        "Earnings Announcement Date (Approx)": [w + timedelta(days=2) for w in filtered_weeks]
                    })
                    st.dataframe(earn_df, use_container_width=True, height=300)
                else:
                    st.info(f"No earnings dates fell within your selected backtest period for {selected_earn_ticker}.")
            else:
                st.info(f"No earnings announcement dates found for {selected_earn_ticker} (common for index ETFs).")
else:
    st.info("👈 Set your tickers & contract configurations and click **🚀 Run Portfolio Backtest** to view results!")
