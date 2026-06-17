import yfinance as yf
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta

try:
    import databento as db
except ImportError:
    db = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_yfinance_data(ticker: str, start_date: str, end_date: str, 
                        rf_ticker: str = "^IRX", vol_ticker: str = "^VIX", 
                        rolling_vol_window: int = 20) -> pd.DataFrame:
    """
    Downloads historical underlying asset price, risk-free rate, and VIX data,
    computes rolling realized volatility, and aligns them into a single DataFrame.
    """
    logger.info(f"Downloading historical price data for underlying: {ticker} from {start_date} to {end_date} (yfinance)")
    
    underlying_df = yf.download(ticker, start=start_date, end=end_date)
    if underlying_df.empty:
        raise ValueError(f"No data returned for ticker {ticker}. Please check the symbol and dates.")
    
    if isinstance(underlying_df.columns, pd.MultiIndex):
        underlying_df.columns = underlying_df.columns.get_level_values(0)
        
    cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = underlying_df[cols_to_keep].copy()
    
    for col in df.columns:
        df[col] = df[col].astype(float)
        
    # Fetch risk-free rate
    rf_data = yf.download(rf_ticker, start=start_date, end=end_date)
    if rf_data.empty:
        logger.warning(f"Could not download risk-free rate data for {rf_ticker}. Defaulting to 0.04 (4%).")
        df['Rate'] = 0.04
    else:
        if isinstance(rf_data.columns, pd.MultiIndex):
            rf_data.columns = rf_data.columns.get_level_values(0)
        rf_close = rf_data['Close'].astype(float) / 100.0
        rf_close = rf_close.reindex(df.index).ffill().bfill()
        df['Rate'] = rf_close
        
    # Fetch VIX
    vol_data = yf.download(vol_ticker, start=start_date, end=end_date)
    if vol_data.empty:
        logger.warning(f"Could not download volatility index data for {vol_ticker}. Will default to Realized Volatility.")
        df['ImpliedVolProxy'] = np.nan
    else:
        if isinstance(vol_data.columns, pd.MultiIndex):
            vol_data.columns = vol_data.columns.get_level_values(0)
        vol_close = vol_data['Close'].astype(float) / 100.0
        vol_close = vol_close.reindex(df.index).ffill().bfill()
        df['ImpliedVolProxy'] = vol_close

    # Calculate rolling realized volatility
    daily_returns = np.log(df['Close'] / df['Close'].shift(1))
    rolling_std = daily_returns.rolling(window=rolling_vol_window).std()
    realized_vol = rolling_std * np.sqrt(252)
    mean_vol = realized_vol.mean()
    if pd.isna(mean_vol):
        mean_vol = 0.20
    df['RealizedVol'] = realized_vol.fillna(mean_vol)
    df['ImpliedVolProxy'] = df['ImpliedVolProxy'].fillna(df['RealizedVol'])
    
    return df

def fetch_earnings_weeks(ticker: str) -> set:
    """
    Downloads historical and upcoming earnings dates for a ticker using yfinance,
    and returns a set of datetime.date objects representing the start of the week
    (Monday) for each week that has an earnings announcement.
    """
    logger.info(f"Fetching historical earnings calendar for {ticker}")
    earnings_weeks = set()
    try:
        t = yf.Ticker(ticker)
        # Fetch up to 100 historical/upcoming earnings dates
        earnings_df = t.get_earnings_dates(limit=100)
        
        if earnings_df is not None and not earnings_df.empty:
            for dt in earnings_df.index:
                # Clean timezone information if present
                dt_naive = dt.tz_localize(None) if dt.tz is not None else dt
                date_obj = dt_naive.date()
                
                # Get start of week (Monday)
                week_start = date_obj - timedelta(days=date_obj.weekday())
                earnings_weeks.add(week_start)
                
            logger.info(f"Found {len(earnings_weeks)} unique earnings weeks for {ticker}")
        else:
            logger.warning(f"No earnings dates found for {ticker} (may be an ETF like SPY)")
    except Exception as e:
        logger.warning(f"Could not retrieve earnings dates for {ticker} (expected for indices/ETFs): {e}")
        
    return earnings_weeks

def fetch_databento_options_data(api_key: str, dataset: str, symbols: list, 
                                 start_date: str, end_date: str, 
                                 schema: str = "trades") -> pd.DataFrame:
    """
    Fetches historical options data directly from Databento.
    """
    if db is None:
        raise ImportError("The 'databento' library is not installed or failed to import.")
    
    key = api_key or os.environ.get("DATABENTO") or os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise ValueError("Databento API key must be provided or set in the environment variable 'DATABENTO' or 'DATABENTO_API_KEY'.")
    
    logger.info(f"Connecting to Databento for dataset {dataset}, schema {schema}")
    client = db.Historical(key=key)
    
    logger.info(f"Querying symbols {symbols} from {start_date} to {end_date}")
    db_data = client.timeseries.get_range(
        dataset=dataset,
        symbols=symbols,
        schema=schema,
        start=start_date,
        end=end_date
    )
    
    df = db_data.to_df()
    logger.info(f"Successfully fetched {len(df)} records from Databento.")
    return df

def load_local_databento_file(filepath: str) -> pd.DataFrame:
    """
    Loads options historical data from a local file downloaded from Databento.
    Supports .csv, .parquet, or .dbn formats.
    """
    logger.info(f"Loading local Databento file: {filepath}")
    _, ext = os.path.splitext(filepath.lower())
    
    if ext == ".parquet":
        return pd.read_parquet(filepath)
    elif ext == ".csv":
        return pd.read_csv(filepath)
    elif ext == ".dbn" or ext == ".zst":
        if db is None:
            raise ImportError("The 'databento' library is required to read .dbn or DBN-compressed files.")
        dbn_store = db.DBNStore.from_file(filepath)
        return dbn_store.to_df()
    else:
        raise ValueError(f"Unsupported file format: {ext}. Please use Parquet, CSV, or DBN.")

if __name__ == "__main__":
    # Test script locally
    try:
        weeks = fetch_earnings_weeks("AAPL")
        print(f"Sample earnings week Mondays: {list(weeks)[:5]}")
    except Exception as e:
        print(f"Error during test: {e}")
