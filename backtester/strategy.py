import logging
from typing import Dict, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class BaseStrategy:
    def __init__(self):
        """
        Base class for options trading strategies.
        Must be subclassed with custom logic.
        """
        self.engine = None  # Will be set by BacktestEngine during run()

    def initialize(self):
        """
        Override this method to initialize parameters, indicators, and setups.
        """
        pass

    def on_data(self, date: datetime, spot: float, rate: float, vol: float):
        """
        Override this method to run custom trading logic on each new bar.
        
        Parameters:
        -----------
        date : datetime - Current timestamp of the data
        spot : float - Current underlying stock closing price
        rate : float - Current risk-free interest rate (decimal)
        vol : float - Current implied/realized volatility (decimal)
        """
        raise NotImplementedError("Strategy must implement on_data method.")

    # --- Convenience Wrappers for Engine Operations ---
    
    def buy_call(self, strike: float, expiry: datetime, quantity: int) -> Optional[str]:
        """
        Convenience wrapper to buy a call option contract.
        """
        return self.engine.open_option("call", "buy", strike, expiry, quantity)

    def sell_call(self, strike: float, expiry: datetime, quantity: int) -> Optional[str]:
        """
        Convenience wrapper to sell (write) a call option contract.
        """
        return self.engine.open_option("call", "sell", strike, expiry, quantity)

    def buy_put(self, strike: float, expiry: datetime, quantity: int) -> Optional[str]:
        """
        Convenience wrapper to buy a put option contract.
        """
        return self.engine.open_option("put", "buy", strike, expiry, quantity)

    def sell_put(self, strike: float, expiry: datetime, quantity: int) -> Optional[str]:
        """
        Convenience wrapper to sell (write) a put option contract.
        """
        return self.engine.open_option("put", "sell", strike, expiry, quantity)

    def close_position(self, position_id: str, quantity: Optional[int] = None) -> bool:
        """
        Convenience wrapper to close an open option position.
        """
        return self.engine.close_option(position_id, quantity)

    def buy_shares(self, quantity: int):
        """
        Convenience wrapper to buy shares of the underlying asset.
        """
        self.engine.buy_shares(quantity)

    def sell_shares(self, quantity: int):
        """
        Convenience wrapper to sell shares of the underlying asset.
        """
        self.engine.sell_shares(quantity)

    # --- Portfolio State Getters ---

    @property
    def cash(self) -> float:
        return self.engine.cash

    @property
    def shares(self) -> int:
        return self.engine.shares

    @property
    def portfolio_value(self) -> float:
        return self.engine.portfolio_value

    @property
    def available_funds(self) -> float:
        return self.engine.available_funds

    @property
    def margin_locked(self) -> float:
        return self.engine.margin_locked

    @property
    def open_positions(self) -> Dict:
        return self.engine.positions

    @property
    def shares_entry_price(self) -> float:
        return self.engine.shares_entry_price
