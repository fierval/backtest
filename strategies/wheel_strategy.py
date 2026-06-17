from backtester.strategy import BaseStrategy
from backtester.data import fetch_earnings_weeks
from datetime import datetime, timedelta
import numpy as np
import logging
from scipy.stats import norm

logger = logging.getLogger(__name__)

class WheelStrategy(BaseStrategy):
    def __init__(self, ticker: str, delta_target: float = 0.30, 
                 earnings_weeks: set = None, avoid_calls_below_cost: bool = False,
                 fixed_qty: int = None):
        """
        ticker: Symbol of the stock to trade
        delta_target: Target option delta (e.g. 0.30)
        earnings_weeks: Pre-fetched set of Monday dates representing earnings announcement weeks
        avoid_calls_below_cost: If True, strike for covered calls will be at least the shares cost basis
        fixed_qty: If set, trade this exact number of contracts instead of calculating it from capital
        """
        super().__init__()
        self.ticker = ticker
        self.delta_target = delta_target
        self.earnings_weeks = earnings_weeks if earnings_weeks is not None else set()
        self.avoid_calls_below_cost = avoid_calls_below_cost
        self.fixed_qty = fixed_qty
        self._last_week = None

    def initialize(self):
        """
        Loads the earnings weeks calendar if not pre-provided.
        """
        if not self.earnings_weeks:
            self.earnings_weeks = fetch_earnings_weeks(self.ticker)
            
    def on_data(self, date: datetime, spot: float, rate: float, vol: float):
        # 1. Identify start of the week
        current_week = date.isocalendar()[1]
        
        if self._last_week is None or self._last_week != current_week:
            self._last_week = current_week
            is_start_of_week = True
        else:
            is_start_of_week = False
            
        if not is_start_of_week:
            return # Only execute new trades at the start of the week
            
        # 2. Check for earnings skip
        week_start = (date - timedelta(days=date.weekday())).date()
        if week_start in self.earnings_weeks:
            logger.info(f"[{date.strftime('%Y-%m-%d')}] Skipping week starting {week_start} due to earnings announcement for {self.ticker}.")
            return
            
        # 3. Expiration calculation (Friday close of this week)
        days_to_friday = 4 - date.weekday()
        if days_to_friday < 0:
            days_to_friday = 0 # If it is Saturday/Sunday (not expected)
        expiry = date + timedelta(days=days_to_friday)
        
        t_years = max((expiry - date).days, 1.0) / 365.0
        
        # 4. Wheel Logic
        # Case A: No shares owned -> Cash Secured Put (CSP) phase
        if self.shares <= 0:
            # If we already have open option positions, wait
            if len(self.open_positions) > 0:
                return
                
            # Find Strike K for Put with target delta
            # Put Delta = N(d1) - 1 = -delta_target => N(d1) = 1 - delta_target => d1 = norm.ppf(1 - delta_target)
            d1 = norm.ppf(1.0 - self.delta_target)
            
            # Solve for Strike: K = S * exp((r + 0.5 * vol^2) * T - d1 * vol * sqrt(T))
            vol_term = vol * np.sqrt(t_years)
            drift_term = (rate + 0.5 * vol**2) * t_years
            strike_calculated = spot * np.exp(drift_term - d1 * vol_term)
            strike = round(strike_calculated)
            
            if strike <= 0:
                logger.warning(f"Calculated invalid strike: {strike} at spot {spot}")
                return
                
            # Determine contracts: use fixed quantity if provided, else max possible
            if self.fixed_qty is not None:
                qty = self.fixed_qty
            else:
                required_collateral = strike * 100.0
                allowed_cash = self.available_funds * (1.0 + self.engine.allocation_tolerance)
                qty = int(np.floor(allowed_cash / required_collateral))
            
            if qty > 0:
                logger.info(f"[{date.strftime('%Y-%m-%d')}] CSP Phase: Selling {qty} Put contracts. Strike: {strike}, Expiry: {expiry.strftime('%Y-%m-%d')} (Spot: {spot:.2f}, Est Delta: -{self.delta_target})")
                self.sell_put(strike, expiry, qty)
            else:
                logger.debug(f"[{date.strftime('%Y-%m-%d')}] CSP Phase: Insufficient funds or zero contracts specified to sell Put strike {strike}.")

        # Case B: Stock shares owned -> Covered Call (CC) phase
        else:
            # If we already have open option positions (e.g. Call is running), wait
            if len(self.open_positions) > 0:
                return
                
            # Find Strike K for Call with target delta
            # Call Delta = N(d1) = delta_target => d1 = norm.ppf(delta_target)
            # Standard retail target is 0.30 delta call, which has d1 = norm.ppf(0.30)
            d1 = norm.ppf(self.delta_target)
            
            vol_term = vol * np.sqrt(t_years)
            drift_term = (rate + 0.5 * vol**2) * t_years
            strike_calculated = spot * np.exp(drift_term - d1 * vol_term)
            strike = round(strike_calculated)
            
            # Optionally avoid selling calls below our stock purchase cost basis
            if self.avoid_calls_below_cost and self.shares_entry_price > 0:
                cost_basis_strike = round(self.shares_entry_price)
                if strike < cost_basis_strike:
                    logger.info(f"[{date.strftime('%Y-%m-%d')}] CC Phase: Call strike {strike} is below cost basis {self.shares_entry_price:.2f}. Adjusting strike to {cost_basis_strike}")
                    strike = cost_basis_strike
                    
            # Determine contracts: use fixed quantity (capped by shares) or all available covered shares
            max_covered_qty = int(self.shares // 100)
            if self.fixed_qty is not None:
                qty = min(self.fixed_qty, max_covered_qty)
            else:
                qty = max_covered_qty
            
            if qty > 0:
                logger.info(f"[{date.strftime('%Y-%m-%d')}] CC Phase: Selling {qty} Covered Call contracts. Strike: {strike}, Expiry: {expiry.strftime('%Y-%m-%d')} (Spot: {spot:.2f}, Est Delta: {self.delta_target})")
                self.sell_call(strike, expiry, qty)
