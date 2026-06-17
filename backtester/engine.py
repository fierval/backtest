import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime

from .bsm import black_scholes_price, calculate_greeks

logger = logging.getLogger(__name__)

class OptionPosition:
    def __init__(self, position_id: str, option_type: str, side: str, 
                 strike: float, expiry: datetime, quantity: int, 
                 entry_date: datetime, entry_price: float):
        """
        id: Unique identifier for the position
        option_type: 'call' or 'put'
        side: 'long' or 'short'
        strike: Option strike price
        expiry: Expiration date
        quantity: Number of contracts (1 contract = 100 shares)
        entry_date: Date option was opened
        entry_price: Premium per share paid or received
        """
        self.id = position_id
        self.option_type = option_type.lower()
        self.side = "long" if side.lower() in ("buy", "long") else "short"
        self.strike = strike
        self.expiry = pd.to_datetime(expiry)
        self.quantity = quantity
        self.entry_date = pd.to_datetime(entry_date)
        self.entry_price = entry_price
        self.current_price = entry_price
        self.greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}

    def mark_to_market(self, spot: float, rate: float, vol: float, current_date: datetime):
        """
        Updates the option price and Greeks using the BSM model.
        """
        current_dt = pd.to_datetime(current_date)
        days_to_expiry = (self.expiry - current_dt).days
        
        if days_to_expiry <= 0:
            # Expired: value is intrinsic
            if self.option_type == "call":
                self.current_price = max(spot - self.strike, 0.0)
            else:
                self.current_price = max(self.strike - spot, 0.0)
            self.greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
        else:
            t_years = days_to_expiry / 365.0
            self.current_price = black_scholes_price(spot, self.strike, t_years, rate, vol, self.option_type)
            self.greeks = calculate_greeks(spot, self.strike, t_years, rate, vol, self.option_type)

    def get_value(self) -> float:
        """
        Returns the absolute mark-to-market value of this option position.
        Long positions have positive value, short positions have negative value.
        """
        mult = 1.0 if self.side == "long" else -1.0
        return mult * self.current_price * 100.0 * self.quantity

    def get_intrinsic_value(self, spot: float) -> float:
        if self.option_type == "call":
            return max(spot - self.strike, 0.0)
        else:
            return max(self.strike - spot, 0.0)


class BacktestEngine:
    def __init__(self, initial_capital: float = 100000.0, 
                 commission_per_contract: float = 0.65, 
                 slippage_pct: float = 0.0, 
                 slippage_flat: float = 0.01,
                 allocation_tolerance: float = 0.0):
        """
        initial_capital: Starting cash balance
        commission_per_contract: Flat fee per option contract traded
        slippage_pct: Slippage as a percentage of option price
        slippage_flat: Flat slippage per contract (applied as price offset per share)
        allocation_tolerance: Fractional wiggle room for capital collateral (e.g. 0.05 for 5%)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.shares = 0  # Underlying shares held
        self.shares_entry_price = 0.0
        
        self.commission_per_contract = commission_per_contract
        self.slippage_pct = slippage_pct
        self.slippage_flat = slippage_flat
        self.allocation_tolerance = allocation_tolerance
        
        self.positions: Dict[str, OptionPosition] = {}
        self.position_counter = 0
        
        # Logging & History
        self.history: List[dict] = []
        self.transaction_log: List[dict] = []
        
        # Current state
        self.current_date: Optional[datetime] = None
        self.current_spot: float = 0.0
        self.current_rate: float = 0.0
        self.current_vol: float = 0.0

    @property
    def portfolio_value(self) -> float:
        """
        Total portfolio value = Cash + Option Values + Underlying Stock Value
        """
        option_val = sum(pos.get_value() for pos in self.positions.values())
        stock_val = self.shares * self.current_spot
        return self.cash + option_val + stock_val

    @property
    def margin_locked(self) -> float:
        """
        Calculates locked capital for options positions.
        - Long options: 0 margin locked (premium is paid upfront out of cash).
        - Short puts (Cash Secured Put model): Locked margin = Strike * 100 * Qty
        - Short calls:
            - If covered by underlying shares (shares >= 100 * Qty): 0 margin locked.
            - If naked short call: Locked margin = Strike * 100 * Qty * 0.20 (20% of strike value proxy)
        """
        margin = 0.0
        covered_shares = self.shares
        
        # Process short options for margin
        for pos in self.positions.values():
            if pos.side == "short":
                if pos.option_type == "put":
                    # Cash-secured put: lock full strike value
                    margin += pos.strike * 100.0 * pos.quantity
                elif pos.option_type == "call":
                    # Check if covered by shares
                    required_shares = pos.quantity * 100
                    if covered_shares >= required_shares:
                        # Covered call: no extra margin, but these shares are "pledged"
                        covered_shares -= required_shares
                    else:
                        # Naked call: lock 20% of the strike value as margin collateral
                        margin += pos.strike * 100.0 * pos.quantity * 0.20
        return margin

    @property
    def maintenance_margin(self) -> float:
        """
        Locked capital + any short options mark-to-market liability.
        """
        # OptionPosition.get_value() returns negative for short positions
        short_liability = sum(-pos.get_value() for pos in self.positions.values() if pos.side == "short")
        return self.margin_locked + short_liability

    @property
    def available_funds(self) -> float:
        """
        Funds available for new trades = Cash - Margin Locked
        """
        return self.cash - self.margin_locked

    def _next_position_id(self) -> str:
        self.position_counter += 1
        return f"OPT_{self.position_counter:05d}"

    def buy_shares(self, quantity: int):
        """
        Buys underlying stock shares at the current spot price.
        """
        cost = quantity * self.current_spot
        commission = 0.0  # Assume zero commission for stock trades for simplicity
        
        if self.cash < cost:
            # Allow margin stock buying? No, keep it cash-only for safety
            logger.warning(f"Insufficient cash to buy {quantity} shares. Cash: {self.cash:.2f}, Cost: {cost:.2f}")
            return
            
        # Update average entry price
        if self.shares > 0:
            self.shares_entry_price = ((self.shares * self.shares_entry_price) + cost) / (self.shares + quantity)
        else:
            self.shares_entry_price = self.current_spot
            
        self.shares += quantity
        self.cash -= cost
        
        self.transaction_log.append({
            'date': self.current_date,
            'type': 'BUY_STOCK',
            'symbol': 'STOCK',
            'strike': np.nan,
            'expiry': None,
            'quantity': quantity,
            'price': self.current_spot,
            'commission': commission,
            'cash_flow': -cost,
            'notes': 'Bought shares of underlying stock'
        })

    def sell_shares(self, quantity: int):
        """
        Sells underlying stock shares at the current spot price.
        """
        if self.shares < quantity:
            logger.warning(f"Attempted to sell {quantity} shares but only hold {self.shares}")
            quantity = self.shares
            
        if quantity <= 0:
            return
            
        revenue = quantity * self.current_spot
        self.shares -= quantity
        self.cash += revenue
        
        if self.shares == 0:
            self.shares_entry_price = 0.0
            
        self.transaction_log.append({
            'date': self.current_date,
            'type': 'SELL_STOCK',
            'symbol': 'STOCK',
            'strike': np.nan,
            'expiry': None,
            'quantity': quantity,
            'price': self.current_spot,
            'commission': 0.0,
            'cash_flow': revenue,
            'notes': 'Sold shares of underlying stock'
        })

    def open_option(self, option_type: str, side: str, strike: float, expiry: datetime, quantity: int) -> Optional[str]:
        """
        Opens an option position.
        """
        if quantity <= 0:
            return None
            
        # Calculate theoretical price
        expiry_dt = pd.to_datetime(expiry)
        days_to_expiry = (expiry_dt - pd.to_datetime(self.current_date)).days
        if days_to_expiry <= 0:
            logger.warning(f"Cannot open option expiring today or in the past: {expiry}")
            return None
            
        t_years = days_to_expiry / 365.0
        theo_price = black_scholes_price(self.current_spot, strike, t_years, self.current_rate, self.current_vol, option_type)
        
        # Apply slippage
        # Buy: pay more, Sell: receive less
        slippage_offset = (theo_price * self.slippage_pct) + self.slippage_flat
        if side == "buy" or side == "long":
            execution_price = theo_price + slippage_offset
            cash_flow = -execution_price * 100.0 * quantity
        else:
            execution_price = max(theo_price - slippage_offset, 0.01) # Price can't be <= 0
            cash_flow = execution_price * 100.0 * quantity
            
        commission = self.commission_per_contract * quantity
        net_cash_flow = cash_flow - commission
        
        # Check margin/cash constraints
        if side == "buy" or side == "long":
            # Premium is paid out of cash
            if self.cash < abs(net_cash_flow):
                logger.warning(f"Insufficient cash to buy option. Cash: {self.cash:.2f}, Required: {abs(net_cash_flow):.2f}")
                return None
        else:
            # Check margin requirement
            temp_pos = OptionPosition("TEMP", option_type, side, strike, expiry, quantity, self.current_date, execution_price)
            # Add to cash temporarily to see if we satisfy margin
            expected_cash = self.cash + net_cash_flow
            
            # Simple margin check
            required_margin = 0.0
            if option_type == "put":
                required_margin = strike * 100.0 * quantity
            elif option_type == "call":
                # Check if covered by shares
                required_shares = quantity * 100
                if self.shares < required_shares:
                    required_margin = strike * 100.0 * quantity * 0.20
            
            # Apply allocation tolerance (wiggle room) for margin trades
            if required_margin > 0.0:
                allowed_collateral_limit = expected_cash * (1.0 + self.allocation_tolerance)
                if allowed_collateral_limit < required_margin:
                    logger.warning(f"Insufficient capital/margin to sell option. Expected cash (with tolerance): {allowed_collateral_limit:.2f}, Required margin: {required_margin:.2f}")
                    return None
                
        # Deduct cash
        self.cash += net_cash_flow
        
        # Save position
        pos_id = self._next_position_id()
        position = OptionPosition(pos_id, option_type, side, strike, expiry, quantity, self.current_date, execution_price)
        position.mark_to_market(self.current_spot, self.current_rate, self.current_vol, self.current_date)
        
        self.positions[pos_id] = position
        
        self.transaction_log.append({
            'date': self.current_date,
            'type': f'OPEN_{side.upper()}_{option_type.upper()}',
            'symbol': f"{strike}_{option_type[0].upper()}_{expiry_dt.strftime('%m%d%y')}",
            'strike': strike,
            'expiry': expiry_dt,
            'quantity': quantity,
            'price': execution_price,
            'commission': commission,
            'cash_flow': net_cash_flow,
            'spot': self.current_spot,
            'option_type': option_type,
            'side': side,
            'delta': position.greeks['delta'],
            'notes': f"Opened {side} {option_type} position"
        })
        
        return pos_id

    def close_option(self, position_id: str, quantity: Optional[int] = None) -> bool:
        """
        Closes an active option position. If quantity is None, closes the entire position.
        """
        if position_id not in self.positions:
            logger.warning(f"Position ID {position_id} not found in open positions.")
            return False
            
        pos = self.positions[position_id]
        if quantity is None or quantity >= pos.quantity:
            quantity = pos.quantity
            remove_position = True
        else:
            remove_position = False
            
        # Apply slippage
        # Long position: sell to close (receive cash, minus slippage)
        # Short position: buy to close (pay cash, plus slippage)
        slippage_offset = (pos.current_price * self.slippage_pct) + self.slippage_flat
        if pos.side == "long":
            execution_price = max(pos.current_price - slippage_offset, 0.01)
            cash_flow = execution_price * 100.0 * quantity
        else:
            execution_price = pos.current_price + slippage_offset
            cash_flow = -execution_price * 100.0 * quantity
            
        commission = self.commission_per_contract * quantity
        net_cash_flow = cash_flow - commission
        
        self.cash += net_cash_flow
        
        self.transaction_log.append({
            'date': self.current_date,
            'type': f'CLOSE_{pos.side.upper()}_{pos.option_type.upper()}',
            'symbol': f"{pos.strike}_{pos.option_type[0].upper()}_{pos.expiry.strftime('%m%d%y')}",
            'strike': pos.strike,
            'expiry': pos.expiry,
            'quantity': quantity,
            'price': execution_price,
            'commission': commission,
            'cash_flow': net_cash_flow,
            'spot': self.current_spot,
            'option_type': pos.option_type,
            'side': pos.side,
            'notes': f"Closed {quantity} contracts of position {position_id} (Entry: {pos.entry_price:.4f}, Exit: {execution_price:.4f})"
        })
        
        if remove_position:
            del self.positions[position_id]
        else:
            pos.quantity -= quantity
            
        return True

    def _settle_expirations(self):
        """
        Checks all open option positions and settles those that have reached or passed their expiration.
        """
        expired_ids = []
        for pos_id, pos in self.positions.items():
            if self.current_date >= pos.expiry:
                # Settle contract
                intrinsic = pos.get_intrinsic_value(self.current_spot)
                
                # Check for assignment / called-away
                is_assignment = False
                is_called_away = False
                
                if pos.side == "short" and intrinsic > 0:
                    if pos.option_type == "put":
                        is_assignment = True
                    elif pos.option_type == "call":
                        is_called_away = True
                
                commission = self.commission_per_contract * pos.quantity
                
                if is_assignment:
                    # Cash-Secured Put Assignment: Buy stock at strike price
                    purchase_cost = pos.strike * 100.0 * pos.quantity
                    net_cash_flow = -purchase_cost - commission
                    self.cash += net_cash_flow
                    
                    new_shares = pos.quantity * 100
                    if self.shares > 0:
                        self.shares_entry_price = ((self.shares * self.shares_entry_price) + purchase_cost) / (self.shares + new_shares)
                    else:
                        self.shares_entry_price = pos.strike
                    self.shares += new_shares
                    
                    outcome = "ASSIGNED (ITM Put)"
                    notes = f"Put expired ITM. Assigned stock: bought {new_shares} shares at ${pos.strike:.2f}"
                elif is_called_away:
                    # Covered Call Called-Away: Sell stock at strike price
                    sale_revenue = pos.strike * 100.0 * pos.quantity
                    net_cash_flow = sale_revenue - commission
                    self.cash += net_cash_flow
                    
                    shares_sold = pos.quantity * 100
                    self.shares -= shares_sold
                    if self.shares <= 0:
                        self.shares = 0
                        self.shares_entry_price = 0.0
                        
                    outcome = "CALLED AWAY (ITM Call)"
                    notes = f"Call expired ITM. Stock called away: sold {shares_sold} shares at ${pos.strike:.2f}"
                else:
                    # Standard Cash Settlement for Long options or OTM short options
                    if pos.side == "long":
                        cash_flow = intrinsic * 100.0 * pos.quantity
                    else:
                        cash_flow = -intrinsic * 100.0 * pos.quantity
                    net_cash_flow = cash_flow - commission
                    self.cash += net_cash_flow
                    
                    outcome = "ITM" if intrinsic > 0 else "OTM"
                    notes = f"Expired {outcome}. Settled in cash at intrinsic value: {intrinsic:.4f}"
                
                expired_ids.append(pos_id)
                
                self.transaction_log.append({
                    'date': self.current_date,
                    'type': f'EXPIRY_{pos.side.upper()}_{pos.option_type.upper()}',
                    'symbol': f"{pos.strike}_{pos.option_type[0].upper()}_{pos.expiry.strftime('%m%d%y')}",
                    'strike': pos.strike,
                    'expiry': pos.expiry,
                    'quantity': pos.quantity,
                    'price': intrinsic if not (is_assignment or is_called_away) else pos.strike,
                    'commission': commission,
                    'cash_flow': net_cash_flow,
                    'spot': self.current_spot,
                    'option_type': pos.option_type,
                    'side': pos.side,
                    'notes': notes
                })
                
        for pos_id in expired_ids:
            del self.positions[pos_id]

    def _record_daily_history(self):
        """
        Appends the current daily portfolio state to history.
        """
        self.history.append({
            'Date': self.current_date,
            'Cash': self.cash,
            'StockValue': self.shares * self.current_spot,
            'OptionsValue': sum(pos.get_value() for pos in self.positions.values()),
            'PortfolioValue': self.portfolio_value,
            'Spot': self.current_spot,
            'MarginLocked': self.margin_locked,
            'OpenPositions': len(self.positions),
            'UnderlyingShares': self.shares,
            'Rate': self.current_rate
        })

    def run(self, df: pd.DataFrame, strategy) -> pd.DataFrame:
        """
        Executes the backtest simulation over the historical dataset.
        
        df: DataFrame with DatetimeIndex and columns:
            ['Open', 'High', 'Low', 'Close', 'Rate', 'ImpliedVolProxy']
        strategy: An initialized instance of a class inheriting from BaseStrategy
        """
        logger.info(f"Starting options backtest from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        
        # Reset state
        self.cash = self.initial_capital
        self.shares = 0
        self.shares_entry_price = 0.0
        self.positions = {}
        self.history = []
        self.transaction_log = []
        
        # Link engine to strategy
        strategy.engine = self
        strategy.initialize()
        
        for idx, (date, row) in enumerate(df.iterrows()):
            # 1. Update current daily market environment
            self.current_date = date
            self.current_spot = float(row['Close'])
            self.current_rate = float(row['Rate'])
            self.current_vol = float(row['ImpliedVolProxy'])
            
            # 2. Daily mark-to-market of existing positions
            for pos in self.positions.values():
                pos.mark_to_market(self.current_spot, self.current_rate, self.current_vol, self.current_date)
                
            # 3. Check and settle expired contracts (before strategy acts)
            self._settle_expirations()
            
            # 4. Check for margin call (portfolio value < maintenance margin)
            # If so, force close all positions to prevent negative balance
            if self.portfolio_value < 0:
                logger.error(f"MARGIN CALL / BANKRUPTCY on {date.strftime('%Y-%m-%d')}. Portfolio value fell below 0.")
                self._record_daily_history()
                break
                
            # 5. Execute Strategy on_data hook
            strategy.on_data(date, self.current_spot, self.current_rate, self.current_vol)
            
            # 6. Record end-of-day history
            self._record_daily_history()
            
        logger.info("Backtest complete.")
        return pd.DataFrame(self.history).set_index('Date')
