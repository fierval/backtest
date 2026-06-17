from backtester.strategy import BaseStrategy
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CashSecuredPutStrategy(BaseStrategy):
    def initialize(self):
        """
        Set up strategy parameters.
        """
        self.target_dte = 30
        self.strike_pct = 0.95         # 5% Out-of-the-money
        self.profit_target_pct = 0.50  # Take profit at 50% max gain
        
    def on_data(self, date: datetime, spot: float, rate: float, vol: float):
        # 1. Check open positions for early exit (Profit Target)
        open_pos_ids = list(self.open_positions.keys())
        for pos_id in open_pos_ids:
            pos = self.open_positions[pos_id]
            
            # Since this is a short position, profit is realized as the option price decreases
            if pos.side == "short":
                profit_pct = (pos.entry_price - pos.current_price) / pos.entry_price
                
                if profit_pct >= self.profit_target_pct:
                    logger.info(f"[{date.strftime('%Y-%m-%d')}] Profit target met ({profit_pct*100:.1f}%). Closing put position {pos.id} at {pos.current_price:.2f} (Entry: {pos.entry_price:.2f})")
                    self.close_position(pos_id)
        
        # 2. Entry logic: If we have no open positions, sell a new cash-secured put
        if len(self.open_positions) == 0:
            # Calculate strike price (rounded to nearest whole dollar)
            strike = round(spot * self.strike_pct)
            
            # Calculate expiration date
            expiry = date + timedelta(days=self.target_dte)
            
            # Calculate capital needed to secure 1 contract (Strike * 100)
            required_margin = strike * 100.0 * 1
            
            if self.available_funds >= required_margin:
                logger.info(f"[{date.strftime('%Y-%m-%d')}] Selling 30-day {self.strike_pct*100:.0f}% OTM Put strike {strike} at spot {spot:.2f} (DTE: {self.target_dte})")
                self.sell_put(strike, expiry, quantity=1)
            else:
                logger.warning(f"[{date.strftime('%Y-%m-%d')}] Insufficient funds to secure Put. Available: {self.available_funds:.2f}, Required: {required_margin:.2f}")
