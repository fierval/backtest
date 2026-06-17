import numpy as np
from scipy.stats import norm

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """
    Calculates the BSM theoretical price for a European option.
    
    Parameters:
    -----------
    S : float - Underlying price
    K : float - Strike price
    T : float - Time to maturity in years (T > 0)
    r : float - Annualized risk-free interest rate (decimal, e.g., 0.05)
    sigma : float - Annualized volatility (decimal, e.g., 0.20)
    option_type : str - "call" or "put"
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be either 'call' or 'put'")
        
    # Boundary cases
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
            
    if sigma <= 0:
        # Volatility is zero: option value is the discounted intrinsic value
        discount = np.exp(-r * T)
        if option_type == "call":
            return max(S - K * discount, 0.0)
        else:
            return max(K * discount - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
    return max(price, 0.0) # Option price cannot be negative

def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> dict:
    """
    Calculates the option Greeks: Delta, Gamma, Theta (daily), Vega (1% vol change), and Rho (1% interest rate change).
    
    Returns:
    --------
    dict: { 'delta': float, 'gamma': float, 'theta': float, 'vega': float, 'rho': float }
    """
    option_type = option_type.lower()
    greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
    
    if T <= 0:
        # At or post expiration, greeks are 0 or step functions
        if option_type == "call":
            greeks['delta'] = 1.0 if S > K else 0.0
        else:
            greeks['delta'] = -1.0 if S < K else 0.0
        return greeks

    if sigma <= 0:
        sigma = 1e-6 # Avoid division by zero in Greeks formulas

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    
    # 1. Delta
    if option_type == "call":
        greeks['delta'] = cdf_d1
    else:
        greeks['delta'] = cdf_d1 - 1.0
        
    # 2. Gamma (Same for call and put)
    greeks['gamma'] = pdf_d1 / (S * sigma * np.sqrt(T))
    
    # 3. Vega (Same for call and put; derivative wrt sigma. 
    # Frequently scaled to represent a 1% change in vol: / 100)
    greeks['vega'] = (S * np.sqrt(T) * pdf_d1) / 100.0
    
    # 4. Theta (Annualized derivative wrt time; scaled to daily decay: / 365)
    if option_type == "call":
        theta_ann = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * cdf_d2
    else:
        theta_ann = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    greeks['theta'] = theta_ann / 365.0
    
    # 5. Rho (Derivative wrt interest rate; scaled for 1% change: / 100)
    if option_type == "call":
        rho_ann = K * T * np.exp(-r * T) * cdf_d2
    else:
        rho_ann = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    greeks['rho'] = rho_ann / 100.0
    
    return greeks

def implied_volatility(target_price: float, S: float, K: float, T: float, r: float, 
                       option_type: str = "call", max_iterations: int = 100, 
                       tolerance: float = 1e-6) -> float:
    """
    Solves for the implied volatility of a European option using Newton-Raphson 
    with a Bisection search fallback.
    
    Returns:
    --------
    float: Implied volatility as a decimal (e.g., 0.25). Returns NaN if solver fails to converge.
    """
    option_type = option_type.lower()
    
    # Quick bounds checks
    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if target_price <= intrinsic:
        return 0.0 # Option is at intrinsic value or underpriced
        
    # We will use Newton-Raphson: sigma_new = sigma_old - (BS_price - target_price) / Vega
    # Starting guess (20%)
    sigma = 0.20
    
    for _ in range(max_iterations):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        diff = price - target_price
        
        if abs(diff) < tolerance:
            return sigma
            
        # Vega is standard derivative wrt sigma (unscaled by 100)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)
        
        # If vega is too small, Newton-Raphson fails, switch to Bisection
        if vega < 1e-4:
            break
            
        sigma = sigma - diff / vega
        
        # Keep guess in reasonable range
        if sigma <= 0 or sigma > 5.0:
            break
            
    # Fallback: Bisection Method
    low_vol = 0.0001
    high_vol = 5.0
    
    # Check if high_vol is enough, otherwise scale up
    if black_scholes_price(S, K, T, r, high_vol, option_type) < target_price:
        return np.nan # Underpriced or invalid inputs
        
    for _ in range(max_iterations):
        mid_vol = 0.5 * (low_vol + high_vol)
        price = black_scholes_price(S, K, T, r, mid_vol, option_type)
        diff = price - target_price
        
        if abs(diff) < tolerance:
            return mid_vol
            
        if diff > 0:
            high_vol = mid_vol
        else:
            low_vol = mid_vol
            
    return mid_vol # Best approximation

if __name__ == "__main__":
    # Unit tests and verification
    S_test = 100.0
    K_test = 100.0
    T_test = 30.0 / 365.0 # 30 days
    r_test = 0.05 # 5%
    sigma_test = 0.20 # 20%
    
    c_price = black_scholes_price(S_test, K_test, T_test, r_test, sigma_test, "call")
    p_price = black_scholes_price(S_test, K_test, T_test, r_test, sigma_test, "put")
    
    print(f"BSM Call Price (S=100, K=100, T=30d, r=5%, vol=20%): {c_price:.4f}")
    print(f"BSM Put Price (S=100, K=100, T=30d, r=5%, vol=20%): {p_price:.4f}")
    
    c_greeks = calculate_greeks(S_test, K_test, T_test, r_test, sigma_test, "call")
    print("\nCall Greeks:")
    for k, v in c_greeks.items():
        print(f"  {k.capitalize()}: {v:.6f}")
        
    solved_vol = implied_volatility(c_price, S_test, K_test, T_test, r_test, "call")
    print(f"\nSolved Implied Volatility for Call Price {c_price:.4f}: {solved_vol*100:.2f}%")
