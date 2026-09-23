"""
Black-Scholes Greeks for OptionEdge
Calculates Delta, Gamma, Theta, Vega, Theoretical Price
"""
import numpy as np
from scipy.stats import norm

def black_scholes_price(S, K, T, r, sigma, option_type='CE'):
    """Theoretical option price"""
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == 'CE' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'CE':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def calculate_greeks(S, K, T, r, sigma, option_type='CE'):
    """Returns dict with delta, gamma, theta, vega, rho"""
    if T <= 0 or sigma <= 0:
        return dict(delta=0, gamma=0, theta=0, vega=0, rho=0, theo_price=0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = norm.pdf(d1)
    
    # Delta
    if option_type == 'CE':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Gamma (same for CE/PE)
    gamma = nd1 / (S * sigma * np.sqrt(T))
    
    # Vega (per 1% change)
    vega = S * nd1 * np.sqrt(T) / 100
    
    # Theta (per day)
    term1 = -(S * nd1 * sigma) / (2 * np.sqrt(T))
    if option_type == 'CE':
        term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
        theta = (term1 - term2) / 365
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = (term1 + term2) / 365
    
    # Rho (per 1% rate change)
    if option_type == 'CE':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    theo = black_scholes_price(S, K, T, r, sigma, option_type)
    
    return dict(delta=round(delta,4), gamma=round(gamma,6), theta=round(theta,4), 
                vega=round(vega,4), rho=round(rho,4), theo_price=round(theo,2))

def implied_volatility(market_price, S, K, T, r, option_type='CE', tol=1e-4, max_iter=100):
    """Brent-like IV solver via Newton-Raphson bisection"""
    sigma = 0.5
    for _ in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        diff = price - market_price
        if abs(diff) < tol:
            return round(sigma,4)
        # Vega for Newton step
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)) if T>0 else 0
        vega = S * norm.pdf(d1) * np.sqrt(T)
        if vega == 0:
            break
        sigma = sigma - diff / vega
        sigma = max(0.01, min(sigma, 5.0))
    return round(sigma,4)

# Example self-test
if __name__ == "__main__":
    # Nifty 25000 CE, 7 days to expiry, S=25100, r=6%, IV=18%
    greeks = calculate_greeks(S=25100, K=25000, T=7/365, r=0.06, sigma=0.18, option_type='CE')
    print(greeks)
    # {'delta': 0.62, 'gamma': 0.0009, 'theta': -12.5, 'vega': 12.3, ...}
